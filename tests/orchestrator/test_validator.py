"""意図検証・計画構築の正しい仕様 (A01/A03/A04/A05/A06/A07 対応)。"""

import pytest

from kotoba_contracts.canonical import canonical_plan_sha256
from kotoba_contracts.intent import parse_intent
from kotoba_contracts.plan import ControllerProfile, ExecutionPlan
from kotoba_contracts.world import ForbiddenRegion, World, WorldTarget
from kotoba_orchestrator.errors import PlanRejected
from kotoba_orchestrator.session import SessionManager
from kotoba_orchestrator.validator import (
    StopKind,
    build_plan,
    make_conversational_hold,
    validate_intent,
)

PROFILE = ControllerProfile(
    name="sim_profile", max_linear_mps=0.85, max_yaw_radps=0.8, max_duration_s=3.0
)


def _world() -> World:
    return World(
        world_version=1,
        scene_sha256="0" * 64,
        targets=[
            WorldTarget(
                id="goal_a", label="あか", position_m=[1.0, 0.0, 0.0], radius_m=0.25
            ),
            WorldTarget(
                id="goal_b", label="あお", position_m=[0.0, 1.0, 0.0], radius_m=0.25
            ),
        ],
        forbidden_regions=[
            ForbiddenRegion(
                id="zone_x",
                label="みずたまり",
                polygon_xy_m=[[0, 0], [1, 0], [1, 1], [0, 1]],
            )
        ],
        capability_profile_sha256="1" * 64,
    )


def _intent_execute(**overrides):
    raw = {
        "schema_version": "1.0",
        "decision": "execute",
        "target_ids": ["goal_a"],
        "avoid_ids": [],
        "explanation": "あかへ進む",
    }
    raw.update(overrides)
    return parse_intent(raw)


def _build(intent, world, session_id="sess", round_id="round", plan_id="plan"):
    return build_plan(
        intent,
        world,
        session_id=session_id,
        round_id=round_id,
        plan_id=plan_id,
        profile=PROFILE,
        created_monotonic=1.0,
    )


# --- A03: duration=0 は明示拒否 ---
def test_zero_duration_profile_is_rejected_not_clamped():
    from pydantic import ValidationError

    # ControllerProfile生成時点でpydanticが拒否（gt=0）。実行計画にduration<=0は存在し得ない
    with pytest.raises(ValidationError):
        ControllerProfile(
            name="broken", max_linear_mps=0.5, max_yaw_radps=0.5, max_duration_s=0.0
        )
    # 防御深度: pydanticを経由しない生成経路に対してもサーバー側で明示拒否する
    bypass = ControllerProfile.model_construct(
        name="bypass", max_linear_mps=0.5, max_yaw_radps=0.5, max_duration_s=0.0
    )
    with pytest.raises(PlanRejected) as err:
        build_plan(
            _intent_execute(),
            _world(),
            session_id="s",
            round_id="r",
            plan_id="p",
            profile=bypass,
            created_monotonic=1.0,
        )
    assert err.value.reason == "zero_duration"


# --- A01: 速度はprofile由来のみ。LLM envelopeには数値経路が無い ---
def test_plan_velocity_bounds_come_from_profile_only():
    plan = _build(_intent_execute(), _world())
    assert plan.profile.max_linear_mps == PROFILE.max_linear_mps
    assert plan.profile.max_yaw_radps == PROFILE.max_yaw_radps
    assert plan.profile.max_duration_s == PROFILE.max_duration_s


# --- A04: 会話上の保留は実行計画・グラントを生成しない ---
def test_conversational_hold_creates_no_executable_plan():
    label = make_conversational_hold("round1")
    assert label == "conversational_hold"
    with pytest.raises(PlanRejected) as err:
        _build(
            parse_intent(
                {
                    "schema_version": "1.0",
                    "decision": "reject",
                    "reason_code": "unsupported_action",
                    "explanation": "保留は実行に変換されない",
                }
            ),
            _world(),
        )
    assert err.value.reason == "not_executable_decision"


# --- A05: 停止3種の分離 ---
def test_stop_kinds_are_distinct():
    assert StopKind.NORMAL_STOP is not StopKind.MANAGED_ABORT
    assert StopKind.MANAGED_ABORT is not StopKind.SIM_PAUSE
    assert {k.value for k in StopKind} == {"normal_stop", "managed_abort", "sim_pause"}


# --- FR-02/03: 存在検査・矛盾。補完はしない ---
def test_unknown_target_rejected():
    with pytest.raises(PlanRejected) as err:
        validate_intent(_intent_execute(target_ids=["goal_nope"]), _world())
    assert err.value.reason == "unknown_target"


def test_unknown_zone_rejected():
    with pytest.raises(PlanRejected) as err:
        validate_intent(_intent_execute(avoid_ids=["zone_nope"]), _world())
    assert err.value.reason == "unknown_zone"


def test_contradiction_target_in_avoid_rejected():
    with pytest.raises(PlanRejected) as err:
        validate_intent(
            _intent_execute(target_ids=["goal_a"], avoid_ids=["goal_a"]), _world()
        )
    assert err.value.reason == "contradictory_constraints"


def test_clarify_candidate_must_exist():
    raw = {
        "schema_version": "1.0",
        "decision": "clarify",
        "question": "どちらですか",
        "candidate_target_ids": ["goal_nope"],
    }
    with pytest.raises(PlanRejected) as err:
        validate_intent(parse_intent(raw), _world())
    assert err.value.reason == "unknown_target"


# --- A06: action/field不整合は新系では構造的に発生しない ---
def test_plan_kind_is_only_walk_or_hold_marker():
    plan = _build(_intent_execute(), _world())
    assert plan.kind == "walk"
    assert plan.stop_policy == "normal_stop"
    assert isinstance(plan, ExecutionPlan)


# --- A07: request/round/plan id は分離 ---
def test_ids_are_separated_per_round():
    mgr = SessionManager()
    session = mgr.create_session()
    round1 = mgr.begin_round(session.session_id)
    round2 = mgr.begin_round(session.session_id)
    assert round1.request_id != round1.round_id != round1.session_id
    assert round2.request_id != round2.round_id
    assert round1.round_id != round2.round_id
    plan1 = _build(_intent_execute(), _world(), round_id=round1.round_id, plan_id="pa")
    plan2 = _build(_intent_execute(), _world(), round_id=round2.round_id, plan_id="pb")
    assert canonical_plan_sha256(plan1) != canonical_plan_sha256(plan2)
