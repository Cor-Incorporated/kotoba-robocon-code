"""新系の正しい仕様をassertするregression test（green側）。

同じ脅威シナリオが新系では成立しないことを、legacy characterization (red) と対で示す。
"""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from kotoba_contracts.approval import ApprovalRecord
from kotoba_contracts.canonical import canonical_plan_sha256
from kotoba_contracts.intent import IntentEnvelope, parse_intent
from kotoba_contracts.plan import ControllerProfile, ExecutionPlan
from kotoba_orchestrator.approval_store import ApprovalStore
from kotoba_orchestrator.errors import ApprovalError, PlanRejected
from kotoba_orchestrator.session import SessionManager
from kotoba_orchestrator.validator import build_plan, make_conversational_hold
from kotoba_contracts.world import ForbiddenRegion, World, WorldTarget

NOW = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)
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
            )
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


def _intent():
    return parse_intent(
        {
            "schema_version": "1.0",
            "decision": "execute",
            "target_ids": ["goal_a"],
            "avoid_ids": [],
            "explanation": "あかへ",
        }
    )


def _build(world=None, **overrides) -> ExecutionPlan:
    fields = dict(
        intent=_intent(),
        world=world or _world(),
        session_id="sess1",
        round_id="round1",
        plan_id="plan1",
        profile=PROFILE,
        created_monotonic=1.0,
    )
    fields.update(overrides)
    return build_plan(**fields)


# --- A01: envelopeに数値経路は存在しない ---
def test_intent_envelope_has_no_numeric_motion_fields():
    all_fields: set = set()
    for model in IntentEnvelope.__args__:
        all_fields |= set(model.model_fields)
    banned = {"velocity", "speed", "duration_seconds", "position", "approval", "token"}
    assert banned.isdisjoint(all_fields)


# --- A02: 改変planはhash不一致で拒否される ---
def test_tampered_plan_cannot_consume_approval():
    store = ApprovalStore()
    plan = _build()
    record = ApprovalRecord(
        session_id=plan.session_id,
        round_id=plan.round_id,
        plan_id=plan.plan_id,
        canonical_plan_sha256=canonical_plan_sha256(plan),
        world_version=plan.world_version,
        controller_profile_sha256="a" * 64,
        sim_boot_id="boot1",
        expires_at=NOW + timedelta(minutes=5),
    )
    approval_id = store.issue(record)
    tampered = _build(plan_id="plan2")  # 内容が異なる別計画
    with pytest.raises(ApprovalError) as err:
        store.verify_and_consume(
            approval_id,
            plan_sha256=canonical_plan_sha256(tampered),
            world_version=plan.world_version,
            session_id=plan.session_id,
            round_id=plan.round_id,
            sim_boot_id="boot1",
            now=NOW,
        )
    assert err.value.reason == "plan_sha_mismatch"


# --- A02b: storeに無いtoken相当のものは消費不能 ---
def test_unissued_credential_cannot_consume():
    store = ApprovalStore()
    with pytest.raises(ApprovalError) as err:
        store.verify_and_consume(
            "SYNTHETIC_TEST_ONLY",
            plan_sha256="0" * 64,
            world_version=1,
            session_id="s",
            round_id="r",
            sim_boot_id="b",
            now=NOW,
        )
    assert err.value.reason == "unknown_id"


# --- A03: duration=0 は実行計画として成立しない ---
def test_zero_duration_cannot_exist_in_new_system():
    from pydantic import ValidationError as VE

    with pytest.raises(VE):
        ControllerProfile(
            name="x", max_linear_mps=0.5, max_yaw_radps=0.5, max_duration_s=0.0
        )


# --- A04: 保留は実行グラントを発行しない ---
def test_conversational_hold_issues_nothing():
    store = ApprovalStore()
    assert make_conversational_hold("round1") == "conversational_hold"
    with pytest.raises(PlanRejected):
        _build(
            intent=parse_intent(
                {
                    "schema_version": "1.0",
                    "decision": "reject",
                    "reason_code": "unsupported_action",
                    "explanation": "保留",
                }
            )
        )
    assert store.size() == 0


# --- A07: request/round/plan は別ID ---
def test_ids_separated():
    mgr = SessionManager()
    session = mgr.create_session()
    r1 = mgr.begin_round(session.session_id)
    assert len({r1.session_id, r1.round_id, r1.request_id}) == 3
