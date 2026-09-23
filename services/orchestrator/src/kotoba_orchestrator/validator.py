"""意図検証と計画構築。数値はprofile由来のみ。無効出力の黙って補完はしない。"""

import math

from enum import Enum
from typing import Union

from kotoba_contracts.canonical import canonical_plan_sha256
from kotoba_contracts.intent import (
    IntentClarify,
    IntentExecute,
    IntentReject,
    parse_intent,
)
from kotoba_contracts.plan import (
    ControllerProfile,
    ExecutionPlan,
    MotionStepSpec,
    PlanKind,
)
from kotoba_contracts.world import World

from kotoba_orchestrator.errors import PlanRejected

StopKindValue = str


class StopKind(str, Enum):
    """停止は3種を厳密に分離する (A05)。"""

    NORMAL_STOP = "normal_stop"  # 減速→観測した静止と姿勢→pd_stand
    MANAGED_ABORT = "managed_abort"  # モデル/通信障害時の管理された中断
    SIM_PAUSE = "sim_pause"  # シミュレーション所有者へのpause（非常停止性能ではない）


PlanKindValue = PlanKind

_DANGEROUS_HINTS = ("velocity", "duration", "approval", "token", "position_m")


def parse_envelope(raw: dict) -> Union[IntentExecute, IntentClarify, IntentReject]:
    """参加者APIが受けた生dictを検証。特権fieldはextra=forbidでここに届く前に拒否。"""
    return parse_intent(raw)


def validate_intent(
    intent: Union[IntentExecute, IntentClarify, IntentReject], world: World
) -> None:
    """ID存在性・矛盾を検査する。成功時はNone。失敗時はPlanRejected。補完はしない。"""
    if isinstance(intent, IntentExecute):
        if intent.target_ids[0] in intent.avoid_ids:
            raise PlanRejected("contradictory_constraints")
        try:
            world.target(intent.target_ids[0])
        except KeyError:
            raise PlanRejected("unknown_target") from None
        for zone_id in intent.avoid_ids:
            try:
                world.region(zone_id)
            except KeyError:
                raise PlanRejected("unknown_zone") from None
    elif isinstance(intent, IntentClarify):
        for candidate in intent.candidate_target_ids:
            try:
                world.target(candidate)
            except KeyError:
                raise PlanRejected("unknown_target") from None
    else:
        return  # IntentReject はそのまま意味上の応答として扱う


def build_plan(
    intent: Union[IntentExecute, IntentClarify, IntentReject],
    world: World,
    *,
    session_id: str,
    round_id: str,
    plan_id: str,
    profile: ControllerProfile,
    created_monotonic: float,
) -> ExecutionPlan:
    """execute以外は実行計画を生成しない。数値はprofileのみから採る (A01)。"""
    validate_intent(intent, world)
    if not isinstance(intent, IntentExecute):
        raise PlanRejected("not_executable_decision")
    if profile.max_duration_s <= 0:
        # duration<=0 は pydantic(gt=0)以前に明示拒否する (A03)
        raise PlanRejected("zero_duration")
    return ExecutionPlan(
        session_id=session_id,
        round_id=round_id,
        plan_id=plan_id,
        world_version=world.world_version,
        goal_target_id=intent.target_ids[0],
        avoid_zone_ids=list(intent.avoid_ids),
        kind="walk",
        profile=profile,
        created_monotonic=created_monotonic,
    )


_STEP_MAX = {"translate": 2.0, "turn": math.pi}
_TURN_DIRS = {"left", "right", "around"}
_TRANSLATE_DIRS = {"forward", "back", "left", "right"}


_VALID_PACES = {"walk", "fast_walk", "run"}


def _valid_pace(action, direction, pace) -> bool:
    """非walk pace は前進のtranslate/jogのみ — 旋回・横・後退への
    一律増幅は plan 生成段階でも拒否する。"""
    if pace not in _VALID_PACES:
        return False
    if pace == "walk":
        return True
    return action in ("translate", "jog") and direction == "forward"


def build_motion_plan(
    parsed_steps: list,
    world: World,
    *,
    session_id: str,
    round_id: str,
    plan_id: str,
    profile: ControllerProfile,
    created_monotonic: float,
) -> ExecutionPlan:
    """決定論parserのstep列を実行planへ。数値はparserの許可表由来の
    実値のみ受理し、ここで再検査する（LLM経路は存在しない — A01）。

    parsed_steps: gameparse出力 [{action, dir, m|deg, action_key, label}]
    """
    specs: list[MotionStepSpec] = []
    total_m = 0.0
    for s in parsed_steps:
        if len(specs) >= 4:
            raise PlanRejected("program_too_many_steps")
        action = s.get("action")
        direction = s.get("dir")
        pace = s.get("pace") or "walk"
        if not _valid_pace(action, direction, pace):
            raise PlanRejected("invalid_step_pace")
        if action == "translate":
            target = float(s.get("m") or 0.0)
            valid_dir = direction in _TRANSLATE_DIRS
        elif action == "turn":
            deg = float(s.get("deg") or 0.0)
            # degの範囲は丸め前に検査する（270°等をπへ飽和して通さない）
            if not math.isfinite(deg) or deg > 180.0 + 1e-9:
                raise PlanRejected("step_out_of_bounds")
            target = deg * math.pi / 180.0
            # round(...,6)がπを僅かに超える値を作らないよう飽和
            # （180°→3.141593がrunner側検証でbad_turn_target化した実害）
            target = min(target, math.pi)
            valid_dir = direction in _TURN_DIRS
        elif action == "jog":
            # 継続移動 — 距離目標なし。実行時の停止はrunnerの
            # heartbeat/境界/jog期限が担う。
            target = 0.0
            valid_dir = direction in _TRANSLATE_DIRS
        else:
            raise PlanRejected("unknown_step_action")
        if not valid_dir:
            raise PlanRejected("invalid_step_direction")
        if action != "jog" and (
            not math.isfinite(target)
            or not (0.0 < target <= _STEP_MAX[action] + 1e-9)
        ):
            raise PlanRejected("step_out_of_bounds")
        if action == "translate":
            total_m += target
        specs.append(
            MotionStepSpec(
                action=action,
                direction=direction,
                target=round(target, 6),
                pace=pace,
                action_key=str(s.get("action_key") or "")[:64],
                label=str(s.get("label") or "")[:128],
            )
        )
    if not specs:
        raise PlanRejected("empty_program")
    if total_m > 2.0 + 1e-9:
        raise PlanRejected("program_distance_too_long")
    if profile.max_duration_s <= 0:
        raise PlanRejected("zero_duration")
    return ExecutionPlan(
        session_id=session_id,
        round_id=round_id,
        plan_id=plan_id,
        world_version=world.world_version,
        goal_target_id=None,
        avoid_zone_ids=[],
        kind="motion_program",
        steps=specs,
        profile=profile,
        created_monotonic=created_monotonic,
    )


def make_conversational_hold(round_id: str) -> str:
    """会話上の保留 (A04)。

    旧実装の `hold`（デジタル全0フレーム1枚）とは意味が異なる。
    保留は実行計画・承認・グラントを一切生成しない対話結果であり、
    SDK状態をidleへ遷移させるものでも送信を伴うものでもない。
    戻り値はUI表示用のラベルのみ。
    """
    del round_id
    return "conversational_hold"
