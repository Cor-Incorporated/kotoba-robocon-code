"""サーバー側実行計画。LLM由来の数値は存在しない（全てControllerProfile由来 — A01）。"""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from kotoba_contracts.intent import IdStr

PlanKind = Literal["walk", "motion_program", "conversational_hold"]
"""`motion_program` はmarker目標を持たない順序付き動作step列
（旋回+歩行等）。`conversational_hold` は実行計画を生成しない対話応答。"""


class MotionStepSpec(BaseModel):
    """実行planの1step。数値は解決済みの実値（m または rad）で、
    決定論parser/profile由来のみ — LLMが生値を出す経路は無い (A01)。
    targetの意味はaction別: translate=距離m, turn=角度rad。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    action: Literal["translate", "turn", "jog"]
    direction: Literal["forward", "back", "left", "right", "around"]
    # jog は継続動作のため target=0（停止はrunner側のheartbeat・
    # 境界・期限が担う）。translate/turnの正値性はvalidatorが再検査する。
    target: float = Field(ge=0, le=4.0)
    # 歩容 — 非walkは前進のtranslate/jogのみ（validatorが再検査）。
    # runは能力検収が別途必要（capabilities.run=verifiedの環境のみ実行可）。
    pace: Literal["walk", "fast_walk", "run"] = "walk"
    action_key: str = Field(default="", max_length=64)
    label: str = Field(default="", max_length=128)


class ControllerProfile(BaseModel):
    """運動可能範囲の単一正本（実測校正に基づき凍結される。LLMは関与しない）。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    name: str = Field(min_length=1, max_length=64)
    max_linear_mps: float = Field(gt=0)
    max_yaw_radps: float = Field(gt=0)
    max_duration_s: float = Field(gt=0)


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    session_id: IdStr
    round_id: IdStr
    plan_id: IdStr
    world_version: int = Field(ge=1)
    goal_target_id: Optional[IdStr]
    avoid_zone_ids: List[IdStr] = Field(default_factory=list, max_length=8)
    kind: PlanKind
    stop_policy: Literal["normal_stop"] = "normal_stop"
    steps: List[MotionStepSpec] = Field(default_factory=list, max_length=4)
    profile: ControllerProfile
    created_monotonic: float = Field(ge=0)

    def to_canonical_dict(self) -> dict:
        """承認hashの対象となる許可fieldのみの辞書。key順はcanonical化で吸収。"""
        return {
            "avoid_zone_ids": list(self.avoid_zone_ids),
            "created_monotonic": self.created_monotonic,
            "goal_target_id": self.goal_target_id,
            "kind": self.kind,
            "plan_id": self.plan_id,
            "profile": {
                "max_duration_s": self.profile.max_duration_s,
                "max_linear_mps": self.profile.max_linear_mps,
                "max_yaw_radps": self.profile.max_yaw_radps,
                "name": self.profile.name,
            },
            "round_id": self.round_id,
            "session_id": self.session_id,
            "steps": [s.model_dump() for s in self.steps],
            "stop_policy": self.stop_policy,
            "world_version": self.world_version,
        }


__all__ = ["ControllerProfile", "ExecutionPlan", "MotionStepSpec", "PlanKind"]
