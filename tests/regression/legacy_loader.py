"""legacy snapshotの読み込み。docs/handoff/evidence/baseline_snapshot は改変禁止。"""

import sys
from pathlib import Path

SNAPSHOT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "handoff"
    / "evidence"
    / "baseline_snapshot"
)

_loaded = False


def load_legacy():
    """namespace packageとして旧モジュールを読む（38probeと同一技法）。"""
    global _loaded
    if not _loaded:
        path = str(SNAPSHOT)
        if path not in sys.path:
            sys.path.insert(0, path)
        _loaded = True
    from engineai_thor_sidecar.application.approve import approve_plan
    from engineai_thor_sidecar.application.validate_plan import validate_plan
    from engineai_thor_sidecar.domain.action_plan import ActionPlan
    from engineai_thor_sidecar.domain.safety import Limits
    from engineai_thor_sidecar.infrastructure.lcm_gamepad import sequence_for_plan

    return {
        "ActionPlan": ActionPlan,
        "approve_plan": approve_plan,
        "validate_plan": validate_plan,
        "sequence_for_plan": sequence_for_plan,
        "SIM": Limits(max_linear_velocity_mps=0.85),
    }


def legacy_plan(vx=0.6, duration=2.5, action="walk", state="walk", approved=True):
    """probe testと同じ形状の旧ActionPlanを構築する。値は全て合成。"""
    legacy = load_legacy()
    data = dict(
        schema_version="1.0.0",
        instruction_id="synthetic-wp02",
        locale="ja-JP",
        instruction_text="合成試験指示",
        action=action,
        target_motion_state=state,
        duration_seconds=duration,
        requires_approval=True,
        safety=dict(person_detected=False, obstacle_in_path=False, reason="synthetic"),
        body_vel_cmd=dict(
            linear_velocity_x=vx, linear_velocity_y=0.0, yaw_velocity=0.0
        ),
    )
    plan = legacy["ActionPlan"].model_validate(data)
    if approved:
        plan = legacy["approve_plan"](
            plan, "synthetic-operator", token="SYNTHETIC_TEST_ONLY"
        )
    return plan
