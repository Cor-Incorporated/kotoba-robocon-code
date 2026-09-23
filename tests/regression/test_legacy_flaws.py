"""旧実装の欠陥が存在することを記録する characterization test。

これらがPASSする = 欠陥の再現（red側の証拠）。旧snapshotは1行も改変していない。
修正後にこのファイルを削除・緩和してはならない（38probeと同じ現状記録の性質）。
"""

import pytest

from tests.regression.legacy_loader import legacy_plan, load_legacy


# --- A02: 承認済みplanの改変がJSON再検証を通ってしまう ---
def test_legacy_approval_survives_plan_tampering():
    legacy = load_legacy()
    plan = legacy_plan(vx=0.6)
    data = plan.model_dump()
    data["body_vel_cmd"]["linear_velocity_x"] = 0.8  # 承認後に内容を改変
    changed = legacy["ActionPlan"].model_validate(data)
    result = legacy["validate_plan"](changed, legacy["SIM"])
    assert result.accepted, "旧実装では改変planが承認付きで通過する（欠陥の再現）"


# --- A02b: 発行されていないtokenでも validator は通る ---
def test_legacy_synthetic_token_accepted():
    legacy = load_legacy()
    data = legacy_plan(approved=False).model_dump()
    data["approval"] = {
        "status": "approved",
        "approver": None,
        "token": "SYNTHETIC_TEST_ONLY",
    }
    changed = legacy["ActionPlan"].model_validate(data)
    assert legacy["validate_plan"](changed, legacy["SIM"]).accepted


# --- A03: duration=0 で非ゼロ指令フレームが生成される ---
def test_legacy_zero_duration_still_emits_one_nonzero_frame():
    legacy = load_legacy()
    plan = legacy_plan(vx=0.6, duration=0.0)
    assert legacy["validate_plan"](plan, legacy["SIM"]).accepted
    frames = legacy["sequence_for_plan"](plan)
    nonzero = sum(1 for f in frames if any(f.analog_states))
    assert nonzero == 1, "0秒移動でも非ゼロ指令が1フレーム出る（欠陥の再現）"


# --- A06: stand+非ゼロ速度がvalidatorを通る（encodeで捨てられるだけ） ---
def test_legacy_nonwalk_nonzero_velocity_passes_gate():
    legacy = load_legacy()
    plan = legacy_plan(vx=0.1, action="stand", state="pd_stand")
    assert legacy["validate_plan"](plan, legacy["SIM"]).accepted, (
        "action/field不整合がvalidatorを通る（欠陥の再現）"
    )
