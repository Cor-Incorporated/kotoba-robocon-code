"""立位ready・転倒ラッチ・通常停止・採点（characterization test_09/10/12/13 の green側）。"""

import pytest

from kotoba_harness.errors import HarnessError
from kotoba_harness.trial import (
    FALL_HEIGHT_M,
    FallLatch,
    NormalStopPolicy,
    Phase,
    ReadyGate,
    Scorer,
    TiltMonitor,
    TrialFSM,
    up_vector_tilt_deg,
)


class Sample:
    def __init__(self, pos, vel=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)):
        self.position = pos
        self.velocity = vel
        self.quaternion_wxyz = quat


# --- 姿勢の数学（A10準拠） ---
def test_pure_yaw_has_zero_tilt():
    import math

    # yaw 90度: (cos45, 0, 0, sin45)
    assert up_vector_tilt_deg(
        (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    ) == pytest.approx(0.0, abs=1e-9)


def test_minus_quat_same_tilt():
    assert up_vector_tilt_deg((1.0, 0.0, 0.0, 0.0)) == pytest.approx(
        up_vector_tilt_deg((-1.0, 0.0, 0.0, 0.0))
    )


def test_fallen_has_large_tilt_or_low_height():
    assert up_vector_tilt_deg((0.0, 1.0, 0.0, 0.0)) == pytest.approx(180.0)


# --- test_09 の対: 転立位から歩行を開始しない ---
def test_ready_gate_never_opens_when_fallen():
    gate = ReadyGate()
    fallen = Sample((0.0, 0.0, 0.12))
    for t in range(100):
        assert gate.observe(fallen, float(t), None) is False
    assert gate.ready_latched is False


def test_ready_requires_continuous_stand_window():
    gate = ReadyGate(hold_s=2.0)
    standing = Sample((0.0, 0.0, 0.82))
    assert gate.observe(standing, 0.0, None) is False
    assert gate.observe(standing, 1.0, None) is False
    assert gate.observe(standing, 2.0, None) is True  # 2秒連続で成立
    # 成立後の姿勢崩れはlatchを無効化する
    gate.observe(Sample((0.0, 0.0, 0.12)), 3.0, None)
    assert gate.ready_latched is False


# --- test_10 の対: 転倒tailはPASSしない ---
def test_scorer_rejects_fallen_tail_even_with_zero_xy_error():
    scorer = Scorer()
    result = scorer.verdict(
        ready_latched=True,
        fall_category=None,
        tail_ok=False,
        tail_held_s=0.0,
        final_sample=Sample((0.5, 0.0, 0.12)),  # XY誤差0でも z=0.07m は転倒
        target_xy=(0.5, 0.0),
    )
    assert result["verdict"] == "FAIL"
    assert any("posture" in r for r in result["reasons"])


def test_scorer_rejects_when_fall_latched():
    scorer = Scorer()
    result = scorer.verdict(
        ready_latched=True,
        fall_category="walk_fall",
        tail_ok=True,
        tail_held_s=5.0,
        final_sample=Sample((0.5, 0.0, 0.82)),
        target_xy=(0.5, 0.0),
    )
    assert result["verdict"] == "FAIL"


def test_scorer_passes_only_full_conditions():
    scorer = Scorer()
    result = scorer.verdict(
        ready_latched=True,
        fall_category=None,
        tail_ok=True,
        tail_held_s=5.0,
        final_sample=Sample((0.5, 0.0, 0.82), (0.0, 0.0, 0.0)),
        target_xy=(0.5, 0.0),
    )
    assert result["verdict"] == "PASS"
    assert result["reasons"] == []


# --- 転倒ラッチ: 準備〜tailを通じて監視し、種別を分ける ---
def test_fall_latch_categories():
    latch = FallLatch()
    latch.observe(Phase.READY_WAIT.value, Sample((0.0, 0.0, 0.12)))
    assert latch.fallen and latch.category == "prep_fall"

    latch2 = FallLatch()
    latch2.observe(Phase.TAIL.value, Sample((0.0, 0.0, 0.12)))
    assert latch2.category == "tail_fall"

    latch3 = FallLatch()
    latch3.observe(Phase.WALK.value, Sample((0.0, 0.0, 0.12)))
    assert latch3.category == "walk_fall"

    # 2回目の観測で上書きしない（ラッチ）
    latch3.observe(Phase.TAIL.value, Sample((0.0, 0.0, 0.82)))
    assert latch3.phases == ["walk"]


def test_fall_latch_uses_tilt_not_only_height():
    latch = FallLatch()
    latch.observe(Phase.WALK.value, Sample((0.0, 0.0, 0.30), quat=(0.0, 1.0, 0.0, 0.0)))
    assert latch.fallen  # 高さ正常でも倒立なら転倒


# --- test_12 の対: 通常停止は減速→静止確認→遷移の順 ---
def test_normal_stop_requires_stillness_before_transition():
    policy = NormalStopPolicy(stop_speed_mps=0.05, still_confirm_s=1.0)
    moving = Sample((0.0, 0.0, 0.82), (0.4, 0.0, 0.0))
    stopped = Sample((0.0, 0.0, 0.82), (0.0, 0.0, 0.0))
    assert policy.decelerate_done(moving, 0.0, 0.0) is False
    assert policy.decelerate_done(stopped, 1.0, 0.0) is False  # まだ静止確認中
    assert policy.decelerate_done(stopped, 2.5, 0.0) is True  # 1秒以上静止で完了


def test_normal_stop_times_out_when_never_stopping():
    policy = NormalStopPolicy(decel_timeout_s=1.0)
    moving = Sample((0.0, 0.0, 0.82), (0.4, 0.0, 0.0))
    with pytest.raises(HarnessError) as err:
        policy.decelerate_done(moving, 5.0, 0.0)
    assert err.value.reason == "deceleration_timeout"


# --- test_13 の対: 校正と評価の分離はmanifestのpurposeで強制 ---
def test_fsm_transitions_are_legal_only():
    fsm = TrialFSM()
    fsm.transition(Phase.READY_WAIT)
    fsm.transition(Phase.WALK)
    fsm.transition(Phase.DECELERATING)
    fsm.transition(Phase.SETTLING)
    fsm.transition(Phase.TAIL)
    fsm.transition(Phase.DONE)
    with pytest.raises(HarnessError):
        fsm.transition(Phase.WALK)  # DONEからの逆行は違法


def test_walk_done_illegal_abort_path_legal():
    """run e2e7d929 の二次欠陥回帰: 失敗経路で WALK→DONE を直接遷移すると
    HarnessError が一次原因（no_progress等）を上書きした。
    失敗時の合法経路は WALK→ABORT→DONE。"""
    fsm = TrialFSM()
    fsm.transition(Phase.READY_WAIT)
    fsm.transition(Phase.WALK)
    with pytest.raises(HarnessError):
        fsm.transition(Phase.DONE)  # 直接DONEは違法
    # 失敗経路の合法形 — ABORTを経由してDONEへ
    fsm.transition(Phase.ABORT)
    fsm.transition(Phase.DONE)
