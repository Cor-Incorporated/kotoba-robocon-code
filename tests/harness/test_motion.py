"""MotionExecutor の単体試験 — 仮想力学で閉ループ挙動を検証。

stick→速度は定数ゲインの単純応答（残動なし）と、release後の指数減衰
（残動あり）の2種で、進捗到達・settling・順序・失敗分類を見る。
"""
import math
import sys

import pytest
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "services" / "harness" / "src"),
)

from kotoba_harness.motion import (
    ExecProfile,
    MotionExecutor,
    MotionStep,
    yaw_from_quat,
)


def _quat_of_yaw(yaw):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


class Sim:
    """簡易2D力学: stick→指令LPF（指数追従 τ）→速度=filtered×gain。

    実機の remote_command_lpf（cutoff 0.1Hz）相当の指令遅れを入れ、
    release後の残動が実在するモデルにする。速度応答自体は即応答。
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, tau=1.6, gain=(0.5, 0.2, 0.8)):
        self.x, self.y, self.yaw = x, y, yaw
        self.cx = self.cy = self.cz = 0.0  # filtered指令
        self.vx = self.vy = self.wz = 0.0
        self.tau = tau
        self.gain = gain

    def step(self, stick, dt):
        tx = ty = tz = 0.0
        if stick is not None:
            tx, ty, tz = stick
        k = min(1.0, dt / self.tau)
        self.cx += (tx - self.cx) * k
        self.cy += (ty - self.cy) * k
        self.cz += (tz - self.cz) * k
        vx, vy, wz = self.cx * self.gain[0], self.cy * self.gain[1], self.cz * self.gain[2]
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (vx * c - vy * s) * dt
        self.y += (vx * s + vy * c) * dt
        self.yaw += wz * dt
        self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi
        self.vx, self.vy, self.wz = vx, vy, wz

    def obs(self):
        return (
            (self.x, self.y, 0.8),
            _quat_of_yaw(self.yaw),
            (self.vx, self.vy, 0.0),
        )


# simのLPF τ=1.6 に整合する残動推定（実機ではThor校正で固定するprofile値）
P = ExecProfile(residual_k_lin=1.7, residual_k_yaw=1.7)


def _run(ex, sim, dt=0.05, max_s=60.0, drain_s=8.0):
    t = 0.0
    ex.begin(*sim.obs()[:2])
    sticks = []
    while not ex.done and t < max_s:
        stick = ex.tick(*sim.obs(), now=t)
        sim.step(stick, dt)
        sticks.append(stick)
        t += dt
    # 完了宣言後の残動を再生して真の静止位置を測る（実機と同じ計測方法）
    for _ in range(int(drain_s / dt)):
        sim.step(None, dt)
        t += dt
    return t, sticks


def test_translate_forward_1m():
    sim = Sim()
    ex = MotionExecutor([MotionStep("translate", "forward", 1.0)], P)
    t, _ = _run(ex, sim)
    assert ex.done and not ex.failed
    assert ex.prog.status == "completed"
    assert math.hypot(sim.x, sim.y) > 0.8  # 実際に1m近く前進した


def test_translate_right_keeps_heading():
    sim = Sim(yaw=0.3)
    ex = MotionExecutor([MotionStep("translate", "right", 0.25)], P)
    _run(ex, sim)
    assert ex.done and not ex.failed
    # 右方向(機体基準-yaw)の変位が正、前後変位は小さい
    c, s = math.cos(0.3), math.sin(0.3)
    fwd = sim.x * c + sim.y * s
    lat = -sim.x * s + sim.y * c
    assert lat < -0.15  # 右 = 機体-y
    assert abs(fwd) < 0.1
    assert abs(sim.yaw - 0.3) < math.radians(15)


def test_turn_right_90():
    sim = Sim()
    ex = MotionExecutor([MotionStep("turn", "right", math.radians(90))], P)
    _run(ex, sim)
    assert ex.done and not ex.failed
    assert abs(abs(sim.yaw) - math.pi / 2) < math.radians(10)
    assert sim.yaw < 0  # 右=負


def test_turn_around_180_across_pi():
    # ±π跨ぎ: 開始yaw=170°→180°右旋回は-170°付近へ（巻戻りで逆回転しない）
    sim = Sim(yaw=math.radians(170))
    ex = MotionExecutor([MotionStep("turn", "around", math.pi)], P)
    _run(ex, sim)
    assert ex.done and not ex.failed
    diff = abs((sim.yaw - math.radians(170) + math.pi) % (2 * math.pi) - math.pi)
    assert abs(math.pi - diff) < math.radians(10)


def test_compound_turn_then_walk_reanchors():
    sim = Sim()
    steps = [
        MotionStep("turn", "right", math.radians(90), action_key="turn_right_90"),
        MotionStep("translate", "forward", 1.0, action_key="fwd_1m"),
    ]
    ex = MotionExecutor(steps, P)
    _run(ex, sim, max_s=120)
    assert ex.done and not ex.failed
    # 右90°後の前方 = 機体-y方向へ1m
    assert sim.y < -0.8 and abs(sim.x) < 0.3
    evs = [e.get("event") for e in ex.events]
    assert evs.index("motion_step_done") < len(evs)


def test_settle_waits_for_stillness():
    # 残動を持つsim — stick解除後も速度が残り、完了は静止確認まで遅れる
    sim = Sim()
    ex = MotionExecutor(
        [MotionStep("translate", "forward", 0.5)],
        ExecProfile(residual_k_lin=0.0, residual_k_yaw=1.7),  # 残動推定なし→目標到達で即解除
    )
    t, sticks = _run(ex, sim)
    assert ex.done
    # 完了時点では静止している（settle条件を通過した）
    assert math.hypot(sim.vx, sim.vy) < 0.10


def test_abort_midway_no_resume():
    sim = Sim()
    ex = MotionExecutor(
        [
            MotionStep("turn", "right", math.radians(90)),
            MotionStep("translate", "forward", 1.0),
        ]
    )
    ex.begin(*sim.obs()[:2])
    t = 0.0
    # 旋回中にabort
    for _ in range(10):
        s = ex.tick(*sim.obs(), now=t)
        sim.step(s, 0.05)
        t += 0.05
    ex.abort("stop")
    assert ex.done and ex.abort_reason == "stop"
    # その後tickしてもstickは出ない
    for _ in range(50):
        s = ex.tick(*sim.obs(), now=t)
        assert s is None
        sim.step(s, 0.05)
        t += 0.05
    # step2（前進）は未実行
    assert abs(sim.x) < 0.2 and abs(sim.y) < 0.2


def test_no_progress_fails():
    # 全く動かない世界 — 進捗が伸びず no_progress で有限失敗
    class Frozen(Sim):
        def step(self, stick, dt):
            pass

    sim = Frozen()
    ex = MotionExecutor(
        [MotionStep("translate", "forward", 1.0)],
        ExecProfile(no_progress_s=1.0, apply_ticks=2),
    )
    t, _ = _run(ex, sim, max_s=10)
    assert ex.failed
    assert ex.prog.reason == "no_progress"


def test_step_timeout_fails():
    # 遅すぎる世界 — deadline超過で有限失敗
    class Slow(Sim):
        def step(self, stick, dt):
            super().step(
                None if stick is None else (stick[0] * 0.02, stick[1] * 0.02, stick[2] * 0.02),
                dt,
            )

    sim = Slow()
    ex = MotionExecutor(
        [MotionStep("translate", "forward", 1.0)],
        ExecProfile(step_deadline_s=2.0, no_progress_s=100),
    )
    _run(ex, sim, max_s=10)
    assert ex.failed
    assert ex.prog.reason == "step_timeout"


def test_turn_settle_waits_for_yaw_decay():
    """その場旋回のsettleは旋回速度で判定する（並進速度≈0で誤完了しない —
    実機で k 過大+指標誤りにより31°で完了した不具合の回帰）。

    residual_k_yaw=0 で目標角到達時にstick解除するが、simのLPF残動で
    旋回は継続する。完了宣言は旋回が止まってからでなければならない。
    （残動は受入±10°内に収める — それ以上はovershoot失敗の領域）
    """
    sim = Sim(tau=0.3, gain=(0.5, 0.2, 0.5))
    ex = MotionExecutor(
        [MotionStep("turn", "right", math.radians(90))],
        ExecProfile(residual_k_yaw=0.0),
    )
    t = 0.0
    ex.begin(*sim.obs()[:2])
    done_wz = None
    while not ex.done and t < 60.0:
        stick = ex.tick(*sim.obs(), now=t)
        sim.step(stick, 0.05)
        t += 0.05
        if ex.done:
            done_wz = abs(sim.wz)
    assert ex.done and not ex.failed
    # 完了時点で旋回速度は実質ゼロ（旧実装は並進速度で判定し即完了した）
    assert done_wz is not None and done_wz < 0.12


def test_undershoot_correction_converges():
    """残動推定が過大（早期解除で不足）でも、静止後の補正バーストで
    目標へ収束する — 実機で観測された不足→早期完了の回帰。

    simの真の残動係数は τ=1.6 — k=2.5 は過大推定で stick を早期解除し
    prog < target のまま settle へ。補正burstの反復で収束することを見る。
    """
    sim = Sim()
    ex = MotionExecutor(
        [MotionStep("translate", "forward", 1.0)],
        ExecProfile(residual_k_lin=2.5),  # 真値1.6より大きい→不足方向
    )
    t, _ = _run(ex, sim, max_s=120)
    assert ex.done and not ex.failed
    assert abs(sim.x - 1.0) < 0.12
    assert any(e.get("event") == "motion_correct" for e in ex.events)


def test_undershoot_exhaustion_fails_honestly():
    """補正回数を使い切っても不足なら正直に失敗する（黙って完了しない）。

    tau≈0の世界は残動がほぼ無い — 残動推定が残動分だけ早期解除して
    不足が残る状態を作り、補正なしでは undershoot 失敗になることを見る。
    """
    sim = Sim(tau=0.02)
    ex = MotionExecutor(
        [MotionStep("translate", "forward", 1.0)],
        ExecProfile(
            residual_k_lin=0.3,
            max_corrections=0,
            under_frac=0.02,
            under_abs_m=0.02,
        ),
    )
    _run(ex, sim, max_s=120)
    assert ex.done and ex.failed
    assert ex.prog.reason == "undershoot"


def test_progress_dict_shape():
    ex = MotionExecutor(
        [MotionStep("turn", "right", math.radians(90),
                    action_key="turn_right_90", label="右に90°向く")]
    )
    sim = Sim()
    ex.begin(*sim.obs()[:2])
    ex.tick(*sim.obs(), now=0.0)
    d = ex.prog.to_dict()
    assert d["action_key"] == "turn_right_90"
    assert d["label"] == "右に90°向く"
    assert d["step_count"] == 1
    assert d["progress"]["target_deg"] == 90.0


def test_yaw_from_quat_matches():
    for deg in (0, 45, 90, 135, 179, -45, -179):
        q = _quat_of_yaw(math.radians(deg))
        assert abs(math.degrees(yaw_from_quat(q)) - deg) < 0.5


def test_overshoot_fails_not_completed():
    """目標超過を completed へ進めない — 受入許容（90°±10°）を超えた
    旋回は failed になる。残動が強いsimで目標を大きく越えさせる。"""
    sim = Sim(tau=4.0, gain=(0.5, 0.2, 1.2))  # 強いyaw残動 → 90°を超過
    ex = MotionExecutor(
        [MotionStep("turn", "right", math.radians(90))],
        ExecProfile(residual_k_yaw=0.1),  # 残動を過小推定 → 解除が遅れ超過
    )
    _run(ex, sim, max_s=120)
    assert ex.done and ex.failed
    assert ex.prog.reason == "overshoot"
    # 実際に許容を超えて回ったことを確認（failの根拠が実測）
    assert abs(sim.yaw) > math.radians(100)


def test_translate_overshoot_fails_not_completed():
    """並進でも同様: 1m目標を大きく越えたまま completed にならない。"""
    sim = Sim(tau=5.0, gain=(0.9, 0.2, 0.8))
    ex = MotionExecutor(
        [MotionStep("translate", "forward", 1.0)],
        ExecProfile(residual_k_lin=0.1),
    )
    _run(ex, sim, max_s=120)
    assert ex.done and ex.failed
    assert ex.prog.reason == "overshoot"
    assert sim.x > 1.15


# ---- R4: jog（継続移動）と 180°π境界 --------------------------------------

def test_turn_180_rounded_pi_boundary():
    """run 54e75a37 回帰: round(π,6)=3.141593 はπを僅かに超えるが
    受理してπへ飽和する（bad_turn_target で FAIL_INTERNAL にならない）。"""
    ms = MotionStep("turn", "around", 3.141593)
    assert ms.target == math.pi
    # それを超える量は依然拒否
    with pytest.raises(ValueError):
        MotionStep("turn", "around", math.pi + 0.01)


def test_jog_step_contract():
    """jog stepは目標量を持たず、translate方向のみ。"""
    MotionStep("jog", "left", 0.0)
    with pytest.raises(ValueError):
        MotionStep("jog", "around", 0.0)
    with pytest.raises(ValueError):
        MotionStep("jog", "forward", 0.5)


def test_jog_runs_continuously_until_request_stop():
    """「前へ」=継続jog — 距離目標なしで走り続け、request_stopで正常終了。"""
    sim = Sim()
    ex = MotionExecutor([MotionStep("jog", "forward", 0.0)])
    ex.begin(*sim.obs()[:2])
    t = 0.0
    for _ in range(200):  # 10s — 2.0m上限を超えても止まらない
        s = ex.tick(*sim.obs(), now=t)
        sim.step(s, 0.05)
        t += 0.05
    assert not ex.done
    assert ex.jogging and sim.x > 1.8
    ex.request_stop("stop")
    while not ex.done and t < 60:
        s = ex.tick(*sim.obs(), now=t)
        sim.step(s, 0.05)
        t += 0.05
    assert ex.done and not ex.failed
    assert ex.prog.status == "completed" and ex.prog.reason == "stop"


def test_jog_deadline_is_graceful_completion():
    """jog絶対期限は失敗ではなく正常終了（stopped表示と区別可能）。"""
    sim = Sim()
    ex = MotionExecutor(
        [MotionStep("jog", "forward", 0.0)],
        ExecProfile(jog_deadline_s=2.0),
    )
    _run(ex, sim, max_s=30)
    assert ex.done and not ex.failed
    assert ex.prog.reason == "jog_timeout"
    assert sim.x > 0.5  # 実際に走行した


def test_jog_stops_before_arena_bound():
    """arena境界に達する前に減速停止へ入る（bound_r越えで止まらない）。"""
    sim = Sim()
    ex = MotionExecutor(
        [MotionStep("jog", "forward", 0.0)],
        bound_center=(0.0, 0.0), bound_r=1.0,
    )
    _run(ex, sim, max_s=30)
    assert ex.done and not ex.failed
    assert ex.prog.reason == "boundary"
    # 停止要求から残動分は進むが、境界の遥か先までは行かない
    assert math.hypot(sim.x, sim.y) < 1.0 + 0.5


def test_jog_lateral_moves_sideways():
    """「少し右」級の横移動が実際に横へ進む（e2e7d929の横stick非線形域の回帰）。"""
    sim = Sim()
    ex = MotionExecutor([MotionStep("jog", "right", 0.0)])
    ex.begin(*sim.obs()[:2])
    t = 0.0
    for _ in range(100):  # 5s
        s = ex.tick(*sim.obs(), now=t)
        sim.step(s, 0.05)
        t += 0.05
    ex.request_stop("stop")
    while not ex.done and t < 30:
        s = ex.tick(*sim.obs(), now=t)
        sim.step(s, 0.05)
        t += 0.05
    assert ex.done and not ex.failed
    # 右 = 機体-y 方向へ実変位
    assert sim.y < -0.3 and abs(sim.x) < 0.2


def test_executor_without_begin_starts_at_first_step():
    """controller経路の回帰: begin()未呼出しで _i=-1 のまま
    steps[-1]を実行し step0を二重実行しない（実害: 180°旋回が2回走った）。"""
    sim = Sim()
    ex = MotionExecutor([MotionStep("turn", "right", math.radians(90))], P)
    # begin()を呼ばず直接tick（controllerの「次tickの観測でアンカー」経路）
    t = 0.0
    while not ex.done and t < 30:
        s = ex.tick(*sim.obs(), now=t)
        sim.step(s, 0.05)
        t += 0.05
    assert ex.done and not ex.failed
    begins = [e for e in ex.events if e.get("event") == "motion_step_begin"]
    assert len(begins) == 1  # 1step指令が1回だけ実行された
    assert begins[0]["step_index"] == 0
    assert abs(abs(sim.yaw) - math.pi / 2) < math.radians(15)
