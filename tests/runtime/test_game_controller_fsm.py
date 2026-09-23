"""game controllerのStrikeFSM仕様固定 — 実消費・中断・フェーズ駆動。

kotoba_game_controllerはThorのcontainer内で動く（lcm依存）ため、
依存をstub化してStrikeFSM単体を実ロジックで検証する:
- pre: 静止・支持姿勢を実測してからdanceへ（送信だけで静止成立にしない）
- dance: task遷移観測=実消費(consumed) / 未観測ならnot_started
- swing: 軌跡記録はobs健全時のみ / 期限でrecoverへ
- recover: walk復帰で完了 / ENDはpending_endを残す
- abort: 全phaseでSTOP/ENDを受け付け、管理中断はrecoverへ
"""

from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CTRL = ROOT / "services" / "runtime" / "kotoba_game_controller.py"
HARNESS_SRC = ROOT / "services" / "harness" / "src"


def _load_controller():
    """lcm依存のrunner/probeをstubしてcontroller moduleを読む。"""
    if str(HARNESS_SRC) not in sys.path:
        sys.path.insert(0, str(HARNESS_SRC))
    runner = types.ModuleType("kotoba_runner")

    class Sample:
        # 実 kotoba_runner.Sample と同じ属性名（FallLatch.observeが参照）
        def __init__(self, pos, vel, quat):
            self.position = pos
            self.velocity = vel
            self.quaternion_wxyz = quat

    runner.Sample = Sample
    probe = types.ModuleType("kotoba_strike_probe")

    class StrikeObserver:  # 未使用（FSM単体試験ではobsを直接渡す）
        pass

    probe.StrikeObserver = StrikeObserver
    sys.modules.setdefault("kotoba_runner", runner)
    sys.modules.setdefault("kotoba_strike_probe", probe)
    spec = importlib.util.spec_from_file_location(
        "kotoba_game_controller", CTRL
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ctrl():
    return _load_controller()


_J = [0.0] * 25  # FKチェーンが参照する関節配列（J12腰+J13-22腕）


def _still():
    # (pos, vel, quat, joints, task) — walk task・静止
    return ((0.0, 0.0, 0.82), (0.0, 0.0, 0.0), (1.0, 0, 0, 0), list(_J), "walk")


def _moving():
    return ((0.0, 0.0, 0.82), (0.4, 0.0, 0.0), (1.0, 0, 0, 0), list(_J), "walk")


def _dancing():
    return ((0.0, 0.0, 0.82), (0.05, 0.0, 0.0), (1.0, 0, 0, 0), list(_J), "dance")


FRAMES = ("DANCE", "WALK", "IDLE")


def test_pre_waits_for_actual_stillness(ctrl):
    """打撃前の静止は実測 — 移動中はpreを抜けない。"""
    f = ctrl.StrikeFSM("s1", 1, None, 0.2)
    frame, done, _ = f.tick(lambda *a: None, FRAMES, _moving())
    assert f.phase == "pre" and not done and frame == "IDLE"


def test_pre_timeout_is_not_started(ctrl, monkeypatch):
    """静止を確認できないまま期限切れ → not_started（消費しない）。"""
    f = ctrl.StrikeFSM("s1", 1, None, 0.2)
    # 時間を進めてpre期限を超過させる
    f.t0 = time.monotonic() - ctrl.STRIKE_PRE_S - 0.1
    frame, done, info = f.tick(lambda *a: None, FRAMES, _moving())
    assert done and info["status"] == "not_started"
    assert info["aborted"] == "precondition_timeout"
    assert not f.consumed


def test_full_lifecycle_consumes_on_dance_seen(ctrl):
    """pre→dance→swing→recover。consumedはtask=dance観測時に確定。"""
    f = ctrl.StrikeFSM("s1", 1, (0, 0, 0), 0.2)
    # pre: 静止をSTRIKE_PRE_HOLD_Sだけ持続 → danceへ
    f.tick(lambda *a: None, FRAMES, _still())
    assert f.phase == "pre" and f.still_since is not None
    f.still_since = time.monotonic() - ctrl.STRIKE_PRE_HOLD_S
    frame, done, _ = f.tick(lambda *a: None, FRAMES, _still())
    assert f.phase == "dance" and frame == "DANCE"
    # dance遷移観測 → consumed確定・swingへ
    frame, done, _ = f.tick(lambda *a: None, FRAMES, _dancing())
    assert f.consumed and f.phase == "swing" and frame == "IDLE"
    # swing中は軌跡を記録する
    assert f.recorder is not None
    f.tick(lambda *a: None, FRAMES, _dancing())
    assert len(f.recorder.samples) == 1
    # swing期限 → recover
    f.t0 = time.monotonic() - ctrl.SWING_WINDOW_S - 0.1
    frame, done, _ = f.tick(lambda *a: None, FRAMES, _dancing())
    assert f.phase == "recover" and frame == "WALK"
    # walk task復帰+RECOVER_HOLD_S経過 → done
    f.t0 = time.monotonic() - ctrl.RECOVER_HOLD_S - 0.1
    frame, done, info = f.tick(lambda *a: None, FRAMES, _still())
    assert done and info["status"] == "done" and info["done"]


def test_dance_not_started_not_consumed(ctrl):
    """dance遷移が観測できない → not_started・非消費（返却相当）。"""
    f = ctrl.StrikeFSM("s1", 1, None, 0.2)
    f.phase = "dance"
    f.t0 = time.monotonic() - ctrl.DANCE_WAIT_S - 0.1
    frame, done, info = f.tick(lambda *a: None, FRAMES, _still())
    assert done and info["status"] == "not_started"
    assert info["aborted"] == "dance_not_started"
    assert not f.consumed


def test_stop_abort_goes_to_recover(ctrl):
    """swing中のSTOPは管理中断 → recoverへ（瞬間停止でなく検証済み回復）。"""
    f = ctrl.StrikeFSM("s1", 1, None, 0.2)
    f.phase = "swing"
    f.consumed = True
    f.abort("stop")
    assert f.phase == "recover" and f.entry["interrupted"] == "stop"
    # 完了時はstatus=done（開始済みの中断は消費）
    f.t0 = time.monotonic() - ctrl.RECOVER_VERIFY_S
    frame, done, info = f.tick(lambda *a: None, FRAMES, _still())
    assert done and info["status"] == "done"


def test_end_abort_marks_pending_end(ctrl):
    """打撃中のENDはrecover完了後に終了へ反映する。"""
    f = ctrl.StrikeFSM("s1", 1, None, 0.2)
    f.phase = "swing"
    f.consumed = True
    f.abort("end")
    assert f.pending_end and f.phase == "recover"


def test_end_during_recover_still_marks_pending_end(ctrl):
    """recover中のENDも終了へ反映する（見落とさない）。"""
    f = ctrl.StrikeFSM("s1", 1, None, 0.2)
    f.phase = "recover"
    f.consumed = True
    f.abort("end")
    assert f.pending_end


def test_abort_during_dance_unverified(ctrl):
    """dance指令送出後・遷移未観測で中断 → unverified（要確認）。
    物理的に振ったか不明 — 推測で消費/非消費にしない。"""
    f = ctrl.StrikeFSM("s1", 1, None, 0.2)
    f.phase = "dance"
    f.entry["sent_dance"] = True
    f.abort("stop")
    f.t0 = time.monotonic() - ctrl.RECOVER_VERIFY_S
    frame, done, info = f.tick(lambda *a: None, FRAMES, _still())
    assert done and info["status"] == "unverified"
    assert not f.consumed


def test_swing_skips_dead_obs(ctrl):
    """観測欠測サンプルは軌跡へ記録しない（偽の手先位置で誤判定しない）。"""
    f = ctrl.StrikeFSM("s1", 1, (0, 0, 0), 0.2)
    f.phase = "swing"
    f.tick(lambda *a: None, FRAMES, _dancing(), obs_ok=True)
    assert len(f.recorder.samples) == 1
    dead = ((0, 0, 0), (0, 0, 0), (0, 0, 0, 1), {}, "")
    f.tick(lambda *a: None, FRAMES, dead, obs_ok=False)
    assert len(f.recorder.samples) == 1  # 欠測は記録されない


# ---- 終了保持の実観測検証（レビューB群反例 P04-P07） ---------------------
class _ScriptedObs:
    """逐次スクリプトを返すobs stub — sim_tは呼出し毎にsim_stepだけ進む。

    latest()は実StrikeObserverと同じ8要素:
    (pos, vel, quat, joints, sim_t, seq, nonce, task)
    """

    def __init__(self, fn, sim_step=0.1):
        self.fn = fn
        self.sim_t = 0.0
        self.sim_step = sim_step
        self.n = 0

    def latest(self):
        pos, vel, quat = self.fn(self.n, self.sim_t)
        out = (pos, vel, quat, list(_J), self.sim_t, self.n, 1, "walk")
        self.sim_t += self.sim_step
        self.n += 1
        return out


_UP = (1.0, 0.0, 0.0, 0.0)


def _hold_case(ctrl, monkeypatch, fn, sim_step=0.1, settle_s=3.0, hold_s=1.0):
    """検証窓を短縮して _verify_end_hold を実ロジックで駆動する。"""
    monkeypatch.setattr(ctrl, "END_SETTLE_S", settle_s)
    monkeypatch.setattr(ctrl, "END_SETTLE_STABLE_S", 0.4)
    monkeypatch.setattr(ctrl, "END_HOLD_S", hold_s)
    obs = _ScriptedObs(fn, sim_step=sim_step)
    latch = ctrl.FallLatch()
    events = []
    ok = ctrl._verify_end_hold(obs, latch, events)
    return ok, latch, events


def test_end_hold_upright_still_verified(ctrl, monkeypatch):
    """P04正例: 直立静止は保持検証を通る（複合条件でも過剰拒否しない）。"""
    ok, latch, events = _hold_case(
        ctrl, monkeypatch,
        lambda n, t: ((0.0, 0.0, 0.82), (0.0, 0.0, 0.0), _UP),
    )
    assert ok is True and not latch.fallen
    assert any(e["event"] == "end_hold_detail" for e in events)


def test_end_hold_coarse_sim_tick_still_verified(ctrl, monkeypatch):
    """実機反例: sim_t刻みが粗い(~0.2s)と窓pop直後スパンが充填条件を
    下回り続け、完全静止でもsettleしなかった（end_hold_unsettled誤判定）。
    popは次点が窓を覆う時のみに制限し、粗い刻みでも安定成立させる。"""
    ok, latch, events = _hold_case(
        ctrl, monkeypatch,
        lambda n, t: ((0.0, 0.0, 0.82), (0.0, 0.0, 0.0), _UP),
        sim_step=0.2, hold_s=1.0,
    )
    assert ok is True


def test_end_hold_constant_drift_rejected(ctrl, monkeypatch):
    """P05反例: 0.10m/s等速ドリフトは「歩き去り」— 失格にする。

    旧変位窓判定はsettleを通してしまい合格扱いだった。複合判定では
    0.5s窓の純変位が閾値を超えsettle自体を通さない（+保持側の逸脱
    上限でも多段的に拒否する）。
    """
    ok, latch, events = _hold_case(
        ctrl, monkeypatch,
        lambda n, t: ((0.10 * t, 0.0, 0.82), (0.10, 0.0, 0.0), _UP),
        sim_step=0.25, settle_s=1.0,
    )
    assert ok is False
    assert events[-1]["event"] == "end_hold_unsettled"


def test_end_hold_drift_starting_during_hold_rejected(ctrl, monkeypatch):
    """settle後に歩き出した場合は保持中の逸脱上限で失格（P05の保持側）。"""
    def fn(n, t):
        if t < 1.5:
            return ((0.0, 0.0, 0.82), (0.0, 0.0, 0.0), _UP)
        return ((0.10 * (t - 1.5), 0.0, 0.82), (0.10, 0.0, 0.0), _UP)

    ok, latch, events = _hold_case(ctrl, monkeypatch, fn, hold_s=4.0)
    assert ok is False
    br = [e for e in events if e["event"] == "end_hold_broken"]
    # dev=閾値到達で破断（round済み記録値のため>=で比較）
    assert br and br[0]["dev"] >= ctrl.END_HOLD_DISPL_M


def test_end_hold_stationary_tilt_rejected(ctrl, monkeypatch):
    """P06反例: 45度傾きの静止は「立ったまま」でない — 失格。

    変位・速度はゼロでも姿勢条件でsettleを通さない。
    """
    import math as _m

    x = _m.sin(_m.radians(22.5))
    w = _m.cos(_m.radians(22.5))  # world upから45度傾いた姿勢
    ok, latch, events = _hold_case(
        ctrl, monkeypatch,
        lambda n, t: ((0.0, 0.0, 0.82), (0.0, 0.0, 0.0), (w, x, 0, 0)),
        settle_s=0.6,
    )
    assert ok is False
    assert events[-1]["event"] == "end_hold_unsettled"
    assert events[-1]["tilt"] > ctrl.END_HOLD_TILT_DEG


def test_end_hold_fall_latches_and_fails(ctrl, monkeypatch):
    """seed15/16反例: 保持中の転倒はfall latchへ記録されFAILへ。

    z≈0.11の倒伏をlatchが拾い、health=ok・fallen=falseの不整合を防ぐ。
    """
    def fn(n, t):
        if t < 1.5:
            return ((0.0, 0.0, 0.82), (0.0, 0.0, 0.0), _UP)
        return ((0.0, 0.0, 0.11), (0.0, 0.0, 0.0), _UP)

    ok, latch, events = _hold_case(ctrl, monkeypatch, fn, hold_s=3.0)
    assert ok is False and latch.fallen
    assert latch.category == "tail_fall"
    br = [e for e in events if e["event"] == "end_hold_broken"]
    assert br and br[0]["fallen"] is True


def test_end_hold_frozen_sim_time_rejected(ctrl, monkeypatch):
    """sim時刻が凍結していれば（pause/観測停滞）保持に算入しない。"""
    ok, latch, events = _hold_case(
        ctrl, monkeypatch,
        lambda n, t: ((0.0, 0.0, 0.82), (0.0, 0.0, 0.0), _UP),
        sim_step=0.0, settle_s=0.6,
    )
    assert ok is False
    assert events[-1]["event"] == "end_hold_unsettled"


def test_end_hold_circling_rejected(ctrl, monkeypatch):
    """settle後に始まる小半径周回は逸脱で捉えきれない — 累積軌跡で失格。"""
    import math as _m

    def circle(n, t):
        if t < 2.0:
            return ((0.0, 0.0, 0.82), (0.0, 0.0, 0.0), _UP)
        # アンカー(0,0)中心のr=0.09周回 — 逸脱≤0.18でdevは発動しないが
        # 周速度0.36m/sの累積軌跡が保持上限を超える（歩き続けている）
        r, om = 0.09, 4.0
        th = om * (t - 2.0)
        return (
            (r * _m.cos(th), r * _m.sin(th), 0.82),
            (0.36, 0.0, 0.0), _UP,
        )

    ok, latch, events = _hold_case(
        ctrl, monkeypatch, circle, sim_step=0.1, hold_s=4.0,
    )
    assert ok is False
    br = [e for e in events if e["event"] == "end_hold_broken"]
    assert br and br[0]["path"] > ctrl.END_HOLD_PATH_M


def test_end_hold_observation_gap_not_counted(ctrl, monkeypatch):
    """P10反例: 保持中の観測断（契約上のstale例外）を成功時間へ加えない。

    旧実装はhold loopのstale例外を握りつぶしhold_start_simを保持したため、
    1.5秒の欠測→復旧後0.3秒の連続確認で sim_s=2.5・verified=true となった。
    新実装は欠測区間を保持へ算入せず、復帰サンプルから積み直す —
    復旧しないままではwall期限で失格（UNKNOWN側へ正直に分類）。
    """
    from kotoba_harness.errors import HarnessError

    class GappyObs:
        """settle後に15回stale→復帰→0.3sim秒だけ観測→再び恒久的stale。"""
        def __init__(self):
            self.sim_t = 0.0
            self.n = 0

        def latest(self):
            n, self.n = self.n, self.n + 1
            self.sim_t += 0.1  # sim時刻は欠測中も進行する
            # n<12: settle+hold開始の正常観測（~1.2sim秒）
            # 12<=n<27: 1.5sim秒の観測断
            # 27<=n<30: 復帰後0.3sim秒のみ正常
            # n>=30: 復旧しない恒久的欠測
            if 12 <= n < 27 or n >= 30:
                raise HarnessError("stale_observation")
            return (
                (0.0, 0.0, 0.82), (0.0, 0.0, 0.0), _UP,
                list(_J), self.sim_t, n, 1, "walk",
            )

    monkeypatch.setattr(ctrl, "END_SETTLE_STABLE_S", 0.4)
    monkeypatch.setattr(ctrl, "END_HOLD_S", 1.0)  # wall期限=4s
    events = []
    latch = ctrl.FallLatch()
    ok = ctrl._verify_end_hold(GappyObs(), latch, events)
    assert ok is False
    kinds = [e["event"] for e in events]
    # 復帰を検出してgapを記録したが、復旧0.3sでは保持を継続し、
    # 最終的に観測が戻らずtimeoutで失格 — 空白を成功時間へ加えていない
    assert "end_hold_observation_gap" in kinds
    assert kinds[-1] == "end_hold_observation_timeout"
    assert not any(e["event"] == "end_hold_detail" for e in events)


def test_end_hold_gap_recovery_restarts_accumulation(ctrl, monkeypatch):
    """観測断からの復旧後、連続した有効sim時間が保持条件を満たせば
    正常終了できる（gapは失格即決ではなく積み直し）。"""
    from kotoba_harness.errors import HarnessError

    class GappyObs:
        """settle後に5回stale(0.5sim秒の空白)→復帰→以後ずっと正常観測。"""
        def __init__(self):
            self.sim_t = 0.0
            self.n = 0

        def latest(self):
            n, self.n = self.n, self.n + 1
            self.sim_t += 0.1
            if 12 <= n < 17:  # 0.5sim秒の空白
                raise HarnessError("stale_observation")
            return (
                (0.0, 0.0, 0.82), (0.0, 0.0, 0.0), _UP,
                list(_J), self.sim_t, n, 1, "walk",
            )

    monkeypatch.setattr(ctrl, "END_SETTLE_STABLE_S", 0.4)
    monkeypatch.setattr(ctrl, "END_HOLD_S", 1.0)
    events = []
    latch = ctrl.FallLatch()
    ok = ctrl._verify_end_hold(GappyObs(), latch, events)
    assert ok is True
    detail = [e for e in events if e["event"] == "end_hold_detail"]
    assert detail and detail[0]["gaps"] == 1
    # sim_sは復帰後の連続有効区間 — 空白を含まない
    assert detail[0]["sim_s"] >= ctrl.END_HOLD_S
    assert any(e["event"] == "end_hold_observation_gap" for e in events)


def test_end_hold_gap_position_jump_still_detected(ctrl, monkeypatch):
    """空白中に動いた場合は、復帰端点のdev/pathへ下限として現れ失格する。"""
    from kotoba_harness.errors import HarnessError

    class JumpObs:
        """settle→空白(1.0sim秒)の間に0.4m移動→復帰。"""
        def __init__(self):
            self.sim_t = 0.0
            self.n = 0

        def latest(self):
            n, self.n = self.n, self.n + 1
            self.sim_t += 0.1
            if 12 <= n < 22:
                raise HarnessError("stale_observation")
            x = 0.4 if n >= 22 else 0.0  # 空白中に動いた分が端点差に現れる
            return (
                (x, 0.0, 0.82), (0.0, 0.0, 0.0), _UP,
                list(_J), self.sim_t, n, 1, "walk",
            )

    monkeypatch.setattr(ctrl, "END_SETTLE_STABLE_S", 0.4)
    monkeypatch.setattr(ctrl, "END_HOLD_S", 3.0)
    events = []
    latch = ctrl.FallLatch()
    ok = ctrl._verify_end_hold(JumpObs(), latch, events)
    assert ok is False
    br = [e for e in events if e["event"] == "end_hold_broken"]
    assert br and br[0]["dev"] >= ctrl.END_HOLD_DISPL_M


def test_end_hold_settle_gap_gets_fresh_window(ctrl, monkeypatch):
    """settle段の観測断は変位窓も失効 — 復帰後に新規0.5s窓から積み直す
    （欠測前の古いサンプルを窓へ残さず、欠測跨ぎでsettle成立にしない）。"""
    from kotoba_harness.errors import HarnessError

    class SettleGapObs:
        """3サンプル後に恒久的stale — 欠測前サンプルだけではsettle不可。"""
        def __init__(self):
            self.sim_t = 0.0
            self.n = 0

        def latest(self):
            n, self.n = self.n, self.n + 1
            self.sim_t += 0.1
            if n >= 3:
                raise HarnessError("stale_observation")
            return (
                (0.0, 0.0, 0.82), (0.0, 0.0, 0.0), _UP,
                list(_J), self.sim_t, n, 1, "walk",
            )

    monkeypatch.setattr(ctrl, "END_SETTLE_S", 1.0)
    events = []
    latch = ctrl.FallLatch()
    ok = ctrl._verify_end_hold(SettleGapObs(), latch, events)
    assert ok is False
    assert events[-1]["event"] == "end_hold_unsettled"
