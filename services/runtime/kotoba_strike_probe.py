#!/usr/bin/env python3
"""打撃能力probe（SDK container内）— 減速→静止→一振り→回復 を実SDKで実測（B2）。

pm01_edu には専用の腕振りtaskが無い。腕を大きく動かすmotionは
dance(RB+B, rl_dance_example) のみで、その軌道(24.4s,50fps)の腕関節
最大偏差は t≈3.1–3.3s に集中する（dance.npz 事前解析）。
よって「一振り」= dance 開始 → SWING_WINDOW_S 実行 → LB+A(pd_stand) で
中断・立位へ回復、として実装し実測する。

入力: <result.json> <manifest.json>
manifest: run_id / purpose="calibration" / arming("1") / expires_in_s / boot_expect

計測: dance中の腕関節偏差(sim_state joints, MJCF順 index 13-17=左腕,18-22=右腕)、
base水平drift・最低z・task遷移・回復settle時間・転倒有無。
契約: 送信は SendGateway 経由のみ。受信workerは独立スレッドで転倒監視を継続。
"""

from __future__ import annotations

import json
import math
import struct
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, "/kotoba/harness_src")

import lcm

from kotoba_harness.auth import RunManifest, SendGateway
from kotoba_harness.errors import AuthorizationRefused, HarnessError
from kotoba_harness.observer import ClockGate
from kotoba_harness.trial import FallLatch, Phase, ReadyGate, TrialFSM

from kotoba_runner import Sample

URL = "udpm://239.255.76.67:7667?ttl=0"
CLOCK_FP = 0x4B544F4241434C31
HEADER = struct.Struct(">qdi")
CLOCK = struct.Struct(">qqqdd")
HZ = 20.0

# dance軌道の腕最大偏差は t≈3.1-3.3s — 余裕を持って 4.5s で中断する
SWING_WINDOW_S = 4.5
DANCE_START_TIMEOUT_S = 4.0
RECOVER_TIMEOUT_S = 10.0
SETTLE_SPEED_MPS = 0.05
STAND_Z_MIN = 0.75
ARM_SWING_MIN_RAD = 0.5
# sim_state joints配列はMJCF順(freejoint除く): 13-17=左腕J13-J17, 18-22=右腕J18-J22
ARM_IDX = list(range(13, 23))


def _decode_task(data: bytes) -> str | None:
    try:
        n = struct.unpack_from(">i", data, 8)[0]
        # task_stateの文字列はNUL終端を含む（実測: "pd_stand\x00"）
        return data[12 : 12 + n].decode("utf-8", "replace").rstrip("\x00")
    except (struct.error, IndexError):
        return None


class StrikeObserver:
    """sim_state(関節込み)+clock+task_state を結合したprobe用観測。"""

    def __init__(self, expected_nonce: int | None = None) -> None:
        self.handle = lcm.LCM(URL)
        self._lock = threading.Lock()
        self._state = None  # (recv, pos, vel, quat, joints)
        self._clock = None
        self._task = None
        self.bound = None  # (recv, pos, vel, quat, joints, sim_t, seq, nonce, task)
        self._gate = ClockGate()
        self._expected_nonce = expected_nonce
        self._running = True
        self._exception = None
        self.handle.subscribe("sim_state", self._on_state)
        self.handle.subscribe("kotoba_sim_clock", self._on_clock)
        self.handle.subscribe("task_state", self._on_task)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _on_state(self, _ch, data):
        try:
            _fp, _ts, n = HEADER.unpack_from(data, 0)
            off = HEADER.size + 3 * n * 8
            pos = struct.unpack_from(">3d", data, off)
            vel = struct.unpack_from(">3d", data, off + 24)
            quat = struct.unpack_from(">4d", data, off + 48)
            joints = struct.unpack_from(f">{n}d", data, HEADER.size)
        except struct.error:
            return
        if not all(map(math.isfinite, (*pos, *vel, *quat))):
            return
        with self._lock:
            self._state = (time.monotonic(), pos, vel, quat, list(joints))
            self._bind()

    def _on_clock(self, _ch, data):
        if len(data) != 40:
            return
        fp, nonce, seq, sim_t, mono = CLOCK.unpack(data)
        if fp != CLOCK_FP:
            return
        with self._lock:
            if self._expected_nonce is not None and nonce != self._expected_nonce:
                return
            ok, boot_changed = self._gate.accept(nonce, seq, sim_t)
            if boot_changed:
                self._state = None
                self.bound = None
            if not ok:
                return
            if self._expected_nonce is None:
                self._expected_nonce = nonce
            self._clock = (time.monotonic(), nonce, seq, sim_t)
            self._bind()

    def _on_task(self, _ch, data):
        name = _decode_task(data)
        if name is None:
            return
        with self._lock:
            self._task = (time.monotonic(), name)

    def _bind(self):
        if self._state is None or self._clock is None:
            return
        s_recv, pos, vel, quat, joints = self._state
        c_recv, nonce, seq, sim_t = self._clock
        if abs(s_recv - c_recv) > 0.03:
            return
        task = self._task[1] if self._task else None
        self.bound = (
            time.monotonic(), pos, vel, quat, joints, sim_t, seq, nonce, task
        )

    def _loop(self):
        try:
            while self._running:
                self.handle.handle_timeout(20)
        except Exception as exc:
            self._exception = exc

    def latest(self, max_age_s: float = 0.4):
        if self._exception is not None:
            raise HarnessError(f"observer_worker_failed:{self._exception}")
        with self._lock:
            b = self.bound
        if b is None:
            raise HarnessError("no_bound_observation")
        recv, pos, vel, quat, joints, sim_t, seq, nonce, task = b
        if time.monotonic() - recv > max_age_s:
            raise HarnessError("stale_observation")
        return pos, vel, quat, joints, sim_t, seq, nonce, task

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)


def _tilt_deg(quat) -> float:
    w, x, y, z = quat
    # z軸のworld鉛直からの傾き
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, up_z))))


def main() -> int:
    result_path = sys.argv[1]
    manifest = json.loads(Path(sys.argv[2]).read_text())
    result = {"run_id": manifest.get("run_id", "unknown"), "events": []}
    events = result["events"]

    expect = manifest.get("boot_expect")
    obs = None
    gateway = None
    try:
        expected = int(expect) if expect and str(expect).isdigit() else None
        obs = StrikeObserver(expected_nonce=expected)
        deadline = time.monotonic() + 15
        while True:
            try:
                pos, vel, quat, joints, sim_t, seq, nonce, task = obs.latest()
                break
            except HarnessError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.1)
        if not expect or str(expect) in ("boot-pending", "pending"):
            raise HarnessError("boot_not_bound")
        if str(nonce) != str(expect):
            raise HarnessError(f"boot_mismatch:{nonce}!={expect}")
        result["boot_nonce"] = nonce
        result["task_at_start"] = task

        rm = RunManifest(
            run_id=manifest["run_id"],
            purpose=manifest.get("purpose", "calibration"),
            profile=__import__(
                "kotoba_harness.auth", fromlist=["SIM_PROFILE"]
            ).SIM_PROFILE,
            sim_boot_id=nonce,
            expires_monotonic=time.monotonic()
            + float(manifest.get("expires_in_s", 240)),
        )
        gateway = SendGateway(
            rm,
            obs.handle,
            "virtual_gamepad/gamepad_keys",
            now_monotonic=time.monotonic(),
            sim_mode_confirmed=True,
            arming_env=manifest.get("arming", ""),
        )

        fsm = TrialFSM()
        latch = FallLatch()
        gate = ReadyGate(hold_s=2.0)
        fsm.transition(Phase.READY_WAIT)

        # 1) 立位確認（静止状態であること）
        ready_deadline = time.monotonic() + 15.0
        while time.monotonic() < ready_deadline:
            pos, vel, quat, joints, *_ = obs.latest()
            sample = Sample(pos, vel, quat)
            latch.observe(Phase.READY_WAIT.value, sample)
            if latch.fallen:
                raise HarnessError("prep_fall")
            if gate.observe(sample, time.monotonic(), None):
                break
        else:
            raise HarnessError("invalid_start_never_ready")
        events.append({"event": "ready", "z": round(pos[2], 3)})
        base_pos0 = pos
        joints0 = list(joints)
        n_joints = len(joints0)

        frame_idle = gateway.prepare("idle")

        def stream(frame, seconds, track=None):
            """seconds間 @20Hz でframe送信しながら観測を継続。
            track: 任意のcallable(pos,vel,quat,joints,task,t)を各stepに呼ぶ。"""
            t0 = time.monotonic()
            steps = int(seconds * HZ)
            for i in range(steps):
                gateway.issue(
                    frame, now_monotonic=time.monotonic(), command_name="stream"
                )
                try:
                    p, v, q, j, _st, _sq, _nn, tsk = obs.latest()
                    latch.observe(Phase.WALK.value, Sample(p, v, q))
                    if latch.fallen:
                        raise HarnessError("walk_fall")
                    if track:
                        track(p, v, q, j, tsk, time.monotonic() - t0)
                except HarnessError as exc:
                    if "walk_fall" in str(exc):
                        raise
                due = t0 + (i + 1) / HZ
                while time.monotonic() < due:
                    time.sleep(0.001)

        # 2) 一振り: dance開始 → 腕振り区間だけ実行 → pd_standで中断
        fsm.transition(Phase.WALK)
        frame_dance = gateway.prepare("combo_dance")
        stream(frame_dance, 0.6)

        dance_t0 = time.monotonic()
        dance_seen = False
        while time.monotonic() - dance_t0 < DANCE_START_TIMEOUT_S:
            gateway.issue(
                frame_idle, now_monotonic=time.monotonic(), command_name="idle"
            )
            try:
                *_r, task = obs.latest()
                if task == "dance":
                    dance_seen = True
                    break
            except HarnessError:
                pass
            time.sleep(0.05)
        result["dance_task_seen"] = dance_seen
        if not dance_seen:
            raise HarnessError("dance_not_started")

        # 腕振り区間の計測（idleを流し続けてgamepadを生存させる）
        arm_dev = [0.0] * n_joints
        arm_peak_t = 0.0
        base_drift = 0.0
        z_min = pos[2]
        tilt_max = 0.0

        def track_swing(p, v, q, j, tsk, t):
            nonlocal arm_peak_t, base_drift, z_min, tilt_max
            for idx in ARM_IDX:
                if idx < len(j):
                    d = abs(j[idx] - joints0[idx])
                    if d > arm_dev[idx]:
                        arm_dev[idx] = d
                        if idx == max(ARM_IDX, key=lambda k: arm_dev[k]):
                            arm_peak_t = t
            base_drift = max(
                base_drift, math.hypot(p[0] - base_pos0[0], p[1] - base_pos0[1])
            )
            z_min = min(z_min, p[2])
            tilt_max = max(tilt_max, _tilt_deg(q))

        stream(frame_idle, SWING_WINDOW_S, track=track_swing)

        # 3) 回復: danceのauto_transition先と同じく walk へ戻して安定化させ、
        # 静止後に pd_stand へ移す（動的姿勢からの直接pd_standは転倒する — 実測）
        fsm.transition(Phase.DECELERATING)
        frame_walk = gateway.prepare("combo_walk")
        stream(frame_walk, 0.6)
        recover_t0 = time.monotonic()
        recovered = False
        while time.monotonic() - recover_t0 < RECOVER_TIMEOUT_S:
            gateway.issue(
                frame_idle, now_monotonic=time.monotonic(), command_name="idle"
            )
            try:
                p, v, q, j, _st, _sq, _nn, tsk = obs.latest()
                latch.observe(Phase.TAIL.value, Sample(p, v, q))
                if latch.fallen:
                    raise HarnessError("recover_fall")
                for idx in ARM_IDX:
                    if idx < len(j):
                        arm_dev[idx] = max(arm_dev[idx], abs(j[idx] - joints0[idx]))
                if math.hypot(v[0], v[1]) < SETTLE_SPEED_MPS and p[2] > STAND_Z_MIN:
                    recovered = True
                    result["recover_task"] = tsk
                    break
            except HarnessError as exc:
                if "recover_fall" in str(exc):
                    raise
            time.sleep(0.05)
        recover_s = time.monotonic() - recover_t0

        # 安定化後に pd_stand へ移して完全静止を確認
        if recovered:
            frame_stand = gateway.prepare("combo_pd_stand")
            stream(frame_stand, 0.6)
            stand_deadline = time.monotonic() + 5.0
            while time.monotonic() < stand_deadline:
                try:
                    p, v, q, j, _st, _sq, _nn, tsk = obs.latest()
                    latch.observe(Phase.TAIL.value, Sample(p, v, q))
                    if latch.fallen:
                        recovered = False
                        result["recover_task"] = tsk
                        break
                    if math.hypot(v[0], v[1]) < SETTLE_SPEED_MPS and p[2] > STAND_Z_MIN:
                        result["recover_task"] = tsk
                        break
                except HarnessError:
                    pass
                time.sleep(0.05)

        fsm.transition(Phase.SETTLING)
        fsm.transition(Phase.TAIL)
        fsm.transition(Phase.DONE)

        arm_max = max(arm_dev[i] for i in ARM_IDX if i < n_joints)
        result["strike"] = {
            "dance_task_seen": dance_seen,
            "swing_window_s": SWING_WINDOW_S,
            "arm_max_dev_rad": round(arm_max, 3),
            "arm_peak_t_s": round(arm_peak_t, 2),
            "arm_joint_dev_rad": {
                f"j{i}": round(arm_dev[i], 3) for i in ARM_IDX if i < n_joints
            },
            "base_drift_m": round(base_drift, 3),
            "z_min_m": round(z_min, 3),
            "tilt_max_deg": round(tilt_max, 1),
            "recover_s": round(recover_s, 2),
            "recovered": recovered,
            "fallen": latch.fallen,
        }
        ok = (
            dance_seen
            and arm_max >= ARM_SWING_MIN_RAD
            and recovered
            and not latch.fallen
        )
        result["verdict"] = "PASS" if ok else "FAIL"
        if not ok:
            result["reasons"] = [
                f"dance={dance_seen}", f"arm_dev={arm_max:.3f}>={ARM_SWING_MIN_RAD}",
                f"recovered={recovered}", f"fallen={latch.fallen}",
            ]
    except (HarnessError, AuthorizationRefused) as exc:
        result["verdict"] = "FAIL_" + getattr(exc, "reason", "error").upper()
        result["reasons"] = [str(exc)]
    except Exception as exc:
        result["verdict"] = "FAIL_INTERNAL"
        result["reasons"] = [f"{type(exc).__name__}: {exc}"]
    finally:
        if gateway is not None:
            try:
                gateway.issue(
                    gateway.prepare("idle"),
                    now_monotonic=time.monotonic(),
                    command_name="final_idle",
                )
                gateway.close()
            except Exception:
                pass
        if obs is not None:
            try:
                obs.stop()
            except Exception:
                pass
        Path(result_path).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return 0 if result.get("verdict") == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
