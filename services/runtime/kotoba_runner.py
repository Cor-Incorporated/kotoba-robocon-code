#!/usr/bin/env python3
"""ことばでロボコン 実行ランナー v2（SDK container内）。

前提: 起動立位マネージャが直立を確立済み（boot-ready.json 存在）。
本ランナーは直立機体に対して walk combo → 閉ループ前進 → 通常停止 → 採点を行う。

入力: <result.json> <target_distance_m(float)> <manifest.json>
manifest: run_id / purpose / arming("1") / expires_in_s / boot_expect(必須・実nonce) /
          tolerance_m / stop_trigger_m / walk_stick

契約（NEXT_INSTRUCTIONS §4）:
- 送信は必ず SendGateway.send を経由（期限・allowlist・単一送信者の検証済み出口）
- 受信workerは独立スレッドで常時動き、送信中も観測・転倒監視が止まらない
- state と kotoba_sim_clock を同じpublisher周期（受信差30ms）で結合した原子snapshotを制御・採点に使用
- target_distance は float として検証（finite・0<x≤2.0・manifest一致）
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
from kotoba_harness.observer import ClockGate
from kotoba_harness.errors import AuthorizationRefused, HarnessError, ScheduleOverrun
from kotoba_harness.scheduler import DeadlineScheduler
from kotoba_harness.trial import (
    FALL_HEIGHT_M,
    STAND_HEIGHT_MAX_M,
    STAND_HEIGHT_MIN_M,
    FallLatch,
    NormalStopPolicy,
    Phase,
    ReadyGate,
    Scorer,
    TiltMonitor,
    TrialFSM,
)

URL = "udpm://239.255.76.67:7667?ttl=0"
CLOCK_FP = 0x4B544F4241434C31
HZ = 20.0
LIVE_PATH = "/kotoba/runtime/live.json"
# 停止制御は二段: APPROACH_ENTER_M 手前で低速アプローチへ落とし、低速域の
# 短く安定した停止距離（実測 v≈0.35m/s → 0.18m）で最終停止する。
# 高速域のまま固定距離トリガを踏むと overshoot する（v=0.57→0.68m, v=0.72→0.72m）。
APPROACH_ENTER_M = 0.8
FINAL_TRIGGER_MIN_M = 0.20
FINAL_TRIGGER_K = 1.0  # 停止距離係数を過大側に固定し、不足分は補正ステップで収束させる
SLOW_CMD = "walk_stick_slow"
# 停止後に残った誤差は低速バースト+静止確認を反復して解消する（真の閉ループ）。
# 反復で誤差が増大した場合（overshoot方向）は補正を打ち切り、正直に採点する。
MAX_CORRECTIONS = 5
CORRECTION_BURST_FRAMES = 8  # 20Hzで約0.4sの低速stick入力


class Observer:
    """独立受信スレッド。state と clock を同一publisher周期で結合した原子snapshotを保持。

    - subscribe を明示的に登録する（登録なしでは何も受信しない — F01）
    - worker の例外は check() で伝播させる
    - stop() はスレッドを終了し join する
    """

    def __init__(self, expected_nonce: int | None = None) -> None:
        self.handle = lcm.LCM(URL)
        self._lock = threading.Lock()
        self._state = None  # (recv_mono, pos, vel, quat)
        self._clock = None  # (recv_mono, nonce, seq, sim_t, src_mono)
        self.bound = None  # (recv_mono, pos, vel, quat, sim_t, seq, nonce)
        self.nonce = None
        self._gate = ClockGate()  # seq/sim_t単調・boot遷移の共通検証
        # 実行中runnerは承認されたbootに固定する。未指定なら最初に受理した
        # nonceへ固定し、以後いかなるboot変更も追従しない（R3-B）。
        self._expected_nonce = expected_nonce
        self._dropped_boot = 0
        self._running = True
        self._exception = None
        self._state_sub = self.handle.subscribe("sim_state", self._on_state)
        self._clock_sub = self.handle.subscribe("kotoba_sim_clock", self._on_clock)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _on_state(self, ch, data):
        try:
            _fp, _ts, n = struct.unpack_from(">qdi", data, 0)
            off = 20 + 3 * n * 8
            pos = struct.unpack_from(">3d", data, off)
            vel = struct.unpack_from(">3d", data, off + 24)
            quat = struct.unpack_from(">4d", data, off + 48)
        except struct.error:
            return
        if not all(map(math.isfinite, (*pos, *vel, *quat))):
            return
        with self._lock:
            recv = time.monotonic()
            self._state = (recv, pos, vel, quat)
            # 同一publisher周期の結合（受信差30ms以内のclockと対にする）
            if self._clock and abs(recv - self._clock[0]) < 0.03:
                c_recv, nonce, seq, sim_t, _sm = self._clock
                self.bound = (recv, pos, vel, quat, sim_t, seq, nonce)

    def _on_clock(self, ch, data):
        if len(data) != 40:
            return
        fp, nonce, seq, sim_t, mono = struct.unpack(">qqqdd", data)
        if fp != CLOCK_FP:
            return
        with self._lock:
            if self._expected_nonce is not None and nonce != self._expected_nonce:
                # 承認外bootのパケットは受理しない（runnerはboot固定 — R3-B）
                self._dropped_boot += 1
                return
            ok, boot_changed = self._gate.accept(nonce, seq, sim_t)
            if boot_changed:
                # 旧bootのstate/boundを新世代へ持ち越さない（30ms結合窓で
                # 旧boot最終stateが新boot clockへ誤結合するのを防ぐ）
                self._state = None
                self.bound = None
            if not ok:
                return
            if self._expected_nonce is None:
                self._expected_nonce = nonce  # 最初の受理bootへ固定
            self.nonce = nonce
            self._clock = (time.monotonic(), nonce, seq, sim_t, mono)
            if self._state and abs(self._state[0] - time.monotonic()) < 0.03:
                s_recv, pos, vel, quat = self._state
                self.bound = (time.monotonic(), pos, vel, quat, sim_t, seq, nonce)

    def _loop(self) -> None:
        try:
            while self._running:
                self.handle.handle_timeout(20)
        except Exception as exc:  # worker異常は check() で伝播
            self._exception = exc

    def check(self) -> None:
        if self._exception is not None:
            raise HarnessError(f"observer_worker_failed:{self._exception}")

    def latest(self, max_age_s: float = 0.4) -> tuple:
        self.check()
        with self._lock:
            b = self.bound
        if b is None:
            raise HarnessError("no_bound_observation")
        recv, pos, vel, quat, sim_t, seq, nonce = b
        age = time.monotonic() - recv
        if age > max_age_s:
            raise HarnessError(f"stale_observation:{age:.2f}s")
        return pos, vel, quat, sim_t, seq, nonce

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)


class Sample:
    def __init__(self, pos, vel, quat):
        self.position = pos
        self.velocity = vel
        self.quaternion_wxyz = quat


def main() -> int:
    result_path = sys.argv[1]
    try:
        target_distance = float(sys.argv[2])
    except ValueError:
        result = {
            "verdict": "FAIL_ARG",
            "reasons": [f"target_distance_not_float:{sys.argv[2]}"],
        }
        Path(result_path).write_text(json.dumps(result, indent=1))
        return 2
    manifest = json.loads(Path(sys.argv[3]).read_text())
    if not math.isfinite(target_distance) or not (0.05 < target_distance <= 2.0):
        result = {
            "verdict": "FAIL_ARG",
            "reasons": [f"target_distance_out_of_range:{target_distance}"],
        }
        Path(result_path).write_text(json.dumps(result, indent=1))
        return 2
    result = {
        "run_id": manifest.get("run_id", "unknown"),
        "target_distance_m": target_distance,
        "events": [],
    }
    events = result["events"]

    expect = manifest.get("boot_expect")
    try:
        # 実行中runnerは承認されたbootへ固定する（R3-B）。
        # boot-pending等の未結合値は pin せず、以後のboot_not_bound検査で拒否される。
        try:
            expected = int(expect) if expect else None
            if expected is None or str(expect) in ("boot-pending", "pending"):
                expected = None
        except (TypeError, ValueError):
            expected = None
        obs = Observer(expected_nonce=expected)
        # 観測の確立（boot一致を必ず検証）
        deadline = time.monotonic() + 15
        while True:
            try:
                pos, vel, quat, sim_t, seq, nonce = obs.latest()
                break
            except HarnessError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.1)
        if not expect or expect in ("boot-pending", "pending"):
            raise HarnessError("boot_not_bound")
        if str(nonce) != str(expect):
            raise HarnessError(f"boot_mismatch:{nonce}!={expect}")
        result["boot_nonce"] = nonce
        anchor = pos
        events.append(
            {
                "event": "bound",
                "anchor_xy": [round(anchor[0], 4), round(anchor[1], 4)],
                "z": round(pos[2], 3),
            }
        )

        # 送信gateway（唯一の出口）
        rm = RunManifest(
            run_id=manifest["run_id"],
            purpose=manifest.get("purpose", "evaluation"),
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
        tiltmon = TiltMonitor()
        stop = NormalStopPolicy()
        sched = DeadlineScheduler()
        scorer = Scorer(tolerance_m=float(manifest.get("tolerance_m", 0.15)))
        fsm.transition(Phase.READY_WAIT)

        # 1) 既存立位の検証（pd_standは送らない——stand managerが確立済み）
        ready_deadline = time.monotonic() + 15.0
        while time.monotonic() < ready_deadline:
            pos, vel, quat, *_ = obs.latest()
            sample = Sample(pos, vel, quat)
            latch.observe(Phase.READY_WAIT.value, sample)
            if latch.fallen:
                raise HarnessError("prep_fall")
            if gate.observe(sample, time.monotonic(), None):
                break
        else:
            raise HarnessError("invalid_start_never_ready")
        events.append({"event": "ready", "z": round(pos[2], 3)})
        anchor = pos
        fsm.transition(Phase.WALK)

        # 2) 目標方向: 初期quaternionからyawを直接計算（較正burst廃止 — F05 §4.2）
        #    R3修正: 機体+X を world XY へ射影する yaw（分母は 1-2(y²+z²)）
        _, _, start_quat, *_ = obs.latest()
        _sqw, _sqx, _sqy, _sqz = start_quat
        _yaw = math.atan2(
            2.0 * (_sqx * _sqy + _sqw * _sqz),
            1.0 - 2.0 * (_sqy * _sqy + _sqz * _sqz),
        )
        heading = (math.cos(_yaw), math.sin(_yaw))
        target = (
            anchor[0] + heading[0] * target_distance,
            anchor[1] + heading[1] * target_distance,
        )
        events.append(
            {
                "event": "target_from_quat",
                "yaw_deg": round(math.degrees(_yaw), 1),
                "target_xy": [round(c, 4) for c in target],
                "manifest_stop_trigger_m": float(manifest.get("stop_trigger_m", 0.45)),
                "approach_enter_m": APPROACH_ENTER_M,
                "final_trigger_min_m": FINAL_TRIGGER_MIN_M,
                "final_trigger_k": FINAL_TRIGGER_K,
            }
        )
        # UIの目的地マーカーと同じplanの目標をAPI経由で公開する
        try:
            Path("/kotoba/run/target.json").write_text(
                json.dumps(
                    {
                        "run_id": manifest.get("run_id"),
                        "target_xy": [round(c, 4) for c in target],
                        "target_distance_m": target_distance,
                    }
                )
            )
        except OSError:
            pass
        frame_walk = gateway.prepare("combo_walk")
        frame_stick = gateway.prepare("walk_stick")
        frame_stick_slow = gateway.prepare("walk_stick_slow")
        frame_idle = gateway.prepare("idle")
        frame_stand = gateway.prepare("combo_pd_stand")

        def publish_frames(frames, names=None):
            t0 = time.monotonic()
            for i, f in enumerate(frames):
                gateway.issue(
                    f,
                    now_monotonic=time.monotonic(),
                    command_name=(names[i] if names else f"seq{i}"),
                )
                due = t0 + (i + 1) / HZ
                while time.monotonic() < due:
                    time.sleep(0.001)  # 受信threadと同時呼出しされないようsleep

        # 3) 閉ループ移動 → 停止トリガ → 減速 → 静止確認 → pd_stand
        publish_frames([frame_walk] * 10)
        stop_triggered = False
        started = time.monotonic()
        decel_started = started
        tol = float(manifest.get("tolerance_m", 0.15))
        corrections = 0
        prev_err = None
        while time.monotonic() - started < 40.0:
            pos, vel, quat, sim_t, seq, _ = obs.latest()
            sample = Sample(pos, vel, quat)
            latch.observe(Phase.WALK.value, sample)
            if latch.fallen:
                raise HarnessError("walk_fall")
            err = math.hypot(pos[0] - target[0], pos[1] - target[1])
            speed = math.hypot(vel[0], vel[1])
            events.append(
                {
                    "event": "loop",
                    "err_m": round(err, 4),
                    "v": round(speed, 3),
                    "z": round(pos[2], 3),
                    "cmd_age": 0,
                }
            )
            if not stop_triggered:
                trigger_dist = max(FINAL_TRIGGER_MIN_M, speed * FINAL_TRIGGER_K)
                if err <= trigger_dist:
                    events.append(
                        {
                            "event": "stop_trigger",
                            "err": round(err, 4),
                            "speed": round(speed, 3),
                            "trigger_dist": round(trigger_dist, 4),
                        }
                    )
                    stop_triggered = True
                    decel_started = time.monotonic()
                    fsm.transition(Phase.DECELERATING)
                    publish_frames([frame_idle] * 5)
                    continue
                # 観測位置・速度に基づく減速: 手前ゾーンでは低速stickで接近する
                if err <= APPROACH_ENTER_M:
                    publish_frames([frame_stick_slow] * 5)
                else:
                    publish_frames([frame_stick] * 5)
            else:
                publish_frames([frame_idle] * 5)
                if stop.decelerate_done(sample, time.monotonic(), decel_started):
                    # 静止確認: 許容誤差外なら低速バーストで再接近（前回より遠ざかったら打ち切り）
                    can_correct = (
                        err > tol
                        and corrections < MAX_CORRECTIONS
                        and (prev_err is None or err <= prev_err + 0.02)
                    )
                    if can_correct:
                        corrections += 1
                        prev_err = err
                        events.append(
                            {
                                "event": "reapproach",
                                "attempt": corrections,
                                "err": round(err, 4),
                            }
                        )
                        fsm.transition(Phase.WALK)
                        publish_frames([frame_stick_slow] * CORRECTION_BURST_FRAMES)
                        publish_frames([frame_idle] * 4)
                        fsm.transition(Phase.DECELERATING)
                        decel_started = time.monotonic()
                        continue
                    events.append(
                        {
                            "event": "final_settle",
                            "err": round(err, 4),
                            "corrections": corrections,
                        }
                    )
                    publish_frames([frame_stand] * 10)
                    fsm.transition(Phase.SETTLING)
                    break
        if fsm.phase == Phase.DECELERATING:
            raise HarnessError("never_stopped")
        time.sleep(1.0)

        # 4) tail保持 5秒（姿勢・速度を連続監視）
        # 保持時間は sim時刻 で計る — wall時計だと sim pause 中の経過時間も
        # 保持成功へ加算されてしまう（R3: 計画的pauseを異常と混同せず、
        # 停止時間を保持に算入しない）。観測断は latest() が stale を上げて失敗。
        fsm.transition(Phase.TAIL)
        _, _, _, tail_start_sim, _, _ = obs.latest()
        tail_ok = True
        tail_held = 0.0
        while tail_held < 5.0:
            pos, vel, quat, sim_t, _, _ = obs.latest()
            tail_held = sim_t - tail_start_sim
            sample = Sample(pos, vel, quat)
            latch.observe(Phase.TAIL.value, sample)
            tiltmon.observe(sample)
            if (
                latch.fallen
                or math.hypot(vel[0], vel[1]) > 0.1
                or tiltmon.max_tilt_deg > 40
            ):
                tail_ok = False
            time.sleep(0.2)

        pos, vel, quat, sim_t, seq, _ = obs.latest()
        verdict = scorer.verdict(
            ready_latched=gate.ready_latched,
            fall_category=latch.category,
            tail_ok=tail_ok,
            tail_held_s=tail_held,
            final_sample=Sample(pos, vel, quat),
            target_xy=target,
        )
        result["scorer"] = verdict
        result["verdict"] = verdict["verdict"]
        result.update({k: v for k, v in verdict.items() if k != "verdict"})
        result["sim_time_s"] = round(sim_t, 4)
        result["anchor_xy"] = [round(anchor[0], 4), round(anchor[1], 4)]
        fsm.transition(Phase.DONE)
    except (HarnessError, ScheduleOverrun, AuthorizationRefused) as exc:
        result["verdict"] = "FAIL_" + getattr(exc, "reason", "error").upper()
        result["reasons"] = [str(exc)]
    except Exception as exc:
        result["verdict"] = "FAIL_INTERNAL"
        result["reasons"] = [f"{type(exc).__name__}: {exc}"]
    finally:
        try:
            gateway.close()
        except Exception:
            pass
        try:
            obs.stop()
        except Exception:
            pass
        Path(result_path).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return 0 if result.get("verdict") == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
