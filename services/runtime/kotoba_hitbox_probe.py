#!/usr/bin/env python3
"""hitbox判定probe（SDK container内）— 手先軌跡×スイカhit volume の実測（B4）。

dance一振りの手先(LINK_ELBOW_END_L/R)world軌跡をFK導出し、
manifest.target のhit_volumeへの最小距離で HIT/MISS を判定する。
target省略時は軌跡包絡のみ報告（狙い位置設計用のmeasureモード）。

入力: <result.json> <manifest.json>
manifest: run_id / purpose="calibration" / arming("1") / expires_in_s /
          boot_expect / target=[x,y,z]? / hit_radius=0.20?
契約: 送信は SendGateway 経由のみ。判定は観測手先位置のみに基づく。
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "/kotoba/harness_src")

from kotoba_harness.auth import RunManifest, SIM_PROFILE, SendGateway
from kotoba_harness.errors import AuthorizationRefused, HarnessError
from kotoba_harness.hitbox import HitRecorder
from kotoba_harness.trial import FallLatch, Phase, ReadyGate, TrialFSM

from kotoba_runner import Sample
from kotoba_strike_probe import (
    DANCE_START_TIMEOUT_S,
    RECOVER_TIMEOUT_S,
    SETTLE_SPEED_MPS,
    STAND_Z_MIN,
    SWING_WINDOW_S,
    StrikeObserver,
)

HZ = 20.0


def main() -> int:
    result_path = sys.argv[1]
    manifest = json.loads(Path(sys.argv[2]).read_text())
    result = {"run_id": manifest.get("run_id", "unknown"), "events": []}
    events = result["events"]
    target = manifest.get("target")
    target_base = manifest.get("target_base")  # base0座標系の狙い位置
    hit_radius = float(manifest.get("hit_radius", 0.20))

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

        rm = RunManifest(
            run_id=manifest["run_id"],
            purpose=manifest.get("purpose", "calibration"),
            profile=SIM_PROFILE,
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
        base_quat0 = quat
        if target is None and target_base is not None:
            # base0座標系の狙い位置 → world:  dance中のbase姿勢変動に
            # 引っ張られないよう、ready時の姿勢で一度だけ変換する
            from kotoba_harness.kinematics import _matvec, _quat_to_rot

            R0 = _quat_to_rot(base_quat0)
            off = _matvec(R0, target_base)
            target = tuple(base_pos0[i] + off[i] for i in range(3))

        frame_idle = gateway.prepare("idle")
        recorder = HitRecorder()

        def stream(frame, seconds, track=None):
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

        # 一振り: dance開始 → swing区間で手先軌跡を記録
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
                *_r, tsk = obs.latest()
                if tsk == "dance":
                    dance_seen = True
                    break
            except HarnessError:
                pass
            time.sleep(0.05)
        result["dance_task_seen"] = dance_seen
        if not dance_seen:
            raise HarnessError("dance_not_started")

        def track_hands(p, v, q, j, tsk, t):
            recorder.record(t, p, q, j)

        stream(frame_idle, SWING_WINDOW_S, track=track_hands)

        # 回復: dance→walk→静止→pd_stand（B2検証済み手順）
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
                if math.hypot(v[0], v[1]) < SETTLE_SPEED_MPS and p[2] > STAND_Z_MIN:
                    recovered = True
                    break
            except HarnessError as exc:
                if "recover_fall" in str(exc):
                    raise
            time.sleep(0.05)

        if recovered:
            frame_stand = gateway.prepare("combo_pd_stand")
            stream(frame_stand, 0.6)

        fsm.transition(Phase.SETTLING)
        fsm.transition(Phase.TAIL)
        fsm.transition(Phase.DONE)

        result["path"] = recorder.path_summary()
        result["recovered"] = recovered
        result["fallen"] = latch.fallen
        result["base_pos"] = [round(v, 3) for v in base_pos0]
        result["base_quat"] = [round(v, 4) for v in base_quat0]

        if target is not None:
            result["judge"] = recorder.judge(target, hit_radius)
            ok = (
                dance_seen
                and recovered
                and not latch.fallen
                and result["judge"]["hit"]
            )
            result["verdict"] = "PASS" if ok else "FAIL"
            if not ok:
                result["reasons"] = [
                    f"dance={dance_seen}", f"hit={result['judge'].get('hit')}",
                    f"min_dist={result['judge'].get('min_dist_m')}",
                    f"recovered={recovered}", f"fallen={latch.fallen}",
                ]
        else:
            # measureモード — 軌跡包絡の報告のみ
            result["verdict"] = "PASS" if (dance_seen and recovered and not latch.fallen) else "FAIL"

        # 軌跡全点も保存（オフライン検証用）
        result["trajectory"] = [
            {"t": round(t, 2), "L": [round(v, 3) for v in hl], "R": [round(v, 3) for v in hr]}
            for t, hl, hr in recorder.samples
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
