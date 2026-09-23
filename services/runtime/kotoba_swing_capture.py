#!/usr/bin/env python3
"""swing軌跡capture（SDK container内）— dance一振りの生観測を記録（仮想棒検証用）。

hitbox probe はFK手先位置のみ記録し関節を残さない。地上スイカ判定に
手先姿勢（固定棒の剛体変換）が要るため、本probeは (t, pos, quat, joints)
の生系列を保存する。判定・解析はオフライン — ここでは観測だけを記録する。

入力: <result.json> <manifest.json>
manifest: run_id / purpose="calibration" / arming("1") / expires_in_s /
          boot_expect
契約: 送信は SendGateway 経由のみ。観測は sim_state の実値。
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
    samples = []  # (t_rel, pos, quat, joints_dict)

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
            purpose="calibration",
            profile=SIM_PROFILE,
            sim_boot_id=nonce,
            expires_monotonic=time.monotonic()
            + float(manifest.get("expires_in_s", 120)),
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
            latch.observe(Phase.READY_WAIT.value, Sample(pos, vel, quat))
            if latch.fallen:
                raise HarnessError("prep_fall")
            if gate.observe(Sample(pos, vel, quat), time.monotonic(), None):
                break
        else:
            raise HarnessError("invalid_start_never_ready")
        events.append({"event": "ready", "z": round(pos[2], 3)})
        result["base_pos0"] = [round(v, 4) for v in pos]
        result["base_quat0"] = [round(v, 6) for v in quat]

        frame_idle = gateway.prepare("idle")

        # walk → dance開始 → swing区間を生記録 → walk → stand
        fsm.transition(Phase.WALK)
        frame_walk = gateway.prepare("combo_walk")
        frame_dance = gateway.prepare("combo_dance")

        for i in range(int(0.6 * HZ)):
            gateway.issue(
                frame_walk, now_monotonic=time.monotonic(), command_name="walk"
            )
            time.sleep(1 / HZ)
        for i in range(int(0.6 * HZ)):
            gateway.issue(
                frame_dance, now_monotonic=time.monotonic(), command_name="dance"
            )
            time.sleep(1 / HZ)

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

        rec_t0 = time.monotonic()
        steps = int(SWING_WINDOW_S * HZ)
        for i in range(steps):
            gateway.issue(
                frame_idle, now_monotonic=time.monotonic(), command_name="idle"
            )
            try:
                p, v, q, j, _st, _sq, _nn, tsk = obs.latest()
                latch.observe(Phase.WALK.value, Sample(p, v, q))
                if latch.fallen:
                    raise HarnessError("walk_fall")
                samples.append(
                    {
                        "t": round(time.monotonic() - rec_t0, 4),
                        "pos": [round(x, 4) for x in p],
                        "quat": [round(x, 6) for x in q],
                        "task": tsk,
                        "joints": [round(float(val), 5) for val in j],
                    }
                )
            except HarnessError as exc:
                if "walk_fall" in str(exc):
                    raise
            due = rec_t0 + (i + 1) / HZ
            while time.monotonic() < due:
                time.sleep(0.001)

        # 回復: walk → 静止 → pd_stand
        fsm.transition(Phase.DECELERATING)
        for i in range(int(0.6 * HZ)):
            gateway.issue(
                frame_walk, now_monotonic=time.monotonic(), command_name="walk"
            )
            time.sleep(1 / HZ)
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
            for i in range(int(0.6 * HZ)):
                gateway.issue(
                    frame_stand,
                    now_monotonic=time.monotonic(),
                    command_name="pd_stand",
                )
                time.sleep(1 / HZ)

        fsm.transition(Phase.SETTLING)
        fsm.transition(Phase.TAIL)
        fsm.transition(Phase.DONE)
        result["recovered"] = recovered
        result["fallen"] = latch.fallen
        result["n_samples"] = len(samples)
        result["samples"] = samples
        result["verdict"] = "PASS" if (dance_seen and recovered and not latch.fallen) else "FAIL"
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
        Path(result_path).write_text(
            json.dumps(result, ensure_ascii=False, indent=1)
        )
    return 0 if result.get("verdict") == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
