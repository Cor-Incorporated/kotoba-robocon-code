#!/usr/bin/env python3
"""kotoba_steps_runner — 通常モードの順序付き動作step実行run（SDK container内）。

`kotoba_runner.py`（単一距離marker歩行）の拡張版。manifestの
`motion_steps`（plan検証済みの実値step列）を MotionExecutor の
観測閉ループで逐次実行し、step間はその都度の観測headingで基準を
固定する（古いheadingの使い回しをしない）。

入力: <result.json> <manifest.json>
manifest: run_id / purpose / arming("1") / expires_in_s / boot_expect(必須・実nonce) /
          motion_steps=[{action,direction,target,action_key,label}]

契約は kotoba_runner / kotoba_game_controller と同一:
- 送信は必ず SendGateway 経由（期限・allowlist・単一送信者）
- 観測は受信workerの原子snapshotのみ（制御・採点は実観測に限定）
- 終了時は idle → pd_stand → 立位保持の実観測確認（送信≠成功）
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "/kotoba/harness_src")

from kotoba_harness.auth import RunManifest, SendGateway
from kotoba_harness.errors import AuthorizationRefused, HarnessError
from kotoba_harness.motion import MotionExecutor, MotionStep
from kotoba_harness.trial import (
    FallLatch,
    Phase,
    ReadyGate,
    TrialFSM,
    up_vector_tilt_deg,
)

from kotoba_runner import Sample
from kotoba_strike_probe import StrikeObserver
from kotoba_game_controller import _verify_end_hold

HZ = 20.0
STATE_WRITE_HZ = 10.0
OBS_LOST_ABORT_S = 2.0
# 通常runの操縦範囲 — run開始位置からの水平距離上限（game arenaと同一値）。
# 継続jogが境界へ近づいたらexecutorが手前で減速停止する。
ARENA_R_M = 5.0
# 継続jogの外部制御 — APIがrun_dirへ書き込む指令・生存確認。
CMD_PATH = Path("/kotoba/run/run_cmd.json")
HB_PATH = Path("/kotoba/run/heartbeat.json")
HB_STALE_S = 2.5
HB_GRACE_S = 3.5


def main() -> int:
    result_path = sys.argv[1]
    manifest = json.loads(Path(sys.argv[2]).read_text())
    result = {"run_id": manifest.get("run_id", "unknown"), "events": []}
    events = result["events"]
    raw_steps = manifest.get("motion_steps") or []

    expect = manifest.get("boot_expect")
    obs = None
    gateway = None
    try:
        # plan側のMotionStepSpec {action,direction,target,pace} → executorの
        # MotionStep（target単位は同一: translate=m / turn=rad）
        steps = [
            MotionStep(
                action=s["action"],
                direction=s["direction"],
                target=float(s["target"]),
                action_key=str(s.get("action_key") or ""),
                label=str(s.get("label") or ""),
                pace=str(s.get("pace") or "walk"),
            )
            for s in raw_steps
        ]
        if not steps:
            raise HarnessError("no_motion_steps")
        result["program"] = [
            {
                "action": s.action,
                "direction": s.direction,
                "target": s.target,
                "action_key": s.action_key,
                "label": s.label,
            }
            for s in steps
        ]

        expected = int(expect) if expect and str(expect).isdigit() else None
        obs = StrikeObserver(expected_nonce=expected)
        deadline = time.monotonic() + 15
        while True:
            try:
                pos, vel, quat, joints, sim_t, seq0, nonce, task = obs.latest()
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
            purpose=manifest.get("purpose", "evaluation"),
            profile=__import__(
                "kotoba_harness.auth", fromlist=["profile_for_capabilities"]
            ).profile_for_capabilities(manifest.get("capabilities")),
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

        frame_idle = gateway.prepare("idle")
        frame_walk = gateway.prepare("combo_walk")

        def issue(frame, name):
            gateway.issue(
                frame, now_monotonic=time.monotonic(), command_name=name
            )

        # walkモードへ入る（方向指令はwalk taskでのみ有効）
        fsm.transition(Phase.WALK)
        for _ in range(int(0.6 * HZ)):
            issue(frame_walk, "combo_walk")
            time.sleep(1 / HZ)
        events.append({"event": "walk_mode"})

        ex = MotionExecutor(
            steps,
            bound_center=(pos[0], pos[1]),
            bound_r=ARENA_R_M,
        )
        ex.begin(pos, quat)
        state_path = Path(result_path).parent / "run_state.json"
        last_state_write = 0.0

        def write_state(t_mono):
            nonlocal last_state_write
            if t_mono - last_state_write < 1 / STATE_WRITE_HZ:
                return
            last_state_write = t_mono
            try:
                d = ex.prog.to_dict()
                d["fallen"] = latch.fallen
                state_path.write_text(json.dumps(d, ensure_ascii=False))
            except Exception:
                pass

        obs_dead_since = None
        shutdown_reason = None
        jog_since_mono = None
        run_id = manifest["run_id"]
        while True:
            t0 = time.monotonic()
            try:
                pos, vel, quat, joints, sim_t, seq_i, nonce_i, task = obs.latest()
            except HarnessError:
                # 観測断: stickを出さず観測復帰を待つ（executorの時計も進めない）
                if obs_dead_since is None:
                    obs_dead_since = t0
                if t0 - obs_dead_since > OBS_LOST_ABORT_S:
                    shutdown_reason = "obs_lost"
                    break
                time.sleep(0.02)
                continue
            obs_dead_since = None
            sample = Sample(pos, vel, quat)
            latch.observe(Phase.WALK.value, sample)
            if latch.fallen:
                shutdown_reason = "fallen"
                break

            # 継続jog中の外部制御: 停止指令（通常停止）と操作者heartbeat。
            # heartbeat欠落は操作者消失・通信断とみなし減速停止する。
            if ex.jogging:
                if jog_since_mono is None:
                    jog_since_mono = t0
                try:
                    cdat = json.loads(CMD_PATH.read_text())
                    if (
                        cdat.get("type") == "stop"
                        and cdat.get("run_id") == run_id
                    ):
                        ex.request_stop("stop")
                except FileNotFoundError:
                    pass
                except (ValueError, TypeError, json.JSONDecodeError):
                    pass
                if t0 - jog_since_mono > HB_GRACE_S:
                    hb_fresh = False
                    try:
                        hb = json.loads(HB_PATH.read_text())
                        hb_t = float(hb.get("t_wall") or 0.0)
                        hb_fresh = (
                            hb.get("run_id") == run_id
                            and 0.0 <= time.time() - hb_t <= HB_STALE_S
                        )
                    except (
                        FileNotFoundError,
                        ValueError,
                        TypeError,
                        json.JSONDecodeError,
                    ):
                        hb_fresh = False
                    if not hb_fresh:
                        ex.request_stop("heartbeat_lost")

            stick = ex.tick(pos, quat, vel, now=t0)
            if stick is not None:
                issue(gateway.prepare_move(*stick), "motion_stick")
            else:
                issue(frame_idle, "idle_tick")
            write_state(t0)
            if ex.done:
                break
            if t0 > rm.expires_monotonic:
                shutdown_reason = "manifest_expired"
                break
            due = t0 + 1 / HZ
            while time.monotonic() < due:
                time.sleep(0.001)

        # executorイベントをrun記録へ（監査: step毎の実測進捗）
        events.extend(ex.events)

        # ---- 終了処理: idle → pd_stand → 立位保持を実観測で確認 ----
        # 失敗・中断時も立位へ戻す試行は行う（転倒していなければ有効）。
        # 一次失敗理由は終了処理の二次失敗で上書きしない。
        end_hold_verified = False
        for _ in range(10):
            try:
                issue(frame_idle, "final_idle")
            except AuthorizationRefused:
                break
            time.sleep(1 / HZ)
        clean_end = ex.done and not ex.failed and shutdown_reason is None
        primary_reason = shutdown_reason or ex.prog.reason
        if clean_end:
            try:
                frame_stand = gateway.prepare("combo_pd_stand")
                for _ in range(10):
                    issue(frame_stand, "combo_pd_stand")
                    time.sleep(1 / HZ)
            except AuthorizationRefused:
                clean_end = False
            fsm.transition(Phase.DECELERATING)
            fsm.transition(Phase.SETTLING)
            fsm.transition(Phase.TAIL)
            if clean_end:
                end_hold_verified = _verify_end_hold(obs, latch, events)
                events.append(
                    {"event": "end_hold", "verified": end_hold_verified}
                )
            fsm.transition(Phase.DONE)
        else:
            # 失敗経路は合法遷移 WALK→ABORT→DONE（WALK→DONEは違法で
            # 一次原因を FAIL_ILLEGAL_TRANSITION へ上書きした実害あり）。
            # 立位回復を試み、保持確認も記録する（転倒検知へ寄与）。
            try:
                frame_stand = gateway.prepare("combo_pd_stand")
                for _ in range(10):
                    issue(frame_stand, "combo_pd_stand")
                    time.sleep(1 / HZ)
            except Exception as exc:
                events.append(
                    {"event": "stand_recover_error",
                     "error": str(exc)[:200]}
                )
            fsm.transition(Phase.ABORT)
            events.append(
                {"event": "abort", "reason": primary_reason}
            )
            try:
                end_hold_verified = _verify_end_hold(obs, latch, events)
            except Exception as exc:
                events.append(
                    {"event": "end_hold_error", "error": str(exc)[:200]}
                )
            events.append(
                {"event": "end_hold", "verified": end_hold_verified}
            )
            fsm.transition(Phase.DONE)

        # 最終観測は保持・停止処理の「後」に取る — 残動中の値を
        # 停止検証と誤認させない（B12）。
        try:
            pos, vel, quat, joints, *_ = obs.latest()
            result["speed_mps"] = round(math.hypot(vel[0], vel[1]), 4)
            result["height_m"] = round(pos[2], 4)
            result["tilt_deg"] = round(up_vector_tilt_deg(quat), 1)
        except Exception:
            pass

        steps_done = ex.prog.step_index + (
            1 if ex.prog.status == "completed" else 0
        )

        result["control"] = {
            "steps_total": len(steps),
            "steps_done": steps_done,
            "motion_status": ex.prog.status,
            "motion_reason": ex.prog.reason,
            "shutdown_reason": shutdown_reason,
            "end_hold_verified": end_hold_verified,
            "fallen": latch.fallen,
        }
        # verdict: 全step完了＋立位保持の実観測でのみPASS。
        if latch.fallen:
            result["verdict"] = "FAIL_FALLEN"
        elif clean_end and end_hold_verified:
            result["verdict"] = "PASS"
        elif clean_end:
            result["verdict"] = "UNKNOWN_END_HOLD"
        else:
            result["verdict"] = "FAIL_" + (
                primary_reason or "no_clean_end"
            ).upper()
            result["reasons"] = [primary_reason or "no_clean_end"]
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
    sys.exit(main())
