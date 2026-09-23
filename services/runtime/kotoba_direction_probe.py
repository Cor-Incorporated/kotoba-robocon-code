#!/usr/bin/env python3
"""方向能力probe（SDK container内）— 実SDK経路で前後・左右・旋回・ゼロ指令を実測。

ことばでスイカ割りの連続操縦（B1）に先立ち、virtual_gamepadの各stick軸が
実シミュレータで期待方向の運動を生むかを実測・照合する。

入力: <result.json> <manifest.json>
manifest: run_id / purpose="calibration" / arming("1") / expires_in_s / boot_expect

手順: 立位確認 → combo_walk → 各方向で burst(2s,20Hz) 送信し body座標系の
変位・yaw変化・peak速度を計測 → idleで静止確認 → 最後に pd_stand。
結果: result.json に方向ごとの実測値・期待符号との照合・総合verdictを残す。

契約（kotoba_runnerと同一）:
- 送信は SendGateway 経由のみ（期限・arming・profile bounds検証済みの出口）
- 受信workerは独立スレッドで送信中も転倒監視が止まらない
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
from kotoba_harness.trial import FallLatch, Phase, ReadyGate, TrialFSM

from kotoba_runner import Observer, Sample

HZ = 20.0
# virtual_gamepadの入力timeoutは200ms — 連続streamが必要。
# SDK側で remote_command_lpf が有効（cutoff 0.1Hz, τ≈1.6s）のため、
# 2秒burstでは定常応答の7割に届かない。能力判定は4秒burstで行う。
BURST_FRAMES = 80
IDLE_FRAMES = 20
SETTLE_TIMEOUT_S = 2.5
SETTLE_SPEED_MPS = 0.05

# (名前, fwd, lat, yaw) — stick空間。implied速度はprofile bounds内に収める。
# 弱い軸は上限まで試す（能力の有無を判定するのが目的）。
DIRECTIONS = [
    ("forward", 0.6, 0.0, 0.0),
    ("backward", -0.85, 0.0, 0.0),
    ("left", 0.0, 1.0, 0.0),
    ("right", 0.0, -1.0, 0.0),
    ("turn_left", 0.0, 0.0, 0.8),
    ("turn_right", 0.0, 0.0, -0.8),
    ("zero", 0.0, 0.0, 0.0),
]

# 判定閾値（実測ノイズを考慮した最小限の符号検証）
FWD_MIN_M = 0.15
BACK_MAX_M = -0.10
LAT_MIN_M = 0.04
TURN_MIN_DEG = 6.0


def _yaw_of(quat) -> float:
    w, x, y, z = quat
    return math.atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _judge(name: str, m: dict) -> tuple[bool, str]:
    """実測値と期待方向の照合。fail理由は反証可能な文字列。"""
    if not m["settle_ok"]:
        return False, "no_settle_after_burst"
    if name == "forward":
        ok = m["disp_fwd_m"] > FWD_MIN_M
        return ok, f"disp_fwd={m['disp_fwd_m']:.3f}>{FWD_MIN_M}"
    if name == "backward":
        ok = m["disp_fwd_m"] < BACK_MAX_M
        return ok, f"disp_fwd={m['disp_fwd_m']:.3f}<{BACK_MAX_M}"
    if name in ("left", "right"):
        ok = abs(m["disp_lat_m"]) > LAT_MIN_M
        return ok, f"|disp_lat|={abs(m['disp_lat_m']):.3f}>{LAT_MIN_M}"
    if name in ("turn_left", "turn_right"):
        ok = abs(m["dyaw_deg"]) > TURN_MIN_DEG
        return ok, f"|dyaw|={abs(m['dyaw_deg']):.1f}>{TURN_MIN_DEG}"
    if name == "zero":
        # ゼロ指令の判定は「最終的に止まること」— LPF減衰中の残存速度は
        # 能力欠陥ではなく停止遅延として t_last_moving_s に記録する。
        ok = m["settle_ok"] and m["disp_norm_m"] < 0.10
        return ok, f"disp={m['disp_norm_m']:.3f},t_last_move={m['t_last_moving_s']}s"
    return False, "unknown_direction"


def main() -> int:
    result_path = sys.argv[1]
    manifest = json.loads(Path(sys.argv[2]).read_text())
    result = {"run_id": manifest.get("run_id", "unknown"), "events": [], "directions": {}}
    events = result["events"]

    expect = manifest.get("boot_expect")
    obs = None
    gateway = None
    try:
        expected = int(expect) if expect and str(expect).isdigit() else None
        obs = Observer(expected_nonce=expected)
        deadline = time.monotonic() + 15
        while True:
            try:
                pos, vel, quat, sim_t, seq, nonce = obs.latest()
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

        # 1) 既存立位の検証（stand managerが確立済みの前提）
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

        def stream(frame, n, command_name="move"):
            """n frames @20Hz送信。送信中も観測・転倒監視を継続する。

            戻り値: (peak_speed, t_last_moving) — t_last_moving は最後に
            |v|>=SETTLE_SPEED_MPS を観測したburst内時刻（停止遅延の計測用）。
            """
            t0 = time.monotonic()
            peak_speed = 0.0
            t_last_moving = 0.0
            for i in range(n):
                gateway.issue(
                    frame,
                    now_monotonic=time.monotonic(),
                    command_name=command_name,
                )
                try:
                    p, v, q, *_ = obs.latest()
                    latch.observe(Phase.WALK.value, Sample(p, v, q))
                    if latch.fallen:
                        raise HarnessError("walk_fall")
                    speed = math.hypot(v[0], v[1])
                    peak_speed = max(peak_speed, speed)
                    if speed >= SETTLE_SPEED_MPS:
                        t_last_moving = time.monotonic() - t0
                except HarnessError as exc:
                    if "walk_fall" in str(exc):
                        raise
                    # 観測の一時的staleはburst継続（latestが復帰を待つ）
                due = t0 + (i + 1) / HZ
                while time.monotonic() < due:
                    time.sleep(0.001)
            return peak_speed, t_last_moving

        def settle() -> bool:
            t_end = time.monotonic() + SETTLE_TIMEOUT_S
            while time.monotonic() < t_end:
                try:
                    _, v, _, *_ = obs.latest()
                    if math.hypot(v[0], v[1]) < SETTLE_SPEED_MPS:
                        return True
                except HarnessError:
                    pass
                time.sleep(0.05)
            return False

        # 2) walk modeへ遷移
        fsm.transition(Phase.WALK)
        frame_walk = gateway.prepare("combo_walk")
        frame_idle = gateway.prepare("idle")
        stream(frame_walk, 10, "combo_walk")
        stream(frame_idle, IDLE_FRAMES, "idle")
        if not settle():
            raise HarnessError("no_settle_after_walk_combo")

        # 3) 各方向を実測
        results = {}
        for name, fwd, lat, yaw in DIRECTIONS:
            pos0, _, quat0, *_ = obs.latest()
            yaw0 = _yaw_of(quat0)
            frame = gateway.prepare_move(fwd, lat, yaw)
            peak_speed, t_last_moving = stream(
                frame, BURST_FRAMES, f"move_{name}"
            )
            # burst終端の変位を計測（stream中の最新観測）
            pos1, _, quat1, *_ = obs.latest()
            yaw1 = _yaw_of(quat1)
            dx, dy = pos1[0] - pos0[0], pos1[1] - pos0[1]
            h = (math.cos(yaw0), math.sin(yaw0))
            disp_fwd = dx * h[0] + dy * h[1]
            disp_lat = -dx * h[1] + dy * h[0]  # heading左90°成分
            dyaw = _wrap_pi(yaw1 - yaw0)
            stream(frame_idle, IDLE_FRAMES, "idle")
            settle_ok = settle()
            m = {
                "stick": [fwd, lat, yaw],
                "disp_fwd_m": round(disp_fwd, 4),
                "disp_lat_m": round(disp_lat, 4),
                "disp_norm_m": round(math.hypot(dx, dy), 4),
                "dyaw_deg": round(math.degrees(dyaw), 2),
                "peak_speed": round(peak_speed, 3),
                "t_last_moving_s": round(t_last_moving, 2),
                "settle_ok": settle_ok,
            }
            ok, detail = _judge(name, m)
            m["pass"] = ok
            m["detail"] = detail
            results[name] = m
            events.append({"event": "direction", "name": name, **m})

        # 4) 左右・旋回の符号一貫性（反対指令は反対変位を生むこと）
        if results["left"]["pass"] and results["right"]["pass"]:
            sign_consistent = (
                results["left"]["disp_lat_m"] * results["right"]["disp_lat_m"] < 0
            )
            results["lateral_sign_consistent"] = sign_consistent
            if not sign_consistent:
                results["left"]["pass"] = results["right"]["pass"] = False
        if results["turn_left"]["pass"] and results["turn_right"]["pass"]:
            sign_consistent = (
                results["turn_left"]["dyaw_deg"] * results["turn_right"]["dyaw_deg"]
                < 0
            )
            results["yaw_sign_consistent"] = sign_consistent
            if not sign_consistent:
                results["turn_left"]["pass"] = results["turn_right"]["pass"] = False

        # 計測結果は先に確定させる（以後のFSM/stand遷移で例外が起きても残る）
        result["directions"] = results
        measured = {k: v for k, v in results.items() if isinstance(v, dict)}
        result["verdict"] = (
            "PASS" if all(m.get("pass") for m in measured.values()) else "FAIL"
        )

        # 5) pd_standへ復帰（FSM: WALK→DECELERATING→SETTLING→TAIL→DONE）
        fsm.transition(Phase.DECELERATING)
        frame_stand = gateway.prepare("combo_pd_stand")
        stream(frame_stand, 10, "combo_pd_stand")
        fsm.transition(Phase.SETTLING)
        fsm.transition(Phase.TAIL)
        fsm.transition(Phase.DONE)
    except (HarnessError, AuthorizationRefused) as exc:
        result["verdict"] = "FAIL_" + getattr(exc, "reason", "error").upper()
        result["reasons"] = [str(exc)]
    except Exception as exc:
        result["verdict"] = "FAIL_INTERNAL"
        result["reasons"] = [f"{type(exc).__name__}: {exc}"]
    finally:
        if gateway is not None:
            try:
                # 中断時も入力を明示的に止める（200ms watchdog任せにしない）
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
