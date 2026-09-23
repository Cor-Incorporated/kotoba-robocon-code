#!/usr/bin/env python3
"""ことばでロボコン 起動立位マネージャ v3（SDK container内・常駐）。

ライフサイクル（ctl先起動+連続pd_stand+直立検証）:
  0. ログwatcher（外部）が ctl の「Entered motion [ passive ]」を検知済み
  1. passive検知フラグを待つ
  2. pd_stand burst（130frame=6.5秒）を送信 → ロボットが立ち上がる
  3. 直立を検証: z∈[0.75,0.90]・|qw|≈1・低速度 を5秒保持
  4. boot-ready.json に boot_nonce を書き出し → API が移動受付を解禁
  5. 以後も転倒監視を継続し、崩れたら boot-lost を書く

権限: 事前承認された sim初期化専用操作（combo_pd_stand のみ許可）。
"""

from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, "/kotoba/harness_src")

import lcm

from kotoba_harness.auth import SIM_PROFILE, RunManifest, SendGateway
from kotoba_harness.trial import (
    STAND_HEIGHT_MAX_M,
    STAND_HEIGHT_MIN_M,
    up_vector_tilt_deg,
)

RUNTIME = Path("/kotoba/runtime")
URL = "udpm://239.255.76.67:7667?ttl=0"
PASSIVE_FLAG = RUNTIME / "passive-detected"
READY_FILE = RUNTIME / "boot-ready.json"
LOST_FILE = RUNTIME / "boot-lost.json"
CLOCK_FP = 0x4B544F4241434C31

samples = []  # (mono, z, |qw|, speed_h, pos, quat)
clock_nonce = None
clock_sim_t = None


def _on_state(ch, data):
    try:
        fp, ts, n = struct.unpack_from(">qdi", data, 0)
        off = 20 + 3 * n * 8
        pos = struct.unpack_from(">3d", data, off)
        vel = struct.unpack_from(">3d", data, off + 24)
        quat = struct.unpack_from(">4d", data, off + 48)
        speed = (vel[0] ** 2 + vel[1] ** 2) ** 0.5
        samples.append((time.monotonic(), pos[2], abs(quat[0]), speed, pos, quat))
    except struct.error:
        pass


def _on_clock(ch, data):
    global clock_nonce, clock_sim_t
    if len(data) == 40:
        fp, nonce, seq, sim_t, mono = struct.unpack(">qqqdd", data)
        if fp == CLOCK_FP:
            clock_nonce = nonce
            clock_sim_t = sim_t


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)  # atomic replace


def main() -> int:
    global clock_nonce
    RUNTIME.mkdir(parents=True, exist_ok=True)
    h = lcm.LCM(URL)
    h.subscribe("sim_state", _on_state)
    h.subscribe("kotoba_sim_clock", _on_clock)

    # arming + sim初期化専用manifest
    manifest = RunManifest(
        run_id="sim-init-stand",
        purpose="sim_initialization",
        profile=SIM_PROFILE,
        sim_boot_id="pending",
        expires_monotonic=time.monotonic() + 120,
    )
    gateway = SendGateway(
        manifest,
        h,
        "virtual_gamepad/gamepad_keys",
        now_monotonic=time.monotonic(),
        sim_mode_confirmed=True,
    )

    # 1) passive検知フラグを待つ（ctl先起動の場合はctl起動直後に既に存在する）
    boot_t = time.monotonic()
    while not PASSIVE_FLAG.exists():
        h.handle_timeout(20)
        if time.monotonic() - boot_t > 60:
            print("NO_PASSIVE_ENTRY", flush=True)
            return 2
    print(f"passive detected t+{time.monotonic() - boot_t:.1f}s", flush=True)

    # 2) pd_stand burst: 130frame（6.5秒）を20Hzで送信
    #    LB+A を6.5秒間保持 — ボタン保持が遷移の成立条件（8/28レシピ）
    frame = gateway.prepare("combo_pd_stand")
    send_t0 = time.monotonic()
    for i in range(130):
        gateway.issue(
            frame, now_monotonic=time.monotonic(), command_name="combo_pd_stand"
        )
        due = send_t0 + (i + 1) / 20.0
        while time.monotonic() < due:
            h.handle_timeout(10)
    print(f"burst sent (130 frames / 6.5s)", flush=True)

    # 3) 直立検証: 5秒連続で高さ窓・姿勢・低速度
    #    F03: 古い標本・sim時刻凍結では検証しない。窓外は蓄積リセット。
    stand_since = None
    verified = False
    last_sim_t = None
    last_sim_change = time.monotonic()
    verify_deadline = time.monotonic() + 25
    verify_t0 = time.monotonic()
    trajectory = []
    last_traj_t = 0.0
    while time.monotonic() < verify_deadline:
        h.handle_timeout(10)
        if not samples:
            continue
        recv, z, qw_abs, speed, pos, quat = samples[-1]
        fresh = (time.monotonic() - recv) <= 0.2
        tilt = up_vector_tilt_deg(quat)
        in_window = (
            fresh
            and clock_nonce is not None
            and STAND_HEIGHT_MIN_M <= z <= STAND_HEIGHT_MAX_M
            and tilt <= 15.0
            and speed <= 0.1
        )
        # sim時刻進行停止の検出
        if clock_sim_t is not None:
            if clock_sim_t == last_sim_t:
                if time.monotonic() - last_sim_change > 1.0:
                    print("CLOCK_STALLED", flush=True)
                    return 5
            else:
                last_sim_t = clock_sim_t
                last_sim_change = time.monotonic()
        if time.monotonic() - last_traj_t >= 0.2:
            last_traj_t = time.monotonic()
            trajectory.append(
                {
                    "t": round(time.monotonic() - verify_t0, 2),
                    "z": round(z, 3),
                    "qw": round(qw_abs, 3),
                    "fresh": fresh,
                    "bursts": 1,
                }
            )
        if in_window:
            if stand_since is None:
                stand_since = time.monotonic()
            if time.monotonic() - stand_since >= 5.0:
                verified = True
                break
        else:
            stand_since = None
    if not verified:
        write_json(RUNTIME / f"stand-trajectory-{int(time.time())}.json", trajectory)
        print("STAND_VERIFY_TIMEOUT", flush=True)
        return 3

    # 4) boot-ready.json 公開（F02修正: 原子的書出し・受信clock由来のnonce）
    write_json(
        READY_FILE,
        {
            "boot_nonce": clock_nonce,
            "z": round(samples[-1][1], 4),
            "tilt_deg": round(up_vector_tilt_deg(samples[-1][5]), 2),
            "verified_at": time.time(),
            "standing": True,
        },
    )
    write_json(RUNTIME / "stand-trajectory-final.json", trajectory)
    print(f"STAND_VERIFIED nonce={clock_nonce} z={samples[-1][1]:.3f}", flush=True)

    # 5) 転倒監視＋ready失効（常駐）
    #    API側の _boot_ready は verified_at が 3600s 以内を要求する。
    #    立位が継続している間はここで verified_at を定期更新し、
    #    「監視プロセスが生存し立位を確認し続けている」ことを証明する
    #    （更新が止まれば鮮度切れ→ゲート閉鎖、転倒/観測断なら即時削除+lost）。
    last_recv = time.monotonic()
    last_msim = clock_sim_t
    last_msim_change = time.monotonic()
    last_refresh = time.monotonic()
    while True:
        h.handle_timeout(50)
        now = time.monotonic()
        if not samples:
            continue
        z = samples[-1][1]
        tilt = up_vector_tilt_deg(samples[-1][5])
        lost = None
        if now - last_recv > 1.0:
            lost = "observation_gap"
        elif z < 0.5 or tilt > 30.0:
            lost = "fall"
        elif clock_sim_t is not None and clock_sim_t == last_msim:
            if now - last_msim_change > 1.0:
                lost = "sim_clock_stalled"
        else:
            last_recv = now
            if clock_sim_t is not None:
                if clock_sim_t != last_msim:
                    last_msim = clock_sim_t
                    last_msim_change = now
            if now - last_refresh >= 30.0 and clock_nonce is not None:
                last_refresh = now
                write_json(
                    READY_FILE,
                    {
                        "boot_nonce": clock_nonce,
                        "z": round(z, 4),
                        "tilt_deg": round(tilt, 2),
                        "verified_at": time.time(),
                        "standing": True,
                    },
                )
        if lost:
            if READY_FILE.exists():
                READY_FILE.unlink()  # ready失効
            write_json(
                LOST_FILE, {"lost_at": time.time(), "reason": lost, "z": round(z, 3)}
            )
            print(f"BOOT_LOST reason={lost}", flush=True)
            return 4


if __name__ == "__main__":
    raise SystemExit(main())
