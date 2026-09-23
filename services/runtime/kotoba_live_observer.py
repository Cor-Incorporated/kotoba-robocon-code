#!/usr/bin/env python3
"""ことばでロボコン 常駐観測サービス（SDK container内・read-only）。

sim_state + kotoba_sim_clock + task_state を単一受信スレッドで結合し、
原子snapshotを /kotoba/runtime/live.json へ定期書き出しする。

用途:
- UIの2D map（/api/obs/live が読む live.json）
- G2 renderer の姿勢フィード（joint_position を含む）
- boot_nonce の変化で sim reset を検出する

送信は一切しない（観測専用）。単一受信所有者＋ロック付きsnapshot。
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

from kotoba_harness.observer import ClockGate

URL = "udpm://239.255.76.67:7667?ttl=0"
CLOCK_FP = 0x4B544F4241434C31
HEADER = struct.Struct(">qdi")
CLOCK = struct.Struct(">qqqdd")
RUNTIME = Path("/kotoba/runtime")
LIVE_PATH = RUNTIME / "live.json"
# The renderer polls at 30Hz. Keep the latest authoritative observation at the
# same cadence; the 500Hz LCM receive/ClockGate path remains unchanged.
WRITE_HZ = 30.0
STATE_FP = 0x2D53D9E29374E48E  # data::SimState getHash() — 受信時に確認のみ


def _decode_task(data: bytes) -> str | None:
    """TaskState = fingerprint(int64) + string current_motion_task_name."""
    try:
        n = struct.unpack_from(">i", data, 8)[0]
        return data[12 : 12 + n].decode("utf-8", "replace")
    except (struct.error, IndexError):
        return None


class LiveObserver:
    """単一受信スレッド + ロック付きsnapshot + 書き出しループ。"""

    def __init__(self) -> None:
        self.handle = lcm.LCM(URL)
        self._lock = threading.Lock()
        self._state = None  # (recv_mono, fp, pos, vel, quat, joints)
        self._clock = None  # (recv_mono, nonce, seq, sim_t, src_mono)
        self._task = None  # (recv_mono, name)
        self._bound = None  # (recv_mono, obs_wall, pos, vel, quat, joints, sim_t, seq, nonce, task)
        self._recv_total = 0
        self._gate = ClockGate()  # seq単調・boot遷移の共通検証
        self._running = True
        self._exception = None
        self.handle.subscribe("sim_state", self._on_state)
        self.handle.subscribe("kotoba_sim_clock", self._on_clock)
        self.handle.subscribe("task_state", self._on_task)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _on_state(self, _ch, data):
        try:
            fp, _ts, n = HEADER.unpack_from(data, 0)
            off = HEADER.size + 3 * n * 8
            pos = struct.unpack_from(">3d", data, off)
            vel = struct.unpack_from(">3d", data, off + 24)
            quat = struct.unpack_from(">4d", data, off + 48)
            joints = struct.unpack_from(f">{n}d", data, HEADER.size)
        except struct.error:
            return
        if not all(map(math.isfinite, (*pos, *vel, *quat))):
            return
        recv = time.monotonic()
        with self._lock:
            self._state = (recv, fp, pos, vel, quat, list(joints))
            self._recv_total += 1
            self._try_bind(recv)

    def _on_clock(self, _ch, data):
        if len(data) != 40:
            return
        fp, nonce, seq, sim_t, mono = CLOCK.unpack(data)
        if fp != CLOCK_FP:
            return
        recv = time.monotonic()
        with self._lock:
            ok, boot_changed = self._gate.accept(nonce, seq, sim_t)
            if boot_changed:
                # 新boot: 旧seq基準・旧state/clock/boundを持ち越さない
                # （旧boot最終stateが新boot clockへ誤結合するのを防ぐ）
                self._state = None
                self._clock = None
                self._bound = None
            if not ok:
                return
            self._clock = (recv, nonce, seq, sim_t, mono)
            self._try_bind(recv)

    def _on_task(self, _ch, data):
        name = _decode_task(data)
        if name is None:
            return
        with self._lock:
            self._task = (time.monotonic(), name)

    def _try_bind(self, recv: float) -> None:
        """同一publisher周期（受信差30ms以内）の state+clock を結合。"""
        if self._state is None or self._clock is None:
            return
        s_recv, _fp, pos, vel, quat, joints = self._state
        c_recv, nonce, seq, sim_t, _mono = self._clock
        if abs(s_recv - c_recv) > 0.03:
            return
        task = self._task[1] if self._task else None
        # obs_wall は「この観測が実際に受信されたwall時刻」。書出し時刻とは分離する。
        self._bound = (recv, time.time(), pos, vel, quat, joints, sim_t, seq, nonce, task)

    def _loop(self) -> None:
        try:
            while self._running:
                self.handle.handle_timeout(50)
        except Exception as exc:
            self._exception = exc

    def snapshot(self) -> dict | None:
        with self._lock:
            b = self._bound
            meta = (
                self._recv_total,
                self._gate.dropped_late,
                self._gate.boot_changes,
                self._gate.dropped_frozen,
                self._gate.dropped_retired,
            )
        if b is None:
            return None
        recv, obs_wall, pos, vel, quat, joints, sim_t, seq, nonce, task = b
        age = time.monotonic() - recv
        return {
            "wall": time.time(),  # 書出し時刻（監査用。fresh判定には使わない）
            "obs_wall": obs_wall,  # 物理観測の受信wall時刻（fresh判定の根拠）
            "pos": [round(v, 4) for v in pos],
            "vel": [round(v, 4) for v in vel],
            "vel_h": round(math.hypot(vel[0], vel[1]), 4),
            "quat_wxyz": [round(v, 4) for v in quat],
            "joint_position": [round(v, 5) for v in joints],
            "sim_time_s": round(sim_t, 4),
            "step_seq": seq,
            "boot_nonce": nonce,
            "task": task,
            "obs_age_s": round(age, 3),
            "obs_recv_total": meta[0],
            "obs_dropped_late": meta[1],
            "boot_changes": meta[2],
            "obs_dropped_frozen": meta[3],
            "obs_dropped_retired": meta[4],
        }

    def check(self) -> None:
        if self._exception is not None:
            raise RuntimeError(f"observer_worker_failed:{self._exception}")

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)


def _write_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def main() -> int:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    obs = LiveObserver()
    period = 1.0 / WRITE_HZ
    next_due = time.monotonic()
    try:
        while True:
            obs.check()
            snap = obs.snapshot()
            if snap is not None:
                _write_atomic(LIVE_PATH, snap)
            next_due += period
            delay = next_due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_due = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        obs.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
