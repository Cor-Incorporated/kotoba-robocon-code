"""制御指令（control command）プロトコル v2（C3改訂）。

2つの輸送路を使い分ける:

- `control.json`（latest-wins 連続操縦チャネル）: move/stop/end。
  最新seqのみ意味を持つ。move は `dur_s` 付きの有界nudge —
  controller は issued_wall+dur_s まで適用し、期限後は自動でidleへ
  戻す（停止上限。送信者が死んでも動き続けない）。issued_wall が
  CMD_STALE_S より古い指令は受付時に拒否する（輸送・解釈遅延の検査）。
- `strike.json`（単発イベント mailbox）: strike_id + expires_wall の
  一度だけ実行するイベント。latest-winsで上書きされず、controller が
  打撃開始をACKした時点で「実消費」が確定する。pendingは最大1件
  （API側がbusyとして返す）。重複strike_id・期限切れは実行しない。

全指令共通の検査（受理前）: lease一致 / seq単調 / 型・有限性 /
issued_wallのfinite性。鮮度・dur境界は check_fresh / dur_s で。
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from kotoba_harness.errors import HarnessError

MOVE = "move"
STOP = "stop"
STRIKE = "strike"
END = "end"
KNOWN_TYPES = (MOVE, STOP, STRIKE, END)
# latest-winsチャネルで扱う型（strikeは別mailbox）
CHANNEL_TYPES = (MOVE, STOP, END)

# 指令の輸送・解釈鮮度: issued_wall からこの秒数より古い指令は受理しない
CMD_STALE_S = 1.5
# move のdur_s許容範囲（秒）— 有界nudgeの上限
MOVE_DUR_MIN_S = 0.1
MOVE_DUR_MAX_S = 6.0
# steps付きmove（観測閉ループ実行）のdur_s外側上限 — stepあたり25s×最大3
STEPS_DUR_MAX_S = 80.0
# step列の境界（parserのallowlistと一致 — ここは輸送層の検査）
MAX_STEPS = 3
STEP_DIST_MIN_M = 0.05
STEP_DIST_MAX_M = 2.0
STEP_ANGLE_MAX_DEG = 180.0
# strikeイベントの有効期限（秒）— controllerが拾うまでの猶予
STRIKE_TTL_S = 8.0


class ControlRejected(HarnessError):
    """指令の検証失敗。reason: bad_json / lease_mismatch / unknown_type /
    nonfinite / seq_replay / missing_field / stale / dur_out_of_bounds /
    bad_strike_id / expired。"""


@dataclass(frozen=True)
class MotionStep:
    """観測閉ループで実行する1動作。数値は検証済みの有界値。"""
    action: str          # "translate" | "turn"
    direction: str       # translate: forward|back|left|right; turn: left|right|around
    target_m: float = 0.0
    target_deg: float = 0.0
    action_key: str = ""
    label: str = ""
    pace: str = "walk"   # walk|fast_walk|run — 前進のみ非walk可


_TRANSLATE_DIRS = ("forward", "back", "left", "right")
_TURN_DIRS = ("left", "right", "around")
_PACES = ("walk", "fast_walk", "run")


def _parse_pace(raw, action, direction) -> str:
    """pace fieldの輸送検査 — 非walkは前進のtranslate/jogのみ。"""
    pace = raw.get("pace")
    if pace is None:
        return "walk"
    if pace not in _PACES:
        raise ControlRejected("unknown_type")
    if pace != "walk" and not (
        action in ("translate", "jog") and direction == "forward"
    ):
        raise ControlRejected("unknown_type")
    return pace


def _parse_step(raw) -> MotionStep:
    if not isinstance(raw, dict):
        raise ControlRejected("missing_field")
    action = raw.get("action")
    direction = raw.get("dir")
    key = raw.get("action_key")
    label = raw.get("label")
    if action == "translate":
        if direction not in _TRANSLATE_DIRS:
            raise ControlRejected("unknown_type")
        m = _float(raw.get("m"))
        if m is None or not math.isfinite(m):
            raise ControlRejected("nonfinite")
        if not (STEP_DIST_MIN_M <= m <= STEP_DIST_MAX_M):
            raise ControlRejected("dur_out_of_bounds")
        return MotionStep(
            action=action, direction=direction, target_m=m,
            action_key=key if isinstance(key, str) else "",
            label=label if isinstance(label, str) else "",
            pace=_parse_pace(raw, action, direction),
        )
    if action == "turn":
        if direction not in _TURN_DIRS:
            raise ControlRejected("unknown_type")
        deg = _float(raw.get("deg"))
        if deg is None or not math.isfinite(deg):
            raise ControlRejected("nonfinite")
        if not (0.0 < deg <= STEP_ANGLE_MAX_DEG):
            raise ControlRejected("dur_out_of_bounds")
        return MotionStep(
            action=action, direction=direction, target_deg=deg,
            action_key=key if isinstance(key, str) else "",
            label=label if isinstance(label, str) else "",
            pace=_parse_pace(raw, action, direction),
        )
    if action == "jog":
        # 継続移動 — 距離目標を持たない。外側期限dur_sと
        # executorのjog_deadline/heartbeat/境界で止まる。
        if direction not in _TRANSLATE_DIRS:
            raise ControlRejected("unknown_type")
        return MotionStep(
            action=action, direction=direction,
            action_key=key if isinstance(key, str) else "",
            label=label if isinstance(label, str) else "",
            pace=_parse_pace(raw, action, direction),
        )
    raise ControlRejected("unknown_type")


@dataclass(frozen=True)
class ControlCommand:
    lease_id: str
    seq: int
    type: str
    fwd: float = 0.0
    lat: float = 0.0
    yaw: float = 0.0
    dur_s: float = 0.0
    steps: tuple = ()
    stop_after: bool = False
    action_key: str = ""
    strike_id: str = ""
    expires_wall: float = 0.0
    issued_wall: float = 0.0


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_command(data: dict, expected_lease_id: str) -> ControlCommand:
    """dict → 検証済み ControlCommand（latest-winsチャネル用）。

    lease不一致・未知型・非finite・strike混入は拒否。鮮度は呼出し側が
    check_fresh で検査する（受理時刻基準）。
    """
    if not isinstance(data, dict):
        raise ControlRejected("bad_json")
    lease = data.get("lease_id")
    if lease != expected_lease_id:
        raise ControlRejected("lease_mismatch")
    try:
        seq = int(data["seq"])
    except (KeyError, TypeError, ValueError):
        raise ControlRejected("missing_field") from None
    cmd = data.get("cmd") or {}
    ctype = cmd.get("type")
    if ctype not in KNOWN_TYPES:
        raise ControlRejected("unknown_type")
    if ctype == STRIKE:
        # strikeは別mailbox — 連続チャネルへの混入は拒否
        raise ControlRejected("unknown_type")
    fwd = lat = yaw = dur = 0.0
    steps = ()
    stop_after = False
    action_key = ""
    if ctype == MOVE:
        raw_steps = cmd.get("steps")
        if raw_steps is not None:
            # 観測閉ループ実行のstep列 — 各stepと合計距離を検査
            if not isinstance(raw_steps, list) or not (
                1 <= len(raw_steps) <= MAX_STEPS
            ):
                raise ControlRejected("missing_field")
            steps = tuple(_parse_step(rs) for rs in raw_steps)
            total_m = sum(s.target_m for s in steps
                          if s.action == "translate")
            if total_m > STEP_DIST_MAX_M + 1e-9:
                raise ControlRejected("dur_out_of_bounds")
            stop_after = bool(cmd.get("stop_after"))
            ak = cmd.get("action_key")
            action_key = ak if isinstance(ak, str) else ""
            dur = _float(cmd.get("dur_s"))
            if dur is None or not math.isfinite(dur):
                raise ControlRejected("missing_field")
            # step列のdurは実行絶対上限（観測閉ループのfail-safe）
            if not (MOVE_DUR_MIN_S <= dur <= STEPS_DUR_MAX_S):
                raise ControlRejected("dur_out_of_bounds")
        else:
            # 旧形式の有界nudge（回帰経路として維持）
            vals = (
                _float(cmd.get("fwd")),
                _float(cmd.get("lat")),
                _float(cmd.get("yaw")),
            )
            if any(v is None for v in vals):
                raise ControlRejected("missing_field")
            fwd, lat, yaw = vals
            if not all(map(math.isfinite, (fwd, lat, yaw))):
                raise ControlRejected("nonfinite")
            dur = _float(cmd.get("dur_s"))
            if dur is None or not math.isfinite(dur):
                raise ControlRejected("missing_field")
            if not (MOVE_DUR_MIN_S <= dur <= MOVE_DUR_MAX_S):
                raise ControlRejected("dur_out_of_bounds")
    issued = _float(data.get("issued_wall"))
    if issued is None or not math.isfinite(issued) or issued <= 0:
        raise ControlRejected("missing_field")
    return ControlCommand(
        lease_id=lease,
        seq=seq,
        type=ctype,
        fwd=fwd,
        lat=lat,
        yaw=yaw,
        dur_s=dur,
        steps=steps,
        stop_after=stop_after,
        action_key=action_key,
        issued_wall=issued,
    )


def parse_strike_event(data: dict, expected_lease_id: str) -> ControlCommand:
    """strike mailbox → 検証済み ControlCommand。

    strike_id（冪等性）と expires_wall（期限）を必須とする。
    """
    if not isinstance(data, dict):
        raise ControlRejected("bad_json")
    lease = data.get("lease_id")
    if lease != expected_lease_id:
        raise ControlRejected("lease_mismatch")
    try:
        seq = int(data["seq"])
    except (KeyError, TypeError, ValueError):
        raise ControlRejected("missing_field") from None
    cmd = data.get("cmd") or {}
    if cmd.get("type") != STRIKE:
        raise ControlRejected("unknown_type")
    strike_id = cmd.get("strike_id")
    if not isinstance(strike_id, str) or not (4 <= len(strike_id) <= 64):
        raise ControlRejected("bad_strike_id")
    expires = _float(cmd.get("expires_wall"))
    if expires is None or not math.isfinite(expires):
        raise ControlRejected("missing_field")
    issued = _float(data.get("issued_wall"))
    if issued is None or not math.isfinite(issued) or issued <= 0:
        raise ControlRejected("missing_field")
    return ControlCommand(
        lease_id=lease,
        seq=seq,
        type=STRIKE,
        strike_id=strike_id,
        expires_wall=expires,
        issued_wall=issued,
    )


def check_fresh(cmd: ControlCommand, now_wall: float) -> None:
    """受理時の鮮度検査 — 古い指令・期限切れイベントを拒否する。"""
    if cmd.type == STRIKE:
        if now_wall >= cmd.expires_wall:
            raise ControlRejected("expired")
    elif now_wall - cmd.issued_wall > CMD_STALE_S:
        raise ControlRejected("stale")


class LatestWinsGate:
    """seq単調のlatest-winsゲート。同seq/巻き戻りは受理しない（replay拒否）。"""

    def __init__(self) -> None:
        self.last_seq = 0

    def accept(self, cmd: ControlCommand) -> bool:
        if cmd.seq <= self.last_seq:
            return False
        self.last_seq = cmd.seq
        return True


def _atomic_write(path: Path, payload: dict) -> dict:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)
    return payload


def write_command(path: Path | str, lease_id: str, seq: int, cmd: dict) -> dict:
    """latest-wins指令を原子的に書き込む（tmp→renameでpartial read防止）。

    API側の唯一の書込入口。cmd は {"type": "move"|"stop"|"end", ...}。
    strike は write_strike_event を使う（別mailbox）。
    """
    payload = {
        "lease_id": lease_id,
        "seq": int(seq),
        "issued_wall": time.time(),
        "cmd": cmd,
    }
    return _atomic_write(Path(path), payload)


def write_strike_event(
    path: Path | str, lease_id: str, seq: int, via: str | None = None
) -> dict:
    """strikeイベントをmailboxへ原子的に書き込む。

    strike_idは呼出しごとに新規（冪等キー）。expires_wallまでは
    controllerが拾う猶予 — 開始ACK無しに消費されない。
    via は解釈経路の監査記録（parser|llm — 検査対象外の補助フィールド）。
    """
    strike_id = uuid.uuid4().hex[:16]
    payload = {
        "lease_id": lease_id,
        "seq": int(seq),
        "issued_wall": time.time(),
        "cmd": {
            "type": STRIKE,
            "strike_id": strike_id,
            "expires_wall": time.time() + STRIKE_TTL_S,
            "via": via or "api",
        },
    }
    return _atomic_write(Path(path), payload)
