"""control指令チャネルの仕様固定 — parse/lease/seq/latest-wins/atomic write。

C3プロトコルv2: move/stop/endは /kotoba/run/control.json の原子上書きで
届き（latest-wins）、controller側で lease一致・seq単調・型・非finite・
issued_wall・dur_s境界・鮮度を検査する。strikeは別mailbox（strike.json）
の単発event — control.jsonを上書きせず、開始ACKのみが実消費になる。
"""

import json
import time

import pytest

from kotoba_harness.control import (
    CMD_STALE_S,
    END,
    MOVE,
    STOP,
    STRIKE,
    ControlCommand,
    ControlRejected,
    LatestWinsGate,
    check_fresh,
    parse_command,
    parse_strike_event,
    write_command,
    write_strike_event,
)

LEASE = "lease-test-1"


def _payload(**kw):
    base = {
        "lease_id": LEASE,
        "seq": 1,
        "issued_wall": time.time(),
        "cmd": {"type": STOP},
    }
    base.update(kw)
    return base


def _move(fwd=0.5, lat=-0.2, yaw=0.1, dur_s=3.0):
    return {"type": "move", "fwd": fwd, "lat": lat, "yaw": yaw, "dur_s": dur_s}


# --- parse: 正常系 ---
def test_parse_move_command():
    cmd = parse_command(_payload(cmd=_move()), LEASE)
    assert cmd.type == MOVE and cmd.fwd == 0.5 and cmd.lat == -0.2
    assert cmd.yaw == 0.1 and cmd.dur_s == 3.0


def test_parse_simple_types():
    for t in (STOP, END):
        cmd = parse_command(_payload(cmd={"type": t}), LEASE)
        assert cmd.type == t and cmd.fwd == 0.0


def test_strike_rejected_on_control_channel():
    """strikeはlatest-winsチャネルに混ぜない — mailboxへ行く（C3反例T06）。"""
    with pytest.raises(ControlRejected) as err:
        parse_command(_payload(cmd={"type": STRIKE}), LEASE)
    assert err.value.reason == "unknown_type"


# --- parse: 拒否系 ---
def test_lease_mismatch_rejected():
    with pytest.raises(ControlRejected) as err:
        parse_command(_payload(lease_id="other-lease"), LEASE)
    assert err.value.reason == "lease_mismatch"


def test_unknown_type_rejected():
    with pytest.raises(ControlRejected) as err:
        parse_command(_payload(cmd={"type": "teleport"}), LEASE)
    assert err.value.reason == "unknown_type"


def test_missing_seq_rejected():
    with pytest.raises(ControlRejected):
        parse_command({"lease_id": LEASE, "cmd": {"type": STOP}}, LEASE)


def test_missing_issued_wall_rejected():
    """issued_wall必須 — 鮮度検査の基準時刻（C3反例T09）。"""
    p = {"lease_id": LEASE, "seq": 1, "cmd": {"type": STOP}}
    with pytest.raises(ControlRejected) as err:
        parse_command(p, LEASE)
    assert err.value.reason == "missing_field"


def test_stale_command_rejected():
    """issued_wallが古い指令は受理しない — 遅い解釈・古いファイルを無効化。"""
    p = _payload(issued_wall=time.time() - CMD_STALE_S - 0.5)
    cmd = parse_command(p, LEASE)
    with pytest.raises(ControlRejected) as err:
        check_fresh(cmd, time.time())
    assert err.value.reason == "stale"
    # 新鮮な指令は通る
    check_fresh(parse_command(_payload(), LEASE), time.time())


def test_non_dict_rejected():
    with pytest.raises(ControlRejected):
        parse_command(["not", "a", "dict"], LEASE)


def test_nonfinite_move_rejected():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ControlRejected) as err:
            parse_command(
                _payload(cmd=_move(fwd=bad, lat=0, yaw=0)),
                LEASE,
            )
        assert err.value.reason == "nonfinite"


def test_move_missing_axis_rejected():
    with pytest.raises(ControlRejected):
        parse_command(_payload(cmd={"type": "move", "fwd": 0.5}), LEASE)


def test_move_dur_bounds():
    """dur_sは有界nudgeの上限 — 無期限・非finite・範囲外は拒否。"""
    with pytest.raises(ControlRejected):
        parse_command(_payload(cmd=_move(dur_s=0)), LEASE)
    with pytest.raises(ControlRejected) as err:
        parse_command(_payload(cmd=_move(dur_s=99.0)), LEASE)
    assert err.value.reason == "dur_out_of_bounds"
    with pytest.raises(ControlRejected):
        parse_command(_payload(cmd=_move(dur_s=float("nan"))), LEASE)
    # dur無しのmoveは拒否（無期限指令を作らない）
    with pytest.raises(ControlRejected):
        parse_command(
            _payload(cmd={"type": "move", "fwd": 0.5, "lat": 0, "yaw": 0}),
            LEASE,
        )


# --- LatestWinsGate: seq単調・replay拒否 ---
def _cmd(seq, type_=STOP):
    return ControlCommand(lease_id=LEASE, seq=seq, type=type_)


def test_gate_accepts_increasing_seq():
    g = LatestWinsGate()
    assert g.accept(_cmd(1))
    assert g.accept(_cmd(5))
    assert g.accept(_cmd(6))
    assert g.last_seq == 6


def test_gate_rejects_same_and_rewind_seq():
    g = LatestWinsGate()
    assert g.accept(_cmd(5))
    assert not g.accept(_cmd(5))  # 同seq replay
    assert not g.accept(_cmd(3))  # 巻き戻り
    assert not g.accept(_cmd(0))
    assert g.last_seq == 5


def test_gate_rejects_first_nonpositive_seq():
    g = LatestWinsGate()
    assert not g.accept(_cmd(0))
    assert not g.accept(_cmd(-1))


# --- write_command: atomic write → parse の往復 ---
def test_write_then_parse_roundtrip(tmp_path):
    p = tmp_path / "control.json"
    write_command(p, LEASE, 7, _move(fwd=0.6, lat=0, yaw=0))
    cmd = parse_command(json.loads(p.read_text()), LEASE)
    assert cmd.seq == 7 and cmd.type == MOVE and cmd.fwd == 0.6
    assert cmd.issued_wall > 0


def test_write_overwrites_latest_only(tmp_path):
    p = tmp_path / "control.json"
    write_command(p, LEASE, 1, _move(fwd=0.6, lat=0, yaw=0))
    write_command(p, LEASE, 2, {"type": "stop"})
    cmd = parse_command(json.loads(p.read_text()), LEASE)
    assert cmd.seq == 2 and cmd.type == STOP  # 最新のみが残る = latest-wins輸送


def test_write_seq_coerced_to_int(tmp_path):
    p = tmp_path / "control.json"
    payload = write_command(p, LEASE, "9", {"type": "stop"})
    assert payload["seq"] == 9 and isinstance(payload["seq"], int)


# --- strike mailbox（単発event） ---
def test_strike_mailbox_roundtrip(tmp_path):
    """strike eventはstrike_id+expires_wallを持ち、冪等・期限検査を通る。"""
    p = tmp_path / "strike.json"
    payload = write_strike_event(p, LEASE, 5)
    assert payload["cmd"]["type"] == STRIKE
    assert payload["cmd"]["strike_id"]
    assert payload["cmd"]["expires_wall"] > time.time()
    ev = parse_strike_event(json.loads(p.read_text()), LEASE)
    assert ev.strike_id == payload["cmd"]["strike_id"]
    check_fresh(ev, time.time())


def test_strike_expired_rejected(tmp_path):
    """期限切れのstrike eventは受理しない — 遅着の二重実行を防ぐ。"""
    p = tmp_path / "strike.json"
    payload = write_strike_event(p, LEASE, 5)
    raw = json.loads(p.read_text())
    raw["cmd"]["expires_wall"] = time.time() - 1.0
    ev = parse_strike_event(raw, LEASE)
    with pytest.raises(ControlRejected) as err:
        check_fresh(ev, time.time())
    assert err.value.reason == "expired"


def test_strike_id_required_and_bounded(tmp_path):
    """strike_idは冪等キー — 欠落・異常な長さは拒否。"""
    p = _payload(cmd={"type": STRIKE})
    with pytest.raises(ControlRejected) as err:
        parse_strike_event(p, LEASE)
    assert err.value.reason == "bad_strike_id"
    p2 = _payload(cmd={"type": STRIKE, "strike_id": "x"})
    with pytest.raises(ControlRejected):
        parse_strike_event(p2, LEASE)


def test_strike_lease_mismatch_rejected(tmp_path):
    p = _payload(lease_id="other",
                 cmd={"type": STRIKE, "strike_id": "abcd1234",
                      "expires_wall": time.time() + 5})
    with pytest.raises(ControlRejected) as err:
        parse_strike_event(p, LEASE)
    assert err.value.reason == "lease_mismatch"
