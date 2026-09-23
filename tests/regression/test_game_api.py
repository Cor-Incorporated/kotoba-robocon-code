"""スイカ割りC2: ゲーム実行・指令経路の回帰試験。

実ProductSession＋実control.jsonで検証（docker起動はstub化）:
- start_game: spawn無し拒否 / lease発行 / run_lock / 二重開始拒否
- game_command: 未実行拒否 / 解釈不能は聞き返し / lease+seq単調で書込 /
  有界cmd / strike回数上限 / done後は拒否
- finish_run: round_end理由（hit/out_of_swings/timeout）→ outcome・message
- abort_run: game ctxを失効（中断後の指令は届かない）
- API: /api/game/start のoperatorゲート・/api/game/state の実状態結合
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "services" / "harness" / "src" / "kotoba_harness"


def _ensure_real_harness_pkg():
    """snapshot汚染対策（test_pr22_rereview_fixesと同じ技法）。

    test_product_characterization_review が収集時に snapshot 版
    kotoba_harness を sys.modules/sys.path へ置くため、service import前に
    現行パッケージを __path__ 付きで明示復元する。
    """
    spec = importlib.util.spec_from_file_location(
        "kotoba_harness",
        HARNESS / "__init__.py",
        submodule_search_locations=[str(HARNESS)],
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["kotoba_harness"] = pkg
    spec.loader.exec_module(pkg)
    return pkg


def _session():
    _ensure_real_harness_pkg()
    from kotoba_api.service import ProductSession

    return ProductSession()


def _spawned(sess, tmp_path, seed=42):
    """session作成 + RoundSpec生成 + game start までの共通下準備。"""
    sid = sess.create_session()
    sess.spawn_round(sid, seed, (0.0, 0.0, 0.82), 0.0)
    run_id = "run-test-1"
    out = sess.start_game(sid, "boot-1", run_id, tmp_path)
    return sid, out


def _read_control(tmp_path):
    return json.loads((tmp_path / "control.json").read_text())


# ---- spawn fixture対照 ----------------------------------------------------
def test_spawn_fixture_target_marks_seed(tmp_path):
    """operator fixture: 明示target指定時はspec.seed=-1で監査標識。"""
    sess = _session()
    sid = sess.create_session()
    spec = sess.spawn_round(sid, 42, (0.0, 0.0, 0.82), 0.0,
                            target=(0.3, 0.35, 1.05))
    assert spec.seed == -1
    assert spec.watermelon.pos == (0.3, 0.35, 1.05)
    # 生成済みのroundは不変。通常seedとの対照は別sessionで行う。
    sid2 = sess.create_session()
    spec2 = sess.spawn_round(sid2, 42, (0.0, 0.0, 0.82), 0.0)
    assert spec2.seed == 42  # 通常経路はseed生成のまま


# ---- start_game ----------------------------------------------------------
def test_start_game_requires_spawn(tmp_path):
    sess = _session()
    sid = sess.create_session()
    out = sess.start_game(sid, "boot-1", "r1", tmp_path)
    assert out["started"] is False and out["reason"] == "no_round_spawned"
    assert sess.run_lock is False


def test_start_game_issues_lease_and_lock(tmp_path):
    sess = _session()
    sid, out = _spawned(sess, tmp_path)
    assert out["started"] is True and out["lease_id"]
    r = sess.rounds[sid]
    assert r.phase == "RUNNING" and r.run_id == "run-test-1"
    assert sess.run_lock is True
    assert r.game["lease_id"] == out["lease_id"]
    assert r.game["boot_expect"] == "boot-1"


def test_start_game_rejects_double_start(tmp_path):
    sess = _session()
    sid, _ = _spawned(sess, tmp_path)
    out2 = sess.start_game(sid, "boot-1", "r2", tmp_path)
    assert out2["started"] is False and out2["reason"] == "run_in_progress"


# ---- game_command --------------------------------------------------------
def test_command_rejected_when_not_running(tmp_path):
    sess = _session()
    sid = sess.create_session()
    out = sess.game_command(sid, "前")
    assert out["accepted"] is False and out["reason"] == "not_running"


def test_command_unclear_returns_clarify(tmp_path):
    sess = _session()
    sid, _ = _spawned(sess, tmp_path)
    out = sess.game_command(sid, "前か後ろか分からない")
    assert out["accepted"] is False and out["reason"] == "unclear"
    assert "message" in out


def test_command_writes_bounded_cmd_with_lease_seq(tmp_path):
    sess = _session()
    sid, st = _spawned(sess, tmp_path)
    out = sess.game_command(sid, "前")
    assert out["accepted"] is True and out["seq"] == 1
    assert out["cmd"]["type"] == "move"
    (st0,) = out["cmd"]["steps"]
    assert st0["action"] == "jog" and st0["dir"] == "forward"
    wire = _read_control(tmp_path)
    assert wire["lease_id"] == st["lease_id"] and wire["seq"] == 1
    assert wire["cmd"]["type"] == "move"
    # controller側検査を実際に通すことを確認（lease+seq+有限値）
    from kotoba_harness.control import parse_command

    cmd = parse_command(wire, st["lease_id"])
    assert cmd.type == "move" and cmd.steps[0].direction == "forward"


def test_command_seq_monotonic_latest_wins(tmp_path):
    sess = _session()
    sid, st = _spawned(sess, tmp_path)
    sess.game_command(sid, "前")
    out = sess.game_command(sid, "少し左")
    assert out["seq"] == 2
    wire = _read_control(tmp_path)
    assert wire["seq"] == 2
    assert wire["cmd"]["steps"][0]["dir"] == "left"


def _read_strike(tmp_path):
    return json.loads((tmp_path / "strike.json").read_text())


def _write_state(tmp_path, **kw):
    st = {"t_wall": time.time()}
    st.update(kw)
    (tmp_path / "control_state.json").write_text(json.dumps(st))


# ---- strike mailbox（C3: 予約→実消費ACK分離） -----------------------------
def test_strike_goes_to_mailbox_not_control_channel(tmp_path):
    """strikeは単発event mailbox（strike.json）へ — control.jsonを上書きしない。"""
    sess = _session()
    sid, st = _spawned(sess, tmp_path)
    sess.game_command(sid, "前")
    wire_before = _read_control(tmp_path)
    out = sess.game_command(sid, "割って")
    assert out["accepted"] is True and out["pending"] is True
    assert out["strike_id"]
    # control.jsonは上書きされない（moveが残る）
    assert _read_control(tmp_path) == wire_before
    swire = _read_strike(tmp_path)
    assert swire["lease_id"] == st["lease_id"]
    assert swire["cmd"]["type"] == "strike"
    assert swire["cmd"]["strike_id"] == out["strike_id"]
    assert swire["cmd"]["expires_wall"] > time.time()
    # controller側検査を実際に通す
    from kotoba_harness.control import check_fresh, parse_strike_event

    ev = parse_strike_event(swire, st["lease_id"])
    check_fresh(ev, time.time())


def test_strike_pending_rejects_second_as_busy(tmp_path):
    """pending(未ACK)のstrikeがある間の追加分はbusy — 二重発行しない。"""
    sess = _session()
    sid, _ = _spawned(sess, tmp_path)
    out = sess.game_command(sid, "割って")
    assert out["accepted"] is True
    out2 = sess.game_command(sid, "割って")
    assert out2["accepted"] is False and out2["reason"] == "strike_busy"


def test_strike_not_consumed_until_controller_starts(tmp_path):
    """振数正本はcontrollerのstrikes_started — 予約・受理では消費しない。

    C3レビュー反例T06: strike→stopを読取り前に繰返しても振数は減らない。
    """
    sess = _session()
    sid, _ = _spawned(sess, tmp_path)
    spec = sess.rounds[sid].round_spec
    # 1回目: 予約のみ（未消費）
    sess.game_command(sid, "割って")
    # controllerが実行中（ACK済・swing）の間はbusy — 消費は1のみ
    _write_state(tmp_path, strike={
        "strike_id": sess.rounds[sid].game["strike_pending"]["strike_id"],
        "phase": "swing", "consumed": True,
    }, strikes_started=1)
    out = sess.game_command(sid, "割って")
    assert out["accepted"] is False and out["reason"] == "strike_busy"
    # 打撃完了（state.strikeがnull、strikes_started=1）→ 次が通る
    _write_state(tmp_path, strike=None, strikes_started=1)
    out = sess.game_command(sid, "割って")
    assert out["accepted"] is True
    # 上限を残り1として再現: started=max_swings-1ならもう1回通る
    _write_state(
        tmp_path,
        strike_expired=[sess.rounds[sid].game["strike_pending"]["strike_id"]],
        strikes_started=spec.max_swings - 1,
    )
    out = sess.game_command(sid, "割って")
    assert out["accepted"] is True
    # 実消費がmaxに達した時点でout_of_swings
    _write_state(
        tmp_path,
        strike_expired=[sess.rounds[sid].game["strike_pending"]["strike_id"]],
        strikes_started=spec.max_swings,
    )
    out = sess.game_command(sid, "割って")
    assert out["accepted"] is False and out["reason"] == "out_of_swings"


def test_strike_expired_by_controller_is_not_consumed(tmp_path):
    """期限切れ・未開始のstrikeは消費しない（返却相当）。"""
    sess = _session()
    sid, _ = _spawned(sess, tmp_path)
    out = sess.game_command(sid, "割って")
    sid_ = out["strike_id"]
    _write_state(tmp_path, strike_expired=[sid_], strikes_started=0)
    out2 = sess.game_command(sid, "割って")
    assert out2["accepted"] is True and out2["strike_id"] != sid_


def test_strike_write_failure_does_not_reserve(tmp_path, monkeypatch):
    """mailbox書込失敗は予約も消費もしない（C3レビュー反例T07）。"""
    sess = _session()
    sid, _ = _spawned(sess, tmp_path)
    import kotoba_api.service as svc

    monkeypatch.setattr(
        svc, "write_strike_event",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )
    out = sess.game_command(sid, "割って")
    assert out["accepted"] is False
    g = sess.rounds[sid].game
    assert g["strike_pending"] is None


def test_stale_command_rejected_by_freshness(tmp_path):
    """issued_wallが古い指令は受理しない（C3レビュー反例T09）。"""
    import time as _t

    from kotoba_harness.control import (
        CMD_STALE_S, ControlRejected, check_fresh, parse_command,
    )

    sess = _session()
    sid, st = _spawned(sess, tmp_path)
    sess.game_command(sid, "前")
    wire = _read_control(tmp_path)
    wire["issued_wall"] = _t.time() - CMD_STALE_S - 1.0
    cmd = parse_command(wire, st["lease_id"])
    with pytest.raises(ControlRejected) as ei:
        check_fresh(cmd, _t.time())
    assert ei.value.reason == "stale"


def test_move_has_bounded_duration(tmp_path):
    """全てのmoveは有限の外側期限 — controller側の絶対上限がある。"""
    sess = _session()
    sid, _ = _spawned(sess, tmp_path)
    out = sess.game_command(sid, "前")
    cmd = out["cmd"]
    assert cmd["type"] == "move" and 0 < cmd["dur_s"] <= 80.0
    # 「少し」は有限step、「左」は継続jog — 同じ指令ではない（反例T05）
    out2 = sess.game_command(sid, "少し左")
    out3 = sess.game_command(sid, "左")
    assert out2["cmd"]["steps"][0]["action"] == "translate"
    assert out3["cmd"]["steps"][0]["action"] == "jog"


def test_stop_bumps_epoch(tmp_path):
    """STOP/END受理でepochが進む（遅いLLM解釈の古い書込を無効化する）。"""
    sess = _session()
    sid, _ = _spawned(sess, tmp_path)
    g = sess.rounds[sid].game
    e0 = g["epoch"]
    sess.game_command(sid, "止まって")
    assert g["epoch"] == e0 + 1
    sess.game_command(sid, "終わって")
    assert g["epoch"] == e0 + 2
    # move/strikeはepochを進めない
    sess.game_command(sid, "前")
    assert g["epoch"] == e0 + 2


def test_command_rejected_after_done(tmp_path):
    sess = _session()
    sid, _ = _spawned(sess, tmp_path)
    sess.finish_run(sid, "run-test-1", {"verdict": "PASS", "events": []})
    out = sess.game_command(sid, "前")
    assert out["accepted"] is False and out["reason"] == "not_running"


# ---- finish / abort ------------------------------------------------------
@pytest.mark.parametrize(
    "reason,expected",
    [("hit", "スイカを割りました！"), ("out_of_swings", "振れる回数を使い切りました"),
     ("timeout", "時間切れです"),
     ("arena_exit", "操縦範囲から出たため終了しました")],
)
def test_finish_run_game_outcome(tmp_path, reason, expected):
    sess = _session()
    sid, _ = _spawned(sess, tmp_path)
    result = {"verdict": "PASS", "events": [{"event": "round_end", "reason": reason}]}
    assert sess.finish_run(sid, "run-test-1", result) is True
    r = sess.rounds[sid]
    assert r.phase == "RESULT" and sess.run_lock is False
    assert r.game["done"] is True and r.game["outcome"] == reason
    assert r.message == expected


def test_finish_run_end_command_is_ended_not_timeout(tmp_path):
    """end指令受理で終了したroundは'timeout'と誤表示しない（実機で観測したbug）。"""
    sess = _session()
    sid, _ = _spawned(sess, tmp_path)
    result = {"verdict": "PASS",
              "events": [{"event": "cmd", "seq": 9, "type": "end"}],
              "control": {"end_received": True}}
    sess.finish_run(sid, "run-test-1", result)
    r = sess.rounds[sid]
    assert r.game["outcome"] == "ended" and r.message == "終了しました"


def test_finish_run_fault_outcome(tmp_path):
    sess = _session()
    sid, _ = _spawned(sess, tmp_path)
    result = {"verdict": "FAIL_FALLEN", "reasons": ["walk_fall"], "events": []}
    sess.finish_run(sid, "run-test-1", result)
    r = sess.rounds[sid]
    assert r.game["outcome"] == "fault" and "異常終了" in r.message


def test_obs_lost_is_fault_not_timeout(tmp_path):
    """観測喪失の異常終了を普通の時間切れにしない（C3レビュー反例T10）。

    controller実式: shutdown_reason=obs_lost → health=obs_lost。
    round_end無し・end_received無し → outcomeはfault。
    """
    sess = _session()
    sid, _ = _spawned(sess, tmp_path)
    result = {
        "verdict": "FAIL_OBS_LOST",
        "events": [],
        "control": {
            "end_received": False,
            "shutdown_reason": "obs_lost",
            "health": "obs_lost",
            "fallen": False,
        },
    }
    sess.finish_run(sid, "run-test-1", result)
    r = sess.rounds[sid]
    assert r.game["outcome"] == "fault"
    assert "obs_lost" in r.message


def test_unverified_end_hold_is_fault(tmp_path):
    """終了後の立位保持を実観測できなかった場合はendedでなくfault（要確認）。"""
    sess = _session()
    sid, _ = _spawned(sess, tmp_path)
    result = {
        "verdict": "UNKNOWN_END_HOLD",
        "events": [{"event": "cmd", "seq": 9, "type": "end"}],
        "control": {"end_received": True, "health": "ok",
                    "end_hold_verified": False},
    }
    sess.finish_run(sid, "run-test-1", result)
    r = sess.rounds[sid]
    assert r.game["outcome"] == "fault"


def test_abort_run_clears_game_ctx(tmp_path):
    sess = _session()
    sid, _ = _spawned(sess, tmp_path)
    sess.abort_run(sid, "中断")
    r = sess.rounds[sid]
    assert r.game is None and r.phase == "FAULT"
    out = sess.game_command(sid, "前")
    assert out["accepted"] is False


# ---- API層 ---------------------------------------------------------------
def _load_app(tmp_path, monkeypatch):
    _ensure_real_harness_pkg()
    monkeypatch.setenv("KOTOBA_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("KOTOBA_HOME", str(tmp_path))
    monkeypatch.setenv("KOTOBA_OFFLINE", "0")
    monkeypatch.setenv("KOTOBA_SELFTEST", "1")
    monkeypatch.setenv("KOTOBA_OPERATOR_TOKEN", "op-token")
    import kotoba_api.app as app_mod

    return importlib.reload(app_mod)


def _write_boot(tmp_path, nonce="boot-1"):
    (tmp_path / "boot-ready.json").write_text(json.dumps({
        "standing": True, "boot_nonce": nonce, "verified_at": time.time(),
    }))


def _spawn_json(app_mod, session_id, boot_nonce="boot-1", **extra):
    """利用者が送信前に読み取った世代をPOSTへ固定する。"""
    return {
        "session_id": session_id,
        "expected_round_id": app_mod.session.rounds[session_id].round_id,
        "expected_boot_nonce": boot_nonce,
        **extra,
    }


def test_spawn_rejects_request_from_previous_round_after_reset(tmp_path, monkeypatch):
    """旧POSTがreset後に到着しても新roundへスイカを生成しない。"""
    from fastapi.testclient import TestClient

    app_mod = _load_app(tmp_path, monkeypatch)
    client = TestClient(app_mod.app)
    sid = client.post("/api/sessions").json()["session_id"]
    old_round = app_mod.session.rounds[sid].round_id
    _write_boot(tmp_path)
    (tmp_path / "live.json").write_text(json.dumps({
        "boot_nonce": "boot-1", "obs_wall": time.time(), "wall": time.time(),
        "pos": [0, 0, 0.82], "quat_wxyz": [1, 0, 0, 0], "step_seq": 1,
    }))
    assert app_mod.session.reset(sid, restart_sim=False)["reset"] is True
    assert app_mod.session.rounds[sid].round_id != old_round

    stale = client.post("/api/round/spawn", json={
        "session_id": sid, "expected_round_id": old_round,
        "expected_boot_nonce": "boot-1",
    }, headers={"x-kotoba-operator": "op-token"})
    assert stale.status_code == 409
    assert stale.json()["detail"] == "expected_round_mismatch"
    assert app_mod.session.rounds[sid].round_spec is None
    assert client.get("/api/display/round").json() == {"active": False}
    missing = client.post("/api/round/spawn", json={"session_id": sid},
                          headers={"x-kotoba-operator": "op-token"})
    assert missing.status_code == 422
    assert app_mod.session.rounds[sid].round_spec is None


def test_spawn_rejects_request_from_previous_boot(tmp_path, monkeypatch):
    """旧bootで作ったPOSTが新bootへ遅着しても生成しない。"""
    from fastapi.testclient import TestClient

    app_mod = _load_app(tmp_path, monkeypatch)
    client = TestClient(app_mod.app)
    sid = client.post("/api/sessions").json()["session_id"]
    round_id = app_mod.session.rounds[sid].round_id
    _write_boot(tmp_path, "boot-2")
    (tmp_path / "live.json").write_text(json.dumps({
        "boot_nonce": "boot-2", "obs_wall": time.time(), "wall": time.time(),
        "pos": [0, 0, 0.82], "quat_wxyz": [1, 0, 0, 0], "step_seq": 1,
    }))
    stale = client.post("/api/round/spawn", json={
        "session_id": sid, "expected_round_id": round_id,
        "expected_boot_nonce": "boot-1",
    }, headers={"x-kotoba-operator": "op-token"})
    assert stale.status_code == 409
    assert stale.json()["detail"] == "expected_boot_mismatch"
    assert app_mod.session.rounds[sid].round_spec is None


def test_spawn_rejects_stale_or_wrong_boot_live_observation(tmp_path, monkeypatch):
    """生成位置は現在bootの新鮮な観測でのみ確定する。"""
    from fastapi.testclient import TestClient

    app_mod = _load_app(tmp_path, monkeypatch)
    client = TestClient(app_mod.app)
    sid = client.post("/api/sessions").json()["session_id"]
    headers = {"x-kotoba-operator": "op-token"}
    _write_boot(tmp_path, "boot-current")
    live_path = tmp_path / "live.json"
    sample = {
        "boot_nonce": "boot-current", "obs_wall": time.time() - 5,
        "wall": time.time(), "pos": [0, 0, 0.82],
        "quat_wxyz": [1, 0, 0, 0], "step_seq": 1,
    }
    live_path.write_text(json.dumps(sample))
    stale = client.post("/api/round/spawn",
                        json=_spawn_json(app_mod, sid, "boot-current"), headers=headers)
    assert stale.status_code == 409 and stale.json()["detail"] == "live_not_ready"
    assert app_mod.session.rounds[sid].round_spec is None

    sample.update(boot_nonce="boot-retired", obs_wall=time.time())
    live_path.write_text(json.dumps(sample))
    wrong_boot = client.post("/api/round/spawn",
                             json=_spawn_json(app_mod, sid, "boot-current"), headers=headers)
    assert wrong_boot.status_code == 409 and wrong_boot.json()["detail"] == "live_boot_mismatch"
    assert app_mod.session.rounds[sid].round_spec is None

    sample.update(boot_nonce="boot-current", obs_wall=time.time())
    live_path.write_text(json.dumps(sample))
    app_mod.session.boot_nonce = "boot-retired"
    retired = client.post("/api/round/spawn",
                          json=_spawn_json(app_mod, sid, "boot-current"), headers=headers)
    assert retired.status_code == 409 and retired.json()["detail"] == "live_boot_mismatch"
    assert app_mod.session.rounds[sid].round_spec is None
    app_mod.session.boot_nonce = None
    accepted = client.post("/api/round/spawn",
                           json=_spawn_json(app_mod, sid, "boot-current"), headers=headers)
    assert accepted.status_code == 200 and accepted.json()["spawned"] is True
    assert client.get(f"/api/game/state/{sid}").json()["spec_boot_nonce"] == "boot-current"


def test_spawn_observation_and_reset_are_serialized(tmp_path, monkeypatch):
    """観測を読んでからspec確定までresetが世代を入れ替えられない。"""
    from fastapi.testclient import TestClient

    app_mod = _load_app(tmp_path, monkeypatch)
    client = TestClient(app_mod.app)
    sid = client.post("/api/sessions").json()["session_id"]
    _write_boot(tmp_path)
    live_path = tmp_path / "live.json"
    live_path.write_text(json.dumps({
        "boot_nonce": "boot-1", "obs_wall": time.time(),
        "wall": time.time(), "pos": [0, 0, 0.82],
        "quat_wxyz": [1, 0, 0, 0], "step_seq": 1,
    }))
    spawn_body = _spawn_json(app_mod, sid)
    read_started = threading.Event()
    resume_read = threading.Event()
    reset_done = threading.Event()
    original_read = Path.read_text
    first_read = True

    def block_live_read(path, *args, **kwargs):
        nonlocal first_read
        data = original_read(path, *args, **kwargs)
        if path == live_path and first_read:
            first_read = False
            read_started.set()
            assert resume_read.wait(5), "spawn live read was not released"
        return data

    monkeypatch.setattr(Path, "read_text", block_live_read)
    result = {}

    def spawn():
        try:
            result["response"] = client.post(
                "/api/round/spawn", json=spawn_body,
                headers={"x-kotoba-operator": "op-token"},
            )
        except BaseException as exc:
            result["error"] = exc

    def reset():
        result["reset"] = app_mod.session.reset(sid, restart_sim=False)
        reset_done.set()

    spawn_thread = threading.Thread(target=spawn)
    reset_thread = threading.Thread(target=reset)
    spawn_thread.start()
    try:
        assert read_started.wait(5)
        reset_thread.start()
        reset_completed_during_read = reset_done.wait(0.5)
    finally:
        resume_read.set()
        spawn_thread.join(5)
        if reset_thread.ident is not None:
            reset_thread.join(5)
    assert not spawn_thread.is_alive() and not reset_thread.is_alive()
    assert not reset_completed_during_read
    assert "error" not in result
    assert result["response"].status_code == 200
    assert result["reset"]["reset"] is True
    assert app_mod.session.rounds[sid].round_spec is None


def test_api_game_start_gates(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    app_mod = _load_app(tmp_path, monkeypatch)
    monkeypatch.setattr(app_mod, "_execute_game", lambda *a: None)
    c = TestClient(app_mod.app)
    sid = c.post("/api/sessions").json()["session_id"]
    # operator token無し → 403
    r = c.post("/api/game/start", json={"session_id": sid})
    assert r.status_code == 403
    # spawn無し → 409
    r = c.post("/api/game/start", json={"session_id": sid},
               headers={"x-kotoba-operator": "op-token"})
    assert r.status_code == 409 and "no_round_spawned" in r.text
    # spawnにも現在bootと新鮮なlive観測が必要。
    no_live = c.post("/api/round/spawn", json=_spawn_json(app_mod, sid, seed=42),
                     headers={"x-kotoba-operator": "op-token"})
    assert no_live.status_code == 409
    (tmp_path / "live.json").write_text(json.dumps({
        "boot_nonce": "boot-1", "obs_wall": time.time(), "wall": time.time(),
        "pos": [0, 0, 0.82], "quat_wxyz": [1, 0, 0, 0], "step_seq": 1,
    }))
    no_boot = c.post("/api/round/spawn", json=_spawn_json(app_mod, sid, seed=42),
                     headers={"x-kotoba-operator": "op-token"})
    assert no_boot.status_code == 409
    _write_boot(tmp_path)
    r = c.post("/api/round/spawn", json=_spawn_json(app_mod, sid, seed=42),
               headers={"x-kotoba-operator": "op-token"})
    assert r.status_code == 200
    again = c.post("/api/round/spawn", json=_spawn_json(app_mod, sid, seed=43),
                   headers={"x-kotoba-operator": "op-token"})
    assert again.status_code == 409 and again.json()["detail"] == "round_not_ready"
    (tmp_path / "boot-lost.json").write_text("{}")
    r = c.post("/api/game/start", json={"session_id": sid},
               headers={"x-kotoba-operator": "op-token"})
    assert r.status_code == 409 and "sim_not_ready" in r.text
    # boot ready → started（_execute_gameはstub化済み）
    (tmp_path / "boot-lost.json").unlink()
    r = c.post("/api/game/start", json={"session_id": sid},
               headers={"x-kotoba-operator": "op-token"})
    assert r.status_code == 200 and r.json()["started"] is True


def test_api_game_state_reports_controller_state(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    app_mod = _load_app(tmp_path, monkeypatch)
    monkeypatch.setattr(app_mod, "_execute_game", lambda *a: None)
    c = TestClient(app_mod.app)
    sid = c.post("/api/sessions").json()["session_id"]
    (tmp_path / "live.json").write_text(json.dumps({
        "boot_nonce": "boot-1", "obs_wall": time.time(), "wall": time.time(),
        "pos": [0, 0, 0.82], "quat_wxyz": [1, 0, 0, 0], "step_seq": 1,
    }))
    _write_boot(tmp_path)
    c.post("/api/round/spawn", json=_spawn_json(app_mod, sid, seed=42),
           headers={"x-kotoba-operator": "op-token"})
    st = c.post("/api/game/start", json={"session_id": sid},
                headers={"x-kotoba-operator": "op-token"}).json()
    run_dir = tmp_path / st["run_id"]
    # controllerの実状態ファイルを再現（applied_seq/残時間/実消費振数/打撃判定）
    (run_dir / "control_state.json").write_text(json.dumps({
        "lease_id": "x", "applied_seq": 3, "current_type": "move",
        "strikes_started": 1, "strikes_done": 1, "rejects": [],
        "strike": {"strike_id": "s1", "phase": "recover", "consumed": True},
        "last_strike": {"hit": False, "min_dist_m": 0.33},
        "round_deadline_wall": time.time() + 55.0,
        "health": "ok",
        "t_wall": time.time(),
    }))
    c.post("/api/game/command", json={"session_id": sid, "text": "前"})
    d = c.get(f"/api/game/state/{sid}").json()
    assert d["phase"] == "RUNNING"
    g = d["game"]
    assert g["seq"] == 1 and g["applied_seq"] == 3
    assert g["current_type"] == "move" and g["strikes_done"] == 1
    assert g["strikes_started"] == 1
    assert g["swings_left"] == d["spec"]["max_swings"] - 1
    assert g["strike_phase"] == "recover"
    assert g["last_strike"]["hit"] is False
    assert 50 < g["remaining_s"] <= 55
    assert g["controller_alive"] is True
    assert g["max_swings"] == d["spec"]["max_swings"]


def test_api_game_command_unknown_session(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    app_mod = _load_app(tmp_path, monkeypatch)
    c = TestClient(app_mod.app)
    r = c.post("/api/game/command", json={"session_id": "nope", "text": "前"})
    assert r.status_code == 404


# ---- LLMフォールバック経路（C3c-1: 有界intent変換・STOP優先・経路区別） ----
def test_llm_fallback_produces_bounded_cmd_with_via(tmp_path, monkeypatch):
    """parser解釈不能な自由文だけLLMへ。出力は固定表の有界値+via=llm。"""
    import kotoba_api.service as svc_mod

    monkeypatch.setattr(
        svc_mod, "interpret_game",
        lambda text: {"type": "move", "direction": "forward", "size": "small"},
    )
    sess = _session()
    sid, st = _spawned(sess, tmp_path)
    out = sess.game_command(sid, "にゅっと動かして")  # parserはNoneを返す文
    assert out["accepted"] is True and out["via"] == "llm"
    assert out["cmd"]["type"] == "move"
    (st0,) = out["cmd"]["steps"]
    assert st0["action"] == "translate" and st0["m"] == 0.25  # small固定表
    wire = _read_control(tmp_path)
    assert wire["cmd"]["via"] == "llm"
    assert wire["cmd"]["steps"][0]["dir"] == "forward"
    from kotoba_harness.control import parse_command
    cmd = parse_command(wire, st["lease_id"])
    assert cmd.type == "move"


def test_parser_path_marks_via_parser(tmp_path, monkeypatch):
    """parserで解釈できる入力はLLMを呼ばず via=parser。"""
    import kotoba_api.service as svc_mod

    def _boom(text):
        raise AssertionError("llm must not be called for parser-covered input")

    monkeypatch.setattr(svc_mod, "interpret_game", _boom)
    sess = _session()
    sid, st = _spawned(sess, tmp_path)
    out = sess.game_command(sid, "前")
    assert out["accepted"] is True and out["via"] == "parser"
    assert _read_control(tmp_path)["cmd"]["via"] == "parser"


def test_stop_is_parser_priority_never_llm(tmp_path, monkeypatch):
    """STOP優先: 「止まって」はparserが即応答 — LLM遅延の影響を受けない。"""
    import kotoba_api.service as svc_mod

    def _boom(text):
        raise AssertionError("stop must not reach llm")

    monkeypatch.setattr(svc_mod, "interpret_game", _boom)
    sess = _session()
    sid, _ = _spawned(sess, tmp_path)
    out = sess.game_command(sid, "止まって")
    assert out["accepted"] is True and out["cmd"]["type"] == "stop"
    assert out["via"] == "parser"


def test_llm_unreachand_invalid_fall_back_to_unclear(tmp_path, monkeypatch):
    """LLM未達・schema違反はunclearへ正直に戻す（発行しない）。"""
    import kotoba_api.service as svc_mod
    from kotoba_api.llm import LlmError

    monkeypatch.setattr(
        svc_mod, "interpret_game",
        lambda text: (_ for _ in ()).throw(LlmError("llm_unreachable")),
    )
    sess = _session()
    sid, _ = _spawned(sess, tmp_path)
    out = sess.game_command(sid, "なんかよくわからん")
    assert out["accepted"] is False and out["reason"] == "unclear"
    # LLMがunclearを返した場合も同様
    monkeypatch.setattr(svc_mod, "interpret_game", lambda text: {"type": "unclear"})
    out2 = sess.game_command(sid, "えーと")
    assert out2["accepted"] is False and out2["reason"] == "unclear"


def test_llm_strike_goes_to_mailbox_with_via(tmp_path, monkeypatch):
    """LLM解釈の打撃も同一mailbox+via記録（経路は変えない）。"""
    import kotoba_api.service as svc_mod

    monkeypatch.setattr(
        svc_mod, "interpret_game", lambda text: {"type": "strike"}
    )
    sess = _session()
    sid, st = _spawned(sess, tmp_path)
    out = sess.game_command(sid, "あの緑のやつ思いっきり")
    assert out["accepted"] is True and out["via"] == "llm" and out["pending"]
    swire = _read_strike(tmp_path)
    assert swire["cmd"]["via"] == "llm"
    from kotoba_harness.control import parse_strike_event
    scmd = parse_strike_event(swire, st["lease_id"])
    assert scmd.type == "strike"


# ---- レビューA群反例: LLM待ちと停止系の分離・明示禁止の非復活 -----------
def test_explicit_prohibition_never_reaches_llm(tmp_path, monkeypatch):
    """P03反例: 明示禁止は決定論的拒否 — LLMへ回して復活させない。

    parserがrejectと返した入力はinterpret_gameを呼ばず、仮にschema
    適合のLLM応答があってもactuationへ格上げしない。
    """
    import kotoba_api.service as svc_mod

    calls = []

    def _spy(text):
        calls.append(text)
        return {"type": "strike"}  # 呼ばれたとしても発行されてはならない

    monkeypatch.setattr(svc_mod, "interpret_game", _spy)
    sess = _session()
    sid, _ = _spawned(sess, tmp_path)
    out = sess.game_command(sid, "割るな")
    assert out["accepted"] is False and out["reason"] == "rejected"
    out2 = sess.game_command(sid, "前に進まないで")
    assert out2["accepted"] is False and out2["reason"] == "rejected"
    out3 = sess.game_command(sid, "前に進むの？")
    assert out3["accepted"] is False and out3["reason"] == "rejected"
    assert calls == []  # LLMは一度も呼ばれていない
    assert not (tmp_path / "strike.json").exists()
    assert not (tmp_path / "control.json").exists()


def test_llm_wait_does_not_block_stop(tmp_path, monkeypatch):
    """P02反例: LLM推論待ちの間もSTOPは即受理され、遅い応答は破棄される。

    推論が_run_muを保持しているとSTOPが数秒〜数十秒ブロックされる。
    新構造では推論はロック外 — STOP受理→epoch/input_seq進行により
    後から返るLLM応答はstaleとして書き込まれない。
    """
    import threading

    import kotoba_api.service as svc_mod

    started = threading.Event()
    release = threading.Event()

    def _slow(text):
        started.set()
        release.wait(15)
        return {"type": "move", "direction": "forward", "size": "normal"}

    monkeypatch.setattr(svc_mod, "interpret_game", _slow)
    sess = _session()
    sid, st = _spawned(sess, tmp_path)
    result = {}
    th = threading.Thread(
        target=lambda: result.setdefault(
            "llm", sess.game_command(sid, "ふわふわ動かして")
        ),
        daemon=True,
    )
    th.start()
    assert started.wait(5), "llm inference did not start"
    # 推論がin-flightのままSTOPを送る — 即受理されること
    t0 = time.monotonic()
    stop = sess.game_command(sid, "止まって")
    elapsed = time.monotonic() - t0
    assert stop["accepted"] is True and stop["cmd"]["type"] == "stop"
    assert elapsed < 2.0, f"stop blocked behind llm for {elapsed:.2f}s"
    release.set()
    th.join(15)
    # STOP受理で世代が進んだため、遅いLLM応答はstale破棄
    assert result["llm"]["accepted"] is False
    assert result["llm"]["reason"] == "stale"
    # control.jsonはSTOPのまま — LLM応答で上書きされていない
    wire = _read_control(tmp_path)
    assert wire["cmd"]["type"] == "stop" and wire["cmd"]["via"] == "parser"


def test_llm_response_discarded_when_newer_input_accepted(tmp_path, monkeypatch):
    """受理順が新旧を決める: 後に受理された入力が古いLLM応答を破棄させる。

    LLMの完了順ではなく受付順 — 先に送った解釈が遅れて返っても、
    その間に受理された指令（parser経路）があれば適用しない。
    """
    import threading

    import kotoba_api.service as svc_mod

    release = threading.Event()

    def _router(text):
        release.wait(15)
        return {"type": "strike"}  # 遅れて返る打撃 — 適用されてはならない

    monkeypatch.setattr(svc_mod, "interpret_game", _router)
    sess = _session()
    sid, st = _spawned(sess, tmp_path)
    result = {}
    th = threading.Thread(
        target=lambda: result.setdefault(
            "llm", sess.game_command(sid, "あの緑のやつ思いっきり")
        ),
        daemon=True,
    )
    th.start()
    # input_seq確保を待つため少し待機し、parser指令を受理させる
    time.sleep(0.2)
    out = sess.game_command(sid, "前")
    assert out["accepted"] is True
    release.set()
    th.join(15)
    assert result["llm"]["accepted"] is False
    assert result["llm"]["reason"] == "stale"
    assert not (tmp_path / "strike.json").exists()


def test_llm_response_applies_when_generation_unchanged(tmp_path, monkeypatch):
    """世代が変わっていなければLLM応答は通常どおり適用される。"""
    import kotoba_api.service as svc_mod

    monkeypatch.setattr(
        svc_mod, "interpret_game",
        lambda text: {"type": "move", "direction": "left", "size": "small"},
    )
    sess = _session()
    sid, st = _spawned(sess, tmp_path)
    out = sess.game_command(sid, "ひらひら")
    assert out["accepted"] is True and out["via"] == "llm"
    wire = _read_control(tmp_path)
    assert wire["cmd"]["type"] == "move"
    assert wire["cmd"]["steps"][0]["dir"] == "left"


# ---- レビューA残欠 P04: 後からの明示禁止・取消しが未適用の旧意図を失効 ----
def test_prohibit_invalidates_pending_llm_intent(tmp_path, monkeypatch):
    """P04反例: LLM待ち→「進まないで」→遅いLLM応答は発行されない。

    旧実装はreject分岐でinput_seqを進めず、世代不変のまま旧LLM応答が
    moveを発行した。prohibit受理時に未適用の解釈世代を失効させ、
    遅着応答はstaleとして破棄する。
    """
    import threading

    import kotoba_api.service as svc_mod

    started = threading.Event()
    release = threading.Event()

    def _slow(text):
        started.set()
        release.wait(15)
        return {"type": "move", "direction": "forward", "size": "normal"}

    monkeypatch.setattr(svc_mod, "interpret_game", _slow)
    sess = _session()
    sid, st = _spawned(sess, tmp_path)
    result = {}
    th = threading.Thread(
        target=lambda: result.setdefault(
            "llm", sess.game_command(sid, "ふわふわ動かして")
        ),
        daemon=True,
    )
    th.start()
    assert started.wait(5), "llm inference did not start"
    # 推論がin-flightのまま明示禁止を送る — 即rejected受理
    t0 = time.monotonic()
    veto = sess.game_command(sid, "前に進まないで")
    assert time.monotonic() - t0 < 2.0
    assert veto["accepted"] is False and veto["reason"] == "rejected"
    release.set()
    th.join(15)
    # 禁止受理で入力世代が進んだため、遅いLLM応答は発行されない
    assert result["llm"]["accepted"] is False
    assert result["llm"]["reason"] == "stale"
    assert not (tmp_path / "control.json").exists()


def test_cancel_invalidates_pending_llm_intent(tmp_path, monkeypatch):
    """取消し語彙もprohibitクラス — 未適用の旧意図を失効する。"""
    import threading

    import kotoba_api.service as svc_mod

    started = threading.Event()
    release = threading.Event()

    def _slow(text):
        started.set()
        release.wait(15)
        return {"type": "strike"}

    monkeypatch.setattr(svc_mod, "interpret_game", _slow)
    sess = _session()
    sid, st = _spawned(sess, tmp_path)
    result = {}
    th = threading.Thread(
        target=lambda: result.setdefault(
            "llm", sess.game_command(sid, "あの緑のやつ思いっきり")
        ),
        daemon=True,
    )
    th.start()
    assert started.wait(5)
    out = sess.game_command(sid, "キャンセル")
    assert out["accepted"] is False and out["reason"] == "rejected"
    release.set()
    th.join(15)
    assert result["llm"]["accepted"] is False
    assert not (tmp_path / "strike.json").exists()


def test_nonop_reject_does_not_invalidate_pending_llm(tmp_path, monkeypatch):
    """非操作入力（質問・引用）は拒否するが未適用解釈の世代を変えない。

    「質問まで全て停止へ変える」必要はない — 別入力のLLM解釈は
    有効期限内で通常どおり適用される（受理順の新旧は保つ）。
    """
    import threading

    import kotoba_api.service as svc_mod

    started = threading.Event()
    release = threading.Event()

    def _slow(text):
        started.set()
        release.wait(15)
        return {"type": "move", "direction": "forward", "size": "small"}

    monkeypatch.setattr(svc_mod, "interpret_game", _slow)
    sess = _session()
    sid, st = _spawned(sess, tmp_path)
    result = {}
    th = threading.Thread(
        target=lambda: result.setdefault(
            "llm", sess.game_command(sid, "ふわふわ動かして")
        ),
        daemon=True,
    )
    th.start()
    assert started.wait(5)
    q = sess.game_command(sid, "今どこにいるの？")
    assert q["accepted"] is False and q["reason"] == "rejected"
    release.set()
    th.join(15)
    # 質問は世代を変えない — 旧意図のLLM応答は有効期限内で適用される
    assert result["llm"]["accepted"] is True
    assert _read_control(tmp_path)["cmd"]["type"] == "move"


def test_llm_ticket_deadline_rejects_late_answer(tmp_path, monkeypatch):
    """受付時に記録した期限を応答時に再検査 — 遅着回答へ新たな
    実行期限を与えない（世代が不変でも期限超過なら破棄）。"""
    import kotoba_api.service as svc_mod

    monkeypatch.setattr(svc_mod, "LLM_TICKET_TTL_S", 0.3)
    gate = {"open": False}

    def _gated(text):
        while not gate["open"]:
            time.sleep(0.05)
        return {"type": "move", "direction": "forward", "size": "small"}

    import threading

    monkeypatch.setattr(svc_mod, "interpret_game", _gated)
    sess = _session()
    sid, st = _spawned(sess, tmp_path)
    result = {}
    th = threading.Thread(
        target=lambda: result.setdefault(
            "llm", sess.game_command(sid, "ふわふわ動かして")
        ),
        daemon=True,
    )
    th.start()
    time.sleep(0.6)  # 受理期限(0.3s)を超えてから応答させる
    gate["open"] = True
    th.join(15)
    assert result["llm"]["accepted"] is False
    assert result["llm"]["reason"] == "expired"
    assert not (tmp_path / "control.json").exists()


def test_reset_records_interrupted_run_evidence(tmp_path):
    """resetで失効する実行中runは監督側で終端記録を残す。

    controllerがresult.jsonを書く前にresetが走ると結果は未観測のまま。
    「成功/失敗」を推測で生成せず、incompleteとして履歴へ記録する
    （C残欠 — reset終了理由の監督記録）。
    """
    sess = _session()
    sid, _ = _spawned(sess, tmp_path)
    r = sess.rounds[sid]
    run_id = r.run_id
    assert run_id == "run-test-1" and r.phase == "RUNNING"

    out = sess.reset(sid, restart_sim=False)
    assert out["reset"] is True
    r2 = sess.rounds[sid]
    assert r2.run_id is None
    assert r2.history and r2.history[-1]["orphaned_run_id"] == run_id
    assert r2.history[-1]["terminated_by"] == "reset"
    assert r2.history[-1]["result"] == "incomplete"


# ---- 2026-09-20 公開試用障害の回帰 -------------------------------------------
def test_boot_ready_rejects_expired_verification(tmp_path, monkeypatch):
    """stand_readyゲートは verified_at の鮮度(3600s)を要求する。

    検証ファイルが残っていても verified_at が古ければ False — stand manager
    は立位継続中に verified_at を定期更新するため、鮮度切れ=監視死亡の
    正直な信号として閉鎖する（公開試用でgame/startが409 sim_not_readyに
    なった障害の契約側。manager側の定期更新は実機で確認する）。
    """
    app_mod = _load_app(tmp_path, monkeypatch)
    _write_boot(tmp_path)
    ready, nonce = app_mod._boot_ready()
    assert ready is True and nonce == "boot-1"
    # verified_at が鮮度窓を超えると閉鎖（boot-lost未検出でも）
    (tmp_path / "boot-ready.json").write_text(json.dumps({
        "standing": True, "boot_nonce": "boot-1",
        "verified_at": time.time() - 3700,
    }))
    ready, _ = app_mod._boot_ready()
    assert ready is False


def test_approval_record_coerces_int_boot_nonce(tmp_path):
    """run result由来のint nonceでもapproveはValidationErrorにならない。

    controllerのresult.jsonは boot_nonce をintで返す。_boot_nonce_fn失効時の
    fallback(self.boot_nonce)がintのままApprovalRecord(sim_boot_id: str)へ
    渡ると500 — worker側では502 backend_unreachableへ正規化されていた。
    """
    import kotoba_api.service as svc_mod
    from kotoba_contracts.intent import parse_intent
    from kotoba_orchestrator.validator import build_plan

    sess = _session()
    sid, _ = _spawned(sess, tmp_path)
    r = sess.rounds[sid]
    # 実controller result.jsonと同じint nonceがfinish_run経由で入る
    sess.finish_run(sid, r.run_id, {"verdict": "PASS", "boot_nonce": 40869349881412})
    assert sess.boot_nonce == "40869349881412"
    # int nonceでもApprovalRecordが構築できる（approveの500回帰）
    intent = parse_intent({
        "schema_version": "1.0", "decision": "execute",
        "target_ids": ["goal_near"], "avoid_ids": [], "explanation": "てまえへ",
    })
    plan = build_plan(
        intent, svc_mod._virtual_world(r.round_id), session_id=sid,
        round_id=r.round_id, plan_id="plan-x", profile=svc_mod.PRODUCT_PROFILE,
        created_monotonic=time.monotonic(),
    )
    rec = svc_mod._approval_record(plan, r, 40869349881412)
    assert rec.sim_boot_id == "40869349881412" and isinstance(rec.sim_boot_id, str)


def test_display_round_returns_latest_spawned_spec(tmp_path, monkeypatch):
    """display/roundは最後にspawnしたroundのspecを返す。

    全session走査でdict先頭に残った古いsession（終了済みfixture等）のspecが
    新しいspawnを隠し、公開UIでスイカ位置が常に同じに見えた障害の対策。
    sim worldのスイカは1つ — 最後のspawnが正本。
    """
    from fastapi.testclient import TestClient

    app_mod = _load_app(tmp_path, monkeypatch)
    c = TestClient(app_mod.app)
    (tmp_path / "live.json").write_text(json.dumps({
        "boot_nonce": "boot-1", "obs_wall": time.time(), "wall": time.time(),
        "pos": [0, 0, 0.82], "quat_wxyz": [1, 0, 0, 0], "step_seq": 1,
    }))
    _write_boot(tmp_path)
    op = {"x-kotoba-operator": "op-token"}
    # 古いsessionにfixture specを残す（昨日の試験sessionの残滓を再現）
    old_sid = c.post("/api/sessions").json()["session_id"]
    c.post("/api/round/spawn",
           json=_spawn_json(app_mod, old_sid, target=[0.3, 0.35, 1.05]),
           headers=op)
    # 新しい利用者が通常spawn → 表示は新しい方の位置を返す
    new_sid = c.post("/api/sessions").json()["session_id"]
    c.post("/api/round/spawn", json=_spawn_json(app_mod, new_sid, seed=7),
           headers=op)
    d = c.get("/api/display/round").json()
    own = c.get(f"/api/game/state/{new_sid}").json()
    assert d["active"] is True
    assert d["watermelon"]["pos"] == own["spec"]["watermelon"]["pos"]
    assert d["watermelon"]["pos"] != [0.3, 0.35, 1.05]
    # reset後は古いfixtureへのフォールバックでスイカを再表示しない。
    assert app_mod.session.reset(new_sid, restart_sim=False)["reset"] is True
    assert c.get("/api/display/round").json() == {"active": False}


def test_game_start_rejects_spec_hidden_by_other_session_reset(tmp_path, monkeypatch):
    """別タブのidle reset後、見えないスイカへのゲーム開始を拒否する。"""
    from fastapi.testclient import TestClient

    app_mod = _load_app(tmp_path, monkeypatch)
    monkeypatch.setattr(app_mod, "_execute_game", lambda *a: None)
    c = TestClient(app_mod.app)
    op = {"x-kotoba-operator": "op-token"}
    owner = c.post("/api/sessions").json()["session_id"]
    other = c.post("/api/sessions").json()["session_id"]
    _write_boot(tmp_path)
    (tmp_path / "live.json").write_text(json.dumps({
        "boot_nonce": "boot-1", "obs_wall": time.time(), "wall": time.time(),
        "pos": [0, 0, 0.82], "quat_wxyz": [1, 0, 0, 0], "step_seq": 1,
    }))
    assert c.post("/api/round/spawn", json=_spawn_json(app_mod, owner, seed=42),
                  headers=op).status_code == 200
    assert c.get("/api/display/round").json()["active"] is True

    # 両ブラウザは同じsimを共有する。otherにはspecがなくてもreset可能。
    assert app_mod.session.reset(other, restart_sim=False)["reset"] is True
    assert c.get("/api/display/round").json() == {"active": False}
    state = c.get(f"/api/game/state/{owner}").json()
    assert state["startable"]["ok"] is False
    assert "world_superseded" in state["startable"]["reasons"]
    denied = c.post("/api/game/start", json={"session_id": owner}, headers=op)
    assert denied.status_code == 409
    assert denied.json()["detail"] == "world_superseded"
    assert app_mod.session.run_lock is False


def test_game_start_requires_spawn_boot_and_current_world(tmp_path, monkeypatch):
    """別タブのspawnや外部boot更新で古いspecを実行しない。"""
    from fastapi.testclient import TestClient

    app_mod = _load_app(tmp_path, monkeypatch)
    monkeypatch.setattr(app_mod, "_execute_game", lambda *a: None)
    c = TestClient(app_mod.app)
    op = {"x-kotoba-operator": "op-token"}
    owner = c.post("/api/sessions").json()["session_id"]
    other = c.post("/api/sessions").json()["session_id"]
    _write_boot(tmp_path, "boot-1")
    (tmp_path / "live.json").write_text(json.dumps({
        "boot_nonce": "boot-1", "obs_wall": time.time(), "wall": time.time(),
        "pos": [0, 0, 0.82], "quat_wxyz": [1, 0, 0, 0], "step_seq": 1,
    }))
    assert c.post("/api/round/spawn", json=_spawn_json(app_mod, owner, seed=42),
                  headers=op).status_code == 200
    assert c.post("/api/round/spawn", json=_spawn_json(app_mod, other, seed=43),
                  headers=op).status_code == 200
    assert c.get(f"/api/game/state/{owner}").json()["startable"]["ok"] is False
    denied = c.post("/api/game/start", json={"session_id": owner}, headers=op)
    assert denied.status_code == 409
    assert denied.json()["detail"] == "world_superseded"

    _write_boot(tmp_path, "boot-2")
    assert c.get("/api/display/round").json() == {"active": False}
    state = c.get(f"/api/game/state/{other}").json()
    assert state["startable"]["ok"] is False
    assert "spec_boot_mismatch" in state["startable"]["reasons"]
    denied = c.post("/api/game/start", json={"session_id": other}, headers=op)
    assert denied.status_code == 409
    assert denied.json()["detail"] == "spec_boot_mismatch"
    assert app_mod.session.run_lock is False


def test_game_start_rejects_stale_caller_boot_after_world_change(tmp_path):
    """APIがbootを読んだ後に世代が変わっても旧nonceのleaseを発行しない。"""
    _ensure_real_harness_pkg()
    from kotoba_api.service import ProductSession

    current_boot = ["boot-2"]
    sess = ProductSession(boot_nonce_fn=lambda: current_boot[0])
    sid = sess.create_session()
    sess.spawn_round(sid, 42, (0.0, 0.0, 0.82), 0.0,
                     boot_nonce="boot-2", expected_round_id=sess.rounds[sid].round_id,
                     expected_boot_nonce="boot-2")
    result = sess.start_game(sid, "boot-1", "stale-run", tmp_path)
    assert result == {"started": False, "reason": "spec_boot_mismatch"}
    assert sess.run_lock is False


def test_api_game_state_startable_dto(tmp_path, monkeypatch):
    """startable DTO — 開始可否の実理由を状態面で公開する（R6-A）。

    UIはこれをdisabled表示に使うだけで、実判定は /api/game/start が
    ロック内で再検査する。reasonsは同じコード列で、利用者表示と
    サーバー409のdetailが一致することを検証する。
    """
    from fastapi.testclient import TestClient

    app_mod = _load_app(tmp_path, monkeypatch)
    monkeypatch.setattr(app_mod, "_execute_game", lambda *a: None)
    c = TestClient(app_mod.app)
    op = {"x-kotoba-operator": "op-token"}
    sid = c.post("/api/sessions").json()["session_id"]

    # spawn前 — no_round_spawned + sim_not_ready が両方見える
    d = c.get(f"/api/game/state/{sid}").json()
    sb = d["startable"]
    assert sb["ok"] is False
    assert "no_round_spawned" in sb["reasons"]
    assert "sim_not_ready" in sb["reasons"]

    (tmp_path / "live.json").write_text(json.dumps({
        "boot_nonce": "boot-1", "obs_wall": time.time(), "wall": time.time(),
        "pos": [0, 0, 0.82], "quat_wxyz": [1, 0, 0, 0], "step_seq": 1,
    }))
    _write_boot(tmp_path)
    c.post("/api/round/spawn", json=_spawn_json(app_mod, sid, seed=42),
           headers=op)

    # spawn+boot準備完了 — 開始可能
    d = c.get(f"/api/game/state/{sid}").json()
    sb = d["startable"]
    assert sb["ok"] is True and sb["reasons"] == []

    # 開始後 — run_in_progress。同じ理由が409 detailにも出る
    r = c.post("/api/game/start", json={"session_id": sid}, headers=op)
    assert r.status_code == 200
    d = c.get(f"/api/game/state/{sid}").json()
    sb = d["startable"]
    assert sb["ok"] is False and "run_in_progress" in sb["reasons"]
    r2 = c.post("/api/game/start", json={"session_id": sid}, headers=op)
    assert r2.status_code == 409 and "run_in_progress" in r2.text


def test_api_request_id_header(tmp_path, monkeypatch):
    """X-Request-Id — 409等の失敗をapi.logへ相関させる診断ID（R6-A）。

    応答全てに付く。client-suppliedのidを受理する（Worker経由で
    端の要求idをそのまま観測するため）。
    """
    from fastapi.testclient import TestClient

    app_mod = _load_app(tmp_path, monkeypatch)
    c = TestClient(app_mod.app)
    sid = c.post("/api/sessions").json()["session_id"]
    r = c.post("/api/game/start", json={"session_id": sid},
               headers={"x-kotoba-operator": "op-token"})
    assert r.status_code == 409
    rid = r.headers.get("x-request-id")
    assert rid and len(rid) >= 8
    # client-supplied idはそのまま返る（端側トレースの透過）
    r2 = c.get("/api/health", headers={"x-request-id": "edge-abc123"})
    assert r2.headers.get("x-request-id") == "edge-abc123"
