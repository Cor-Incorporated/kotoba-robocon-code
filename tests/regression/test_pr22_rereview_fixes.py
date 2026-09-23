"""PR #22 再検収レビュー（R2-A/B, R3-A/B）の反例→修正の回帰試験。

前回のAST抜粋試験に加え、本ファイルは実モジュールの入口で検証する:
- 実 ProductSession（reset中のstart_run拒否・遅着finishのboot非復帰）
- 実 ClockGate（sim_t凍結/逆行/退役nonce）
- 実 kotoba_runner.Observer（boot固定・凍結sim_tのstale化）
- 実 kotoba_live_observer.LiveObserver（退役boot遅着の拒否）
- 実 FastAPI app（TestClient経由のHTTP入口: operator gate・reset中run拒否）

対象反例（kotoba-pr22-rereview-2026-09-17/run_probes.py）:
- C08: reset中（pre_restart窓）のstart_runが受付される → 修正後は拒否
- C09: 旧run終了が boot だけ旧値へ戻す → 修正後はboot不変
- C11: seq増+sim_t凍結をlatestが501回受理 → 修正後はstale
- C12: A→B→旧A の時計をClockGateが受理 → 修正後は退役drop
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import struct
import sys
import threading
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "services" / "harness" / "src" / "kotoba_harness"
RUNTIME = ROOT / "services" / "runtime"


def _ensure_real_harness_pkg():
    """snapshot汚染対策: 現行 kotoba_harness パッケージを __path__ 付きで登録。

    test_product_characterization_review が収集時に snapshot 版を
    sys.modules/sys.path へ置くため、明示的に現行版へ差し替える。
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


_ensure_real_harness_pkg()

from kotoba_harness.observer import ClockGate  # noqa: E402


def _load_module(alias: str, path: Path):
    """現行ソースを別名モジュールとして読む（sys.path汚染に非依存）。

    読み込み中だけ kotoba_harness を現行版へ差し替え、完了後に
    既存sys.modules状態を復元する（snapshot characterization試験と共存）。
    """
    saved = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k == "kotoba_harness" or k.startswith("kotoba_harness.")
    }
    try:
        _ensure_real_harness_pkg()
        spec = importlib.util.spec_from_file_location(alias, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[alias] = mod
        spec.loader.exec_module(mod)
    finally:
        # 読み込まれたモジュールには現行参照が結び付いたまま。
        # 他テスト用に sys.modules を元の状態へ戻す。
        for k in [k for k in sys.modules if k == "kotoba_harness" or k.startswith("kotoba_harness.")]:
            sys.modules.pop(k)
        sys.modules.update(saved)
    return mod


@pytest.fixture()
def lcm_stub(monkeypatch):
    stub = types.ModuleType("lcm")
    stub.LCM = lambda _url: None
    monkeypatch.setitem(sys.modules, "lcm", stub)
    return stub


@pytest.fixture()
def runner_mod(lcm_stub):
    return _load_module("kotoba_runner_current", RUNTIME / "kotoba_runner.py")


@pytest.fixture()
def live_mod(lcm_stub):
    return _load_module("kotoba_live_current", RUNTIME / "kotoba_live_observer.py")


def _make_session():
    from kotoba_api.service import ProductSession

    class _SimCtl:
        def __init__(self):
            self.restarts = 0

        def restart(self):
            self.restarts += 1
            return True, "boot-new"

    return ProductSession(sim_ctl=_SimCtl(), live_accepted=False, selftest=True)


def _put_plan(session, sid, plan_id="p1"):
    """実planをREVIEW状態へ置く（決定論validator、LLM不使用）。"""
    from kotoba_api.service import PRODUCT_PROFILE, _virtual_world
    from kotoba_contracts.intent import parse_intent
    from kotoba_orchestrator.validator import build_plan

    r = session.rounds[sid]
    intent = parse_intent(
        {
            "schema_version": "1.0",
            "decision": "execute",
            "target_ids": ["goal_near"],
            "avoid_ids": [],
            "explanation": "てまえへ",
        }
    )
    plan = build_plan(
        intent,
        _virtual_world(r.round_id),
        session_id=sid,
        round_id=r.round_id,
        plan_id=plan_id,
        profile=PRODUCT_PROFILE,
        created_monotonic=time.monotonic(),
    )
    r.plan_obj = plan
    r.pending_plan_id = plan.plan_id
    r.phase = "REVIEW"
    return plan


# ============================ R2-A: reset窓の実行受付 ========================
def test_c08_start_run_during_reset_window_is_rejected():
    """実session: resetのpre_restart中にstart_run → 拒否・承認未消費・lock不残留。"""
    session = _make_session()
    sid = session.create_session()
    _put_plan(session, sid)

    consumed = []

    class Store:
        def verify_and_consume(self, *a, **k):
            consumed.append(k)
            return types.SimpleNamespace(model_dump=lambda: {"grant": True})

    session.store = Store()
    during = {}

    def cleanup_interleaving():
        # pre_restart（runner kill等）の最中に実行要求が届く順序を再現
        during["phase"] = session.rounds[sid].phase
        during["res"] = session.start_run(sid, "p1", "valid-approval", "boot-old")

    out = session.reset(sid, restart_sim=True, pre_restart=cleanup_interleaving)
    assert out["reset"] is True
    assert during["phase"] == "RESETTING"
    assert during["res"]["started"] is False  # 受付閉鎖中は起動しない
    assert consumed == []  # 有効な旧承認を消費していない
    r = session.rounds[sid]
    assert r.phase == "READY" and session.run_lock is False
    # reset完了後も旧roundのpendingは失効済み → 同じ要求は引き続き拒否
    res2 = session.start_run(sid, "p1", "valid-approval", "boot-old")
    assert res2["started"] is False and consumed == []


def test_c08b_admission_reopens_after_reset_with_new_plan():
    """reset完了後の受付再開: 新roundのREVIEW+新承認なら正常に開始する。"""
    session = _make_session()
    sid = session.create_session()
    session.reset(sid, restart_sim=True)
    _put_plan(session, sid, plan_id="p2")

    class Store:
        def verify_and_consume(self, *a, **k):
            return types.SimpleNamespace(model_dump=lambda: {"grant": True})

    session.store = Store()
    res = session.start_run(sid, "p2", "a", "boot-new")
    assert res["started"] is True
    assert session.rounds[sid].phase == "RUNNING" and session.run_lock is True


# ============================ R2-B: 遅着finishの副作用 ======================
def test_c09_stale_finish_does_not_revert_boot_nonce():
    """実session: reset+新run後の旧run終了はbootを旧値へ戻さない。"""
    session = _make_session()
    sid = session.create_session()
    r = session.rounds[sid]
    r.phase, r.run_id = "RUNNING", "run-old"
    session.run_lock = True

    session.reset(sid, restart_sim=True)  # boot_nonce='boot-new'をライフサイクルが設定
    assert session.boot_nonce == "boot-new"
    r2 = session.rounds[sid]
    r2.phase, r2.run_id = "RUNNING", "run-new"
    session.run_lock = True

    accepted = session.finish_run(
        sid, "run-old", {"run_id": "run-old", "verdict": "PASS", "boot_nonce": "boot-old"}
    )
    assert accepted is False
    assert session.boot_nonce == "boot-new"  # bootはライフサイクル所有・旧値へ戻らない
    assert r2.phase == "RUNNING" and session.run_lock is True

    # 現行runの正常finishはbootを反映する（陽性対照）
    accepted = session.finish_run(
        sid, "run-new", {"verdict": "PASS", "boot_nonce": "boot-new"}
    )
    assert accepted is True and session.boot_nonce == "boot-new"


def test_c09b_abort_run_closes_admission_and_invalidates():
    """pause経路（abort_run）: 世代失効+受付閉鎖を一つの区間で確定。"""
    session = _make_session()
    sid = session.create_session()
    _put_plan(session, sid)
    r = session.rounds[sid]
    r.phase, r.run_id = "RUNNING", "run-x"
    session.run_lock = True

    session.abort_run(sid, "中断")
    assert session._admission_open is False
    assert r.phase == "FAULT" and r.run_id is None and session.run_lock is False
    # FAULT中のstart_runは拒否（有効な未使用承認でも）
    res = session.start_run(sid, "p1", "a", "boot-x")
    assert res["started"] is False
    # 遅着finishは履歴隔離・boot不変
    session.boot_nonce = "boot-new"
    assert (
        session.finish_run(sid, "run-x", {"verdict": "PASS", "boot_nonce": "boot-old"})
        is False
    )
    assert session.boot_nonce == "boot-new"


# ============================ R3-A: sim_t進行の検証 ==========================
def test_c11_clock_gate_rejects_frozen_sim_time_with_advancing_seq():
    """実ClockGate: seq増・sim_t凍結の入力を受理しない（C11反例）。"""
    g = ClockGate()
    assert g.accept(777, 1, 0.002) == (True, False)
    for i in range(500):
        ok, changed = g.accept(777, i + 2, 0.002)  # seqは進むがsim_t=0.002凍結
        assert ok is False and changed is False
    assert g.dropped_frozen == 500
    # 正当な進行は通る（陽性対照）
    assert g.accept(777, 502, 0.004) == (True, False)


def test_c11b_clock_gate_rejects_sim_time_regression_and_nonfinite():
    g = ClockGate()
    g.accept(777, 1, 1.0)
    assert g.accept(777, 2, 0.5) == (False, False)  # sim_t逆行
    assert g.accept(777, 3, float("nan")) == (False, False)
    assert g.accept(777, 4, float("inf")) == (False, False)
    assert g.dropped_frozen == 3
    assert g.accept(777, 5, 1.002) == (True, False)


def test_c11c_runner_latest_rejects_frozen_sim_stream(runner_mod):
    """実runner Observer: seq増+sim_t凍結streamはlatestがstaleを上げる（実入口）。"""

    class FakeClock:
        def __init__(self):
            self.t = 100.0

        def monotonic(self):
            return self.t

    clock = FakeClock()
    mod = runner_mod
    mod.time = types.SimpleNamespace(monotonic=clock.monotonic)
    obs = mod.Observer.__new__(mod.Observer)
    obs._lock = threading.Lock()
    obs._state = obs._clock = obs.bound = None
    obs.nonce = None
    obs._gate = ClockGate()
    obs._exception = None
    obs._expected_nonce = None
    obs._dropped_boot = 0

    state = struct.pack(">qdi", 0x2D53D9E29374E48E, 0.0, 0)
    state += struct.pack(">3d3d4d", 0.0, 0.0, 0.82, 0.0, 0.0, 0.0, 1.0, 0, 0, 0)

    accepted = rejected = 0
    for i in range(501):
        clock.t = 100.0 + i * 0.01
        obs._on_state("sim_state", state)
        obs._on_clock(
            "kotoba_sim_clock",
            struct.pack(">qqqdd", 0x4B544F4241434C31, 777, i + 1, 0.0, clock.t),
        )
        try:
            obs.latest()
            accepted += 1
        except Exception:
            rejected += 1
    # 最初の標本のみ受理・以後は凍結sim_tでdrop → boundが古くなりstale
    assert rejected > 0 and obs._gate.dropped_frozen == 500
    assert obs.bound[5] == 1  # boundは最初のseq=1標本のまま


def test_runner_observer_pins_to_expected_boot(runner_mod):
    """実runner Observer: 承認boot以外のパケットは受理しない（R3-B）。"""
    mod = runner_mod
    mod.time = types.SimpleNamespace(monotonic=lambda: 200.0)
    obs = mod.Observer.__new__(mod.Observer)
    obs._lock = threading.Lock()
    obs._state = obs._clock = obs.bound = None
    obs.nonce = None
    obs._gate = ClockGate()
    obs._exception = None
    obs._expected_nonce = 777  # 承認されたboot
    obs._dropped_boot = 0

    pkt = lambda n, s, t: struct.pack(">qqqdd", 0x4B544F4241434C31, n, s, t, 200.0)
    obs._on_clock("kotoba_sim_clock", pkt(888, 1, 0.1))  # 別boot
    obs._on_clock("kotoba_sim_clock", pkt(999, 1, 0.1))  # 別boot
    assert obs._dropped_boot == 2 and obs._clock is None and obs.bound is None
    obs._on_clock("kotoba_sim_clock", pkt(777, 1, 0.1))
    assert obs._clock is not None  # 承認bootのみ受理


# ============================ R3-B: 退役bootの扱い ==========================
def test_c12_clock_gate_rejects_retired_boot_replay():
    """実ClockGate: A→B→旧A は退役drop（C12反例）。live側のboot追従契約。"""
    g = ClockGate()
    assert g.accept(111, 100, 1.0) == (True, False)
    assert g.accept(222, 1, 0.002) == (True, True)  # 正当な新boot
    ok, changed = g.accept(111, 101, 1.002)  # 退役bootの遅着
    assert (ok, changed) == (False, False)
    assert g.nonce == 222 and g.dropped_retired == 1 and g.boot_changes == 1


def test_c12b_live_observer_ignores_retired_boot_packets(live_mod):
    """実live observer: 新boot追従後、退役bootの遅着はboundを戻さない。"""
    mod = live_mod
    mod.time = types.SimpleNamespace(monotonic=lambda: 300.0, time=lambda: 1000.0)
    obs = mod.LiveObserver.__new__(mod.LiveObserver)
    obs._lock = threading.Lock()
    obs._state = obs._clock = obs._task = obs._bound = None
    obs._recv_total = 0
    obs._gate = ClockGate()
    obs._running = True
    obs._exception = None

    pkt = lambda n, s, t: struct.pack(">qqqdd", 0x4B544F4241434C31, n, s, t, 300.0)
    obs._on_clock("c", pkt(111, 100, 1.0))
    obs._on_clock("c", pkt(222, 1, 0.002))  # 新bootへ追従
    obs._on_clock("c", pkt(111, 101, 1.002))  # 退役Aの遅着
    assert obs._gate.nonce == 222 and obs._gate.dropped_retired == 1
    # clockは新bootのseq=1のまま（退役Aのseq=101で上書きされない）
    assert obs._clock[2] == 1


# ============================ 実API入口（TestClient） ========================
def _load_app(tmp_path, monkeypatch, **env):
    """実 kotoba_api.app をenv付きで読み込む（TestClient用）。"""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("KOTOBA_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("KOTOBA_HOME", str(tmp_path))
    import kotoba_api.app as app_mod

    return importlib.reload(app_mod)


def test_api_operator_gate_real_http(tmp_path, monkeypatch):
    """実API: LIVE未受入+SELFTEST → 無token 403 / 正tokenで先へ進む。"""
    from fastapi.testclient import TestClient

    app_mod = _load_app(
        tmp_path,
        monkeypatch,
        KOTOBA_OFFLINE="0",
        KOTOBA_SELFTEST="1",
        KOTOBA_OPERATOR_TOKEN="tok-real",
    )
    monkeypatch.setattr(app_mod, "_boot_ready", lambda: (True, "boot-x"))
    monkeypatch.setattr(app_mod, "_execute_run", lambda *a, **k: None)

    session = app_mod.session
    sid = session.create_session()
    _put_plan(session, sid)
    ap = session.approve(sid, "p1")
    assert ap["approved"] is True
    approval_id = ap["approval_id"]
    # /api/intents が作る_pendingを再現（本試験はrun経路に限定）
    app_mod._pending["p1"] = {"distance_m": 0.45, "session_id": sid}

    client = TestClient(app_mod.app)
    # boot-ready.json を用意（実ファイル経路を通す）
    (tmp_path / "boot-ready.json").write_text(
        json.dumps({"standing": True, "boot_nonce": "boot-x", "verified_at": time.time()})
    )

    body = {"session_id": sid, "plan_id": "p1", "approval_id": approval_id}
    r = client.post("/api/runs", json=body)
    assert r.status_code == 403 and r.json()["detail"] == "operator_required"
    r = client.post("/api/runs", json=body, headers={"x-kotoba-operator": "bad"})
    assert r.status_code == 403
    # 正token: 実session・実store経路でrun開始まで進む
    r = client.post(
        "/api/runs", json=body, headers={"x-kotoba-operator": "tok-real"}
    )
    assert r.status_code == 200 and "run_id" in r.json()
    assert session.rounds[sid].phase == "RUNNING" and session.run_lock is True


def test_api_reset_window_rejects_concurrent_run(tmp_path, monkeypatch):
    """実API: reset中に/api/runsが届いても409・承認未消費・lock不残留。"""
    from fastapi.testclient import TestClient

    app_mod = _load_app(
        tmp_path,
        monkeypatch,
        KOTOBA_OFFLINE="0",
        KOTOBA_SELFTEST="1",
        KOTOBA_OPERATOR_TOKEN="tok-real",
    )
    monkeypatch.setattr(app_mod, "_boot_ready", lambda: (True, "boot-x"))
    monkeypatch.setattr(app_mod, "_execute_run", lambda *a, **k: None)

    def kill_and_rearm():
        # C08旧バグ形状の再現: reset窓内で旧roundのpendingが残ったままの
        # 実行要求が届く（旧実装はpending非消去で受理してしまった）。
        # 窓内でpendingを再武装し、admission gate単独の拒否を検証する。
        r = session.rounds[sid]
        r.pending_plan_id = "p1"
        r.plan_obj = plan
        time.sleep(0.5)

    # reset中のcleanup窓を広げる（実運用のrunner kill I/Oに相当）
    monkeypatch.setattr(app_mod, "_kill_runners", kill_and_rearm)

    session = app_mod.session
    session._sim_ctl = types.SimpleNamespace(
        restart=lambda: (time.sleep(0.4), (True, "boot-new"))[1]
    )
    sid = session.create_session()
    plan = _put_plan(session, sid)
    ap = session.approve(sid, "p1")
    approval_id = ap["approval_id"]
    # /api/intents が作る_pendingを再現 — これが無いとstale_planで早期409になり
    # admission gateの検査にならない（修正前コードでも通る偽陰性を防ぐ）
    app_mod._pending["p1"] = {"distance_m": 0.45, "session_id": sid}

    client = TestClient(app_mod.app)
    results = {}

    def do_reset():
        results["reset"] = client.post("/api/round/reset", json={"session_id": sid})

    th = threading.Thread(target=do_reset)
    th.start()
    time.sleep(0.15)  # resetがRESETTINGに入るまで待つ
    r = client.post(
        "/api/runs",
        json={"session_id": sid, "plan_id": "p1", "approval_id": approval_id},
        headers={"x-kotoba-operator": "tok-real"},
    )
    th.join(timeout=30)
    assert results["reset"].status_code == 200
    # 実行受付閉鎖（reset_in_progress）由来の409であることを明示 — 単なる
    # stale_plan/run_in_progressではなくR2-A経路を踏んだことを区別する
    assert r.status_code == 409 and r.json()["detail"] == "reset_in_progress"
    assert session.store.is_consumed(approval_id) is False
    assert session.run_lock is False
    assert session.rounds[sid].phase == "READY"


def test_kiosk_exit_gate_rejects_run_and_reset_window():
    session = _make_session()
    sid = session.create_session()

    assert session.reserve_kiosk_exit(hold_s=10)
    assert session.start_game(sid, "boot", "run", Path("/unused"))["reason"] == "kiosk_exiting"
    session.rounds[sid].phase = "REVIEW"
    assert session.start_run(sid, "plan", "approval", "boot")["reason"] == "kiosk_exiting"

    session._kiosk_exit_until = 0
    session.rounds[sid].phase = "RESETTING"
    assert session.reserve_kiosk_exit() is False
    session.rounds[sid].phase = "READY"
    session._admission_open = False
    assert session.reserve_kiosk_exit() is False
    session._admission_open = True
    session.run_lock = True
    assert session.reserve_kiosk_exit() is False


def test_kiosk_exit_gate_http_requires_operator_and_idle(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    app_mod = _load_app(
        tmp_path, monkeypatch, KOTOBA_OFFLINE="0", KOTOBA_SELFTEST="1",
        KOTOBA_OPERATOR_TOKEN="tok-real",
    )
    app_mod.session.create_session()
    client = TestClient(app_mod.app)
    assert client.post("/api/kiosk/exit-ready").status_code == 403
    headers = {"x-kotoba-operator": "tok-real"}
    assert client.post("/api/kiosk/exit-ready", headers=headers).json() == {"safe": True}
    app_mod.session._admission_open = False
    assert client.post("/api/kiosk/exit-ready", headers=headers).status_code == 409


def test_spawn_cannot_replace_world_during_run_or_exit():
    from kotoba_api.service import SpawnRejected

    session = _make_session()
    first = session.create_session()
    second = session.create_session()
    spec = session.spawn_round(first, 7, (0, 0, 0.82), 0.0)
    history = list(session._spawn_history)

    with pytest.raises(SpawnRejected, match="round_not_ready"):
        session.spawn_round(first, 8, (0, 0, 0.82), 0.0)
    session.rounds[first].phase = "RUNNING"
    session.run_lock = True
    with pytest.raises(SpawnRejected, match="run_in_progress"):
        session.spawn_round(second, 9, (0, 0, 0.82), 0.0)
    session.run_lock = False
    session.rounds[first].phase = "RESULT"
    assert session.reserve_kiosk_exit()
    with pytest.raises(SpawnRejected, match="kiosk_exiting"):
        session.spawn_round(second, 9, (0, 0, 0.82), 0.0)
    assert session.rounds[first].round_spec is spec
    assert session.rounds[second].round_spec is None
    assert session._latest_spec_session == first
    assert session._spawn_history == history


def test_reset_rejects_other_session_run_before_global_restart():
    session = _make_session()
    owner = session.create_session()
    other = session.create_session()
    session.rounds[owner].phase = "RUNNING"
    session.rounds[owner].run_id = "owner-run"
    session.run_lock = True
    cleanup = []

    out = session.reset(other, restart_sim=True, pre_restart=lambda: cleanup.append(True))
    assert out == {"reset": False, "reason": "other_session_running"}
    assert cleanup == []
    assert session._sim_ctl.restarts == 0
    assert session._admission_open is True
    assert session.run_lock is True
    assert session.rounds[owner].run_id == "owner-run"


def test_reset_http_rejects_other_session_run(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    app_mod = _load_app(tmp_path, monkeypatch, KOTOBA_OFFLINE="0")
    owner = app_mod.session.create_session()
    other = app_mod.session.create_session()
    app_mod.session.rounds[owner].phase = "RUNNING"
    app_mod.session.run_lock = True
    monkeypatch.setattr(app_mod, "_kill_runners", lambda: pytest.fail("runner killed"))
    response = TestClient(app_mod.app).post(
        "/api/round/reset", json={"session_id": other}
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "other_session_running"
