"""PR #22 C13/C14（遅延launch取消・重複reset）の反例→修正の回帰試験。

実入口で検証する（AST抜粋ではない）:
- 実 ProductSession.reset: 並行resetの直列化・受付早期再開の防止（C14）
- 実 FastAPI /api/round/reset: 同時POSTがrestartを重複させない（実HTTP）
- 実 _execute_run: launch中に世代失効 → 起こしたcontainerを即終了（C13）
- 実 _execute_run: launch前世代失効 → docker runを呼ばない（C13）
"""

from __future__ import annotations

import importlib
import json
import subprocess
import threading
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _make_session():
    from kotoba_api.service import ProductSession

    class _SimCtl:
        def __init__(self):
            self.restarts = 0
            self.in_restart = threading.Event()

        def restart(self):
            self.restarts += 1
            self.in_restart.set()
            time.sleep(0.4)  # 実sim再起動相当の窓
            self.in_restart.clear()
            return True, "boot-new"

    return ProductSession(sim_ctl=_SimCtl(), live_accepted=False, selftest=True)


def _put_plan(session, sid, plan_id="p1"):
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


# ============================ C14: 重複resetの直列化 ========================
def test_c14_concurrent_reset_same_session_coalesces():
    """実session: 同一sessionへの並行resetはrestartを重複させない。"""
    session = _make_session()
    sid = session.create_session()
    results = {}

    def do_reset(tag):
        results[tag] = session.reset(sid, restart_sim=True)

    th1 = threading.Thread(target=do_reset, args=("r1",))
    th2 = threading.Thread(target=do_reset, args=("r2",))
    th1.start()
    time.sleep(0.05)  # r1が_reset_muを取るまでの僅かな差
    th2.start()
    th1.join(timeout=30)
    th2.join(timeout=30)
    assert results["r1"]["reset"] is True and results["r2"]["reset"] is True
    # 2回目は直前resetの併合 → sim再起動は1回だけ（二重boot乱発を防ぐ）
    assert session._sim_ctl.restarts == 1
    assert session.rounds[sid].phase == "READY"
    assert session._admission_open is True


def test_c14_concurrent_reset_other_session_serializes_not_coalesces():
    """別sessionのresetは併合されず直列実行される（各roundが確実に新規化）。"""
    session = _make_session()
    sid1 = session.create_session()
    sid2 = session.create_session()
    results = {}

    def do_reset(tag, s):
        results[tag] = session.reset(s, restart_sim=True)

    th1 = threading.Thread(target=do_reset, args=("r1", sid1))
    th2 = threading.Thread(target=do_reset, args=("r2", sid2))
    th1.start()
    time.sleep(0.05)
    th2.start()
    th1.join(timeout=30)
    th2.join(timeout=30)
    assert results["r1"]["reset"] is True and results["r2"]["reset"] is True
    # 別sessionは併合対象外 → 2回restart、ただし同時には走らない
    assert session._sim_ctl.restarts == 2
    assert session.rounds[sid1].phase == "READY" and session.rounds[sid2].phase == "READY"


def test_c14_admission_stays_closed_while_reset_in_progress():
    """reset全体（外部I/O含む）の間、受付は閉じたまま — 早期再開なし。"""
    session = _make_session()
    sid = session.create_session()
    _put_plan(session, sid)
    seen = {}

    def do_reset():
        seen["res"] = session.reset(sid, restart_sim=True)

    th = threading.Thread(target=do_reset)
    th.start()
    assert session._sim_ctl.in_restart.wait(timeout=10)
    # restartの外部I/O最中: 受付は閉じている（旧実装はここで再開し得た）
    assert session._admission_open is False
    res = session.start_run(sid, "p1", "a", "boot-old")
    assert res["started"] is False and res.get("reason") == "reset_in_progress"
    th.join(timeout=30)
    assert seen["res"]["reset"] is True
    assert session._admission_open is True


# ============================ C13: 遅延launchの取消し =======================
def _load_app(tmp_path, monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("KOTOBA_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("KOTOBA_HOME", str(tmp_path))
    import kotoba_api.app as app_mod

    return importlib.reload(app_mod)


def _armed_run(app_mod, sid, monkeypatch):
    """実sessionで承認済みrunを開始し _execute_run の引数を返す。"""
    monkeypatch.setattr(app_mod, "_boot_ready", lambda: (True, "boot-x"))
    session = app_mod.session
    plan = _put_plan(session, sid)
    ap = session.approve(sid, "p1")
    assert ap["approved"] is True
    app_mod._pending["p1"] = {
        "distance_m": 0.45,
        "session_id": sid,
        "anchor": [0.0, 0.0],
        "heading_yaw": 0.0,
        "target_id": "goal_near",
    }
    started = session.start_run(sid, "p1", ap["approval_id"], "boot-x")
    assert started["started"] is True
    return started["run_id"], app_mod._pending["p1"], started["grant"]["sim_boot_id"]


def test_c13_prelaunch_generation_loss_skips_docker_run(tmp_path, monkeypatch):
    """実_execute_run: launch前に世代失効 → subprocess(docker run)を呼ばない。"""
    app_mod = _load_app(
        tmp_path, monkeypatch,
        KOTOBA_OFFLINE="0", KOTOBA_SELFTEST="1", KOTOBA_OPERATOR_TOKEN="tok",
    )
    sid = app_mod.session.create_session()
    run_id, pending, boot = _armed_run(app_mod, sid, monkeypatch)

    calls = []

    class FakeCompleted:
        def __init__(self, rc=0):
            self.returncode = rc
            self.stdout = ""
            self.stderr = ""

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return FakeCompleted()

    monkeypatch.setattr(subprocess, "run", fake_run)
    # launch前に世代を失効させる（pause/reset相当）
    app_mod.session.abort_run(sid, "中断")
    app_mod._execute_run(sid, run_id, pending, boot)
    assert calls == []  # docker run すら呼ばない


def test_c13_midlaunch_invalidation_kills_spawned_container(tmp_path, monkeypatch):
    """実_execute_run: docker run完了〜wait前の窓でpause/reset → 起こした
    containerを直ちに rm -f し、waitせず結果を隔離する（C13反例）。"""
    app_mod = _load_app(
        tmp_path, monkeypatch,
        KOTOBA_OFFLINE="0", KOTOBA_SELFTEST="1", KOTOBA_OPERATOR_TOKEN="tok",
    )
    sid = app_mod.session.create_session()
    run_id, pending, boot = _armed_run(app_mod, sid, monkeypatch)
    cname = f"kotoba-runner-{run_id[:8]}"

    calls = []

    class FakeCompleted:
        def __init__(self, rc=0):
            self.returncode = rc
            self.stdout = ""
            self.stderr = ""

    def fake_run(cmd, **kw):
        calls.append(cmd)
        joined = " ".join(cmd) if isinstance(cmd, list) else cmd
        if "docker run" in joined:
            # launch中にpause/resetが走った状態を再現（kill対象に載らない窓）
            app_mod.session.abort_run(sid, "中断")
        return FakeCompleted()

    monkeypatch.setattr(subprocess, "run", fake_run)
    app_mod._execute_run(sid, run_id, pending, boot)

    flat = [" ".join(c) if isinstance(c, list) else c for c in calls]
    assert any("docker run" in c for c in flat)
    # 世代失効を検出 → 自分が起こしたcontainerを即終了（孤児runnerを残さない）
    assert any("docker" in c and "rm" in c and cname in c for c in flat)
    # wait も result.json 読みもしていない（起動していないものを待たない）
    assert not any("docker wait" in c for c in flat)
    # 結果は隔離（現roundはFAULTのまま、履歴へ）
    r = app_mod.session.rounds[sid]
    assert r.phase == "FAULT"
    assert r.history and r.history[-1]["orphaned_run_id"] == run_id
    assert r.history[-1]["result"]["verdict"] == "ABORTED_AT_LAUNCH"


def test_c13_normal_launch_waits_and_finishes(tmp_path, monkeypatch):
    """陽性対照: 世代が有効なままなら wait→result読取→finish が走る。"""
    app_mod = _load_app(
        tmp_path, monkeypatch,
        KOTOBA_OFFLINE="0", KOTOBA_SELFTEST="1", KOTOBA_OPERATOR_TOKEN="tok",
    )
    sid = app_mod.session.create_session()
    run_id, pending, boot = _armed_run(app_mod, sid, monkeypatch)

    calls = []

    class FakeCompleted:
        def __init__(self, rc=0):
            self.returncode = rc
            self.stdout = ""
            self.stderr = ""

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return FakeCompleted()

    monkeypatch.setattr(subprocess, "run", fake_run)
    # runnerが書く result.json を run dir に置く（wait後に読まれる）
    run_dir = Path(app_mod.RUNTIME_DIR) / run_id

    orig_run = fake_run

    def fake_run2(cmd, **kw):
        out = orig_run(cmd, **kw)
        joined = " ".join(cmd) if isinstance(cmd, list) else cmd
        if "docker wait" in joined:
            run_dir.mkdir(exist_ok=True)
            (run_dir / "result.json").write_text(
                json.dumps({"verdict": "PASS", "err_m": 0.05, "boot_nonce": "boot-x"})
            )
        return out

    monkeypatch.setattr(subprocess, "run", fake_run2)
    app_mod._execute_run(sid, run_id, pending, boot)

    flat = [" ".join(c) if isinstance(c, list) else c for c in calls]
    assert any("docker run" in c for c in flat)
    assert any("docker wait" in c for c in flat)
    r = app_mod.session.rounds[sid]
    assert r.phase == "RESULT" and r.result["verdict"] == "PASS"
    assert app_mod.session.run_lock is False


# ================== 実HTTP入口: 同時reset POST ===============================
def test_api_concurrent_reset_posts_do_not_double_restart(tmp_path, monkeypatch):
    """実API: /api/round/reset の同時POST → restart 1回・双方200。"""
    from fastapi.testclient import TestClient

    app_mod = _load_app(
        tmp_path, monkeypatch,
        KOTOBA_OFFLINE="0", KOTOBA_SELFTEST="1", KOTOBA_OPERATOR_TOKEN="tok",
    )
    restarts = []

    class FakeSim:
        def restart(self):
            restarts.append(1)
            time.sleep(0.3)
            return True, "boot-new"

        def pause(self):
            return True

    app_mod.session._sim_ctl = FakeSim()
    monkeypatch.setattr(app_mod, "_kill_runners", lambda: None)
    monkeypatch.setattr(app_mod, "_execute_run", lambda *a, **k: None)

    client = TestClient(app_mod.app)
    sid = app_mod.session.create_session()
    results = {}

    def do_reset(tag):
        results[tag] = client.post("/api/round/reset", json={"session_id": sid})

    ths = [threading.Thread(target=do_reset, args=(f"r{i}",)) for i in range(2)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(timeout=30)
    assert results["r0"].status_code == 200 and results["r1"].status_code == 200
    assert len(restarts) == 1
    assert app_mod.session.rounds[sid].phase == "READY"
