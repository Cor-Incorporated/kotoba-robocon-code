"""PR #22 検収レビュー（R1/R2/R3）の反例→修正の回帰試験。

- C4: reset中の旧runの遅着finishが新roundを上書きしない
- C6: 凍結seq/sim時刻の再配信を観測が新規標本にしない（ClockGate共有実装）
- C7: 古い観測をファイル書出し時刻でfreshへ戻さない
- R2: run_lock の check-and-set は同時要求から保護される
"""

import importlib.util
import threading
import time
from pathlib import Path

import pytest

from kotoba_orchestrator.freshness import live_freshness

ROOT = Path(__file__).resolve().parents[2]


def _load_clock_gate():
    """現行 observer.py を直接読む（snapshot sys.path汚染への耐性）。"""
    spec = importlib.util.spec_from_file_location(
        "kotoba_harness_observer_current",
        ROOT / "services" / "harness" / "src" / "kotoba_harness" / "observer.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ClockGate


ClockGate = _load_clock_gate()


class _SimCtl:
    def __init__(self, ok=True):
        self.ok = ok
        self.restarts = 0

    def restart(self):
        self.restarts += 1
        return self.ok, "boot-new"


def _make_session():
    from kotoba_api.service import ProductSession

    return ProductSession(sim_ctl=_SimCtl(), live_accepted=False, selftest=True)


def _running_round(session, sid, run_id):
    r = session.rounds[sid]
    r.phase = "RUNNING"
    r.run_id = run_id
    session.run_lock = True


# --- C4: reset後の旧run finish は新roundを汚さない -------------------------
def test_late_finish_from_old_generation_is_orphaned():
    session = _make_session()
    sid = session.create_session()
    _running_round(session, sid, "run-old")

    assert session.reset(sid, restart_sim=False)["reset"] is True
    _running_round(session, sid, "run-new")  # 新roundでrun-new開始

    accepted = session.finish_run(
        sid, "run-old", {"run_id": "run-old", "verdict": "PASS", "err_m": 0.08}
    )
    r = session.rounds[sid]
    assert accepted is False
    assert r.phase == "RUNNING"          # 新runの状態を壊さない
    assert r.run_id == "run-new"
    assert r.result is None              # 旧結果を新roundへ書き戻さない
    assert session.run_lock is True      # 新runのlockを解除しない
    assert r.history and r.history[-1]["orphaned_run_id"] == "run-old"

    # 正しい世代のfinishは受理される
    assert session.finish_run(sid, "run-new", {"verdict": "PASS"}) is True
    assert r.phase == "RESULT" and session.run_lock is False


def test_pause_invalidates_run_generation():
    session = _make_session()
    sid = session.create_session()
    _running_round(session, sid, "run-old")
    r = session.rounds[sid]
    r.run_id = None  # pause経路相当の世代失効
    r.phase = "FAULT"
    session.release_run_lock(sid)
    accepted = session.finish_run(sid, "run-old", {"verdict": "PASS"})
    assert accepted is False and r.phase == "FAULT"


# --- R2: run_lock のcheck-and-setはlock保護 -------------------------------
def _real_plan(session, sid, r):
    """実planをREVIEW状態へ置く（LLMを介さず決定論validatorで構築）。"""
    from kotoba_api.service import PRODUCT_PROFILE, _virtual_world
    from kotoba_contracts.intent import parse_intent
    from kotoba_orchestrator.validator import build_plan

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
        plan_id="p1",
        profile=PRODUCT_PROFILE,
        created_monotonic=time.monotonic(),
    )
    r.plan_obj = plan
    r.pending_plan_id = plan.plan_id


def test_run_lock_check_and_set_is_serialized():
    session = _make_session()
    sid = session.create_session()
    r = session.rounds[sid]
    r.phase = "REVIEW"
    _real_plan(session, sid, r)

    class _Store:
        def verify_and_consume(self, *a, **k):
            time.sleep(0.01)  # 競合窓を広げる
            return type("G", (), {"model_dump": lambda s: {}})()

    session.store = _Store()
    starts = []
    threads = [
        threading.Thread(
            target=lambda: starts.append(
                session.start_run(sid, "p1", "a", "boot").get("started", False)
            )
        )
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert starts.count(True) == 1  # 8並行でも1つだけ起動
    assert starts.count(False) == 7


# --- C6: 凍結seqの再配信は受理しない（runner/live observer共通のClockGate） ---
def test_clock_gate_drops_frozen_and_reordered_seq():
    g = ClockGate()
    assert g.accept(nonce=777, seq=1, sim_t=0.002) == (True, False)
    assert g.accept(nonce=777, seq=2, sim_t=0.004) == (True, False)
    # 凍結したseq=1を受信時刻だけ新しくして再配信 → 新規標本にしない
    for _ in range(500):
        assert g.accept(nonce=777, seq=1, sim_t=0.002) == (False, False)
    assert g.dropped_late == 500


def test_clock_gate_resets_seq_baseline_on_new_boot():
    g = ClockGate()
    g.accept(nonce=777, seq=100, sim_t=0.5)
    # 新boot(nonce変更)ではseq基準をreset — 小さいseqも受理する
    ok, changed = g.accept(nonce=999, seq=3, sim_t=0.01)
    assert (ok, changed) == (True, True)
    assert g.boot_changes == 1
    # 新boot内では単調性を維持
    assert g.accept(nonce=999, seq=3, sim_t=0.01) == (False, False)
    assert g.accept(nonce=999, seq=4, sim_t=0.02) == (True, False)


# --- C7: 書出し時刻で観測をfreshへ戻さない ---------------------------------
def test_live_freshness_uses_observation_wall_not_write_wall():
    now = time.time()
    # 5秒前に受信した観測を今書き直した（wall=now, obs_wall=now-5）
    payload = {"wall": now, "obs_wall": now - 5.0}
    fresh, age = live_freshness(payload, now)
    assert fresh is False and age == pytest.approx(5.0)

    fresh, age = live_freshness({"wall": now - 0.2, "obs_wall": now - 0.2}, now)
    assert fresh is True and age == pytest.approx(0.2)

    # obs_wall欠落（旧形式）は物理受信時刻を証明できない → 明示stale
    fresh, age = live_freshness({"wall": now}, now)
    assert fresh is False and age is None


# ===========================================================================
# レビューと同じ手法: 現行の関数本体をAST抽出し、合成依存で反例を再実行する。
# （excerpt方式 — 修正前コードでは FAIL し、修正後で PASS することが必須）
# ===========================================================================
import ast
import json
import struct
import types
import threading


APP = ROOT / "services" / "orchestrator" / "src" / "kotoba_api" / "app.py"
RUNNER = ROOT / "services" / "runtime" / "kotoba_runner.py"


class _HTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code, self.detail = status_code, detail


def _exec_function(path, name, ns):
    """現行ソースから関数本体を抽出してns内でexec（reviewer手法）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node = ast.fix_missing_locations(
        ast.FunctionDef(
            name=node.name,
            args=node.args,
            body=node.body,
            decorator_list=[],
            returns=node.returns,
        )
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns[name]


def _exec_class(path, cls_name, ns):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == cls_name
    )
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns
    )
    return ns[cls_name]


# --- C2 再実行: SELFTEST=True だけでは実行経路が開かない -------------------
def _start_run_ns(**env):
    launches = []

    class FakeThread:
        def __init__(self, **kw):
            self.kw = kw

        def start(self):
            launches.append(self.kw["args"])

    class FakeRequest:
        def __init__(self, headers):
            self.headers = headers

    ns = {
        "RunIn": types.SimpleNamespace,
        "Request": FakeRequest,
        "HTTPException": _HTTPException,
        "_pending": {"plan-syn": {"session_id": "sess-syn"}},
        "_boot_ready": lambda: (True, "boot-syn"),
        "session": types.SimpleNamespace(
            start_run=lambda *a: {
                "started": True,
                "run_id": "run-syn",
                "grant": {"sim_boot_id": "boot-syn"},
            }
        ),
        "threading": types.SimpleNamespace(Thread=FakeThread),
        "_execute_run": lambda *a: None,
        "_run_display": {},
        "OPERATOR_TOKEN": env.get("token", ""),
        "OFFLINE_MODE": env.get("offline", False),
        "LIVE_ACCEPTED": env.get("live", False),
        "SELFTEST_MODE": env.get("selftest", False),
    }
    fn = _exec_function(APP, "start_run", ns)
    return fn, FakeRequest, launches


def _call(fn, FakeRequest, headers):
    body = types.SimpleNamespace(
        session_id="sess-syn", plan_id="plan-syn", approval_id="appr-syn"
    )
    try:
        return {"response": fn(body, FakeRequest(headers))}
    except _HTTPException as exc:
        return {"status": exc.status_code, "detail": exc.detail}


def test_c2_selftest_alone_no_longer_opens_run_path():
    """レビュー反例C2の再実行: SELFTEST=True+無トークン → 403。"""
    fn, Req, launches = _start_run_ns(
        offline=False, live=False, selftest=True, token="tok-abc"
    )
    r = _call(fn, Req, headers={})
    assert r["status"] == 403 and r["detail"] == "operator_required"
    assert launches == []
    # 誤トークンも拒否
    r = _call(fn, Req, headers={"x-kotoba-operator": "wrong"})
    assert r["status"] == 403 and launches == []


def test_c2b_selftest_token_unset_is_closed():
    """OPERATOR_TOKEN未設定ならSELFTESTでも常に拒否（既定=閉）。"""
    fn, Req, launches = _start_run_ns(
        offline=False, live=False, selftest=True, token=""
    )
    r = _call(fn, Req, headers={"x-kotoba-operator": "anything"})
    assert r["status"] == 403 and launches == []


def test_c2c_authorized_operator_selftest_executes():
    """有効な操作者トークンならselftest実行経路は維持される。"""
    fn, Req, launches = _start_run_ns(
        offline=False, live=False, selftest=True, token="tok-abc"
    )
    r = _call(fn, Req, headers={"x-kotoba-operator": "tok-abc"})
    assert "response" in r and r["response"]["run_id"] == "run-syn"
    assert len(launches) == 1


def test_c2d_live_accepted_participant_path_unchanged():
    """LIVE受入後は通常経路（トークン不要）。"""
    fn, Req, launches = _start_run_ns(
        offline=False, live=True, selftest=False
    )
    r = _call(fn, Req, headers={})
    assert "response" in r and len(launches) == 1


# --- C6 再実行: 現行runner Observerに凍結streamを送る -----------------------
def test_c6_frozen_clock_stream_does_not_rebind():
    """凍結seq/sim_tの再配信はboundを更新しない → latestはstaleを上げる。"""

    class FakeClock:
        def __init__(self):
            self.t = 100.0

        def monotonic(self):
            return self.t

    clock = FakeClock()
    ns = {
        "lcm": None,
        "struct": struct,
        "math": __import__("math"),
        "time": types.SimpleNamespace(monotonic=clock.monotonic),
        "threading": threading,
        "ClockGate": ClockGate,
        "CLOCK_FP": 0x4B544F4241434C31,
        "HarnessError": type("HarnessError", (Exception,), {}),
    }
    Observer = _exec_class(RUNNER, "Observer", ns)
    obs = Observer.__new__(Observer)
    obs._lock = threading.Lock()
    obs._state = obs._clock = obs.bound = None
    obs.nonce = None
    obs._gate = ClockGate()
    obs._exception = None
    obs._expected_nonce = None
    obs._dropped_boot = 0

    state = struct.pack(">qdi", 0x2D53D9E29374E48E, 0.0, 0)
    state += struct.pack(">3d3d4d", 0.0, 0.0, 0.82, 0.0, 0.0, 0.0, 1.0, 0, 0, 0)

    # 先に有効な1標本（seq=1）を結合させる
    obs._on_clock("kotoba_sim_clock", struct.pack(">qqqdd", ns["CLOCK_FP"], 777, 1, 0.0, clock.t))
    obs._on_state("sim_state", state)
    first = obs.latest()
    assert first[4] == 1

    # seq=1/sim_t=0 を受信時刻だけ進めて501回再配信（C6反例と同じ入力）
    for i in range(501):
        clock.t = 100.0 + i * 0.01
        obs._on_state("sim_state", state)
        obs._on_clock(
            "kotoba_sim_clock",
            struct.pack(">qqqdd", ns["CLOCK_FP"], 777, 1, 0.0, clock.t),
        )
        if i < 40:  # 直近はboundの経過時間内 — 受理自体はしないがまだfresh
            pass
    # boundはseq=1標本のまま古くなる → latest() は stale を上げる
    with pytest.raises(Exception, match="stale_observation"):
        obs.latest()
    assert obs._gate.dropped_late == 501
    # 結合済み最終値は凍結seqのまま進んでいない（受信があっても現在扱いしない）
    clock.t = 100.0 + 501 * 0.01 - 0.1
    b = obs.bound
    assert b[5] == 1 and b[4] == 0.0


def test_c6b_boot_change_clears_bound_until_new_pair():
    """runnerはboot固定: 別bootパケットは受理せず、boundはstale化する。"""
    clock = type("C", (), {})()
    clock.t = 200.0
    ns = {
        "lcm": None,
        "struct": struct,
        "math": __import__("math"),
        "time": types.SimpleNamespace(monotonic=lambda: clock.t),
        "threading": threading,
        "ClockGate": ClockGate,
        "CLOCK_FP": 0x4B544F4241434C31,
        "HarnessError": type("HarnessError", (Exception,), {}),
    }
    Observer = _exec_class(RUNNER, "Observer", ns)
    obs = Observer.__new__(Observer)
    obs._lock = threading.Lock()
    obs._state = obs._clock = obs.bound = None
    obs.nonce = None
    obs._gate = ClockGate()
    obs._exception = None
    obs._expected_nonce = 777  # 承認されたboot
    obs._dropped_boot = 0

    state = struct.pack(">qdi", 0x2D53D9E29374E48E, 0.0, 0)
    state += struct.pack(">3d3d4d", 0.0, 0.0, 0.82, 0.0, 0.0, 0.0, 1.0, 0, 0, 0)
    obs._on_clock("kotoba_sim_clock", struct.pack(">qqqdd", ns["CLOCK_FP"], 777, 100, 5.0, clock.t))
    obs._on_state("sim_state", state)
    assert obs.latest()[4] == 100
    # 別boot(nonce=999)のパケットはrunnerが受理しない — boundは更新されずstaleへ
    obs._on_clock("kotoba_sim_clock", struct.pack(">qqqdd", ns["CLOCK_FP"], 999, 5, 0.1, clock.t))
    assert obs._dropped_boot == 1
    clock.t = 201.0  # boundの受信時刻が古くなる
    with pytest.raises(Exception, match="stale_observation"):
        obs.latest()
    # 承認bootの有効対は引き続き受理（陽性対照）
    clock.t = 202.0
    obs._on_state("sim_state", state)
    obs._on_clock("kotoba_sim_clock", struct.pack(">qqqdd", ns["CLOCK_FP"], 777, 101, 5.1, clock.t))
    got = obs.latest()
    assert got[4] == 101 and got[5] == 777


# --- C7 再実行: 現行 _read_latest_state_file に古い観測ファイルを与える -------
def test_c7_old_observation_not_marked_fresh(tmp_path):
    """5秒前に受信した観測を今書き込んだファイル → fresh=False のまま。"""
    now = 1700000000.0
    live = tmp_path / "live.json"
    live.write_text(
        json.dumps(
            {
                "wall": now,  # 今書き込んだ
                "obs_wall": now - 5.0,  # 観測自体は5秒前
                "pos": [0, 0, 0.82],
            }
        )
    )
    ns = {
        "json": json,
        "time": types.SimpleNamespace(time=lambda: now),
        "RUNTIME_DIR": tmp_path,
        "Path": Path,
        "live_freshness": live_freshness,
        "_active_target": lambda: None,
    }
    fn = _exec_function(APP, "_read_latest_state_file", ns)
    data = fn()
    assert data["fresh"] is False
    assert data["age_s"] == 5.0
    assert data["file_age_s"] == 0.0

    # 新鮮な観測は fresh=True（陽性対照）
    live.write_text(json.dumps({"wall": now, "obs_wall": now - 0.1, "pos": [0, 0, 0.82]}))
    data = fn()
    assert data["fresh"] is True and data["age_s"] == 0.1


# --- C4 再実行: 現行 finish_run に旧世代の結果を送る ------------------------
def test_c4_excerpt_level_stale_finish_does_not_overwrite(tmp_path):
    """現行 finish_run 本体へ reset後の旧run結果を送っても新roundを壊さない。"""
    session = _make_session()
    sid = session.create_session()
    _running_round(session, sid, "run-old")
    session.reset(sid, restart_sim=False)
    _running_round(session, sid, "run-new")
    # service.pyの現行finish_runをAST抽出しても同じ契約であることを確認
    src = ROOT / "services" / "orchestrator" / "src" / "kotoba_api" / "service.py"
    fn = _exec_function(src, "finish_run", {})
    accepted = fn(session, sid, "run-old", {"run_id": "run-old", "verdict": "PASS"})
    r = session.rounds[sid]
    assert accepted is False
    assert r.phase == "RUNNING" and r.run_id == "run-new"
    assert r.result is None and session.run_lock is True
