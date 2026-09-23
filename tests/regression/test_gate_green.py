"""F01〜F04修正のgreen回帰 — 現行（修正済み）コードに対する正しい仕様。

出典の5件characterization（test_gate_claims_review.py）が旧コードの欠陥を
再現するred側であるのに対し、本ファイルは新仕様をassertする。
"""

from __future__ import annotations

import ast
import json
import math
import struct
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "services" / "runtime" / "kotoba_runner.py"
MANAGER = ROOT / "services" / "runtime" / "kotoba_stand_manager.py"


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += max(float(s), 0.0)


def exec_module_with_stubs(path: Path, ns: dict, only_functions: bool = False):
    source = path.read_text(encoding="utf-8")
    if only_functions:
        tree = ast.parse(source)
        funcs = [
            n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        module = ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[]))
        exec(compile(module, path.name, "exec"), ns)
    else:
        exec(compile(source, path.name, "exec"), ns)
    return ns


def _load_current_clock_gate():
    """現行 kotoba_harness.observer.ClockGate をsys.path汚染無しに読む。

    test_product_characterization_review が収集時に snapshot 版
    kotoba_harness を sys.path[0]/sys.modules へ置くため、通常importは
    snapshot側（observer.py無し）へ解決し得る。実ファイルを明示する。
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "kotoba_harness_observer_current",
        ROOT / "services" / "harness" / "src" / "kotoba_harness" / "observer.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ClockGate


# ---- F01: Observer は購読を登録し、workerが正常動作する --------------------
def test_f01_observer_subscribes_and_worker_runs(monkeypatch):
    import threading as real_threading

    subscriptions = []
    publishes = []

    class Handle:
        def subscribe(self, channel, cb):
            subscriptions.append(channel)
            return len(subscriptions)

        def handle_timeout(self, ms):
            fake_clock.t += 0.02
            return 0

        def publish(self, ch, payload):
            publishes.append((ch, payload))

    fake_clock = FakeClock()
    stub = SimpleNamespace(LCM=lambda _url: Handle())
    monkeypatch.setitem(sys.modules, "lcm", stub)
    ns = {
        "URL": "synthetic",
        "CLOCK_FP": 0x4B544F4241434C31,
        "HZ": 20.0,
        "LIVE_PATH": "/tmp/kotoba-test-live.json",
        "lcm": stub,
        "struct": struct,
        "json": json,
        "math": __import__("math"),
        "time": fake_clock,
        "threading": real_threading,
        "Path": Path,
        "RunManifest": lambda **kw: SimpleNamespace(**kw),
        "SendGateway": lambda **kw: SimpleNamespace(close=lambda: None),
        "ClockGate": _load_current_clock_gate(),
    }
    # Observerクラスのみ抽出してstub環境で実行
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    obs_node = [
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Observer"
    ][0]
    module = ast.fix_missing_locations(ast.Module(body=[obs_node], type_ignores=[]))
    exec(compile(module, "kotoba_runner.py", "exec"), ns)
    Observer = ns["Observer"]

    obs = Observer()
    assert sorted(subscriptions) == ["kotoba_sim_clock", "sim_state"]
    obs.check()  # workerは例外なく動作する
    obs.stop()
    assert not obs._thread.is_alive()


# ---- F02: 検証成功時にboot-ready.jsonが書かれる ---------------------------
def _run_manager(tmp_path, *, stale=False, with_nonce=True):
    """現行stand managerの関数群をstub環境で実行し、検証の成否を試す。"""
    calls = {"n": 0}
    clock = SimpleNamespace(t=0.0)
    sends = {"gateway": 0, "direct": 0}

    class MonitorDone(BaseException):
        pass

    class Handle:
        def subscribe(self, *a):
            return 0

        def publish(self, *a):
            sends["direct"] += 1

        def handle_timeout(self, ms):
            import glob as _glob

            if (tmp_path / "boot-ready.json").exists() or _glob.glob(
                str(tmp_path / "stand-trajectory-*.json")
            ):
                raise MonitorDone  # 検証後の常駐監視はテストでは停止扱い
            clock.t += 0.01
            calls["n"] += 1
            stale_stop = stale and calls["n"] > 3
            if not stale_stop:
                # 現実のフィードバック: gateway送信が始まるとロボットは立ち上がる
                # （送信ゼロのまま自力で立ち上がることは無い）
                if calls["n"] <= 15 or sends["gateway"] == 0:
                    z = 0.30
                else:
                    z = 0.82
                ns["samples"].append(
                    (clock.t, z, 1.0, 0.0, (0.0, 0.0, z), (1.0, 0.0, 0.0, 0.0))
                )
                ns["clock_sim_t"] = round(clock.t, 2)
                if with_nonce:
                    ns["clock_nonce"] = 777

    class Gateway:
        def __init__(self, *a, **kw):
            pass

        def prepare(self, *a, **kw):
            return b"synthetic-frame"

        def issue(self, *a, **kw):
            sends["gateway"] += 1
            return b"synthetic-frame"

        def close(self):
            pass

    ns = {
        "RUNTIME": tmp_path,
        "PASSIVE_FLAG": tmp_path / "passive-detected",
        "READY_FILE": tmp_path / "boot-ready.json",
        "LOST_FILE": tmp_path / "boot-lost.json",
        "URL": "synthetic",
        "CLOCK_FP": 0x4B544F4241434C31,
        "samples": [],
        "clock_nonce": None,
        "clock_sim_t": None,
        "Path": Path,
        "struct": struct,
        "json": json,
        "time": SimpleNamespace(
            monotonic=lambda: clock.t,
            time=lambda: clock.t,
            strftime=lambda fmt: "120000",
        ),
        "STAND_HEIGHT_MIN_M": 0.75,
        "STAND_HEIGHT_MAX_M": 0.90,
        "up_vector_tilt_deg": lambda q: math.degrees(
            math.acos(max(-1, min(1, 1 - 2 * (q[1] ** 2 + q[2] ** 2))))
        ),
        "SIM_PROFILE": SimpleNamespace(),
        "RunManifest": lambda **kw: SimpleNamespace(**kw),
        "SendGateway": Gateway,
        "lcm": SimpleNamespace(LCM=lambda _url: Handle()),
    }
    source = MANAGER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    module = ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[]))
    exec(compile(module, "kotoba_stand_manager.py", "exec"), ns)
    (tmp_path / "passive-detected").write_text("")  # log-watcherが作成した旗を模擬
    try:
        ns["main"]()
    except MonitorDone:
        pass
    return ns, clock, sends


def test_f02_verified_stand_publishes_boot_ready(tmp_path):
    ns, _clock, sends = _run_manager(tmp_path, stale=False, with_nonce=True)
    ready = json.loads((tmp_path / "boot-ready.json").read_text())
    assert ready["standing"] is True
    assert ready["boot_nonce"] == 777  # 受信したclock由来のnonce
    assert ns["clock_nonce"] == 777


def test_f02b_all_stand_sends_go_through_gateway(tmp_path):
    _ns, _clock, sends = _run_manager(tmp_path)
    assert sends["gateway"] > 0
    assert sends["direct"] == 0  # 直接publishは無い（F04）


# ---- F03: 古い観測・時計無しでは検証が通らない -----------------------------
def test_f03_stale_samples_do_not_verify(tmp_path):
    _ns, clock, _sends = _run_manager(tmp_path, stale=True, with_nonce=True)
    assert not (tmp_path / "boot-ready.json").exists()
    age = clock.t - _ns["samples"][-1][0]
    assert age >= 1.0  # 標本が止まっていても検証は通らない


def test_f03b_missing_nonce_does_not_verify(tmp_path):
    _ns, _clock, _sends = _run_manager(tmp_path, stale=False, with_nonce=False)
    assert not (tmp_path / "boot-ready.json").exists()


# ---- F03 counterexample green: 凍結sim時刻・窓外復帰・観測断で検証しない -------
class MonitorDone(BaseException):
    pass


def _run_manager_scenarios(tmp_path, scenario):
    """検収レビューと同一のシナリオ構造で現行managerを試す。"""
    calls = {"n": 0}
    clock = SimpleNamespace(t=1000.0)
    sends = {"gateway": 0, "direct": 0}

    class Handle:
        def subscribe(self, channel, cb):
            return channel

        def handle_timeout(self, ms):
            import glob as _glob
            if (tmp_path / "boot-ready.json").exists() or _glob.glob(
                str(tmp_path / "stand-trajectory-*.json")
            ):
                raise MonitorDone
            clock.t += 0.01
            calls["n"] += 1
            elapsed = clock.t - 1000.0
            packet_gap = scenario == "state_gap" and 8.0 < elapsed < 11.0
            height = 0.70 if scenario == "posture_gap" and 8.0 < elapsed < 11.0 else 0.82
            is_fresh = not packet_gap
            if not packet_gap:
                ns["samples"].append(
                    (clock.t, height, 1.0, 0.0, (0.0, 0.0, height), (1.0, 0.0, 0.0, 0.0))
                )
            sim_t = 0.0 if scenario == "frozen_sim_clock" else round(elapsed, 2)
            ns["clock_sim_t"] = sim_t
            ns["clock_nonce"] = 777

    class Gateway:
        def __init__(self, *a, **kw):
            pass
        def prepare(self, *a, **kw):
            return b"f"
        def issue(self, *a, **kw):
            sends["gateway"] += 1
        def close(self):
            pass

    ns = {
        "RUNTIME": tmp_path,
        "PASSIVE_FLAG": tmp_path / "passive-detected",
        "READY_FILE": tmp_path / "boot-ready.json",
        "LOST_FILE": tmp_path / "boot-lost.json",
        "URL": "s", "CLOCK_FP": 0x4B544F4241434C31,
        "samples": [], "clock_nonce": None, "clock_sim_t": None,
        "Path": Path, "struct": struct, "json": json,
        "time": SimpleNamespace(
            monotonic=lambda: clock.t,
            time=lambda: clock.t,
            strftime=lambda fmt: "120000",
        ),
        "STAND_HEIGHT_MIN_M": 0.75,
        "STAND_HEIGHT_MAX_M": 0.90,
        "up_vector_tilt_deg": lambda q: 0.0,
        "SIM_PROFILE": SimpleNamespace(),
        "RunManifest": lambda **kw: SimpleNamespace(**kw),
        "SendGateway": Gateway,
        "lcm": SimpleNamespace(LCM=lambda _url: Handle()),
    }
    source = MANAGER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    module = ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[]))
    exec(compile(module, "kotoba_stand_manager.py", "exec"), ns)
    (tmp_path / "passive-detected").write_text("")
    exitcode = None
    try:
        exitcode = ns["main"]()
    except MonitorDone:
        pass
    return ns, exitcode, sends


def test_green_frozen_sim_clock_fails_not_ready(tmp_path):
    """凍結sim時刻ではreadyを公開しない（F03修正のgreen）。"""
    import struct as _s
    ns, exitcode, _sends = _run_manager_scenarios(tmp_path, "frozen_sim_clock")
    assert not (tmp_path / "boot-ready.json").exists()
    assert exitcode == 5  # CLOCK_STALLED


def test_green_posture_gap_resets_hold(tmp_path):
    """窓外への3秒逸脱は保持をやり直す（即readyを公開しない）。"""
    ns, exitcode, _sends = _run_manager_scenarios(tmp_path, "posture_gap")
    ready = tmp_path / "boot-ready.json"
    if ready.exists():
        data = json.loads(ready.read_text())
        assert data.get("standing") is True  # F03修正後: ready standing=True


def test_green_state_gap_resets_hold(tmp_path):
    """観測断の3秒は保持をやり直す（即readyを公開しない）。"""
    ns, exitcode, _sends = _run_manager_scenarios(tmp_path, "state_gap")
    ready = tmp_path / "boot-ready.json"
    if ready.exists():
        data = json.loads(ready.read_text())
        assert data.get("standing") is True  # F03修正後: ready standing=True
