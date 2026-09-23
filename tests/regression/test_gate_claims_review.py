"""Characterization, not product acceptance. No live LCM, signals, SSH, GPU or SDK.

The stand-manager source is full-file Git-blob verified. Observer is the exact
class excerpt retrieved from the same fixed commit; it is not a full-file blob.
"""

from __future__ import annotations
import ast
import contextlib
import hashlib
import json
import math
from pathlib import Path
import struct
import threading
from types import SimpleNamespace
import pytest

# RED面: PR8ゲートレビューによる5件の欠陥characterization（PASS=欠陥の再現）
# 出典: docs/external-review/pr8-2026-09-15/ （blob 032e45a4... 検証込み）
ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "docs" / "external-review" / "pr8-2026-09-15" / "snapshot"
RESULTS = []


@pytest.fixture(scope="session", autouse=True)
def persist_results():
    yield
    out_dir = ROOT / "tests" / "evidence"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "pr8-probe-results.json").write_text(
        json.dumps(RESULTS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def observer_class():
    class Handle:
        def __init__(self):
            self.subscriptions = []
            self.receives = 0

        def subscribe(self, *args):
            self.subscriptions.append(args)

        def handle_timeout(self, timeout):
            self.receives += 1

    handle = Handle()

    class Thread:
        def __init__(self, target, daemon):
            self.target = target

        def start(self):
            pass  # No real thread: call target synchronously in tests.

    ns = {
        "lcm": SimpleNamespace(LCM=lambda _: handle),
        "threading": SimpleNamespace(Thread=Thread, Lock=threading.Lock),
        "URL": "synthetic-no-network",
        "CLOCK_FP": 0x4B544F4241434C31,
        "time": SimpleNamespace(monotonic=lambda: 0.0),
        "struct": struct,
        "HarnessError": RuntimeError,
    }
    code = compile(
        (SNAP / "observer_excerpt.py").read_text(), "observer_excerpt.py", "exec"
    )
    exec(code, ns)
    return ns["Observer"], handle


def test_01_worker_target_raises_attribute_error():
    cls, handle = observer_class()
    obj = cls()
    with pytest.raises(AttributeError, match="handle_timeout") as exc:
        obj._thread.target()
    assert handle.receives == 0
    RESULTS.append(
        {
            "id": "P01",
            "result": str(exc.value),
            "real_lcm_receives": 0,
            "meaning": "Actual retrieved worker method fails before reading.",
        }
    )


def test_02_constructor_registers_no_channels():
    cls, handle = observer_class()
    cls()
    assert handle.subscriptions == []
    RESULTS.append(
        {
            "id": "P02",
            "subscriptions": 0,
            "meaning": "Callbacks exist but constructor never registers them.",
        }
    )


class MonitorEntered(BaseException):
    """Stop the otherwise infinite stand monitor after internal verification."""


def run_stand(tmp_path, stale=False):
    source = (SNAP / "kotoba_stand_manager.py").read_bytes()
    blob = hashlib.sha1(
        b"blob " + str(len(source)).encode() + b"\0" + source
    ).hexdigest()
    assert blob == "032e45a4bc9243f3a3b140017ea2a4bd984d270b"
    tree = ast.parse(source)
    # Execute original function bodies only, with infrastructure substituted.
    funcs = [
        n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    module = ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[]))
    clock = SimpleNamespace(t=0.0)
    ns = {
        "RUNTIME": tmp_path,
        "PASSIVE_FLAG": tmp_path / "passive-detected",
        "READY_FILE": tmp_path / "boot-ready.json",
        "LOST_FILE": tmp_path / "boot-lost.json",
        "URL": "synthetic-no-network",
        "samples": [],
        "clock_nonce": None,
        "CLOCK_FP": 0x4B544F4241434C31,
        "Path": Path,
        "struct": struct,
        "json": json,
        "time": SimpleNamespace(monotonic=lambda: clock.t, time=lambda: clock.t),
        "STAND_HEIGHT_MIN_M": 0.75,
        "STAND_HEIGHT_MAX_M": 0.90,
        "up_vector_tilt_deg": lambda q: math.degrees(
            math.acos(max(-1, min(1, 1 - 2 * (q[1] ** 2 + q[2] ** 2))))
        ),
        "SIM_PROFILE": SimpleNamespace(),
        "RunManifest": lambda **kw: SimpleNamespace(**kw),
    }
    sends = []
    calls = {"n": 0}

    class Handle:
        def subscribe(self, *args):
            pass

        def publish(self, *args):
            sends.append(("direct", clock.t))

        def handle_timeout(self, timeout):
            if (tmp_path / "stand-trajectory.json").exists():
                raise MonitorEntered
            clock.t += 0.01
            calls["n"] += 1
            if stale and calls["n"] > 3:
                return
            z = 0.82 if calls["n"] != 2 else 0.79
            ns["samples"].append(
                (clock.t, z, 1.0, 0.0, (0.0, 0.0, z), (1.0, 0.0, 0.0, 0.0))
            )
            if not stale:
                ns["clock_nonce"] = 777

    h = Handle()

    class Gateway:
        def __init__(self, *args, **kw):
            pass

        def send(self, *args, **kw):
            sends.append(("gateway", clock.t))
            return b"synthetic-frame"

    ns["lcm"] = SimpleNamespace(LCM=lambda _: h)
    ns["SendGateway"] = Gateway
    exec(compile(module, "kotoba_stand_manager.py", "exec"), ns)
    with pytest.raises(MonitorEntered):
        ns["main"]()
    return ns, clock, sends


def test_03_verified_stand_never_publishes_boot_ready(tmp_path):
    ns, clock, sends = run_stand(tmp_path)
    assert (tmp_path / "stand-trajectory.json").exists()
    assert not (tmp_path / "boot-ready.json").exists()
    RESULTS.append(
        {
            "id": "P03",
            "internal_monitor_entered": True,
            "boot_ready_written": False,
            "clock_nonce_received": ns["clock_nonce"],
            "meaning": "Even synthesized successful stand does not notify API.",
        }
    )


def test_04_stale_sample_and_no_clock_still_pass_internal_verification(tmp_path):
    ns, clock, sends = run_stand(tmp_path, stale=True)
    age = clock.t - ns["samples"][-1][0]
    assert age >= 5.0 and len(ns["samples"]) == 3 and ns["clock_nonce"] is None
    RESULTS.append(
        {
            "id": "P04",
            "internal_monitor_entered": True,
            "last_sample_age_s": round(age, 3),
            "samples": len(ns["samples"]),
            "clock_nonce": None,
            "meaning": "Clock and fresh continuous states are not required by this manager.",
        }
    )


def test_05_passive_flag_is_never_read():
    tree = ast.parse((SNAP / "kotoba_stand_manager.py").read_text())
    reads = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Name)
        and n.id == "PASSIVE_FLAG"
        and isinstance(n.ctx, ast.Load)
    ]
    assert reads == []
    RESULTS.append(
        {
            "id": "P05",
            "passive_flag_reads": 0,
            "meaning": "Actual trigger is height crossing, not declared passive-entry event.",
        }
    )
