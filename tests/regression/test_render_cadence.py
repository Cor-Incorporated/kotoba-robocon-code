"""描画周期最適化がboot境界と観測鮮度を壊さないための軽量回帰試験。

renderer本体はThor専用のMuJoCo/Viserをimportするため、姿勢識別関数だけを
ASTから実行する。live observerはLCMモジュールを隔離して実ソースを読む。
"""

import ast
import importlib.util
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def _pose_signature():
    source = ROOT / "ops/thor/render/pm01_render_server.py"
    tree = ast.parse(source.read_text())
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_pose_signature"
    )
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_pose_signature"]


def _live_module():
    source = ROOT / "services/runtime/kotoba_live_observer.py"
    harness = str(ROOT / "services/harness/src")
    sys.path.insert(0, harness)
    try:
        with patch.dict(sys.modules, {"lcm": types.ModuleType("lcm")}):
            spec = importlib.util.spec_from_file_location("live_cadence_probe", source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(harness)


class RenderCadenceTest(unittest.TestCase):
    def test_static_pose_is_same_but_new_boot_or_motion_reapplies(self):
        signature = _pose_signature()
        pose = {"boot_nonce": 11, "pos": [0, 0, 0.82],
                "quat_wxyz": [1, 0, 0, 0]}
        joints = [0.0] * 24
        first = signature(pose, joints)
        self.assertEqual(signature(dict(pose), list(joints)), first)
        self.assertNotEqual(signature({**pose, "boot_nonce": 12}, joints), first)
        moved = list(joints)
        moved[5] = 0.01
        self.assertNotEqual(signature(pose, moved), first)
        self.assertNotEqual(
            signature({**pose, "pos": [0.01, 0, 0.82]}, joints), first
        )

    def test_rewriting_snapshot_does_not_refresh_source_time(self):
        module = _live_module()
        observer = object.__new__(module.LiveObserver)
        observer._lock = threading.Lock()
        source_time = time.time() - 5
        observer._bound = (time.monotonic(), source_time, (0, 0, 0.82),
                           (0, 0, 0), (1, 0, 0, 0), [0.0] * 24,
                           1.0, 100, 11, None)
        observer._recv_total = 100
        observer._gate = module.ClockGate()
        first = observer.snapshot()
        time.sleep(0.02)
        second = observer.snapshot()
        self.assertGreater(second["wall"], first["wall"])
        self.assertEqual(second["obs_wall"], source_time)
        self.assertGreaterEqual(second["obs_age_s"], first["obs_age_s"])

    def test_clock_gate_rejects_frozen_and_retired_boot(self):
        gate = _live_module().ClockGate()
        self.assertEqual(gate.accept(11, 100, 1.0), (True, False))
        self.assertEqual(gate.accept(11, 101, 1.0), (False, False))
        self.assertEqual(gate.accept(12, 1, 0.0), (True, True))
        self.assertEqual(gate.accept(11, 102, 2.0), (False, False))


if __name__ == "__main__":
    unittest.main()
