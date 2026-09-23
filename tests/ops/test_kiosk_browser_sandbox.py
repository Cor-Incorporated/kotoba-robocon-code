"""Fail-closed decision tests for the native Firefox sandbox gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "ops/thor/deploy/kiosk-browser-sandbox-check.py"
SPEC = importlib.util.spec_from_file_location("kiosk_browser_sandbox_check", SCRIPT)
assert SPEC and SPEC.loader
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


class SandboxDecisionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.host = "user:[1]"
        self.parent = {
            "pid": 10,
            "ppid": 1,
            "leader": True,
            "profile": "firefox (unconfined)",
            "userns": self.host,
            "content": False,
            "forkserver": False,
            "seccomp": 0,
            "filters": 0,
            "no_new_privs": 0,
        }
        self.renderer = {
            "pid": 12,
            "ppid": 11,
            "leader": False,
            "profile": "firefox (unconfined)",
            "userns": "user:[2]",
            "content": True,
            "forkserver": False,
            "seccomp": 2,
            "filters": 1,
            "no_new_privs": 1,
        }
        self.fork_server = {
            **self.parent,
            "pid": 11,
            "ppid": 10,
            "leader": False,
            "forkserver": True,
        }

    def test_accepts_named_profile_and_isolated_web_content(self) -> None:
        self.assertTrue(GATE.sandbox_result(
            [self.parent, self.fork_server, self.renderer], self.host
        )[0])

    def test_refuses_missing_profile_or_web_content(self) -> None:
        self.assertFalse(GATE.sandbox_result([self.renderer], self.host)[0])
        self.assertFalse(GATE.sandbox_result([self.parent], self.host)[0])
        bad_parent = {**self.parent, "profile": "unconfined"}
        self.assertFalse(GATE.sandbox_result(
            [bad_parent, self.fork_server, self.renderer], self.host
        )[0])
        bystander = {**self.parent, "leader": False}
        self.assertFalse(GATE.sandbox_result(
            [bystander, self.fork_server, self.renderer], self.host
        )[0])

    def test_refuses_web_content_outside_own_forkserver_tree(self) -> None:
        orphan = {**self.renderer, "ppid": 99}
        success, reason = GATE.sandbox_result(
            [self.parent, self.fork_server, orphan], self.host
        )
        self.assertFalse(success)
        self.assertIn("forkserver", reason)

    def test_refuses_any_unisolated_web_renderer(self) -> None:
        for changed in (
            {"userns": self.host},
            {"seccomp": 0},
            {"filters": 0},
            {"no_new_privs": 0},
            {"profile": "unconfined"},
        ):
            with self.subTest(changed=changed):
                unsafe = {**self.renderer, **changed}
                self.assertFalse(
                    GATE.sandbox_result(
                        [self.parent, self.fork_server, self.renderer,
                         {**unsafe, "pid": 13}], self.host
                    )[0]
                )

    @unittest.skipUnless(sys.platform == "linux" and shutil.which("cc"), "Linux /proc + C compiler")
    def test_gui_parent_identity_binds_profile_and_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sleeper.c"
            browser = Path(directory) / "firefox"
            source.write_text("#include <unistd.h>\nint main(void) { for (;;) pause(); }\n")
            subprocess.run(["cc", str(source), "-o", str(browser)], check=True)
            profile = Path(directory) / "profile"
            profile.mkdir()
            entry = "http://127.0.0.1:8780/"
            proc = subprocess.Popen(
                [str(browser), "--new-instance", "--profile", str(profile), "--kiosk", entry]
            )
            try:
                for _ in range(100):
                    if GATE.kiosk_parent_matches(proc.pid, browser, profile, entry):
                        break
                    time.sleep(0.01)
                self.assertTrue(GATE.kiosk_parent_matches(proc.pid, browser, profile, entry))
                self.assertFalse(GATE.kiosk_parent_matches(
                    proc.pid, browser, profile, "http://127.0.0.1:9999/"
                ))
                self.assertFalse(GATE.kiosk_parent_matches(
                    proc.pid, browser, Path(directory) / "other-profile", entry
                ))
            finally:
                proc.terminate()
                proc.wait(timeout=2)


if __name__ == "__main__":
    unittest.main()
