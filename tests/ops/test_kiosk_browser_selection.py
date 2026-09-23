"""Kiosk Firefox selector: shell syntax and isolated launcher/rollback probes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "ops" / "thor" / "deploy"
SHA = "7dd9425eafa0decf61c0f6bc56dc71cba84595495dc01395d3eea38a18aaf710"


class KioskBrowserSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name) / "home"
        self.bin_dir = Path(self.tmp.name) / "bin"
        self.home.mkdir()
        self.bin_dir.mkdir()
        self.log = Path(self.tmp.name) / "browser.log"
        self.aa_log = Path(self.tmp.name) / "aa.log"
        self.gate_log = Path(self.tmp.name) / "gate.log"
        self.deploy = Path(self.tmp.name) / "deploy"
        self.deploy.mkdir()
        # Keep fixed production paths. Substitute only in this isolated copy.
        for name in (
            "kiosk-browser-common.sh",
            "kiosk-browser-select.sh",
            "kiosk-browser-rollback.sh",
            "kiosk-launch.sh",
        ):
            source = (DEPLOY / name).read_text()
            if name == "kiosk-launch.sh":
                self.assertEqual(source.count("/snap/bin/firefox"), 2)
                source = source.replace("/snap/bin/firefox", str(self.bin_dir / "firefox"))
            if name == "kiosk-browser-common.sh":
                self.assertEqual(source.count("/usr/bin/aa-exec"), 1)
                source = source.replace("/usr/bin/aa-exec", str(self.bin_dir / "aa-exec"))
            (self.deploy / name).write_text(source)
        shutil.copy2(DEPLOY / "kiosk-browser-policies.json", self.deploy)
        (self.deploy / "kiosk-browser-sandbox-check.py").write_text(
            "import os, subprocess, sys\n"
            "from pathlib import Path\n"
            "launch = len(sys.argv) == 5 and sys.argv[1] == '--launch'\n"
            "if launch:\n"
            " if not Path(sys.argv[3]).is_dir(): sys.exit(3)\n"
            "else:\n"
            " if len(sys.argv) != 3 or not Path(sys.argv[2]).is_dir(): sys.exit(3)\n"
            "with Path(os.environ['KOTOBA_BROWSER_TEST_GATE_LOG']).open('a') as log: "
            "log.write('launched\\n' if launch else 'checked\\n')\n"
            "failure = int(os.environ.get('KOTOBA_BROWSER_TEST_SANDBOX_EXIT', '0'))\n"
            "if failure: sys.exit(failure)\n"
            "if launch: sys.exit(subprocess.run([os.environ['KOTOBA_BROWSER_TEST_AA_EXEC'], "
            "'-p', 'firefox', '--', sys.argv[2], '--new-instance', '--profile', "
            "sys.argv[3], '--kiosk', sys.argv[4]]).returncode)\n"
        )
        for name, body in {
            "pgrep": "exit 1",
            "curl": "exit 0",
            "sleep": "exit 0",
            "sudo": "exit 0",
            "xrandr": "exit 0",
            "wmctrl": "exit 0",
            "id": 'case "$1" in -un) echo cloudia;; -u) echo "$KOTOBA_BROWSER_TEST_UID";; esac',
            "getent": 'echo "cloudia:x:$KOTOBA_BROWSER_TEST_UID:$KOTOBA_BROWSER_TEST_UID::${HOME}:/bin/bash"',
            "loginctl": "exit 1",
            "firefox": 'printf "snap:%s\\n" "$*" >> "$KOTOBA_BROWSER_TEST_LOG"',
            "aa-exec": 'printf "aa:%s\\n" "$*" >> "$KOTOBA_BROWSER_TEST_AA_LOG"\nshift 3\nexec "$@"',
        }.items():
            self._script(self.bin_dir / name, body)
        self.env = os.environ.copy()
        self.env.update(
            HOME=str(self.home),
            PATH=f"{self.bin_dir}:{os.environ['PATH']}",
            KOTOBA_BROWSER_TEST_LOG=str(self.log),
            KOTOBA_BROWSER_TEST_AA_LOG=str(self.aa_log),
            KOTOBA_BROWSER_TEST_GATE_LOG=str(self.gate_log),
            KOTOBA_BROWSER_TEST_UID=str(os.getuid()),
            KOTOBA_BROWSER_TEST_AA_EXEC=str(self.bin_dir / "aa-exec"),
        )

    @staticmethod
    def _script(path: Path, body: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)

    def _run(self, name: str, *args: str, timeout: int = 5) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.deploy / name), *args],
            env=self.env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )

    def _choice(self, value: str) -> None:
        path = self.home / ".config/kotoba/kiosk-browser"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value + "\n")

    def _install_mock_native(self, version: str = "156.0") -> Path:
        root = self.home / ".local/opt/kotoba-firefox/156.0"
        native = root / "firefox/firefox"
        self._script(
            native,
            f'''if [ "${{1:-}}" = --version ]; then printf "Mozilla Firefox {version}\\n"; exit 0; fi
printf "native:%s\\n" "$*" >> "$KOTOBA_BROWSER_TEST_LOG"''',
        )
        (root / "VERIFIED_SHA256").write_text(SHA + "\n")
        policy = root / "firefox/distribution/policies.json"
        policy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DEPLOY / "kiosk-browser-policies.json", policy)
        return root

    def test_shell_syntax(self) -> None:
        for name in (
            "kiosk-browser-common.sh",
            "kiosk-browser-select.sh",
            "kiosk-browser-rollback.sh",
            "kiosk-launch.sh",
        ):
            with self.subTest(name=name):
                result = subprocess.run(["bash", "-n", str(DEPLOY / name)], capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_managed_policy_does_not_override_first_run_or_accept_terms(self) -> None:
        policy = json.loads((DEPLOY / "kiosk-browser-policies.json").read_text())
        self.assertEqual(
            policy,
            {"policies": {
                "DisableAppUpdate": True,
                "DontCheckDefaultBrowser": True,
            }},
        )

    def test_default_and_missing_native_fail_safe_to_snap(self) -> None:
        self.assertEqual(self._run("kiosk-launch.sh").returncode, 0)
        self._choice("native-156.0")
        result = self._run("kiosk-launch.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.log.read_text().count("snap:"), 2)
        self.assertIn("unavailable", result.stderr)

    def test_partial_deploy_without_common_file_uses_existing_firefox(self) -> None:
        orphan = Path(self.tmp.name) / "orphan/kiosk-launch.sh"
        orphan.parent.mkdir()
        shutil.copy2(self.deploy / "kiosk-launch.sh", orphan)
        result = subprocess.run(
            ["bash", str(orphan)], env=self.env, text=True, capture_output=True, timeout=5
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.log.read_text(), "snap:--kiosk http://127.0.0.1:8780/\n")

    def test_invalid_marker_policy_or_version_fails_safe(self) -> None:
        root = self._install_mock_native()
        self._choice("native-156.0")
        for file, content in (
            (root / "VERIFIED_SHA256", "wrong\n"),
            (root / "firefox/distribution/policies.json", '{"policies":{}}\n'),
            (root / "firefox/firefox", '#!/bin/sh\necho "Mozilla Firefox 155.0"\n'),
            (root / "firefox/firefox", '#!/bin/sh\necho "Mozilla Firefox 156.0"\nexit 1\n'),
        ):
            with self.subTest(file=file):
                old = file.read_text()
                file.write_text(content)
                result = self._run("kiosk-launch.sh")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(self.log.read_text().splitlines()[-1].startswith("snap:"))
                file.write_text(old)

    def test_verified_native_uses_separate_profile_and_local_origin(self) -> None:
        self._install_mock_native()
        self._choice("native-156.0")
        result = self._run("kiosk-launch.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.log.read_text(),
            "native:--new-instance --profile "
            f"{self.home}/.local/share/kotoba-firefox/156.0/profile "
            "--kiosk http://127.0.0.1:8780/\n",
        )
        self.assertIn("aa:-p firefox --", self.aa_log.read_text())
        self.assertEqual(self.gate_log.read_text(), "checked\nlaunched\n")

    def test_sandbox_gate_failure_refuses_selection_and_falls_back_to_snap(self) -> None:
        self._install_mock_native()
        self.env["KOTOBA_BROWSER_TEST_SANDBOX_EXIT"] = "1"
        self.assertEqual(self._run("kiosk-browser-select.sh", "native-156.0").returncode, 1)
        self.assertFalse((self.home / ".config/kotoba/kiosk-browser").exists())
        self._choice("native-156.0")
        result = self._run("kiosk-launch.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("snap:", self.log.read_text())
        self.assertEqual(self.gate_log.read_text(), "checked\nchecked\n")

    def test_missing_aa_exec_refuses_selection_and_falls_back_to_snap(self) -> None:
        self._install_mock_native()
        (self.bin_dir / "aa-exec").unlink()
        self.assertEqual(self._run("kiosk-browser-select.sh", "native-156.0").returncode, 1)
        self._choice("native-156.0")
        result = self._run("kiosk-launch.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("snap:", self.log.read_text())
        self.assertFalse(self.gate_log.exists())

    def test_selector_refuses_unverified_and_rollback_fails_without_gui_session(self) -> None:
        result = self._run("kiosk-browser-select.sh", "native-156.0")
        self.assertEqual(result.returncode, 1)
        self.assertFalse((self.home / ".config/kotoba/kiosk-browser").exists())
        self._install_mock_native()
        self.assertEqual(self._run("kiosk-browser-select.sh", "native-156.0").returncode, 0)
        self.assertEqual(
            (self.home / ".config/kotoba/kiosk-browser").read_text(), "native-156.0\n"
        )
        # A broken or absent physical session must still select snap for next login.
        rollback = self._run("kiosk-browser-rollback.sh")
        self.assertEqual(rollback.returncode, 1, rollback.stderr)
        self.assertIn("X11 session", rollback.stderr)
        self.assertIn("選択だけsnapに変更済み", rollback.stderr)
        self.assertEqual((self.home / ".config/kotoba/kiosk-browser").read_text(), "snap\n")
        self.assertFalse(self.log.exists())

    def test_non_cloudia_context_fails_before_mutation(self) -> None:
        self._script(self.bin_dir / "id", 'case "$1" in -un) echo root;; -u) echo 0;; esac')
        self.assertEqual(self._run("kiosk-browser-select.sh", "snap").returncode, 1)
        self.assertEqual(self._run("kiosk-browser-rollback.sh").returncode, 1)
        self.assertFalse((self.home / ".config/kotoba/kiosk-browser").exists())

    def _linux_gui_session_mocks(self) -> Path:
        uid = os.getuid()
        runtime = Path(self.tmp.name) / "run/user" / str(uid)
        x11 = Path(self.tmp.name) / "X11-unix"
        (runtime / "gdm").mkdir(parents=True)
        x11.mkdir()
        (runtime / "gdm/Xauthority").write_bytes(b"test")
        for path in (runtime / "bus", x11 / "X0"):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(str(path))
            sock.listen()
            self.addCleanup(sock.close)
        self.env.update(
            KOTOBA_RUNTIME_ROOT=str(runtime.parent),
            KOTOBA_X11_ROOT=str(x11),
            KOTOBA_SNAP_ROOT=str(Path(self.tmp.name) / "snap/firefox"),
            KOTOBA_SNAP_LAUNCHER=str(self.bin_dir / "firefox"),
            KOTOBA_BROWSER_TEST_SNAP_PID=str(Path(self.tmp.name) / "snap.pid"),
            KOTOBA_BROWSER_TEST_DISPATCH_LOG=str(Path(self.tmp.name) / "dispatch.log"),
            KOTOBA_BROWSER_TEST_STOP_LOG=str(Path(self.tmp.name) / "stop.log"),
        )
        self._script(
            self.bin_dir / "loginctl",
            f'''case "$1" in
list-sessions) echo "2 {uid} cloudia seat0 tty2" ;;
show-session) case "$4" in
  Active) echo yes;; Remote) echo no;; Type) echo x11;; Seat) echo seat0;; Display) echo :0;;
esac ;;
esac''',
        )
        self._script(
            self.bin_dir / "systemd-run",
            '''case " $* " in *" /bin/true "*) exit 0;; esac
printf '%s\\n' "$*" > "$KOTOBA_BROWSER_TEST_DISPATCH_LOG"
if [ "${KOTOBA_BROWSER_TEST_DISPATCH_FAIL:-}" = 1 ]; then exit 1; fi
if [ "${KOTOBA_BROWSER_TEST_NO_WINDOW:-}" = 1 ]; then exit 0; fi
"$KOTOBA_BROWSER_TEST_SNAP_BIN" --kiosk http://127.0.0.1:8780/ </dev/null >/dev/null 2>&1 &
printf '%s\\n' "$!" > "$KOTOBA_BROWSER_TEST_SNAP_PID"''',
        )
        self._script(
            self.bin_dir / "xprop",
            '''if [ ! -s "$KOTOBA_BROWSER_TEST_SNAP_PID" ]; then exit 1; fi
case "$1" in
  -root) echo '_NET_ACTIVE_WINDOW(WINDOW): window id # 0x01' ;;
  -id) printf '_NET_WM_PID(CARDINAL) = %s\\n' "$(head -n 1 "$KOTOBA_BROWSER_TEST_SNAP_PID")" ;;
esac''',
        )
        self._script(
            self.bin_dir / "systemctl",
            'printf "%s\\n" "$*" > "$KOTOBA_BROWSER_TEST_STOP_LOG"',
        )
        source = Path(self.tmp.name) / "fake_firefox.c"
        source.write_text(
            '#include <stdio.h>\n#include <string.h>\n#include <unistd.h>\n'
            'int main(int argc, char **argv) {\n'
            ' if (argc > 1 && !strcmp(argv[1], "--version")) { puts("Mozilla Firefox 156.0"); return 0; }\n'
            ' for (;;) pause();\n}\n'
        )
        snap = Path(self.env["KOTOBA_SNAP_ROOT"]) / "8926/firefox"
        snap.parent.mkdir(parents=True)
        subprocess.run(["cc", str(source), "-o", str(snap)], check=True, capture_output=True)
        self.env["KOTOBA_BROWSER_TEST_SNAP_BIN"] = str(snap)
        self.addCleanup(self._terminate_snap_mock)
        return source

    def _terminate_snap_mock(self) -> None:
        path = Path(self.env.get("KOTOBA_BROWSER_TEST_SNAP_PID", "/nonexistent"))
        if path.is_file():
            try:
                os.kill(int(path.read_text()), 15)
            except ProcessLookupError:
                pass

    @staticmethod
    def _stop_process(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

    @unittest.skipUnless(sys.platform == "linux" and shutil.which("cc"), "Linux /proc + C compiler")
    def test_rollback_terminates_only_matching_native_kiosk(self) -> None:
        root = self._install_mock_native()
        native = root / "firefox/firefox"
        source = self._linux_gui_session_mocks()
        subprocess.run(["cc", str(source), "-o", str(native)], check=True, capture_output=True)
        self._choice("native-156.0")
        process = subprocess.Popen(["bash", str(self.deploy / "kiosk-launch.sh")], env=self.env)
        self.addCleanup(self._stop_process, process)

        def native_child_running() -> bool:
            for entry in Path("/proc").iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    if (entry / "exe").resolve(strict=True) == native:
                        return True
                except (FileNotFoundError, PermissionError):
                    continue
            return False

        for _ in range(100):
            if native_child_running():
                break
            time.sleep(0.02)
        else:
            self.fail("native kiosk did not start")
        rollback = self._run("kiosk-browser-rollback.sh", timeout=60)
        self.assertEqual(rollback.returncode, 0, rollback.stderr)
        process.wait(timeout=5)
        self.assertEqual((self.home / ".config/kotoba/kiosk-browser").read_text(), "snap\n")
        dispatch = Path(self.env["KOTOBA_BROWSER_TEST_DISPATCH_LOG"]).read_text()
        self.assertIn("--setenv=DISPLAY=:0", dispatch)
        self.assertIn("--setenv=XAUTHORITY=", dispatch)
        self.assertIn("--setenv=KOTOBA_KIOSK_HEALTH_ATTEMPTS=8", dispatch)
        self.assertIn("/bin/bash", dispatch)

    @unittest.skipUnless(sys.platform == "linux" and shutil.which("cc"), "Linux /proc + C compiler")
    def test_rollback_preserves_existing_snap_kiosk(self) -> None:
        root = self._install_mock_native()
        native = root / "firefox/firefox"
        source = self._linux_gui_session_mocks()
        subprocess.run(["cc", str(source), "-o", str(native)], check=True, capture_output=True)
        snap = subprocess.Popen(
            [self.env["KOTOBA_BROWSER_TEST_SNAP_BIN"], "--kiosk", "http://127.0.0.1:8780/"],
            env=self.env,
        )
        self.addCleanup(self._stop_process, snap)
        Path(self.env["KOTOBA_BROWSER_TEST_SNAP_PID"]).write_text(str(snap.pid) + "\n")
        kiosk = subprocess.Popen(
            [str(native), "--new-instance", "--profile",
             str(self.home / ".local/share/kotoba-firefox/156.0/profile"),
             "--kiosk", "http://127.0.0.1:8780/"],
            env=self.env,
        )
        self.addCleanup(self._stop_process, kiosk)
        self._choice("native-156.0")
        rollback = self._run("kiosk-browser-rollback.sh", timeout=30)
        self.assertEqual(rollback.returncode, 0, rollback.stderr)
        kiosk.wait(timeout=5)
        self.assertIsNone(snap.poll())
        self.assertFalse(Path(self.env["KOTOBA_BROWSER_TEST_DISPATCH_LOG"]).exists())
        self.assertEqual((self.home / ".config/kotoba/kiosk-browser").read_text(), "snap\n")

    @unittest.skipUnless(sys.platform == "linux" and shutil.which("cc"), "Linux /proc + C compiler")
    def test_rollback_fails_if_snap_dispatch_fails(self) -> None:
        self._linux_gui_session_mocks()
        self._choice("native-156.0")
        self.env["KOTOBA_BROWSER_TEST_DISPATCH_FAIL"] = "1"
        rollback = self._run("kiosk-browser-rollback.sh")
        self.assertEqual(rollback.returncode, 1, rollback.stderr)
        self.assertEqual((self.home / ".config/kotoba/kiosk-browser").read_text(), "snap\n")

    @unittest.skipUnless(sys.platform == "linux" and shutil.which("cc"), "Linux /proc + C compiler")
    def test_rollback_fails_if_snap_window_never_appears(self) -> None:
        self._linux_gui_session_mocks()
        self.env["KOTOBA_BROWSER_TEST_NO_WINDOW"] = "1"
        rollback = self._run("kiosk-browser-rollback.sh")
        self.assertEqual(rollback.returncode, 1, rollback.stderr)
        self.assertIn("実window", rollback.stderr)
        self.assertIn("--user stop kotoba-kiosk-rollback-", Path(self.env["KOTOBA_BROWSER_TEST_STOP_LOG"]).read_text())

    @unittest.skipUnless(sys.platform == "linux" and shutil.which("cc"), "Linux /proc + C compiler")
    def test_rollback_refuses_unidentified_native_process(self) -> None:
        native = self._install_mock_native() / "firefox/firefox"
        source = self._linux_gui_session_mocks()
        subprocess.run(["cc", str(source), "-o", str(native)], check=True, capture_output=True)
        wrong_profile = self.home / "other-profile"
        process = subprocess.Popen(
            [str(native), "--new-instance", "--profile", str(wrong_profile), "--kiosk", "http://127.0.0.1:8780/"],
            env=self.env,
        )
        self.addCleanup(self._stop_process, process)
        self._choice("native-156.0")
        rollback = self._run("kiosk-browser-rollback.sh", timeout=30)
        self.assertEqual(rollback.returncode, 1, rollback.stderr)
        self.assertIn("native Firefox", rollback.stderr)
        self.assertIsNone(process.poll())
        self.assertFalse(Path(self.env["KOTOBA_BROWSER_TEST_DISPATCH_LOG"]).exists())


if __name__ == "__main__":
    unittest.main()
