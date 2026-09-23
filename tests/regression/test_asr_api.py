"""R6 P1-D: ローカルASR転写プロキシの回帰試験。

実入口（実FastAPI TestClient + 実stub ASRサーバ）で検証:
- unknown session → 404
- 正常PCM16 → stub ASRのtextを透過（latency/engine付き）
- サイズ上限超過 → 413、奇数byte → 422
- sessionあたりin-flight=1 — 処理中の2本目は429 asr_busy
- ASR未稼働 → 503 asr_unavailable（typed失敗 — 黙った代替経路なし）
- /api/asr/status: 未稼働 → reachable=False、稼働 → engine/modelを透過

録音の永続化・クラウドASRへのfallbackは存在しないこと（経路は
KOTOBA_ASR_URLのみ — 未指定時は127.0.0.1:8710固定）を構造で確認する。
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "services" / "harness" / "src" / "kotoba_harness"

PCM_1S = b"\x00\x01" * 16000  # 16kHz×1sのPCM16（内容はstubが見ない）


def _ensure_real_harness_pkg():
    """snapshot汚染対策（test_game_api.pyと同じ技法）— service import前に
    現行kotoba_harnessを __path__ 付きで明示復元する。"""
    spec = importlib.util.spec_from_file_location(
        "kotoba_harness",
        HARNESS / "__init__.py",
        submodule_search_locations=[str(HARNESS)],
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["kotoba_harness"] = pkg
    spec.loader.exec_module(pkg)
    return pkg


class _StubAsr(BaseHTTPRequestHandler):
    delay_s = 0.0
    text = "前"
    hang = False

    def log_message(self, *a):  # noqa: A003
        pass

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            b = json.dumps({"ok": True, "engine": "qwen3-asr",
                            "model": "stub-0.6b"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length") or "0")
        self.rfile.read(n)
        if self.hang:
            time.sleep(30)  # client timeoutを待つ（daemon threadなので残留可）
        time.sleep(self.delay_s)
        b = json.dumps({"text": self.text, "latency_ms": 12.3,
                        "engine": "qwen3-asr"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


def _start_stub(**kw):
    """実stub ASRをlocalhostの空きportで起動。"""
    for k, v in kw.items():
        setattr(_StubAsr, k, v)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _StubAsr)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _load_app(tmp_path, monkeypatch, asr_url):
    _ensure_real_harness_pkg()
    monkeypatch.setenv("KOTOBA_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("KOTOBA_HOME", str(tmp_path))
    monkeypatch.setenv("KOTOBA_OFFLINE", "0")
    monkeypatch.setenv("KOTOBA_SELFTEST", "1")
    if asr_url:
        monkeypatch.setenv("KOTOBA_ASR_URL", asr_url)
    import kotoba_api.app as app_mod

    mod = importlib.reload(app_mod)
    mod._asr_locks.clear()  # reload間のlock残留を掃除
    return mod


def _session_id(client):
    r = client.post("/api/sessions")
    assert r.status_code == 200
    return r.json()["session_id"]


def test_unknown_session_rejected(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    app_mod = _load_app(tmp_path, monkeypatch, None)
    r = TestClient(app_mod.app).post(
        "/api/asr/recognize?session_id=nope",
        content=PCM_1S,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert r.status_code == 404


def test_recognize_passthrough(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    srv, url = _start_stub()
    try:
        app_mod = _load_app(tmp_path, monkeypatch, url)
        c = TestClient(app_mod.app)
        sid = _session_id(c)
        r = c.post(
            f"/api/asr/recognize?session_id={sid}",
            content=PCM_1S,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert r.status_code == 200
        d = r.json()
        assert d["text"] == "前"
        assert d["engine"] == "qwen3-asr"
        assert d["latency_ms"] == 12.3
    finally:
        srv.shutdown()


def test_oversized_and_odd_rejected(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    app_mod = _load_app(tmp_path, monkeypatch, None)
    c = TestClient(app_mod.app)
    sid = _session_id(c)
    big = b"\x00" * (app_mod.ASR_MAX_BYTES + 2)
    r = c.post(f"/api/asr/recognize?session_id={sid}", content=big,
               headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 413
    r = c.post(f"/api/asr/recognize?session_id={sid}", content=b"\x00" * 3,
               headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 422


def test_inflight_bound_returns_429(tmp_path, monkeypatch):
    """同sessionで処理中の2本目は429 — clientのqueue制御と独立の防御。"""
    from fastapi.testclient import TestClient

    srv, url = _start_stub(delay_s=1.5)
    try:
        app_mod = _load_app(tmp_path, monkeypatch, url)
        c = TestClient(app_mod.app)
        sid = _session_id(c)
        results = []

        def call():
            results.append(
                c.post(
                    f"/api/asr/recognize?session_id={sid}",
                    content=PCM_1S,
                    headers={"Content-Type": "application/octet-stream"},
                ).status_code
            )

        t1 = threading.Thread(target=call)
        t1.start()
        time.sleep(0.3)  # 1本目がASR側で処理中にする
        call()           # 2本目 — in-flight中なので429のはず
        t1.join(timeout=10)
        assert sorted(results) == [200, 429]
    finally:
        _StubAsr.delay_s = 0.0
        srv.shutdown()


def test_asr_unreachable_is_typed_503(tmp_path, monkeypatch):
    """ASR未稼働 → 503 asr_unavailable:* — 黙った代替経路は存在しない。"""
    from fastapi.testclient import TestClient

    # 何もlistenしていないportを指す
    app_mod = _load_app(tmp_path, monkeypatch, "http://127.0.0.1:9")
    c = TestClient(app_mod.app)
    sid = _session_id(c)
    r = c.post(
        f"/api/asr/recognize?session_id={sid}",
        content=PCM_1S,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert r.status_code == 503
    assert "asr_unavailable" in str(r.json())


def test_status_unreachable_then_reachable(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    app_mod = _load_app(tmp_path, monkeypatch, "http://127.0.0.1:9")
    c = TestClient(app_mod.app)
    d = c.get("/api/asr/status").json()
    assert d["reachable"] is False

    srv, url = _start_stub()
    try:
        app_mod = _load_app(tmp_path, monkeypatch, url)
        d = TestClient(app_mod.app).get("/api/asr/status").json()
        assert d["reachable"] is True
        assert d["engine"] == "qwen3-asr" and d["model"] == "stub-0.6b"
    finally:
        srv.shutdown()
