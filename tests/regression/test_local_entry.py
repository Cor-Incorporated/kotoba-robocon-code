"""R7 1.5/2.1: loopback local entry の回帰試験。

実FastAPI TestClient + 実stub APIサーバ（:8700の代役）で検証:
- sessionなし: GET参照系と POST /api/sessions・/api/asr/recognize は通る
- sessionなし: 変異系（game/start等）は 403 local_operator_required で
  upstreamへ届かない（公開403のローカル版）
- session開始後: 変異系が upstream へ x-kotoba-operator 付きで届く
  （値はenvのserver側token — browser送信値は採用されない）
- browser送信の偽 x-kotoba-operator は剥がされる（session無しでは upstream
  へoperator headerが届かない）
- Hostがlocalhost系でない要求は拒否（DNS rebinding対策）
- 変異系でforeign Originは拒否（SameSite=Strict cookieと二重防御）
- session終了でoperator権を失う
- /diag・/local/diag はsession無しで参照可（診断は開く・復旧はsession要）

upstream stubが実際に受けたheader/bodyを検査するため、
「注入したつもり」「剥がしたつもり」の偽PASSを防ぐ。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "services" / "local_entry" / "kotoba_local_entry.py"
TEST_TOKEN = "test-operator-token-0123456789abcdef"

_seen: list[dict] = []


class _StubApi(BaseHTTPRequestHandler):
    def log_message(self, *a):  # noqa: A003
        pass

    def _rec(self):
        n = int(self.headers.get("Content-Length") or "0")
        body = self.rfile.read(n) if n else b""
        _seen.append({
            "method": self.command,
            "path": self.path,
            "operator": self.headers.get("x-kotoba-operator"),
            "cookie": self.headers.get("cookie"),
        })
        return body

    def _ok(self, extra=None):
        self._rec()
        b = json.dumps({"ok": True, **(extra or {})}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):  # noqa: N802
        self._ok({"path": self.path, "phase": "IDLE"})

    def do_POST(self):  # noqa: N802
        self._ok({"path": self.path})


@pytest.fixture()
def stub_api():
    _seen.clear()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _StubApi)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


@pytest.fixture()
def client(stub_api, tmp_path, monkeypatch):
    os.environ["KOTOBA_API_URL"] = stub_api
    os.environ["KOTOBA_OPERATOR_TOKEN"] = TEST_TOKEN
    os.environ["KOTOBA_LOCAL_PORT"] = "8780"
    monkeypatch.setenv("KOTOBA_KIOSK_DIST", str(tmp_path))
    spec = importlib.util.spec_from_file_location("kotoba_local_entry", ENTRY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kotoba_local_entry"] = mod
    spec.loader.exec_module(mod)
    from fastapi.testclient import TestClient

    c = TestClient(mod.app)
    # TestClient既定Hostはtestserver — 実loopback名を明示してHost検査を通す
    c.headers.update({"host": "localhost:8780"})
    yield c
    mod._sessions.clear()


def test_readonly_paths_pass_without_session(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    r = client.post("/api/sessions")
    assert r.status_code == 200
    r = client.post("/api/asr/recognize", content=b"\x00\x01" * 100)
    assert r.status_code == 200


def test_render_document_is_served_from_local_origin(client, tmp_path):
    """Kioskのiframeはlocal originのHTTP GETでvendored clientを読む。"""
    render_dir = tmp_path / "render"
    render_dir.mkdir()
    html = b"<!doctype html><title>Viser fixture</title>"
    (render_dir / "index.html").write_bytes(html)

    r = client.get("/render/index.html")
    assert r.status_code == 200
    assert r.content == html
    assert r.headers["content-type"].startswith("text/html")
    assert client.get("/render/missing.js").status_code == 404


def test_static_path_cannot_escape_dist_prefix_sibling(client, tmp_path):
    """文字列prefixが一致する隣接dirも静的公開しない。"""
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("synthetic private fixture")
    mod = sys.modules["kotoba_local_entry"]
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        mod.static_files(f"../{outside.name}")
    assert exc.value.status_code == 404


def test_mutation_requires_session(client):
    r = client.post("/api/game/start", json={})
    assert r.status_code == 403
    assert r.json()["detail"] == "local_operator_required"
    # upstreamへ届いていないことをstub側で確認（偽PASS防止）
    assert all(s["path"] != "/api/game/start" for s in _seen)


def test_session_grants_operator_token_at_proxy(client):
    assert client.post("/local/session/start").status_code == 200
    assert client.get("/local/session").json()["operator"] is True
    r = client.post("/api/game/start", json={"session_id": "s1"})
    assert r.status_code == 200
    hit = [s for s in _seen if s["path"] == "/api/game/start"]
    assert hit and hit[-1]["operator"] == TEST_TOKEN


def test_spoofed_operator_header_not_forwarded(client):
    # session無しで偽operator headerを送ってもupstreamへ届かない
    client.post("/api/sessions", headers={"x-kotoba-operator": "forged"})
    hit = [s for s in _seen if s["path"] == "/api/sessions"]
    assert hit and hit[-1]["operator"] is None
    # session有りでも upstream のoperator値はserver側token固定
    client.post("/local/session/start")
    client.post("/api/game/start", headers={"x-kotoba-operator": "forged"},
                json={})
    hit = [s for s in _seen if s["path"] == "/api/game/start"]
    assert hit and hit[-1]["operator"] == TEST_TOKEN


def test_foreign_host_and_origin_rejected(client):
    client.post("/local/session/start")
    # DNS rebinding想定: Hostがevil.exampleなら拒否
    r = client.post("/api/game/start", json={}, headers={
        "host": "evil.example:8780"})
    assert r.status_code == 403
    # foreign Originの変異系は拒否
    r = client.post("/api/game/start", json={}, headers={
        "origin": "https://evil.example"})
    assert r.status_code == 403
    # local originは受理
    r = client.post("/api/game/start", json={}, headers={
        "origin": "http://localhost:8780"})
    assert r.status_code == 200


def test_session_end_revokes(client):
    client.post("/local/session/start")
    assert client.get("/local/session").json()["operator"] is True
    client.post("/local/session/end")
    assert client.get("/local/session").json()["operator"] is False
    r = client.post("/api/game/start", json={})
    assert r.status_code == 403


def test_diag_open_but_restart_requires_session(client):
    page = client.get("/diag")
    assert page.status_code == 200
    assert '<a class="back" href="/">展示画面へ戻る</a>' in page.text
    assert page.text.index('id="exitKiosk"') < page.text.index('<h2>サービス</h2>')
    r = client.get("/local/diag")
    assert r.status_code == 200
    assert r.json()["operator"] is False
    r = client.post("/local/service/restart", json={"unit": "kotoba-asr"})
    assert r.status_code == 403
    # 未知unitはsession有りでも拒否（allowlist）
    client.post("/local/session/start")
    r = client.post("/local/service/restart", json={"unit": "cron"})
    assert r.status_code == 400


def test_token_never_leaks_to_browser(client):
    # static/local応答・session応答にtoken文字列が出ないこと
    for path in ("/local/session", "/diag"):
        r = client.get(path)
        assert TEST_TOKEN not in r.text
    r = client.post("/local/session/start")
    assert TEST_TOKEN not in r.text
    assert TEST_TOKEN not in r.headers.get("set-cookie", "")


def test_kiosk_exit_requires_operator_and_idle_round(client, monkeypatch):
    mod = sys.modules["kotoba_local_entry"]
    from fastapi.responses import Response

    killed = []
    monkeypatch.setattr(mod, "_kiosk_pids", lambda: [1234])
    monkeypatch.setattr(mod.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    assert client.post("/local/kiosk/exit").status_code == 403
    assert client.post("/local/session/start").status_code == 200
    assert client.post("/local/kiosk/exit", headers={
        "origin": "https://outside.example"}).status_code == 403
    monkeypatch.setattr(mod, "_proxy_http", lambda *_: Response(status_code=409))
    assert client.post("/local/kiosk/exit").status_code == 409
    monkeypatch.setattr(mod, "_proxy_http", lambda *_: Response(status_code=503))
    assert client.post("/local/kiosk/exit").status_code == 503
    assert killed == []

    monkeypatch.setattr(mod, "_proxy_http", lambda *_: Response(status_code=200))
    assert client.post("/local/kiosk/exit").json() == {"closing": True}
    assert killed == [(1234, mod.signal.SIGTERM)]


def test_kiosk_pid_selection_uses_exact_origin_and_user(client, tmp_path, monkeypatch):
    mod = sys.modules["kotoba_local_entry"]
    for pid, argv in {
        "101": b"/snap/firefox/current/firefox\0--kiosk\0http://127.0.0.1:8780/\0",
        "102": b"/snap/firefox/current/firefox\0--kiosk\0https://other.example/\0",
        "103": b"python3\0--kiosk\0http://127.0.0.1:8780/\0",
    }.items():
        folder = tmp_path / pid
        folder.mkdir()
        (folder / "cmdline").write_bytes(argv)
    monkeypatch.setattr(mod.os, "getuid", lambda: tmp_path.stat().st_uid)
    assert mod._kiosk_pids(tmp_path) == [101]
