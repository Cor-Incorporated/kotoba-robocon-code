"""ことばでロボコン — Thor一台完結のローカル入口（R7 1.5/2.1）。

公開経路（Cloudflare Access→Worker→Tunnel→:8700）とは別に、
127.0.0.1:8780 のみにbindする展示専用のローカルorigin。

設計:
- kiosk静的配信 + /api/* HTTP proxy + /render/* WS proxy（upstream :8700）。
- 「この端末で操作を開始」の明示操作でserver側所有のoperator sessionを
  発行する（HttpOnly・SameSite=Strict cookie。URL/bundle/localStorageへ
  秘密を置かない）。
- session保持者の変異系 /api/* には proxy が x-kotoba-operator を注入する。
  ブラウザ送信の operator header は常に剥がす（偽装不可）。
- sessionなしでは GET 参照系 + POST /api/sessions + POST /api/asr/recognize
  のみ通す（閲覧は自由・制御はsession保持者のみ — Workerの
  readonly/operator分離と同じ構造をloopback内で再現）。
- Hostがlocalhost系でない要求は拒否（DNS rebinding対策）。変異系は
  Origin一致も必須（SameSite=Strictの二重防御）。
- 「127.0.0.1からの接続だからoperator」という判定はしない — API:8700の
  operator gateはtoken検査のまま。cloudflared等のloopback由来接続は
  この入口を通らない限り権限を得ない。
- /diag: サービス状態・デバイス・モデルの診断と限定restart
  （sudoersで kotoba-* unitのみ許可）。復旧がGUIで完結するための画面。

起動: KOTOBA_API_URL=http://127.0.0.1:8700 を読み、uvicornで
127.0.0.1:8780 にbind。operator tokenはenv KOTOBA_OPERATOR_TOKENからのみ
読み、応答・ログ・静的ファイルへ一切出さない。
"""

import json
import os
import secrets
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

API_URL = os.environ.get("KOTOBA_API_URL", "http://127.0.0.1:8700").rstrip("/")
API_WS = API_URL.replace("http://", "ws://").replace("https://", "wss://")
OPERATOR_TOKEN = os.environ.get("KOTOBA_OPERATOR_TOKEN", "")
KIOSK_DIST = Path(
    os.environ.get(
        "KOTOBA_KIOSK_DIST",
        Path(__file__).resolve().parents[2] / "apps" / "kiosk" / "dist",
    )
)
LISTEN_PORT = int(os.environ.get("KOTOBA_LOCAL_PORT", "8780"))
SESSION_TTL_S = 12 * 3600  # 展示日の稼働時間をカバー。冷起動後は新session。
COOKIE = "kotoba_local_session"

# session無しでも通す公開経路（閲覧・session骨組み・参加者ASR転写）。
# これ以外の変異系はlocal operator session必須。
_PUBLIC_POST = ("^/api/sessions$", "^/api/asr/recognize$")
import re as _re

_PUBLIC_POST_RE = [_re.compile(p) for p in _PUBLIC_POST]

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}
_sessions: dict[str, float] = {}  # sid -> expiry（server側所有。in-memoryのみ）

app = FastAPI(title="kotoba-local-entry", version="0.1.0-r7")


def _host_ok(request: Request) -> bool:
    from urllib.parse import urlsplit

    host = (request.headers.get("host") or "").lower()
    if not host:
        return False
    try:
        # Hostは "name[:port]" — IPv6は [..] 形式
        parsed = urlsplit(f"//{host}")
        return (parsed.hostname or "").lower() in {
            "localhost", "127.0.0.1", "::1",
        }
    except ValueError:
        return False


def _origin_ok(request: Request) -> bool:
    """変異系のCSRF防御 — Originがあればlocal origin一致を要求。
    SameSite=Strict cookieと合わせて二重防御。非browser local client
    （curl等）はOriginを送らない — loopback bind済みなので許容する。"""
    from urllib.parse import urlsplit

    origin = request.headers.get("origin")
    if origin is None:
        return True
    try:
        p = urlsplit(origin)
        if p.scheme != "http":
            return False
        host = (p.hostname or "").lower()
        port = p.port or 80
        return host in {"localhost", "127.0.0.1", "::1"} and port == LISTEN_PORT
    except ValueError:
        return False


def _session_ok(request: Request) -> bool:
    sid = request.cookies.get(COOKIE)
    if not sid:
        return False
    exp = _sessions.get(sid)
    if exp is None:
        return False
    if time.time() > exp:
        _sessions.pop(sid, None)
        return False
    return True


def _proxy_http(method: str, path: str, request: Request, body: bytes) -> Response:
    """/api/* をAPI:8700へ中継。session保持者のみoperator tokenを注入。"""
    url = f"{API_URL}/api/{path}"
    if request.url.query:
        url += f"?{request.url.query}"
    headers = {
        "content-type": request.headers.get("content-type", "application/json"),
        "accept": request.headers.get("accept", "*/*"),
    }
    rid = request.headers.get("x-request-id")
    if rid:
        headers["x-request-id"] = rid
    if _session_ok(request) and OPERATOR_TOKEN:
        headers["x-kotoba-operator"] = OPERATOR_TOKEN
    req = urllib.request.Request(url, data=body if body else None,
                                 headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            payload = r.read()
            out_headers = {
                "content-type": r.headers.get("content-type", "application/json"),
                "cache-control": "no-store",
            }
            upstream_rid = r.headers.get("x-request-id")
            if upstream_rid:
                out_headers["x-request-id"] = upstream_rid
            return Response(payload, status_code=r.status, headers=out_headers)
    except urllib.error.HTTPError as e:
        payload = e.read()
        out_headers = {
            "content-type": e.headers.get("content-type", "application/json")
            if e.headers else "application/json",
            "cache-control": "no-store",
        }
        upstream_rid = e.headers.get("x-request-id") if e.headers else None
        if upstream_rid:
            out_headers["x-request-id"] = upstream_rid
        return Response(payload, status_code=e.code, headers=out_headers)
    except Exception as exc:
        return JSONResponse(
            {"detail": f"api_unreachable:{type(exc).__name__}"}, status_code=502
        )


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def api_proxy(path: str, request: Request):
    if not _host_ok(request):
        return JSONResponse({"detail": "bad_host"}, status_code=403)
    is_public_post = (
        request.method == "POST"
        and any(p.match(f"/api/{path}") for p in _PUBLIC_POST_RE)
    )
    if request.method != "GET" and not is_public_post:
        if not _origin_ok(request):
            return JSONResponse({"detail": "bad_origin"}, status_code=403)
        if not _session_ok(request):
            return JSONResponse({"detail": "local_operator_required"},
                                status_code=403)
    body = await request.body()
    import asyncio

    return await asyncio.to_thread(_proxy_http, request.method, path, request, body)


@app.websocket("/render/{path:path}")
async def render_ws(websocket: WebSocket, path: str):
    """/render/* WSをAPI:8700のbridgeへ中継（読み取り専用の描画stream）。"""
    import asyncio
    import websockets

    host = (websocket.headers.get("host") or "").split(":")[0].lower()
    if host not in _LOCAL_HOSTS:
        await websocket.close(code=1008)
        return
    upstream_url = f"{API_WS}/render/{path}"
    if websocket.url.query:
        upstream_url += f"?{websocket.url.query}"
    client_subprotocols = list(websocket.scope.get("subprotocols") or [])
    try:
        async with websockets.connect(
            upstream_url,
            subprotocols=client_subprotocols or None,
            max_size=None,
            open_timeout=15,
            ping_interval=20,
            ping_timeout=60,
            close_timeout=10,
        ) as upstream:
            await websocket.accept(subprotocol=upstream.subprotocol)

            async def up() -> None:
                while True:
                    msg = await websocket.receive()
                    if msg.get("bytes") is not None:
                        await upstream.send(msg["bytes"])
                    elif msg.get("text") is not None:
                        await upstream.send(msg["text"])
                    elif msg.get("type") == "websocket.disconnect":
                        return

            async def down() -> None:
                async for frame in upstream:
                    if isinstance(frame, bytes):
                        await websocket.send_bytes(frame)
                    else:
                        await websocket.send_text(frame)

            tasks = [asyncio.create_task(up()), asyncio.create_task(down())]
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
    except Exception:
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


# ---- local session（server側所有・明示操作のみで発行） ----------------------

@app.post("/local/session/start")
def session_start(request: Request):
    if not _host_ok(request) or not _origin_ok(request):
        raise HTTPException(403, "bad_origin")
    sid = secrets.token_urlsafe(32)
    _sessions[sid] = time.time() + SESSION_TTL_S
    r = JSONResponse({"operator": True})
    # Secureは付けない — http://localhostではSecure cookieが保存されない
    # 環境があり得る。SameSite=Strict+HttpOnly+Host検査で防御する。
    r.set_cookie(COOKIE, sid, httponly=True, samesite="strict", path="/",
                 max_age=SESSION_TTL_S)
    return r


@app.get("/local/session")
def session_state(request: Request):
    return {"operator": _session_ok(request)}


@app.post("/local/session/end")
def session_end(request: Request):
    sid = request.cookies.get(COOKIE)
    if sid:
        _sessions.pop(sid, None)
    r = JSONResponse({"operator": False})
    r.delete_cookie(COOKIE, path="/")
    return r


# ---- 診断・安全復旧（/diag） ------------------------------------------------
_DIAG_UNITS = [
    "kotoba-api", "kotoba-render", "kotoba-asr", "kotoba-ollama",
    "kotoba-local-entry", "kotoba-sim-boot", "docker",
]
# restart許可対象（sudoersで更に kotoba-* のみへ絞る）
# simコンテナ群(net0/ctl/mujoco/stand/live)はAPI内SimControlの所有物のため
# 個別再起動させない。「kotoba-sim-boot」をrestartするとoneshotが再実行され
# API経由で世界全体が健全に再起動する。
_RESTARTABLE = {
    "kotoba-api", "kotoba-render", "kotoba-asr", "kotoba-ollama",
    "kotoba-sim-boot", "kotoba-local-entry",
}


def _kiosk_pids(proc_root: Path = Path("/proc")) -> list[int]:
    """Find only this user's Firefox kiosk for the local exhibition origin."""
    entry = os.environ.get("KOTOBA_LOCAL_ENTRY", f"http://127.0.0.1:{LISTEN_PORT}")
    expected_url = (entry.rstrip("/") + "/").encode()
    matches = []
    for proc in proc_root.iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            argv = (proc / "cmdline").read_bytes().split(b"\0")
            if (Path(os.fsdecode(argv[0])).name == "firefox"
                    and b"--kiosk" in argv and expected_url in argv):
                matches.append(int(proc.name))
        except (OSError, IndexError):
            continue
    return matches


@app.post("/local/kiosk/exit")
def kiosk_exit(request: Request):
    """Close only the local Firefox kiosk so mouse users can reach GNOME."""
    if not _host_ok(request) or not _origin_ok(request):
        raise HTTPException(403, "bad_origin")
    if not _session_ok(request):
        raise HTTPException(403, "local_operator_required")
    gate = _proxy_http("POST", "kiosk/exit-ready", request, b"")
    if gate.status_code != 200:
        raise HTTPException(gate.status_code, "kiosk_exit_unsafe")
    pids = _kiosk_pids()
    if len(pids) != 1:
        raise HTTPException(409, "kiosk_not_unique")
    try:
        os.kill(pids[0], signal.SIGTERM)
    except ProcessLookupError:
        raise HTTPException(409, "kiosk_already_closed") from None
    return {"closing": True}


def _unit_state(unit: str) -> str:
    try:
        r = subprocess.run(
            ["systemctl", "is-active", f"{unit}.service"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@app.get("/local/diag")
def local_diag(request: Request):
    if not _host_ok(request):
        raise HTTPException(403, "bad_host")
    services = {u: _unit_state(u) for u in _DIAG_UNITS}
    containers = []
    try:
        r = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}} {{.Status}}"],
            capture_output=True, text=True, timeout=10,
        )
        containers = [l for l in r.stdout.splitlines() if l.startswith("kotoba")]
    except Exception:
        pass
    asr = {"reachable": False}
    try:
        with urllib.request.urlopen(f"{API_URL}/api/asr/status", timeout=3) as r:
            asr = json.loads(r.read().decode())
    except Exception:
        pass
    render = {"reachable": False}
    try:
        with urllib.request.urlopen(f"{API_URL}/api/render/status", timeout=3) as r:
            render = json.loads(r.read().decode())
    except Exception:
        pass
    api = {"ok": False}
    try:
        with urllib.request.urlopen(f"{API_URL}/api/health", timeout=3) as r:
            api = {"ok": True, **json.loads(r.read().decode())}
    except Exception:
        pass
    return {
        "operator": _session_ok(request),
        "services": services,
        "containers": containers,
        "api": api,
        "asr": asr,
        "render": render,
        "ts": time.time(),
    }


@app.post("/local/service/restart")
async def service_restart(request: Request):
    if not _host_ok(request) or not _origin_ok(request):
        raise HTTPException(403, "bad_origin")
    if not _session_ok(request):
        raise HTTPException(403, "local_operator_required")
    body = await request.json()
    unit = str(body.get("unit", ""))
    if unit not in _RESTARTABLE:
        raise HTTPException(400, "unknown_unit")
    # sudoersは /bin/systemctl (re)start kotoba-* のみ許可済み
    r = subprocess.run(
        ["sudo", "-n", "systemctl", "restart", f"{unit}.service"],
        capture_output=True, text=True, timeout=60,
    )
    ok = r.returncode == 0
    return {"restarted": ok, "unit": unit,
            "state": _unit_state(unit),
            "error": (r.stderr or "")[-200:] if not ok else None}


@app.get("/diag", response_class=HTMLResponse)
def diag_page():
    return _DIAG_HTML


_DIAG_HTML = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ことばでロボコン — 診断・安全復旧</title>
<style>
body{font-family:system-ui,sans-serif;background:#101418;color:#e8edf2;margin:0;padding:24px;max-width:720px}
h1{font-size:20px} h2{font-size:15px;margin-top:24px}
table{border-collapse:collapse;width:100%} td,th{border:1px solid #2a323c;padding:6px 10px;font-size:13px;text-align:left}
.ok{color:#69c46d}.bad{color:#e06c75}.warn{color:#d9a53d}
button{background:#2a4a8c;color:#fff;border:0;border-radius:6px;padding:8px 14px;font-size:14px;cursor:pointer}
button:disabled{opacity:.4} pre{background:#0a0d10;padding:8px;font-size:12px;overflow:auto}
.back{display:inline-block;background:#2a4a8c;color:#fff;border-radius:6px;padding:12px 18px;
text-decoration:none;font-weight:700}
.nav{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:8px 0 16px}
#exitKiosk{padding:12px 18px;background:#71502f}
</style></head><body>
<h1>ことばでロボコン — 診断・安全復旧</h1>
<div class="nav"><a class="back" href="/">展示画面へ戻る</a>
<button id="exitKiosk" onclick="exitKiosk()" disabled>展示画面を閉じてデスクトップへ戻る</button></div>
<div id="op"></div>
<h2>サービス</h2><table id="svc"></table>
<h2>コンテナ</h2><pre id="ctr">-</pre>
<h2>API / ASR / 描画</h2><pre id="api">-</pre>
<h2>復旧操作（この端末の操作者のみ）</h2>
<div id="ops"></div>
<h2>画面とOS設定</h2>
<p>ネットワークやBluetoothの設定を開くときは、先にラウンドを終了してください。</p>
<p style="font-size:12px;color:#8a94a0">自動更新5s。この画面はlocalhostのみ有効です。
全体を止めたいときはOSから電源操作してください。ロボットの通常停止はメイン画面のSTOPです。</p>
<script>
async function j(u,o){const r=await fetch(u,o);return r.json().catch(()=>({}))}
async function refresh(){
  const d=await j('/local/diag');
  document.getElementById('exitKiosk').disabled=!d.operator;
  document.getElementById('op').innerHTML = d.operator
    ? '<span class="ok">● この端末の操作者として認識中</span> <button onclick="endOp()">操作権を終了</button>'
    : '<button onclick="startOp()">この端末で操作を開始</button>';
  let rows='<tr><th>unit</th><th>状態</th></tr>';
  for(const [k,v] of Object.entries(d.services||{}))
    rows+=`<tr><td>${k}</td><td class="${v==='active'?'ok':'bad'}">${v}</td></tr>`;
  document.getElementById('svc').innerHTML=rows;
  document.getElementById('ctr').textContent=(d.containers||[]).join('\\n')||'(なし)';
  document.getElementById('api').textContent=
    'api: '+JSON.stringify(d.api)+'\\nasr: '+JSON.stringify(d.asr)+'\\nrender: '+JSON.stringify(d.render);
  const ops=document.getElementById('ops');
  ops.innerHTML='';
  const lbl={'kotoba-sim-boot':'シミュレーション世界全体を再起動','kotoba-asr':'ASR(音声認識)を再起動','kotoba-ollama':'LLM(文章解釈)を再起動','kotoba-render':'3D描画を再起動','kotoba-api':'APIを再起動','kotoba-local-entry':'この画面を再起動'};
  for(const u of ['kotoba-sim-boot','kotoba-asr','kotoba-ollama','kotoba-render','kotoba-api','kotoba-local-entry']){
    const b=document.createElement('button');
    b.textContent=lbl[u]||u; b.style.marginRight='8px';
    b.disabled=!d.operator;
    b.onclick=async()=>{b.disabled=true;const r=await j('/local/service/restart',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({unit:u})});alert(u+': '+(r.restarted?'再起動しました':'失敗 '+(r.error||'')));refresh()};
    ops.appendChild(b);
  }
}
async function startOp(){await j('/local/session/start',{method:'POST'});refresh()}
async function endOp(){await j('/local/session/end',{method:'POST'});refresh()}
async function exitKiosk(){
  if(!confirm('展示画面を閉じてデスクトップへ戻ります。ラウンドが終了していることを確認しましたか？'))return;
  const r=await fetch('/local/kiosk/exit',{method:'POST',keepalive:true});
  if(!r.ok)alert('画面を閉じられませんでした。状態を確認してください。');
}
refresh(); setInterval(refresh,5000);
</script></body></html>"""


# ---- static UI（KIOSK_DIST — API:8700と同じ生成物を配信） --------------------

@app.get("/")
def index():
    index_html = KIOSK_DIST / "index.html"
    if index_html.exists():
        return FileResponse(index_html)
    raise HTTPException(503, "kiosk not built")


@app.get("/{path:path}")
def static_files(path: str):
    if path.startswith(("api/", "local/")):
        raise HTTPException(404, "not_found")
    dist = KIOSK_DIST.resolve()
    target = (dist / path).resolve()
    if target.is_relative_to(dist) and target.is_file():
        return FileResponse(target)
    raise HTTPException(404, "not_found")


def main() -> int:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=LISTEN_PORT, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
