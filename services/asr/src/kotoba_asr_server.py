#!/usr/bin/env python3
"""kotoba ローカルASRサービス — Qwen3-ASRをPCM16/16kHz発話区間へ適用する転写専用サーバ。

設計境界（R6 P1-D）:
- 転写のみ。指令化・実行・motor経路は一切持たない。
- モデルは起動時にローカル固定パスから読む（KOTOBA_ASR_MODEL）。
  ダウンロード済みsnapshotへの絶対パスを要求し、起動時に実在検査する。
- 推論は直列（単一worker・ロック1本）— GPUメモリと遅延の再現性を優先。
- 応答: {"text": str, "latency_ms": float, "engine": str}
- /health: モデル読込完了後のみ ok — 起動中は reachable だが ready=False。

起動（Thor上・隔離venv）:
    KOTOBA_ASR_MODEL=/home/cloudia/kotoba/models/qwen3-asr-0.6b \
    /home/cloudia/kotoba/venv-asr/bin/python kotoba_asr_server.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = os.environ.get("KOTOBA_ASR_HOST", "127.0.0.1")
PORT = int(os.environ.get("KOTOBA_ASR_PORT", "8710"))
MODEL_PATH = os.environ.get("KOTOBA_ASR_MODEL", "")
MAX_UTT_BYTES = 600 * 1024

_model = None
_model_lock = threading.Lock()
_model_ready = threading.Event()
_model_error: str | None = None
_model_device: str | None = None  # 実際にロードしたdevice（"cuda:0"/"cpu"）
_model_device_name: str | None = None
ENGINE = "qwen3-asr"


def _load_model() -> None:
    """モデルを起動時に読む。失敗は /health で可視化し、プロセスは残す。"""
    global _model, _model_error, _model_device, _model_device_name
    try:
        if not MODEL_PATH or not Path(MODEL_PATH).is_dir():
            raise RuntimeError(f"KOTOBA_ASR_MODEL not a dir: {MODEL_PATH!r}")
        for required in ("config.json",):
            if not (Path(MODEL_PATH) / required).exists():
                raise RuntimeError(f"model dir missing {required}: {MODEL_PATH}")
        t0 = time.monotonic()
        # qwen-asr パッケージの正式API（transformers経路 — vLLM不要）。
        # 実API: Qwen3ASRModel.from_pretrained + transcribe(audio=(ndarray, sr))
        # → List[ASRTranscription](.text/.language)。from_pretrainedは
        # ローカルdirを受理する。
        import torch
        from qwen_asr import Qwen3ASRModel  # type: ignore

        _model_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _model_device_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
        _model = Qwen3ASRModel.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            device_map=_model_device,
            max_inference_batch_size=4,   # 発話単発用途 — OOM回避で小さく
            max_new_tokens=64,            # 号令は短文 — 生成を早く打ち切る
        )
        dt = time.monotonic() - t0
        print(f"[asr] model loaded path={MODEL_PATH} load_s={dt:.1f}", flush=True)
        _model_ready.set()
    except Exception as exc:  # noqa: BLE001 — 起動失敗をhealthで見せる
        _model_error = f"{type(exc).__name__}: {exc}"
        print(f"[asr] model load FAILED {_model_error}", flush=True)


def _transcribe(pcm: bytes) -> dict:
    """PCM16 LE 16kHz mono → テキスト。推論は直列化。"""
    import numpy as np

    if len(pcm) > MAX_UTT_BYTES or len(pcm) % 2:
        raise ValueError("bad_pcm16")
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size < 1600:  # 100ms未満
        return {"text": "", "latency_ms": 0.0, "engine": ENGINE}
    with _model_lock:  # 推論直列化 — queue深度はHTTP層のclient側が制御
        t0 = time.monotonic()
        # 実API: transcribe(audio=(ndarray, sr)) → List[ASRTranscription]。
        # 言語は日本語固定 — auto判定の揺らぎとextra token生成を避ける。
        res = _model.transcribe((audio, 16000), language="Japanese")
        dt = (time.monotonic() - t0) * 1000.0
    text = ""
    if isinstance(res, (list, tuple)) and res:
        first = res[0]
        text = str(getattr(first, "text", "") or "")
    return {"text": text.strip(), "latency_ms": round(dt, 1), "engine": ENGINE}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        print(f"[asr] {fmt % args}", flush=True)

    def _json(self, code: int, obj: dict) -> None:
        b = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json(
                200,
                {
                    "ok": _model_ready.is_set(),
                    "engine": ENGINE,
                    "model": os.path.basename(MODEL_PATH.rstrip("/")) or None,
                    # GPU→CPU低下をUIが「検収済み構成」と誤表示しないよう
                    # 実ロードdeviceを明示する（R7 1.2）。
                    "device": _model_device,
                    "device_name": _model_device_name,
                    "error": _model_error,
                },
            )
            return
        self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/recognize":
            self._json(404, {"error": "not_found"})
            return
        if not _model_ready.is_set():
            self._json(503, {"error": "model_not_ready", "detail": _model_error})
            return
        try:
            n = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            self._json(411, {"error": "bad_length"})
            return
        if n <= 0 or n > MAX_UTT_BYTES:
            self._json(413 if n > MAX_UTT_BYTES else 422, {"error": "bad_pcm16"})
            return
        pcm = self.rfile.read(n)
        try:
            self._json(200, _transcribe(pcm))
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": f"asr_failed:{type(exc).__name__}"})


def main() -> int:
    threading.Thread(target=_load_model, daemon=True).start()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[asr] listening http://{HOST}:{PORT} model={MODEL_PATH}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
