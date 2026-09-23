"""スイカ割り対応A4: renderer実状態statusの回帰試験。

実入口（実FastAPI TestClient + 実RUNTIME_DIRファイル）で検証:
- render_status.json 不在 → unreachable（白viewerをreadyにしない）
- ready + live boot一致 → reachable + boot_match
- ready + boot不一致 → boot_match=False（旧boot姿勢を新bootとして出さない）
- status tsが古い → reachable=False + stale
- loading_model / syncing の中間状態をそのまま返す
"""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_app(tmp_path, monkeypatch):
    monkeypatch.setenv("KOTOBA_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("KOTOBA_HOME", str(tmp_path))
    monkeypatch.setenv("KOTOBA_OFFLINE", "0")
    monkeypatch.setenv("KOTOBA_SELFTEST", "1")
    import kotoba_api.app as app_mod

    return importlib.reload(app_mod)


def _write_live(tmp_path, boot_nonce):
    (tmp_path / "live.json").write_text(json.dumps({
        "boot_nonce": boot_nonce,
        "obs_wall": time.time(),
        "wall": time.time(),
        "pos": [0.0, 0.0, 0.82],
        "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "step_seq": 10,
    }))


def _write_status(tmp_path, **kw):
    st = {"state": "ready", "instance_id": "i1", "boot_nonce": 111,
          "model_nq": 31, "api_ok": True, "markers_stale": False,
          "pose_age_s": 0.1, "ts": time.time()}
    st.update(kw)
    (tmp_path / "render_status.json").write_text(json.dumps(st))


def test_status_missing_is_unreachable(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    app_mod = _load_app(tmp_path, monkeypatch)
    _write_live(tmp_path, 111)
    r = TestClient(app_mod.app).get("/api/render/status")
    assert r.status_code == 200
    d = r.json()
    assert d["reachable"] is False and d["state"] == "unreachable"
    assert d["boot_match"] is False


def test_status_ready_and_boot_match(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    app_mod = _load_app(tmp_path, monkeypatch)
    _write_live(tmp_path, 111)
    _write_status(tmp_path, boot_nonce=111)
    d = TestClient(app_mod.app).get("/api/render/status").json()
    assert d["reachable"] is True and d["state"] == "ready"
    assert d["boot_match"] is True and d["instance_id"] == "i1"
    assert d["model_nq"] == 31


def test_status_ready_old_boot_is_rejected(tmp_path, monkeypatch):
    """rendererがready報告でもboot不一致 → boot_match=False（UIはreadyにしない）。"""
    from fastapi.testclient import TestClient

    app_mod = _load_app(tmp_path, monkeypatch)
    _write_live(tmp_path, 222)          # 新しいboot
    _write_status(tmp_path, boot_nonce=111)  # rendererの適用poseは旧boot
    d = TestClient(app_mod.app).get("/api/render/status").json()
    assert d["reachable"] is True and d["state"] == "ready"
    assert d["boot_match"] is False


def test_status_stale_ts_is_unreachable(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    app_mod = _load_app(tmp_path, monkeypatch)
    _write_live(tmp_path, 111)
    _write_status(tmp_path, ts=time.time() - 10)
    d = TestClient(app_mod.app).get("/api/render/status").json()
    assert d["reachable"] is False and d["state"] == "stale"


def test_status_intermediate_states_passthrough(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    app_mod = _load_app(tmp_path, monkeypatch)
    _write_live(tmp_path, 111)
    for st in ("loading_model", "syncing"):
        _write_status(tmp_path, state=st, boot_nonce=None)
        d = TestClient(app_mod.app).get("/api/render/status").json()
        assert d["reachable"] is True and d["state"] == st
        assert d["boot_match"] is False


def test_display_targets_no_fallback_needed(tmp_path, monkeypatch):
    """A3: live.json が無くてもdisplay APIが自前計算せずmarkers空を返す
    （rendererはmarker非表示/staleのまま — 現在poseから目標を捏造しない）。"""
    from fastapi.testclient import TestClient

    app_mod = _load_app(tmp_path, monkeypatch)
    # live.json を書かない → anchor/yaw 取得不可
    d = TestClient(app_mod.app).get("/api/display/targets").json()
    assert d["markers"] == {} and d["anchor"] is None
