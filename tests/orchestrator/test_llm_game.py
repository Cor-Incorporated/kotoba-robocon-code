"""interpret_game: LLM出力の厳格検証 — schema外・数値混入は拒否。"""

import pytest

from kotoba_api import llm as llm_mod
from kotoba_api.llm import LlmError, interpret_game


def _fake(raw):
    return lambda *a, **k: raw


def test_valid_intents_pass(monkeypatch):
    monkeypatch.setattr(
        llm_mod, "_chat", _fake({"type": "move", "direction": "forward", "size": "small"})
    )
    assert interpret_game("少し前") == {"type": "move", "direction": "forward", "size": "small"}
    monkeypatch.setattr(llm_mod, "_chat", _fake({"type": "strike"}))
    assert interpret_game("割って") == {"type": "strike"}
    monkeypatch.setattr(llm_mod, "_chat", _fake({"type": "unclear"}))
    assert interpret_game("なんだろ") == {"type": "unclear"}


def test_move_without_direction_rejected(monkeypatch):
    monkeypatch.setattr(llm_mod, "_chat", _fake({"type": "move"}))
    with pytest.raises(LlmError):
        interpret_game("動いて")


def test_unknown_type_rejected(monkeypatch):
    monkeypatch.setattr(llm_mod, "_chat", _fake({"type": "jump"}))
    with pytest.raises(LlmError):
        interpret_game("跳んで")


def test_extra_fields_rejected(monkeypatch):
    # 数値・余計なkeyを混入させても発行しない（速度生成の抑止）
    monkeypatch.setattr(
        llm_mod, "_chat",
        _fake({"type": "move", "direction": "forward", "speed": 9.9}),
    )
    with pytest.raises(LlmError):
        interpret_game("速く")


def test_bad_enum_values_rejected(monkeypatch):
    monkeypatch.setattr(
        llm_mod, "_chat", _fake({"type": "move", "direction": "diagonal"})
    )
    with pytest.raises(LlmError):
        interpret_game("斜め")
    monkeypatch.setattr(
        llm_mod, "_chat", _fake({"type": "move", "direction": "forward", "size": "huge"})
    )
    with pytest.raises(LlmError):
        interpret_game("めっちゃ")


def test_non_dict_payload_rejected(monkeypatch):
    monkeypatch.setattr(llm_mod, "_chat", _fake(["move", "forward"]))
    with pytest.raises(LlmError):
        interpret_game("前")
