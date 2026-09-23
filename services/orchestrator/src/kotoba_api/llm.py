"""ローカルLLM（展示専用Ollama）との意味解釈インターフェース。

- 専用インスタンス: KOTOBA_OLLAMA_HOST 環境変数（既定 127.0.0.1:11435）。業務Ollama(11434)へは触れない
- Ollama structured outputs で intent.schema.json へ生成を制約（format=schema）
- think=false（qwen3の思考モードが予算を消費するため）
- 無効出力の補修はしない（検証失敗は reject として扱う）
"""

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from kotoba_contracts.intent import parse_intent

DEFAULT_HOST = "http://127.0.0.1:11435"
DEFAULT_MODEL = "qwen3:4b"
TIMEOUT_S = 40.0
_INTENT_SCHEMA = None


def _intent_schema() -> dict:
    global _INTENT_SCHEMA
    if _INTENT_SCHEMA is None:
        path = Path(__file__).resolve().parent / "intent.schema.json"
        _INTENT_SCHEMA = json.loads(path.read_text(encoding="utf-8"))
    return _INTENT_SCHEMA


@dataclass(frozen=True)
class TargetInfo:
    target_id: str
    label: str
    distance_m: float


def _system_prompt(targets, clarify_context: str | None) -> str:
    tdesc = "\n".join(
        f"- id: {t.target_id} / ラベル: {t.label} / 前進距離: {t.distance_m}m"
        for t in targets
    )
    ctx = ""
    if clarify_context:
        ctx = (
            f"\n直前の聞き返し: 「{clarify_context}」\n"
            "利用者の今回の入力は、この聞き返しへの回答です。回答が特定の目的地を指せば execute してください。\n"
        )
    return f"""あなたは展示イベント「ことばでロボコン」の意図解釈器です。
来場者の日本語の指示を、指定されたJSONスキーマに従うJSONへ変換してください。

利用できる目的地（これ以外は存在しない）:
{tdesc}

判定ルール:
- ラベルやその言い換え（例: 「手前」「奥」「近い方」「遠い方」「{targets[0].label}」）が一つに特定できる → decision=execute
- 「手前」は近い方（距離が短い方）、「奥」は遠い方（距離が長い方）を意味する
- 本当に目的地を特定できない場合だけ decision=clarify（質問文は「手前と奥のどちらに進みますか？」のような形）
- 存在しない場所・対応外の動作（回転・ジャンプ・物を持つ等）・矛盾した指示 → decision=reject
- 「〜ではなく」「やっぱり」「キャンセルして〜にして」等の訂正・否定・言い換えを正しく扱う
- 座標・速度・秒数を作らない。目的地のidだけを返す
- execute の explanation は確認画面に見せる短い日本語の要約（例: 「手前のマーカーまで前進します」）
- JSON以外の出力は禁止{ctx}"""


_GAME_SCHEMA = None
_GAME_TYPES = ("move", "stop", "end", "strike", "program", "unclear")
_GAME_DIRS = ("forward", "back", "left", "right",
              "turn_left", "turn_right", "turn_around")
_GAME_SIZES = ("normal", "small")
# LLMが出してよい量キー（実数値はgameparseの固定表から供給 — LLM生成禁止）
_GAME_AMOUNT_KEYS = (
    "m_025", "m_05", "m_1", "m_15", "m_2",
    "deg_15", "deg_30", "deg_45", "deg_90", "deg_180",
)
_GAME_STEP_DIRS = _GAME_DIRS


def _game_schema() -> dict:
    global _GAME_SCHEMA
    if _GAME_SCHEMA is None:
        path = Path(__file__).resolve().parent / "game_cmd.schema.json"
        _GAME_SCHEMA = json.loads(path.read_text(encoding="utf-8"))
    return _GAME_SCHEMA


_GAME_SYSTEM = """あなたは「ことばでスイカ割り」ゲームの意図解釈器です。
来場者の日本語を、指定JSONスキーマの指令へ変換してください。

指令type:
- move: 移動（direction=forward/back/left/right/turn_left/turn_right,
  size=normal または small=「少し」「ちょっと」等の小幅指示）
- stop: 移動の中止（「止まって」「ストップ」「待って」）
- strike: 一回だけ腕を振ってスイカを打つ（「割って」「打って」「叩いて」）
- end: ラウンド自体を終える（「終わって」「やめて」「もういい」）
- program: 順序付きの複合動作（「右に90°回ってから1m歩いて」等）。
  steps: [{direction, amount_key}] を順序どおりに（最大3）。
  directionは moveと同じ列挙値。amount_keyは距離 m_025/m_05/m_1/m_15/m_2、
  旋回は deg_15/deg_30/deg_45/deg_90/deg_180 のみ。
- unclear: 上記に確定できない・否定形・質問・雑談・複数候補で曖昧

厳格ルール:
- 否定・禁止（「進まないで」「打たないで」）・質問・能力確認は unclear
- 速度・距離・秒数・座標を数値で出力しない
  （direction/size/amount_keyの列挙値のみ）
- 「後ろを向いて」「振り向いて」は turn_around（後退移動ではない）
- 曖昧なら unclear（推測で動かさない）
- JSON以外の出力は禁止"""


def _chat(text: str, system: str, schema: dict, host, model) -> dict:
    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "think": False,
            "format": schema,
            "options": {"temperature": 0.1, "num_predict": 400},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LlmError(f"llm_unreachable:{type(exc).__name__}") from None
    content = payload.get("message", {}).get("content", "")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise LlmError("llm_invalid_json") from None


def interpret_game(
    text: str,
    host: str | None = None,
    model: str | None = None,
) -> dict | None:
    """自由文→ゲームsemantic intent dict。解釈不能は {"type":"unclear"}。

    LLMは型・方向・大小の列挙値のみを返す。数値は gameparse.from_intent
    が固定表から供給するため、LLMは速度・時間を生成できない。
    """
    host = host or os.environ.get("KOTOBA_OLLAMA_HOST", DEFAULT_HOST)
    model = model or DEFAULT_MODEL
    raw = _chat(text, _GAME_SYSTEM, _game_schema(), host, model)
    if not isinstance(raw, dict):
        raise LlmError("llm_schema_violation")
    t = raw.get("type")
    if t not in _GAME_TYPES:
        raise LlmError("llm_schema_violation")
    if t == "program":
        # 順序付き複合 — stepsは列挙値のみ（最大3、direction+amount_key）
        if set(raw) - {"type", "steps"}:
            raise LlmError("llm_schema_violation")
        steps = raw.get("steps")
        if not isinstance(steps, list) or not (1 <= len(steps) <= 3):
            raise LlmError("llm_schema_violation")
        for st in steps:
            if not isinstance(st, dict) or set(st) - {"direction", "amount_key"}:
                raise LlmError("llm_schema_violation")
            if st.get("direction") not in _GAME_STEP_DIRS:
                raise LlmError("llm_schema_violation")
            if st.get("amount_key") not in _GAME_AMOUNT_KEYS + _GAME_SIZES:
                raise LlmError("llm_schema_violation")
        return raw
    if set(raw) - {"type", "direction", "size", "amount_key"}:
        raise LlmError("llm_schema_violation")
    if "direction" in raw and raw["direction"] not in _GAME_DIRS:
        raise LlmError("llm_schema_violation")
    if "size" in raw and raw["size"] not in _GAME_SIZES:
        raise LlmError("llm_schema_violation")
    if "amount_key" in raw and raw["amount_key"] not in (
        _GAME_AMOUNT_KEYS + _GAME_SIZES
    ):
        raise LlmError("llm_schema_violation")
    if t == "move" and raw.get("direction") not in _GAME_DIRS:
        raise LlmError("llm_schema_violation")
    return raw


def interpret(
    text: str,
    targets,
    clarify_context: str | None = None,
    host: str | None = None,
    model: str | None = None,
) -> dict:
    """自由文 → 検証済みintent dict。失敗時は LlmError。"""
    host = host or os.environ.get("KOTOBA_OLLAMA_HOST", DEFAULT_HOST)
    model = model or DEFAULT_MODEL
    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "think": False,
            "format": _intent_schema(),
            "options": {"temperature": 0.1, "num_predict": 400},
            "messages": [
                {"role": "system", "content": _system_prompt(targets, clarify_context)},
                {"role": "user", "content": text},
            ],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LlmError(f"llm_unreachable:{type(exc).__name__}") from None
    content = payload.get("message", {}).get("content", "")
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        raise LlmError("llm_invalid_json") from None
    try:
        intent = parse_intent(raw)
    except Exception:
        raise LlmError("llm_schema_violation") from None
    return intent.model_dump()


class LlmError(Exception):
    pass
