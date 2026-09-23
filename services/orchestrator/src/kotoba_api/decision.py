"""LocalJev互換 typed-decision adapter（R5-06 — shadow専用）。

L0の決定論parser・STOPの前段に置かず、辞書外の運動表現とASR候補の
順位付けを「評価記録のみ」で行う。actuationへの接続は段階2の限定
SELFTESTまで行わない — ここでの応答は制御経路を変更しない。

契約:
- wire: POST {model, state: <JSON文字列>, questions: {...}} —
  githubnext/localjev@3f23e36 のREADME形式に倣う（応答envelopeは
  未検証のため寛容にparseし、rawを記録する）。
- deadline: 1要求 1500ms。超過は破棄。malformed retryなし。
- queue: inflight 1 + 最新候補1件のみ。新しい入力は旧候補を取消す
  （古い指令を溜めない — epochで採用時再検査する設計の前提）。
- 数値・速度・角度をmodelへ生成させない — 候補programはコードが作り、
  modelはChoice（どれが発話の意味か）とNoul（指示か/曖昧か）だけを返す。
- 候補にない解釈・unsupported_runの表現を消してwalkへ置換しない。

有効化: KOTOBA_DECISION_URL（例 http://127.0.0.1:11436 等の隔離port）。
未設定なら shadow 自体を呼ばない（辞書のみ動作・no-op）。
"""

import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

DEADLINE_S = 1.5
MAX_CANDIDATES = 4
MODEL = os.environ.get("KOTOBA_DECISION_MODEL", "jev-local")
_shadow_log = os.environ.get(
    "KOTOBA_DECISION_SHADOW_LOG", ""
)


class DecisionClient:
    """単一inflight＋最新候補1件のshadow呼出し。決して制御を待たせない。"""

    def __init__(self, url: str | None = None, deadline_s: float = DEADLINE_S):
        self.url = url or os.environ.get("KOTOBA_DECISION_URL") or ""
        self.deadline_s = deadline_s
        self._mu = threading.Lock()
        self._pending = None  # (state, questions, meta) — 最新1件のみ
        self._inflight = False

    def enabled(self) -> bool:
        return bool(self.url)

    def submit_shadow(self, text: str, candidates: list, meta: dict) -> bool:
        """評価用に非同期dispatch。受理したらTrue（無効/混雑はFalse）。

        meta: session_id/round_id/run_id/control_epoch/input_seq/
              parser_why 等の所有識別子 — 応答の世代照合に使う。
        """
        if not self.enabled():
            return False
        with self._mu:
            if self._inflight:
                # inflight中は最新1件だけを待機として保持（積まない）
                self._pending = (text, candidates, meta)
                return True
            self._inflight = True
        threading.Thread(
            target=self._run, args=(text, candidates, meta), daemon=True
        ).start()
        return True

    def _run(self, text, candidates, meta):
        try:
            self._call_and_log(text, candidates, meta)
        finally:
            nxt = None
            with self._mu:
                if self._pending is not None:
                    nxt = self._pending
                    self._pending = None
                else:
                    self._inflight = False
            if nxt is not None:
                self._run(*nxt)

    def _call_and_log(self, text, candidates, meta):
        state = {
            "utterance": text,
            "candidates": {
                f"c{i}": c.get("label") or c.get("action_key") or "?"
                for i, c in enumerate(candidates)
            },
            "capabilities": {
                "walk": True, "fast_walk": True, "run": False,
                "turn": True, "translate": True,
            },
        }
        questions = {
            "candidate_program": {
                "type": "choice",
                "instructions": (
                    "utteranceが要求する動作に一致するcandidatesのIDを選ぶ。"
                    "否定・質問・雑談はnone。対応不能な要求はunsupported。"
                ),
                "criteria": {
                    **{
                        f"c{i}": c.get("label") or "?"
                        for i, c in enumerate(candidates)
                    },
                    "none": "どれでもない・情報不足",
                    "unsupported": "ロボットの能力に無い要求",
                },
            },
            "is_command": {
                "type": "noul",
                "instructions": (
                    "utteranceはロボットに今実行してほしい動作の指示か。"
                    "引用・否定・質問・雑談は指示ではない。"
                ),
            },
            "needs_clarification": {
                "type": "noul",
                "instructions": (
                    "左右・量・参照先が不足していて聞き返しが必要か。"
                ),
            },
        }
        body = json.dumps(
            {
                "model": MODEL,
                "state": json.dumps(state, ensure_ascii=False),
                "questions": questions,
            }
        ).encode("utf-8")
        rec = {
            "t_wall": time.time(),
            "request": {"text": text, "meta": meta, "state": state},
            "response": None,
            "error": None,
            "latency_ms": None,
        }
        t0 = time.monotonic()
        try:
            req = urllib.request.Request(
                self.url,
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.deadline_s) as resp:
                rec["response"] = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            rec["error"] = f"unreachable:{type(exc).__name__}"
        except json.JSONDecodeError:
            rec["error"] = "invalid_json"
        rec["latency_ms"] = round((time.monotonic() - t0) * 1000, 1)
        _append_shadow_log(rec)


def _append_shadow_log(rec: dict) -> None:
    path = _shadow_log
    if not path:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


_CLIENT: DecisionClient | None = None
_CLIENT_MU = threading.Lock()


def decision_client() -> DecisionClient:
    global _CLIENT
    with _CLIENT_MU:
        if _CLIENT is None:
            _CLIENT = DecisionClient()
        return _CLIENT


def candidates_for_unclear(text: str) -> list:
    """辞書外入力に対する候補programをコードで生成する（評価用）。

    modelへ意味を生成させないため、検出できた方向語・動作語から
    少数のcandidateを作る。作れない文は空列 — 呼び出し側は聞き返し。
    """
    cands = []
    has_back = any(w in text for w in ("後ろ", "うしろ", "バック"))
    has_turnish = any(
        w in text for w in ("向い", "振り", "ふり", "回っ", "まわっ", "旋回")
    )
    has_moveish = any(
        w in text for w in ("進ん", "すすん", "歩い", "あるい", "行っ", "いっ")
    )
    if has_back and has_turnish:
        cands.append(
            {
                "action_key": "turn:around",
                "label": "その場で180度旋回（後ろを向く）",
            }
        )
        cands.append({"action_key": "jog:back", "label": "後ろへ継続移動"})
    elif has_back:
        cands.append({"action_key": "jog:back", "label": "後ろへ継続移動"})
        cands.append(
            {"action_key": "turn:around", "label": "その場で180度旋回"}
        )
    if has_moveish and not cands:
        cands.append({"action_key": "jog:forward", "label": "前へ継続移動"})
    return cands[:MAX_CANDIDATES]
