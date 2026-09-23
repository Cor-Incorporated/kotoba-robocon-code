"""決定論ゲームパーサの仕様固定 — 語彙→有界step列・未知語はNone。

v2: moveは観測閉ループ実行のstep列（{"type":"move","steps":[...]}）。
旧dur固定nudge（fwd/lat/yaw直書き）は回帰経路として残すが、parserは
stepsを出す。全文消費型 — 未解釈の数値・単位・残り文字は裸の方向へ
縮退させない（unclear/reject）。
"""

import pytest

from kotoba_harness.control import parse_command
from kotoba_harness.gameparse import describe, parse, parse_detailed, from_intent

LEASE = "l1"


def _accept(cmd):
    """parser出力がcontrol.pyの検査を通ることを保証する。"""
    import time

    return parse_command(
        {"lease_id": LEASE, "seq": 1, "issued_wall": time.time(), "cmd": cmd},
        LEASE,
    )


def _steps(text):
    cmd = parse(text)
    assert cmd["type"] == "move"
    return cmd["steps"]


@pytest.mark.parametrize(
    "text,dirn",
    [("前", "forward"), ("進んで", "forward"),
     ("まっすぐ進んで", "forward"),
     ("後ろ", "back"), ("バック", "back"), ("後退", "back"),
     ("行き過ぎ、後ろ", "back")],
)
def test_move_fwd_back_jog(text, dirn):
    # 裸の方向指示は継続jog（止まってまで走る）— 勝手に0.5mで止めない
    (st,) = _steps(text)
    assert st["action"] == "jog" and st["dir"] == dirn and st["m"] == 0.0
    _accept(parse(text))


@pytest.mark.parametrize(
    "text,dirn",
    [("左", "left"), ("ひだり", "left"),
     ("右", "right"), ("みぎ", "right")],
)
def test_move_lateral_jog(text, dirn):
    (st,) = _steps(text)
    assert st["action"] == "jog" and st["dir"] == dirn
    _accept(parse(text))


@pytest.mark.parametrize(
    "text,dirn,m",
    [("少し左", "left", 0.25), ("少し右", "right", 0.25),
     ("少し右にずれて", "right", 0.25), ("0.25m前", "forward", 0.25)],
)
def test_move_lateral_bounded(text, dirn, m):
    # 「少し」は有限bounded移動のまま（jogへは昇格しない）
    (st,) = _steps(text)
    assert st["action"] == "translate" and st["dir"] == dirn
    assert st["m"] == m
    _accept(parse(text))


def test_jog_inside_compound_is_unclear():
    # 継続動作を含む複合は意味が曖昧 — 静かに半分実行せず聞き返す
    for t in ("右に90°して前へ", "右に90度向いてから前へ", "前へ、右に45度"):
        v = parse_detailed(t)
        assert v["kind"] == "unclear", t
        assert parse(t) is None, t
    # 有限stepのみの複合は従来どおり受理
    v = parse_detailed("右に90度向いてから0.5m前へ")
    assert v["kind"] == "cmd" and len(v["cmd"]["steps"]) == 2


@pytest.mark.parametrize(
    "text,dirn,deg",
    [("左を向いて", "left", 45.0), ("左回転", "left", 45.0),
     ("右を向いて", "right", 45.0), ("右回転", "right", 45.0),
     ("右に90°向いて", "right", 90.0), ("左に90度向いて", "left", 90.0),
     ("右に15度", "right", 15.0), ("左に30度回って", "left", 30.0),
     ("後ろを向いて", "around", 180.0), ("振り向いて", "around", 180.0),
     ("90°右", "right", 90.0)],
)
def test_turn(text, dirn, deg):
    (st,) = _steps(text)
    assert st["action"] == "turn" and st["dir"] == dirn
    assert st["deg"] == deg
    _accept(parse(text))


def test_turn_beats_lateral():
    # 「右を向いて」は「右」(横歩行)より旋回が優先される
    assert _steps("右を向いて")[0]["action"] == "turn"
    assert _steps("左を向いて")[0]["action"] == "turn"


def test_turn_around_not_back_translate():
    # 「後ろを向いて」は後退ではなく180°旋回（反例: 振り向きの取り違え）
    (st,) = _steps("後ろを向いて")
    assert st["action"] == "turn" and st["dir"] == "around" and st["deg"] == 180.0
    # 「後ろに下がって」「バック」は後退jog（止まってまで継続）
    for t in ("後ろに下がって", "バック", "後退"):
        (s,) = _steps(t)
        assert s["action"] == "jog" and s["dir"] == "back"


@pytest.mark.parametrize(
    "text", ["止まって", "ストップ", "止まれ", "停止して"],
)
def test_stop(text):
    assert parse(text)["type"] == "stop"


@pytest.mark.parametrize(
    "text", ["打って", "叩いて", "スイカを割って", "振って", "割れ", "スイカ割り"],
)
def test_strike(text):
    assert parse(text)["type"] == "strike"


@pytest.mark.parametrize(
    "text",
    [
        # C3レビュー反例: 否定・禁止・能力質問・依頼は打撃にしない
        "割ってはいけない", "打ってはいけない", "打つな", "割るな",
        "進んではいけない", "行くな", "触らないで", "止まらないで",
        "スイカを見せて", "スイカはどこですか", "スイカを叩けますか",
        "打てるかな", "どうやって割るの", "「割って」と言った",
    ],
)
def test_negation_question_no_actuation(text):
    assert parse(text) is None


def test_strike_requires_explicit_hit_verb():
    # 「スイカを」単独の言及は打撃語ではない
    assert parse("スイカを") is None
    assert parse("スイカを見せて") is None
    # 明示的な打撃動詞は有効
    assert parse("スイカを割って")["type"] == "strike"
    assert parse("スイカ割り")["type"] == "strike"


@pytest.mark.parametrize(
    "text", ["終了", "やめて", "終わって", "終われ"],
)
def test_end(text):
    assert parse(text)["type"] == "end"


@pytest.mark.parametrize(
    "text", ["", "ジャンプして", "トルクを出して", "関節を動かして",
             "前か後ろか分からない", "何がいい？"],
)
def test_unknown_returns_none(text):
    assert parse(text) is None


def test_no_torque_or_joint_words_accepted():
    # 安全語彙外の低レベル操作語は解釈不能 → None（聞き返しへ）
    for t in ("トルク10で", "joint command", "関節角を直接"):
        assert parse(t) is None


def test_move_has_bounded_duration():
    # 全てのmoveは有限の外側期限（dur_s）— 無期限の連続指令ではない。
    # steps形式は観測閉ループで逐次実行するためstepあたり25sの上限
    cmd = parse("前")
    assert cmd["type"] == "move" and 0 < cmd["dur_s"] <= 80.0
    _accept(cmd)


def test_slightly_is_smaller_target_not_same():
    # 「少し左」は有限0.25m移動、「左」は継続jog — 同じ指令ではない（反例T05）
    (a,), (b,) = _steps("少し左"), _steps("左")
    assert a["dir"] == b["dir"] == "left"
    assert a["action"] == "translate" and a["m"] == 0.25
    assert b["action"] == "jog"


def test_describe_all_types():
    assert describe(parse("前")) == "前進（「止まって」まで継続）"
    assert describe(parse("後ろ")) == "後退（「止まって」まで継続）"
    assert describe(parse("左")) == "左へ（「止まって」まで継続）"
    assert describe(parse("右")) == "右へ（「止まって」まで継続）"
    assert describe(parse("左を向いて")) == "左に45°向く"
    assert describe(parse("止まって")) == "止まる"
    assert describe(parse("打って")) == "打つ"
    assert describe(parse("終了")) == "終了する"


def test_describe_marks_slightly():
    assert describe(parse("少し左")).startswith("左へ0.25m")


# ---- 順序付き複合・停止条件・範囲外値の規則 -------------------------------

def test_compound_ordered():
    # 「右に90°旋回して1m歩く」は2stepの順序付き計画 — 旋回が先
    steps = _steps("右に90°旋回して1m歩いて")
    assert [s["action"] for s in steps] == ["turn", "translate"]
    assert steps[0]["dir"] == "right" and steps[0]["deg"] == 90.0
    assert steps[1]["dir"] == "forward" and steps[1]["m"] == 1.0
    _accept(parse("右に90°旋回して1m歩いて"))


def test_compound_connectors():
    for t in ("右に90度向いてから1m歩いて", "右に45度回ってから左に30度",
              "左に45度回転して、1m前に歩いて"):
        steps = _steps(t)
        assert len(steps) == 2


def test_stop_after_goal():
    # 「1m歩いて止まって」は到達後の停止条件 — 即時停止へ退化しない
    cmd = parse("1m歩いて止まって")
    assert cmd["type"] == "move" and cmd["stop_after"] is True
    assert cmd["steps"][0]["m"] == 1.0
    cmd = parse("前に0.5m進んで止まって")
    assert cmd["type"] == "move" and cmd["stop_after"] is True


def test_stop_then_more_is_stop_only():
    # STOPの後の内容は自動実行しない（後半を捨てて即時停止のみ）
    for t in ("止まって、そのあと右へ", "ストップ、やっぱり前へ"):
        v = parse_detailed(t)
        assert v["kind"] == "cmd" and v["cmd"]["type"] == "stop"
        assert v.get("truncated") is True


def test_cancel_mid_input_resets_steps():
    # 「やっぱり止まって」は前言撤回 — 溜まったstepを破棄して停止のみ
    v = parse_detailed("前に進んで、やっぱり止まって")
    assert v["kind"] == "cmd" and v["cmd"]["type"] == "stop"


def test_out_of_allowlist_rejected_not_degraded():
    # 範囲外の数値・単位・速度語は裸方向へ縮退しない
    for t in ("前に999m", "右999度", "右に速さ無限で", "右へ100m",
              "右に90メートル回転"):
        assert parse_detailed(t)["kind"] == "unclear", t
        assert parse(t) is None, t


def test_full_width_and_units():
    # 全角数字・°・cm を正規化して解釈
    assert _steps("右に９０度")[0]["deg"] == 90.0
    assert _steps("100センチ前")[0]["m"] == 1.0
    assert _steps("90°右！")[0]["action"] == "turn"


def test_ambiguous_angle_unclear():
    # 「90度か180度か分からない」は勝手に選ばない
    assert parse_detailed("90度か180度か分からない")["kind"] == "unclear"


def test_mixed_strike_not_half_executed():
    # 移動+打撃の複合は半分だけ実行しない
    assert parse_detailed("前に1m歩いて割って")["kind"] == "unclear"


def test_marker_words_unclear_in_game():
    # ゲームにはマーカーが無い — 言及は即時停止へ退化させず不明瞭へ
    assert parse_detailed("奥のマーカーまで行って止まって")["kind"] == "unclear"


def test_stop_request_vs_prohibit():
    # 裸の「進まないで」「動かないで」は現行運動の通常停止
    for t in ("進まないで", "動かないで"):
        v = parse_detailed(t)
        assert v["kind"] == "cmd" and v["cmd"]["type"] == "stop"
    # 方向付き禁止は禁止のまま（特定動作の禁止であって停止要求ではない）
    for t in ("前に進まないで", "右に進まないで"):
        assert parse_detailed(t) == {"kind": "reject", "why": "prohibit"}


# ---- parse_detailed: 3-way判定（cmd / reject / unclear） ---------------

@pytest.mark.parametrize(
    "text,kind",
    [
        ("前", "cmd"), ("少し左", "cmd"), ("止まって", "cmd"),
        ("終わって", "cmd"), ("割って", "cmd"), ("進まないで", "cmd"),
        # 明示禁止・質問・引用 → 決定論的拒否（LLMへ回して復活させない）
        ("割るな", "reject"), ("動くな", "reject"), ("前に進まないで", "reject"),
        ("前に進むの？", "reject"), ("スイカはどこ", "reject"),
        ("「割って」と言った", "reject"), ("", "reject"),
        # 語彙外・曖昧 → 要追加解釈
        ("ふわふわ動かして", "unclear"), ("なんかよくわからん", "unclear"),
        ("前か後ろか分からない", "unclear"),
    ],
)
def test_parse_detailed_three_way(text, kind):
    """明示禁止=reject・語彙外=unclearを分離 — rejectがLLMへ回らないよう区別。"""
    assert parse_detailed(text)["kind"] == kind


def test_parse_detailed_reject_is_not_llm_target():
    """rejectとunclearが同じNoneにつぶれない（レビュー反例P03）。"""
    v = parse_detailed("割るな")
    assert v["kind"] == "reject" and v["why"] == "prohibit"
    assert parse_detailed("ふわふわ")["kind"] == "unclear"
    # 旧来のparse()互換 — reject/unclearは両方None
    assert parse("割るな") is None and parse("ふわふわ") is None


def test_reject_why_distinguishes_prohibit_from_nonop():
    """禁止・取消し（prohibit）と非操作入力（nonop）を区別する。

    prohibitは受理待ちの旧意図を失効させるクラス、nonopは世代を
    変えないクラス（P04: 後からの禁止で未適用の旧意図を失効）。
    """
    # 明示禁止 → prohibit
    for t in ("打つな", "割るな", "動いてはいけない", "右へ行かないで"):
        assert parse_detailed(t) == {"kind": "reject", "why": "prohibit"}
    # 取消し語彙 → prohibit
    for t in ("キャンセル", "取り消し", "今のを取り消して"):
        assert parse_detailed(t)["why"] == "prohibit"
    # 質問・引用・空 → nonop（世代は変えない）
    for t in ("どこに行くの？", "「前」と言った", "", "進み方を教えて"):
        assert parse_detailed(t) == {"kind": "reject", "why": "nonop"}


# ---- from_intent: LLM semantic intent → 有界cmd（数値は固定表のみ） ----

def test_from_intent_move_uses_bounded_table():
    cmd = from_intent({"type": "move", "direction": "forward", "size": "normal"})
    assert cmd["type"] == "move"
    (st,) = cmd["steps"]
    assert st["action"] == "translate" and st["dir"] == "forward" and st["m"] == 0.5
    small = from_intent({"type": "move", "direction": "left", "size": "small"})
    assert small["steps"][0]["m"] == 0.25
    turn = from_intent({"type": "move", "direction": "turn_right", "size": "normal"})
    assert turn["steps"][0]["action"] == "turn" and turn["steps"][0]["dir"] == "right"
    _accept(cmd); _accept(small); _accept(turn)


def test_from_intent_amount_keys():
    # 明示量キー（LLMは列挙値のみ、実数値は固定表から）
    cmd = from_intent(
        {"type": "move", "direction": "turn_left", "amount_key": "deg_90"})
    assert cmd["steps"][0]["deg"] == 90.0
    cmd = from_intent(
        {"type": "move", "direction": "forward", "amount_key": "m_1"})
    assert cmd["steps"][0]["m"] == 1.0
    cmd = from_intent(
        {"type": "move", "direction": "turn_around"})
    assert cmd["steps"][0]["dir"] == "around" and cmd["steps"][0]["deg"] == 180.0


def test_from_intent_program_ordered():
    cmd = from_intent({"type": "program", "steps": [
        {"direction": "turn_right", "amount_key": "deg_90"},
        {"direction": "forward", "amount_key": "m_1"},
    ]})
    assert cmd["type"] == "move" and len(cmd["steps"]) == 2
    assert cmd["steps"][0]["deg"] == 90.0
    assert cmd["steps"][1]["dir"] == "forward" and cmd["steps"][1]["m"] == 1.0
    _accept(cmd)


def test_from_intent_oneway_types():
    for t in ("stop", "end", "strike"):
        assert from_intent({"type": t})["type"] == t


def test_from_intent_rejects_unbounded_input():
    # LLMが数値・未知type・未知方向・未知量キーを出しても発行しない
    assert from_intent({"type": "move", "direction": "forward", "size": "huge"}) is None
    assert from_intent({"type": "move", "direction": "forward",
                        "amount_key": "deg_999"}) is None
    assert from_intent({"type": "move", "direction": "forward",
                        "amount_key": "deg_90"}) is None  # 移動に角度キーは不整合
    assert from_intent({"type": "move", "direction": "diagonal"}) is None
    assert from_intent({"type": "teleport"}) is None
    assert from_intent({"type": "move"}) is None
    assert from_intent("move forward") is None
    assert from_intent({"type": "unclear"}) is None
    # programの検査 — 量超過・未知値・空は発行しない
    assert from_intent({"type": "program", "steps": []}) is None
    assert from_intent({"type": "program", "steps": [
        {"direction": "forward", "amount_key": "m_1"},
        {"direction": "forward", "amount_key": "m_1"},
        {"direction": "forward", "amount_key": "m_1"},
    ]}) is None  # 合計3m > 2.0m上限
    assert from_intent({"type": "program", "steps": [
        {"direction": "hyperjump", "amount_key": "m_1"}]}) is None
