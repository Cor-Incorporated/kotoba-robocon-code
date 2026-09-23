"""ゲーム指令の構造化parser（改善版 v2）— 参加者のことば→有界step列。

v1は部分一致で「右に90°」を右横移動へ落とし得た。v2はトークン化して
全文を消費した構文として検証する — 解釈できない数値・単位・残り文字を
裸の方向へ縮退させない。

出力cmd（move）: {"type":"move","steps":[MotionStep相当のdict...],
"dur_s":外側期限,"action_key","label"}。stepsを持つmoveはcontrollerの
MotionExecutorが観測閉ループで実行する（旧dur固定nudgeと区別）。

語彙（決定論・LLM不使用の厳格経路）:
  移動: 前/進む/歩く/行く/ずれる/下がる/バック/後退
  旋回: 向く/回る/旋回/回転/振り向く/振り返る/向き直る
  走行: 走る/駆け足（pace=run — 能力未検収時は実行層で拒否）
  速歩: 速く/早く歩く、早歩き/速歩（pace=fast_walk — 前進のみ）
  停止: 止まって/止まれ/ストップ/停止/止めて
  終了: 終了/終わって/やめて/中止
  打撃: 割って/打って/叩いて/振って
  修飾: 少し/ちょっと
  量:   1m/100cm/90度/90°/110°（全角数字・単位を正規化、
        漢数字合成 百十=110、かな数詞 ひゃくじゅう=110 — 位取り詞を
        必須として助詞「に」等との誤認を防ぐ）
  角度範囲: 1.0〜180.0°（0.1°刻み — 列挙値制限は廃止。範囲外は拒否、
        丸めて別値へ変えない）
  順序: 「〜て」「〜てから」「その後」「そして」

禁止・取消しは受理待ちの旧意図を失効するクラス(prohibit)、質問・引用は
世代を変えない非操作入力(nonop) — 既存の分離を維持する。
実移動中の「進まないで」「動かないで」は現行運動の通常停止を出す
（新たな移動を出さず、受理待ち世代も失効させる）。
"""

from __future__ import annotations

import re
import unicodedata

# ---- 量の範囲（サーバーprofile由来の数値。clampしない — 範囲外は拒否）
ALLOWED_DIST_M = (0.25, 0.5, 1.0, 1.5, 2.0)
# 角度は列挙値を廃止し連続範囲で検証する（R5 — 「右に110°」等）。
# 入力解像度0.1°は表現の精度であり実角度精度の約束ではない。
ANGLE_MIN_DEG = 1.0
ANGLE_MAX_DEG = 180.0
ANGLE_STEP_DEG = 0.1
DEFAULT_DIST_M = 0.5        # 量なし移動の初期profile値
DEFAULT_TURN_DEG = 45.0     # 量なし旋回の初期profile値
SMALL_DIST_M = 0.25         # 「少し」の移動
SMALL_TURN_DEG = 15.0       # 「少し」の旋回
MAX_STEPS = 3
MAX_TOTAL_DIST_M = 2.0

# 歩容（pace）— 「走る」は能力検収と別に認識される（実行可否は
# サーバーのcapability判定。語彙不足と能力未検収を別reasonで返す）。
PACES = ("walk", "fast_walk", "run")
# fast_walk/run は前進profileのみ — 横・後退・旋回への一律増幅はしない
PACE_FORWARD_ONLY = ("fast_walk", "run")

# stepの外側期限（輸送鮮度ではなく実行の絶対上限 — controller側でも検査）
STEP_DEADLINE_S = 25.0

# ---- 否定・禁止・取消し・質問・引用（既存防御を維持）
_NEGATION = (
    "はいけない", "てはいけ", "ないで", "なくていい", "禁止",
    "打つな", "うつな", "割るな", "わるな", "進むな", "すすむな",
    "行くな", "いくな", "触るな", "さわるな", "振るな", "ふるな",
    "動くな", "うごくな", "叩くな", "たたくな", "撃つな",
)
_CANCEL = ("キャンセル", "取り消し", "取り消して", "とりけし", "取消")
# 実移動中の停止要求 — 禁止と区別し、現行運動の通常停止を出す
_STOP_REQUEST = ("進まないで", "すすまないで", "動かないで", "うごかないで")
_QUESTION = (
    "ですか", "ますか", "かな", "？", "?", "見せて", "みせて",
    "教えて", "おしえて", "どうやって", "どうすれば", "どこ", "何が",
    "なぜ", "どうして",
)

# ---- トークン規則（長い語を先に）
_TOK_SPEC = [
    # 順序・接続（先に結合詞）
    ("CONN", r"(てから|してから|その後|そのあと|それから|そして|から|、|。|．|！|!)"),
    ("RESET", r"(やっぱり|やはり|じゃなくて|じゃない|いや、|いや|違う)"),
    # 停止
    ("STOP", r"(止まって|とまって|止まれ|とまれ|ストップ|停止|止めて|とめて)"),
    # 終了（「終わって」が「割って」を含むため先に）
    ("END", r"(終了|終わって|おわって|終わり|おわり|終われ|おわれ|やめて|やめる|中止)"),
    # 打撃
    ("STRIKE", r"(打って|うって|打て|うて|打ちな|たたいて|叩いて|たたけ|叩け|"
               r"振って|ふって|割って|わって|撃って|うってみて|割れ|われ|"
               r"スイカ割り|すいか割り|スイカ割|すいか割|たたき|割る|打つ|叩く)"),
    # 修飾（速さは pace 修飾 — 移動動詞の前に置く）
    ("MOD", r"(ほんの少し|少し|すこし|ちょっと|ちょい)"),
    ("FAST", r"(速く|はやく|早く|速めに|早めに|速め|早め)"),
    # 助詞・依頼（fillerとして消費する）— NUMより先に置く:
    # 「みぎにひゃくじゅうど」の「に」をかな数詞の先頭桁（にひゃく=210）
    # へ誤吸収させない。数詞文字とFILLER語のprefix衝突は無い。
    ("FILLER", r"(お願い|おねがい|ください|下さい|は|を|に|へ|が|の|で|と|"
               r"です|だ|よ|ね|かな|まで|ぐらい|くらい|位|ほど|だけ|します|して|しまって|"
               r"スイカ|すいか|西瓜|棒|ぼう|目標|的)"),
    # 数値（アラビア + 合成漢数字 + 位取り詞を含むかな数詞）
    ("NUM", r"([0-9０-９]+(?:[\.．][0-9０-９]+)?|[一二三四五六七八九〇十百千]+|"
            r"(?:(?:きゅう|はち|なな|ろく|ご|よん|し|さん|に|いち)?"
            r"(?:ひゃく|びゃく|ぴゃく|せん|ぜん|じゅう))+"
            r"(?:きゅう|はち|なな|ろく|ご|よん|し|さん|に|いち)?)"),
    # 単位（「ど」は「ひゃくじゅうど」のような読みの濁音表記）
    ("UNIT_DEG", r"(度|°|ど)"),
    ("UNIT_M", r"(メートル|ｍ|m)"),
    ("UNIT_CM", r"(センチメートル|センチ|ｃｍ|cm)"),
    # 訂正マーカー（「行き過ぎ」は内容を変えずskip — 後続の方向が効く）
    ("OVERSHOOT", r"(行き過ぎ|いきすぎ|行きすぎ|過ぎ|すぎ)"),
    # 方向
    ("DIR", r"(真っ直ぐ|まっすぐ|前方|前|後ろ|うしろ|後方|左|ひだり|右|みぎ)"),
    # 動詞（旋回系を先に — 「向いて」は移動より先に消費）
    ("V_TURN", r"(向き直って|向き直り|振り向いて|ふりむいて|振り向き|ふりむき|"
               r"振り返って|ふりかえって|振り返り|ふりかえり|振り返る|ふりかえる|"
               r"振り向く|ふりむく|"
               r"向いて|むいて|向いて|向く|むく|向け|回って|まわって|回り|まわり|"
               r"回る|まわる|回れ|旋回して|旋回|回転して|回転)"),
    # 走行動詞（pace=run）— 方向は前進のみ。walk動詞と分けて pace を
    # 意味値として保持する（「走る」は jog の別名ではなく歩容要求）
    ("V_RUN", r"(駆け足で|かけあしで|駆け足|かけあし|駆けて|かけて|"
              r"走って|はしって|走り|はしり|走れ|はしれ|走る|はしる|ランして|ラン)"),
    # 速歩動詞（pace=fast_walk の語そのもの — 「早歩きして」等）
    ("V_FAST", r"(早歩きして|はやあるきして|早歩き|はやあるき|"
               r"速歩きして|速歩き|速歩して|速歩|はやあし)"),
    ("V_MOVE", r"(歩いて|あるいて|歩く|あるく|歩け|進んで|すすんで|進む|すすむ|"
               r"進め|すすめ|進み|行って|いって|行く|いく|行き|"
               r"ずれて|ずれろ|ずれる|移動して|移動|動いて|うごいて|動き|進|行|歩|動け|"
               r"下がって|さがって|下がれ|さがれ|下がり|バック|後退|"
               r"戻って|もどって|戻り|戻れ|ゴー|go)"),
    # 目標指定（マーカー回帰用の名詞 — ゲームでは未対応としてunclear）
    ("MARKER", r"(マーカー|まーかー)"),
    ("NEAR", r"(手前|てまえ|近く|ちかく)"),
    ("FAR", r"(奥|おく|遠く|とおく)"),
    ("SPACE", r"[\s　]+"),
]
_TOK_RE = re.compile("|".join(f"(?P<{n}>{p})" for n, p in _TOK_SPEC))

# 漢数字の合成規則（位取り式）— 列挙表ではなく構造で読む。
# 百十=110・四十五=45・百八十=180 のように位取り詞が数字を掛ける。
_KDIGIT = {
    "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_KPLACE = {"十": 10, "百": 100, "千": 1000}
# かな数詞 — NUM正規表現が位取り詞（じゅう/ひゃく系/せん系）を必須に
# しているため、助詞の「に」「し」等と衝突しない。
_KANA_MORPH = {
    "いち": 1, "に": 2, "さん": 3, "し": 4, "よん": 4, "ご": 5,
    "ろく": 6, "なな": 7, "はち": 8, "きゅう": 9,
    "じゅう": 10, "ひゃく": 100, "びゃく": 100, "ぴゃく": 100,
    "せん": 1000, "ぜん": 1000,
}
_KANA_MORPH_RE = re.compile(
    "ひゃく|びゃく|ぴゃく|せん|ぜん|じゅう|きゅう|はち|なな|ろく|"
    "ご|よん|さん|し|に|いち"
)


def _jp_num(s: str):
    """合成漢数字/かな数詞 → 数値。解析不能・空・位取り誤りは None。"""
    if not s:
        return None
    if not all(ch in _KDIGIT or ch in _KPLACE for ch in s):
        # かな数詞 — 語素列へ分解して同じ位取り規則で評価
        morphs = _KANA_MORPH_RE.findall(s)
        if not morphs or "".join(morphs) != s:
            return None
        vals = [_KANA_MORPH[m] for m in morphs]
        total, section, digit = 0, 0, None
        for v in vals:
            if v >= 10:
                section += (digit if digit is not None else 1) * v
                digit = None
            else:
                digit = v
        return float(total + section + (digit or 0))
    total, section, digit = 0, 0, None
    for ch in s:
        if ch in _KDIGIT:
            digit = _KDIGIT[ch]
        elif ch == "千":
            total += (digit if digit is not None else 1) * 1000
            section = 0
            digit = None
        else:
            section += (digit if digit is not None else 1) * _KPLACE[ch]
            digit = None
    return float(total + section + (digit or 0))
# 動詞自体が方向を含意する語群（裸動詞の既定方向解決用）
_VERB_BACK = ("下が", "さが", "バック", "後退", "戻っ", "もどっ", "戻り", "戻れ")
_VERB_FWD = ("進", "すす", "歩", "ある", "行っ", "いっ", "行く", "いく", "行き", "ゴー", "go")
_VERB_TURN_AROUND = ("振り向", "ふりむ", "向き直", "振り返", "ふりかえ")

_DIR_MAP = {
    "前": "forward", "前方": "forward", "まっすぐ": "forward",
    "真っ直ぐ": "forward",
    "後ろ": "back", "うしろ": "back", "後方": "back",
    "左": "left", "ひだり": "left",
    "右": "right", "みぎ": "right",
}


def _num_val(raw: str):
    """NUMトークン→数値。アラビア数字・合成漢数字・かな数詞を受理。
    非有限・負号・NaNは数値として受理しない（None→拒否経路）。"""
    import math as _m
    v = _jp_num(raw)
    if v is not None:
        return v
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v if _m.isfinite(v) else None


def _deg_ok(deg: float) -> bool:
    """明示角度の範囲契約: 1.0〜180.0°・0.1°刻み。
    180超をπへ飽和させず、0.1°未満の刻み外も受理しない。"""
    if not (ANGLE_MIN_DEG - 1e-9 <= deg <= ANGLE_MAX_DEG + 1e-9):
        return False
    tenths = deg / ANGLE_STEP_DEG
    return abs(tenths - round(tenths)) < 1e-6


def _normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    return t.strip()


def _tokenize(t: str):
    """全文をトークン列へ。認識不能な文字は UNKNOWN トークンとして残す
    （落として構文を成立させない — 全文消費検査で拾う）。"""
    toks = []
    i = 0
    while i < len(t):
        m = _TOK_RE.match(t, i)
        if m is None:
            # 未認識の1文字をUNKNOWNとして残す
            toks.append(("UNKNOWN", t[i]))
            i += 1
            continue
        kind = m.lastgroup
        val = m.group()
        i = m.end()
        if kind in ("SPACE", "FILLER"):
            continue
        toks.append((kind, val))
    return toks


_PACE_LABEL = {"walk": "", "fast_walk": "速歩", "run": "走行"}


def _label_for(step: dict) -> str:
    a, d = step["action"], step["dir"]
    pace = _PACE_LABEL.get(step.get("pace", "walk"), "")
    pace_sfx = f"（{pace}）" if pace else ""
    if a == "jog":
        base = {
            "forward": "前進", "back": "後退", "left": "左へ", "right": "右へ"
        }[d]
        return f"{base}{pace_sfx}（「止まって」まで継続）"
    if a == "translate":
        base = {
            "forward": "前へ", "back": "後ろへ", "left": "左へ", "right": "右へ"
        }[d]
        return f"{base}{step['m']:g}m進む{pace_sfx}"
    name = {"left": "左", "right": "右", "around": "後ろ"}[d]
    return f"{name}に{step['deg']:g}°向く"


# jog（継続移動）の指令側外側期限 — executor側 jog_deadline_s と整合。
# dur_sは停止確認の上限時間であり、事前の終了時刻としては表示しない。
JOG_DUR_S = 30.0


def _pace_ok(direction, pace):
    """fast_walk/run は前進profileのみ — 横・後退・旋回を一律増幅しない。"""
    if pace == "walk":
        return True
    return direction == "forward"


def _make_jog(direction, pace="walk"):
    """量なしの方向指示 → 継続移動step（明示停止・境界・期限で止まる）。"""
    if not _pace_ok(direction, pace):
        return None, "pace_dir"
    step = {"action": "jog", "dir": direction, "m": 0.0, "deg": None,
            "pace": pace}
    step["action_key"] = f"jog_{direction}" + (
        f"_{pace}" if pace != "walk" else "")
    step["label"] = _label_for(step)
    return step, None


def _make_step(action, direction, value, small=False, pace="walk"):
    if not _pace_ok(direction, pace):
        return None, "pace_dir"
    if action == "translate":
        m = SMALL_DIST_M if small else (value if value is not None else DEFAULT_DIST_M)
        if m not in ALLOWED_DIST_M and value is not None:
            return None, "bad_dist"
        if m not in ALLOWED_DIST_M:
            return None, "bad_dist"
        step = {"action": "translate", "dir": direction, "m": m}
    else:
        deg = SMALL_TURN_DEG if small else (
            value if value is not None else DEFAULT_TURN_DEG
        )
        if direction == "around":
            deg = value if value is not None else 180.0
        if not _deg_ok(deg):
            return None, "bad_angle"
        deg = round(deg, 1)
        step = {"action": "turn", "dir": direction, "deg": deg}
    if pace != "walk":
        step["pace"] = pace
    step["action_key"] = f"{action}_{direction}_{step.get('m', step.get('deg')):g}" + (
        f"_{pace}" if pace != "walk" else "")
    step["label"] = _label_for(step)
    return step, None


def _parse_tokens(toks):
    """トークン列→step列。完全消費のみ受理。

    戻り値: (steps, stop_after, err) — errは失敗理由（None=成功）。
    stop_after: 終端が明示停止で締まるか（「1m歩いて止まって」型）。
    """
    steps = []
    stop_after = False
    i = 0
    n = len(toks)
    while i < n:
        kind, val = toks[i]
        if kind in ("CONN", "OVERSHOOT"):
            i += 1
            continue
        if kind == "RESET":
            # 「やっぱり」は前言撤回 — 溜まったstepを捨てて後半を読む
            steps = []
            i += 1
            continue
        if kind == "STOP":
            # 末尾の停止 = 計画の終端条件（「歩いて止まって」）
            # 中間の停止で後続がある = 即時停止のみ（後半を自動実行しない）
            if i == n - 1 or all(k == "CONN" for k, _ in toks[i + 1:]):
                stop_after = True
                i += 1
                continue
            return steps, "stop_interrupt", None
        if kind == "END":
            if i == n - 1 or all(k == "CONN" for k, _ in toks[i + 1:]):
                return steps, "end", None
            return steps, "end_interrupt", None
        if kind == "STRIKE":
            return steps, "strike", None
        # 修飾（MOD=小幅、FAST=速歩pace）— 方向/動詞の前に置く副詞。
        small = False
        pace = "walk"
        while i < n and toks[i][0] in ("MOD", "FAST"):
            if toks[i][0] == "MOD":
                if small:
                    return None, None, "dangling_mod"
                small = True
            else:
                if pace != "walk":
                    return None, None, "dangling_mod"
                pace = "fast_walk"
            i += 1
            if i >= n:
                return None, None, "dangling_mod"
        kind, val = toks[i]
        # NUM UNIT DIR? V_TURN? — 「90°右」「1m前」「右に90度」等
        if kind == "NUM":
            num = _num_val(val)
            if num is None:
                return None, None, "bad_num"
            i += 1
            unit = None
            if i < n and toks[i][0] in ("UNIT_DEG", "UNIT_M", "UNIT_CM"):
                unit = toks[i][0]
                i += 1
            # 方向
            direction = None
            if i < n and toks[i][0] == "DIR":
                direction = _DIR_MAP[toks[i][1]]
                i += 1
            # 動詞（任意だが残れば消費 — pace語は走行要求へ繋げる）
            vkind = toks[i][0] if i < n else None
            if vkind in ("V_TURN", "V_MOVE", "V_RUN", "V_FAST"):
                i += 1
            elif vkind is not None and vkind not in ("CONN",):
                return None, None, "trailing_after_num"
            if vkind == "V_RUN":
                pace = "run"
            elif vkind == "V_FAST":
                pace = "fast_walk"
            if unit == "UNIT_DEG":
                if pace != "walk":
                    return None, None, "pace_turn"
                if not _deg_ok(num):
                    return None, None, "bad_angle"
                d = direction if direction in ("left", "right") else (
                    "around" if num == 180 else "right"
                )
                step, err = _make_step("turn", d, num)
                if err:
                    return None, None, err
                steps.append(step)
            elif unit in ("UNIT_M", "UNIT_CM"):
                m = num if unit == "UNIT_M" else num / 100.0
                if m not in ALLOWED_DIST_M:
                    return None, None, "bad_dist"
                d = direction or "forward"
                if d in ("left", "right") and vkind == "V_TURN":
                    return None, None, "dist_with_turn"
                step, err = _make_step("translate", d, m, pace=pace)
                if err:
                    return None, None, err
                steps.append(step)
            else:
                # 単位なしの数値 — 解釈不能（裸方向へ縮退しない）
                return None, None, "unitless_num"
            continue
        if kind == "DIR":
            direction = _DIR_MAP[val]
            i += 1
            # DIR NUM UNIT? — 「右に90度」「前に1m」
            num = None
            unit = None
            if i < n and toks[i][0] == "NUM":
                num = _num_val(toks[i][1])
                if num is None:
                    return None, None, "bad_num"
                i += 1
                if i < n and toks[i][0] in ("UNIT_DEG", "UNIT_M", "UNIT_CM"):
                    unit = toks[i][0]
                    i += 1
                else:
                    return None, None, "unitless_num"
            vkind = toks[i][0] if i < n else None
            if vkind in ("V_TURN", "V_MOVE", "V_RUN", "V_FAST"):
                i += 1
            if vkind == "V_RUN":
                pace = "run"
            elif vkind == "V_FAST":
                pace = "fast_walk"
            elif vkind is not None and vkind not in (
                "V_TURN", "V_MOVE", "CONN", "STOP", "END", "STRIKE",
            ):
                # 方向の後の不明トークンは裸方向へ畳まない
                return None, None, f"trailing_after_dir:{vkind}"
            # 意味決定:
            #  - 「後ろを向く」→ turn 180（around）
            #  - 「後ろに下がる/バック」→ translate back
            #  - 角度指定あり → turn
            #  - 距離指定あり → translate
            #  - V_TURN → turn（量なし=45°、少し=15°、後ろ=180°）
            #  - V_RUN → pace=run / V_FAST・速く+V_MOVE → fast_walk
            #  - V_MOVE/裸DIR → translate or jog（後ろ=back、左右=側方）
            if unit == "UNIT_DEG":
                if pace != "walk":
                    return None, None, "pace_turn"
                if not _deg_ok(num):
                    return None, None, "bad_angle"
                if direction in ("left", "right"):
                    step, err = _make_step("turn", direction, num)
                elif direction == "back" or num == 180:
                    step, err = _make_step("turn", "around", num)
                else:
                    return None, None, "forward_turn"
                if err:
                    return None, None, err
                steps.append(step)
            elif unit in ("UNIT_M", "UNIT_CM"):
                if vkind == "V_TURN":
                    return None, None, "dist_with_turn"
                m = num if unit == "UNIT_M" else num / 100.0
                if m not in ALLOWED_DIST_M:
                    return None, None, "bad_dist"
                step, err = _make_step("translate", direction, m,
                                       small=False, pace=pace)
                if err:
                    return None, None, err
                steps.append(step)
            if unit is None and num is None:
                if vkind == "V_TURN":
                    if direction == "back":
                        step, err = _make_step("turn", "around", None)
                    else:
                        step, err = _make_step("turn", direction, None, small)
                    if err:
                        return None, None, err
                    steps.append(step)
                elif small or pace != "walk":
                    if small:
                        # 「少し右」等 → 有限の微調整（0.25m）
                        step, err = _make_step("translate", direction, None,
                                               small, pace)
                    else:
                        # pace付き量なし移動（「前へ走って」等）→ 継続
                        step, err = _make_jog(direction, pace)
                    if err:
                        return None, None, err
                    steps.append(step)
                else:
                    # 量なしの方向指示（「前」「右へ」「進んで」等）→ 継続移動。
                    # 0.5m暗黙stepへ縮退しない — 「止まって」まで進み続ける
                    # のが利用者の意図（R4）。
                    step, err = _make_jog(direction)
                    if err:
                        return None, None, err
                    steps.append(step)
            continue
        if kind in ("V_RUN", "V_FAST"):
            # 裸の走行/速歩動詞 — 方向未指定は前進。「走って1m」のように
            # 後続の距離指定を伴い得る。
            vpace = "run" if kind == "V_RUN" else "fast_walk"
            i += 1
            num = None
            unit = None
            if i < n and toks[i][0] == "NUM":
                num = _num_val(toks[i][1])
                if num is None:
                    return None, None, "bad_num"
                i += 1
                if i < n and toks[i][0] in ("UNIT_DEG", "UNIT_M", "UNIT_CM"):
                    unit = toks[i][0]
                    i += 1
                else:
                    return None, None, "unitless_num"
            if unit is not None:
                if unit == "UNIT_DEG":
                    return None, None, "pace_turn"
                m = num if unit == "UNIT_M" else num / 100.0
                if m not in ALLOWED_DIST_M:
                    return None, None, "bad_dist"
                step, err = _make_step("translate", "forward", m,
                                       pace=vpace)
                if err:
                    return None, None, err
                steps.append(step)
                continue
            step, err = _make_jog("forward", vpace)
            if err:
                return None, None, err
            steps.append(step)
            continue
        if kind == "V_TURN":
            # 振り向く/向き直るは単独で180°、それ以外の裸旋回は方向不明
            if any(val.startswith(w) or val.startswith(w.replace("ふ","フ"))
                   for w in _VERB_TURN_AROUND):
                step, err = _make_step("turn", "around", 180.0)
                if err:
                    return None, None, err
                steps.append(step)
                i += 1
                continue
            return None, None, "turn_no_dir"
        if kind == "V_MOVE":
            # 方向を含意する裸動詞は既定方向へ、含意しないものはunclear。
            # 少し付き=有限step、量なし=継続jog（上のDIR分岐と同じ意味論）
            if any(val.startswith(w) for w in _VERB_BACK):
                d = "back"
            elif any(val.startswith(w) for w in _VERB_FWD):
                d = "forward"
            else:
                return None, None, "move_no_dir"
            if small:
                step, err = _make_step("translate", d, None, small, pace)
            else:
                step, err = _make_jog(d, pace)
            if err:
                return None, None, err
            steps.append(step)
            i += 1
            continue
        if kind in ("MARKER", "NEAR", "FAR"):
            return None, None, "marker_word"
        return None, None, f"unconsumed:{kind}"
    return steps, stop_after, None


def parse_detailed(text: str) -> dict:
    """参加者テキスト→3-way判定（+step列）。

    {"kind":"cmd","cmd":{...}}          — 発行してよい指令
    {"kind":"reject","why":"prohibit"}  — 明示禁止・取消し（旧意図を失効）
    {"kind":"reject","why":"nonop"}     — 質問・引用・空（世代不変）
    {"kind":"unclear","why":...}        — 語彙外・曖昧・範囲外（LLM経路候補）
    """
    t = _normalize(text)
    if not t:
        return {"kind": "reject", "why": "nonop"}
    # 実移動中の停止要求（裸の「進まないで」「動かないで」）は通常停止を
    # 出す — 「前に進まないで」のような方向付き禁止は _NEGATION の
    # 禁止経路へ残すため、残りテキストの意味内容を確認する。
    for g in _STOP_REQUEST:
        if g in t:
            rest = (t[:t.index(g)] + t[t.index(g) + len(g):])
            rtoks = [k for k, _ in _tokenize(rest)
                     if k not in ("CONN", "SPACE")]
            if not rtoks:
                return {
                    "kind": "cmd",
                    "cmd": {"type": "stop", "action_key": "stop_request",
                            "label": "止まる"},
                    "stop_only": True,
                }
            break
    # 禁止・取消し（質問と混在する入力は禁止側へ）
    for g in _NEGATION + _CANCEL:
        if g in t:
            return {"kind": "reject", "why": "prohibit"}
    for g in _QUESTION:
        if g in t:
            return {"kind": "reject", "why": "nonop"}
    if "「" in t or "」" in t or "'" in t or '"' in t:
        return {"kind": "reject", "why": "nonop"}

    toks = _tokenize(t)
    if not toks:
        return {"kind": "reject", "why": "nonop"}
    if any(k == "UNKNOWN" for k, _ in toks):
        return {"kind": "unclear", "why": "unknown_tokens"}
    steps, tail, err = _parse_tokens(toks)
    if err is not None:
        return {"kind": "unclear", "why": err}
    # 継続jogは単独指令のみ — 「右90°して前へ」のような継続終端の複合は
    # 未対応を明示する（途中で止まらないstepを列の途中に置かない）。
    if any(s["action"] == "jog" for s in steps) and (
        len(steps) > 1 or tail in ("end", "end_interrupt")
    ):
        return {"kind": "unclear", "why": "continuous_composite"}

    # 停止/終了/打撃の処理
    if tail == "stop_interrupt":
        # 「止まって、そのあと右へ」→ 即時停止のみ。後半は実行しない
        return {
            "kind": "cmd",
            "cmd": {"type": "stop", "action_key": "stop", "label": "止まる"},
            "truncated": True,
        }
    if tail == "end_interrupt" or tail == "end":
        if steps:
            # 「Xして終わって」= 計画の終端 → steps実行後にend相当（stop扱い）
            return _move_result(steps, stop_after=True)
        return {
            "kind": "cmd",
            "cmd": {"type": "end", "action_key": "end", "label": "終了する"},
        }
    if tail == "strike":
        if steps:
            # 移動+打撃の複合は現段階で対象外 — 半分だけ実行しない
            return {"kind": "unclear", "why": "mixed_strike"}
        return {
            "kind": "cmd",
            "cmd": {"type": "strike", "action_key": "strike",
                    "label": "打つ"},
        }
    if tail is True:
        if not steps:
            return {
                "kind": "cmd",
                "cmd": {"type": "stop", "action_key": "stop",
                        "label": "止まる"},
            }
        return _move_result(steps, stop_after=True)
    if not steps:
        return {"kind": "unclear", "why": "empty"}
    if len(steps) > MAX_STEPS:
        return {"kind": "unclear", "why": "too_many_steps"}
    total = sum(s["m"] for s in steps if s["action"] == "translate")
    if total > MAX_TOTAL_DIST_M + 1e-9:
        return {"kind": "unclear", "why": "too_much_distance"}
    return _move_result(steps, stop_after=False)


def _move_result(steps, stop_after=False):
    """step列→move cmd dict。"""
    if any(s["action"] == "jog" for s in steps):
        dur = JOG_DUR_S
    else:
        dur = STEP_DEADLINE_S * len(steps)
    label = " → ".join(s["label"] for s in steps)
    if stop_after:
        label += " → 止まる"
    key = "+".join(s["action_key"] for s in steps)
    return {
        "kind": "cmd",
        "cmd": {
            "type": "move",
            "steps": [
                {
                    "action": s["action"],
                    "dir": s["dir"],
                    "m": s.get("m"),
                    "deg": s.get("deg"),
                    "pace": s.get("pace", "walk"),
                    "action_key": s["action_key"],
                    "label": s["label"],
                }
                for s in steps
            ],
            "stop_after": bool(stop_after),
            "dur_s": dur,
            "action_key": key,
            "label": label,
        },
    }


def parse(text: str) -> dict | None:
    """参加者テキスト→control指令dict。解釈不能・拒否はNone。"""
    v = parse_detailed(text)
    return v["cmd"] if v["kind"] == "cmd" else None


# ---- LLM経路の意味intent→有界指令（列挙値のみ — 数値はこの表から供給）
_SEM_DIRS = {
    "forward": ("translate", "forward", None),
    "back": ("translate", "back", None),
    "left": ("translate", "left", None),
    "right": ("translate", "right", None),
    "turn_left": ("turn", "left", None),
    "turn_right": ("turn", "right", None),
    "turn_around": ("turn", "around", 180.0),
}
# LLMが出してよい量キー → 実数値（サーバーprofile由来 — LLM生成禁止）
_AMOUNT_DIST = {"m_025": 0.25, "m_05": 0.5, "m_1": 1.0, "m_15": 1.5, "m_2": 2.0,
                "small": 0.25, "normal": 0.5}
_AMOUNT_DEG = {"deg_15": 15.0, "deg_30": 30.0, "deg_45": 45.0,
               "deg_90": 90.0, "deg_180": 180.0,
               "small": 15.0, "normal": 45.0}
_SEM_TYPES = ("move", "stop", "end", "strike")


# LLM経路のpaceキー → 内部pace（数値生成はさせない — 列挙のみ）
_SEM_PACES = {"walk": "walk", "normal": "walk",
              "fast": "fast_walk", "fast_walk": "fast_walk",
              "run": "run"}


def _step_from_sem(action, direction, amount_key, pace_key=None):
    """LLMの列挙値→step dict。未知値はNone。

    移動方向に量キーが無い場合は継続jog（テキスト経路の裸方向と同じ
    意味論 — 「前」は0.5mでなく「止まって」まで進む）。pace は列挙値
    からのみ受理し、非対応方向への走行/速歩は step を作らない。"""
    base = _SEM_DIRS.get(direction)
    if base is None:
        return None
    pace = _SEM_PACES.get(pace_key or "walk")
    if pace is None:
        return None
    act, d, fixed = base
    if act == "translate":
        if amount_key is None:
            step, err = _make_jog(d, pace)
            return None if err else step
        m = _AMOUNT_DIST.get(amount_key)
        if m is None:
            return None
        step, err = _make_step("translate", d, m, pace=pace)
    else:
        if pace != "walk":
            return None
        deg = fixed if fixed is not None else _AMOUNT_DEG.get(
            amount_key or "normal")
        if deg is None:
            return None
        step, err = _make_step("turn", d, deg)
    if err:
        return None
    return step


def from_intent(sem: dict) -> dict | None:
    """検証済みsemantic intent → parserと同一の有界cmd dict。

    {type:"move", direction, size}（旧単発）または
    {type:"program", steps:[{action? direction, amount_key}]}（順序付き）。
    未知type/方向/量・余計な値はNone（発行しない）。
    """
    if not isinstance(sem, dict):
        return None
    t = sem.get("type")
    if t in ("stop", "end", "strike"):
        return {"type": t, "action_key": t,
                "label": {"stop": "止まる", "end": "終了する",
                          "strike": "打つ"}[t]}
    if t == "program":
        raw_steps = sem.get("steps")
        if not isinstance(raw_steps, list) or not (1 <= len(raw_steps) <= MAX_STEPS):
            return None
        steps = []
        total = 0.0
        for rs in raw_steps:
            if not isinstance(rs, dict):
                return None
            st = _step_from_sem(rs.get("action"), rs.get("direction"),
                                rs.get("amount_key"), rs.get("pace"))
            if st is None:
                return None
            steps.append(st)
            if st["action"] == "translate":
                total += st["m"]
        # 継続jogを列の途中へ混ぜない（parserと同一規則）
        if any(s["action"] == "jog" for s in steps) and len(steps) > 1:
            return None
        if total > MAX_TOTAL_DIST_M + 1e-9:
            return None
        return _move_result(steps)["cmd"]
    if t != "move":
        return None
    st = _step_from_sem(
        "move", sem.get("direction"),
        sem.get("amount_key") or sem.get("size"), sem.get("pace"),
    )
    if st is None:
        return None
    return _move_result([st])["cmd"]


def marker_directive(text: str) -> str | None:
    """明示marker参照の方向ヒントだけを返す。

    "near"/"far" — 方向が一意に決まる明示参照（決定的に解決可）。
    "ambiguous"   — marker語はあるが方向が決まらない
                   （「マーカーまで」「手前と奥どっち」等）→ 聞き返し対象。
    None          — marker参照ではない。
    """
    toks = _tokenize(_normalize(text))
    if not any(k == "MARKER" for k, _ in toks):
        return None
    near = any(k == "NEAR" for k, _ in toks)
    far = any(k == "FAR" for k, _ in toks)
    if near and not far:
        return "near"
    if far and not near:
        return "far"
    return "ambiguous"


def describe(cmd: dict) -> str:
    """UI表示用の一言説明（何を指令したか正直に示す）。"""
    t = cmd.get("type")
    if t == "move":
        if cmd.get("label"):
            return cmd["label"]
        if cmd.get("steps"):
            return " → ".join(s.get("label", "?") for s in cmd["steps"])
        # 旧形式のflat move
        if cmd.get("fwd", 0) > 0:
            return "前へ進む"
        if cmd.get("fwd", 0) < 0:
            return "後ろへ下がる"
        if cmd.get("lat", 0) > 0:
            return "左へ動く"
        if cmd.get("lat", 0) < 0:
            return "右へ動く"
        if cmd.get("yaw", 0) > 0:
            return "左を向く"
        if cmd.get("yaw", 0) < 0:
            return "右を向く"
        return "動く"
    return cmd.get("label") or {
        "stop": "止まる", "strike": "打つ", "end": "終了する"
    }.get(t, t or "?")
