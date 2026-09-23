"""製品セッション状態機械 — 画面フローとサーバー側契約の接続。

READY → INTERPRETING → (CLARIFY) → REVIEW → RUNNING → RESULT → (reset) → READY
       └──────────── reject/error → READY

- 承認はサーバー内部store（kotoba_orchestrator.ApprovalStore）で管理
- 承認は plan hash・world・session・round・boot・期限・単回消費に結合
- 実行は単一run lock（二重実行不可）
"""

import json
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from kotoba_api.llm import LlmError, interpret, interpret_game
from kotoba_api.world import COURSE
# harness依存はトップレベルで実モジュールへ束縛する — メソッド内の遅延importは
# sys.modules汚染（凍結snapshot試験が同名パッケージを差し替える）を拾い得る
from kotoba_harness import gameparse
from kotoba_harness.control import write_command, write_strike_event
from kotoba_harness.roundworld import make_round
from kotoba_contracts.canonical import canonical_plan_sha256
from kotoba_contracts.intent import parse_intent
from kotoba_api.decision import candidates_for_unclear, decision_client
from kotoba_contracts.plan import ControllerProfile
from kotoba_orchestrator.approval_store import ApprovalStore
from kotoba_orchestrator.errors import ApprovalError, PlanRejected
from kotoba_orchestrator.session import SessionManager
from kotoba_orchestrator.validator import build_motion_plan, build_plan

PRODUCT_PROFILE = ControllerProfile(
    name="sim_profile_product_v1",
    max_linear_mps=0.85,
    max_yaw_radps=0.8,
    max_duration_s=6.0,
)

# 能力台帳（R5 — profile/実測に束縛。証明できていないものは verified にしない）:
#   walk:      実機計測済みの基本歩容（stick≦0.8）
#   fast_walk: walk policy内のstick上限までの速歩。run policyではない。
#              Thor実測(F1)で速度差を確認するまで experimental。
#   run:       走行。PM01 task/policyに run が存在しないため off —
#              「走って」を受理して歩行へ黙って置換しない（R5-01）。
CAPABILITIES = {
    "walk": "verified",
    "fast_walk": "experimental",
    "run": "off",
}

_PACE_MESSAGES = {
    "run": (
        "走行はまだ実機で検証できていません。"
        "「速く歩いて」なら速歩きで実行できます。"
    ),
    "fast_walk": "速歩きは現在無効です。普通の速さで指示してください。",
}


def _pace_gate(cmd: dict) -> str | None:
    """move cmdのpace要求を能力台帳と照合。未対応ならpace名を返す。"""
    for s in cmd.get("steps") or []:
        pace = s.get("pace") or "walk"
        if pace == "walk":
            continue
        if CAPABILITIES.get(pace) not in ("experimental", "verified"):
            return pace
    return None


# parserが具体的に特定した不足・範囲外 → 参加者への聞き返し。
# 内部ID・schema語を露出せず、直せる形で伝える（R5-02）。
_UNCLEAR_MESSAGES = {
    "bad_angle": "角度は1〜180度で指定してください（例: 右に110度）。",
    "unitless_num": "数字には単位を付けてください（例: 110度・1m）。",
    "bad_dist": "距離は0.25/0.5/1/1.5/2mで指定してください。",
    "too_much_distance": "一度に進めるのは合計2mまでです。",
    "dist_with_turn": "旋回の角度には距離の単位を付けないでください。",
    "pace_dir": "走る・速歩きは前への移動にだけ使えます。",
    "pace_turn": "走る・速歩きは旋回には使えません。",
    "forward_turn": "前向きへの角度指定はできません。左右または後ろを向く指示にしてください。",
    "too_many_steps": "一度に組み合わせられるのは3動作までです。",
    "continuous_composite": "進み続ける指示は他の動作と組み合わせられません。",
    "mixed_strike": "移動と打撃の組み合わせには対応していません。",
    "turn_no_dir": "どちらへ向くか教えてください（例: 右を向いて・後ろを向いて）。",
    "move_no_dir": "どちらへ動くか教えてください。",
}
_UNCLEAR_DEFAULT = (
    "その表現はまだ解釈できません。"
    "「前」「少し右」「右に110度回って」「止まって」などで指示してください。"
)

SIM_COMPOSE_PROJECT = "kotoba"

# 非同期LLM解釈の受付有効期限（秒）— 受理時に記録し、応答時に再検査する。
# 遅れて返った解釈へ新たな実行期限を無条件に与えない（A残欠 P04）。
LLM_TICKET_TTL_S = 10.0


class SpawnRejected(Exception):
    """ラウンド生成を現在の実行世代に対して安全に拒否した理由。"""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class RoundState:
    round_id: str
    phase: str = "READY"  # READY/INTERPRETING/CLARIFY/REVIEW/RUNNING/RESULT/FAULT
    message: str = "目的地をことばで教えてください"
    last_intent: Optional[dict] = None
    clarify_context: Optional[str] = None
    pending_plan_id: Optional[str] = None
    plan_obj: Optional[object] = None
    run_id: Optional[str] = None
    result: Optional[dict] = None
    history: list = field(default_factory=list)
    # ことばでスイカ割りのラウンド世界（RoundWorld正本、生成後は不変）
    round_spec: Optional[object] = None
    # 生成位置を取得したsim boot。別bootへ切り替わった後は旧座標を使わない。
    spec_boot_nonce: Optional[str] = None
    # ゲーム実行context（start_gameで発行）: lease_id/seq/strikes_issued/
    # boot_expect/control_path/started_wall/done/outcome
    game: Optional[dict] = None


class ProductSession:
    def __init__(
        self,
        sim_ctl=None,
        boot_nonce_fn=None,
        live_accepted: bool = False,
        selftest: bool = False,
    ) -> None:
        self.sessions = SessionManager()
        self.store = ApprovalStore()
        self.rounds: Dict[str, RoundState] = {}
        self.boot_nonce: Optional[str] = None
        self.run_lock = False
        # run_lock の check-and-set をHTTP同時要求から守る（FastAPIの通常defは
        # threadpoolで並行実行される — 逐次409だけを競合保証にしない）
        self._run_mu = threading.Lock()
        # 実行受付の開閉。reset/pause開始で閉じ、新round READYで再開する。
        # _run_mu 区間内でのみ遷移する（RESETTING途中のstart_run受付を防ぐ — R2-A）
        self._admission_open = True
        # kioskを閉じる直前だけ新規実行を抑止する。短時間で自動失効し、
        # デスクトップから展示画面を開き直した後の操作を妨げない。
        self._kiosk_exit_until = 0.0
        # reset全体の直列化ロック（C14: 並行resetによる受付早期再開を防ぐ）。
        # 外部I/Oを含むreset全体を覆うが _run_mu とは別 — 停止操作は塞がない。
        self._reset_mu = threading.Lock()
        self._last_reset_mono = 0.0
        self._last_reset_session: Optional[str] = None
        self._sim_ctl = sim_ctl  # SimControl 呼び出し可能オブジェクト（app側で注入）
        self._boot_nonce_fn = boot_nonce_fn  # 現在bootの実nonceを返す（未準備はNone）
        # 最後にspawnしたroundを持つsession。sim worldのスイカは1つしかないため
        # /api/display/round はこのsessionのspecを正本として返す（古いsessionの
        # fixture specがdict先頭に残って新しいspawnを隠す事故の対策）。
        self._latest_spec_session: Optional[str] = None
        # 直近の通常spawn位置（履歴分離 — 同一世界で連続して近い場所に
        # 出さない。fixtureは含めない・再起動で揮発する揮発履歴）
        self._spawn_history: list = []
        self._live_accepted = live_accepted
        self._selftest = selftest

    # ---- sessions -------------------------------------------------------
    def create_session(self) -> str:
        session = self.sessions.create_session()
        self._new_round(session.session_id)
        return session.session_id

    def _new_round(self, session_id: str) -> str:
        round_ = self.sessions.begin_round(session_id)
        self.rounds[session_id] = RoundState(round_id=round_.round_id)
        return round_.round_id

    def reserve_kiosk_exit(self, hold_s: float = 10.0) -> bool:
        """実行/リセットがないことを確かめ、新規run受付を一時的に塞ぐ。"""
        with self._run_mu:
            if (not self._admission_open or self.run_lock or any(
                r.phase in {"RUNNING", "RESETTING"} for r in self.rounds.values()
            )):
                return False
            self._kiosk_exit_until = time.monotonic() + hold_s
            return True

    def spawn_round(self, session_id: str, seed: int, robot_pos, robot_yaw,
                    target=None, boot_nonce: Optional[str] = None,
                    expected_round_id: Optional[str] = None,
                    expected_boot_nonce: Optional[str] = None):
        """ことばでスイカ割り: RoundWorldでラウンド世界を生成し固定する。

        seedは呼び出し側が選ぶ（検証の再現性のため）。生成後のspecは
        ラウンド中不変 — renderer/API/採点が同一world座標を共有する。
        target 指定は operator fixture 対照用の明示world位置（監査用に
        spec.seed=-1 標識）。通常経路は seed 生成のみ。
        """
        # 生成物の差し替えは実行targetと画面targetの乖離を招く。全sessionの
        # run受付・exit予約との判定から正本更新までを同じロックで不可分にする。
        with self._run_mu:
            r = self.rounds[session_id]
            if boot_nonce is not None:
                # APIが読んだ観測が新鮮でも、要求作成後のresetを推測で
                # 新roundへ適用しない。直接fixture呼び出しだけ期待世代を省く。
                if expected_round_id != r.round_id:
                    raise SpawnRejected("expected_round_mismatch")
                if expected_boot_nonce != boot_nonce:
                    raise SpawnRejected("expected_boot_mismatch")
            if not self._admission_open or r.phase == "RESETTING":
                raise SpawnRejected("reset_in_progress")
            if time.monotonic() < self._kiosk_exit_until:
                raise SpawnRejected("kiosk_exiting")
            if self.run_lock or any(
                other.phase == "RUNNING" for other in self.rounds.values()
            ):
                raise SpawnRejected("run_in_progress")
            if r.phase != "READY" or r.round_spec is not None:
                raise SpawnRejected("round_not_ready")
            if (boot_nonce is not None and self._boot_nonce_fn
                    and self._boot_nonce_fn() != boot_nonce):
                raise SpawnRejected("live_boot_mismatch")
            # 360°配置は turn-around の実測受入後のみ（明示フラグ）。
            full_circle = os.environ.get("KOTOBA_SPAWN_FULL_CIRCLE") == "1"
            spec = make_round(
                r.round_id, seed, tuple(robot_pos), robot_yaw,
                target=target, history=self._spawn_history,
                full_circle=full_circle,
            )
            r.round_spec = spec
            r.spec_boot_nonce = boot_nonce
            self._latest_spec_session = session_id
            if target is None:
                # 通常spawnのみ履歴へ（fixtureは分離母集団）
                self._spawn_history.append(spec.watermelon.pos)
                self._spawn_history = self._spawn_history[-8:]
            return spec

    def display_spec(self, boot_nonce: Optional[str]):
        """現在のsim世界に属するspecだけを描画側へ公開する。"""
        with self._run_mu:
            if not self._admission_open:
                return None
            r = self.rounds.get(self._latest_spec_session or "")
            if r is None or r.round_spec is None:
                return None
            if r.spec_boot_nonce is not None and r.spec_boot_nonce != boot_nonce:
                return None
            return r.round_spec

    def _game_world_rejection(
        self, session_id: str, r: RoundState, boot_nonce: Optional[str]
    ) -> Optional[str]:
        """描画の正本とゲーム開始の対象が一致するかを判定する。_run_mu内。"""
        if r.round_spec is None:
            return "no_round_spawned"
        if self._latest_spec_session != session_id:
            return "world_superseded"
        if r.spec_boot_nonce is not None and r.spec_boot_nonce != boot_nonce:
            return "spec_boot_mismatch"
        return None

    # ---- game（ことばでスイカ割り） ----------------------------------------
    def start_game(self, session_id: str, boot_nonce: str, run_id: str, control_dir) -> dict:
        """スイカ割りラウンドの実行開始 — lease発行とrun登録を原子的に行う。

        spawn_round で固定済みの RoundSpec に対し指令経路の lease_id を発行する。
        lease は round+run+boot に結び付き、別世代の指令は controller 側が
        lease_mismatch で拒否する。plan/approval 経路（単発run用）とは別系統 —
        連続指令は lease+seq+bounds+期限 の fail-closed 検査で縛る。
        """
        with self._run_mu:
            r = self.rounds[session_id]
            if not self._admission_open:
                return {"started": False, "reason": "reset_in_progress"}
            if time.monotonic() < self._kiosk_exit_until:
                return {"started": False, "reason": "kiosk_exiting"}
            current_boot = self._boot_nonce_fn() if self._boot_nonce_fn else boot_nonce
            if current_boot != boot_nonce:
                return {"started": False, "reason": "spec_boot_mismatch"}
            world_rejection = self._game_world_rejection(session_id, r, current_boot)
            if world_rejection is not None:
                return {"started": False, "reason": world_rejection}
            if r.phase == "RUNNING" or self.run_lock:
                return {"started": False, "reason": "run_in_progress"}
            self.run_lock = True
            r.phase = "RUNNING"
            r.run_id = run_id
            r.game = {
                "lease_id": uuid.uuid4().hex,
                "seq": 0,
                # epoch: STOP/END/resetで進める — 遅いLLM解釈が古い
                # 世代へ指令を書き込まないための世代番号
                "epoch": 0,
                "boot_expect": boot_nonce,
                "control_path": str(control_dir / "control.json"),
                "strike_path": str(control_dir / "strike.json"),
                "state_path": str(control_dir / "control_state.json"),
                # strike予約: mailboxへ書き込んだが実消費（開始ACK）が
                # まだ無い strike_id。実消費正本はcontrollerの
                # strikes_started（control_state.json）。
                "strike_pending": None,
                "strike_unconfirmed": False,
                "started_wall": time.time(),
                "done": False,
                "outcome": None,
            }
            r.message = "ラウンド中 — ことばで指示してください"
            return {
                "started": True,
                "run_id": run_id,
                "lease_id": r.game["lease_id"],
            }

    def startable(self, session_id: str) -> dict:
        """ゲーム開始可否のtyped DTO — UIのdisabled理由表示用。

        ここは表示専用の観測面。実判定は start_game が同一条件を
        _run_muロック内で再検査する（check-then-actの競合は残るが、
        拒否理由は同じコード列を返すので利用者表示は一致する）。
        sim起動/offlineのようなこのオブジェクトが持たない条件は
        呼出し側（app層）がreasonsへ追加する。
        """
        with self._run_mu:
            r = self.rounds.get(session_id)
            if r is None:
                return {"ok": False, "reasons": ["unknown_session"]}
            boot_nonce = self._boot_nonce_fn() if self._boot_nonce_fn else None
            reasons = []
            world_rejection = self._game_world_rejection(session_id, r, boot_nonce)
            if world_rejection is not None:
                reasons.append(world_rejection)
            if not self._admission_open:
                reasons.append("reset_in_progress")
            if r.phase == "RUNNING" or self.run_lock:
                reasons.append("run_in_progress")
            return {"ok": not reasons, "reasons": reasons}

    def _game_controller_state(self, g: dict) -> dict:
        """controllerが書く実状態（存在しなければ空dict）。"""
        try:
            return json.loads(Path(g["state_path"]).read_text())
        except (OSError, json.JSONDecodeError, KeyError):
            return {}

    def _strike_pending_resolve(self, g: dict, st: dict) -> None:
        """pending中のstrike予約をcontroller実状態と照合して解消する。

        - state.strike.strike_id が一致し consumed/done なら予約完了
        - strike_expired に含まれれば「未開始」確定 → 予約解放（消費しない）
        - expires_wall 経過後もACK無し: controller生存なら未到達として解放、
          不明なら strike_unconfirmed を立てて推測で消費も返却もしない
        """
        pend = g.get("strike_pending")
        if pend is None:
            return
        sid = pend.get("strike_id")
        cur = st.get("strike") or {}
        if cur.get("strike_id") == sid and (
            cur.get("consumed") or cur.get("phase") in ("done", "recover")
        ):
            g["strike_pending"] = None
            return
        if sid in (st.get("strike_unverified") or []):
            # dance送出後に開始未確認で中断 — 推測で消費/返却しない
            g["strike_pending"] = None
            g["strike_unconfirmed"] = True
            return
        # 完了したstrike_id（消費・非消費を問わず予約は解消）
        if sid == st.get("last_strike_done"):
            g["strike_pending"] = None
            return
        if sid in (st.get("strike_expired") or []):
            g["strike_pending"] = None
            return
        if time.time() > float(pend.get("expires_wall", 0)) + 1.0:
            alive = bool(st) and (time.time() - st.get("t_wall", 0)) < 3.0
            g["strike_pending"] = None
            if not alive or not st.get("strike"):
                g["strike_unconfirmed"] = True

    def game_heartbeat(self, session_id: str):
        """継続jogの生存確認受付 — (lease_id, control_dir) または None。"""
        with self._run_mu:
            r = self.rounds.get(session_id)
            if (
                r is None
                or r.phase != "RUNNING"
                or r.game is None
                or r.game.get("done")
            ):
                return None
            g = r.game
            return (g["lease_id"], str(Path(g["control_path"]).parent))

    def run_command_accept(self, session_id: str):
        """通常runへの指令が届くrun_id — 非実行中はNone（世代結合）。"""
        with self._run_mu:
            r = self.rounds.get(session_id)
            if r is None or r.phase != "RUNNING" or r.run_id is None:
                return None
            return r.run_id

    def game_command(self, session_id: str, text: str) -> dict:
        """参加者のことば→有界指令→control/strike mailbox。

        解釈は決定論parser優先（STOP/ENDを即応答＝STOP優先）、parserが
        「要追加解釈」と返した入力のみローカルLLM経路へフォールバックする。
        明示禁止・取消し・質問・引用は決定論的拒否 — LLMへ回して
        actuationへ格上げしない。さらに禁止・取消しは受理待ちの旧意図を
        失効する（未適用の非同期解釈が禁止をすり抜けて発行されない）。
        LLMは型・方向・大小の列挙値だけを返し、数値は
        gameparse.from_intent の固定表から供給される（速度・時間をLLMが
        生成しない）。経路は cmd["via"]="parser"|"llm" としてpayload・
        応答へ記録し、監査で区別できるようにする。

        ロック構造: _run_mu の保持区間は受付・世代確保・書込のみで、
        LLM呼出しはロック外 — 推論待ち（最大40s）がSTOP/END/resetの
        受付を塞がない。LLM応答後に再lockし、round/run生存・epoch・
        受理番号(input_seq)を再検査 — 新しい入力やSTOP/ENDが先に
        受理されていれば古いLLM応答は破棄する（LLMの完了順ではなく
        受付順が新旧を決める）。

        C3: 振数は「実際に打撃動作を開始した回数」（controllerの
        strikes_started）が正本。mailboxへの書込みは予約であり、
        開始ACK前の上書き・期限切れ・配信失敗では消費しない。
        STOP/END受理時にepochを進め、遅い非同期解釈の書込を無効化する。
        """
        r = self.rounds.get(session_id)
        if r is None:
            return {"accepted": False, "reason": "unknown_session"}
        with self._run_mu:
            if r.phase != "RUNNING" or r.game is None or r.game.get("done"):
                return {"accepted": False, "reason": "not_running"}
            verdict = gameparse.parse_detailed(text)
            if verdict["kind"] == "cmd":
                cmd = verdict["cmd"]
                if cmd["type"] == "move":
                    gated = _pace_gate(cmd)
                    if gated:
                        return {
                            "accepted": False,
                            "reason": f"unsupported_{gated}",
                            "message": _PACE_MESSAGES[gated],
                        }
                return self._game_write(r, cmd, "parser")
            if verdict["kind"] == "reject":
                # 明示禁止・取消し（prohibit）は、受理待ちの旧意図を失効する:
                # LLM推論待ちのmove/strikeが禁止のあとから発行される順序を
                # 閉じるため、未適用の解釈世代（input_seq）を進める。
                # 質問・引用などの非操作入力（nonop）は世代を変えない。
                if verdict.get("why") == "prohibit":
                    r.game["input_seq"] = int(r.game.get("input_seq") or 0) + 1
                    return {
                        "accepted": False,
                        "reason": "rejected",
                        "message": "承知しました。動きの指示は出しません。",
                    }
                return {
                    "accepted": False,
                    "reason": "rejected",
                    "message": "その指示には従えません。「前」「少し左」「止まって」「割って」などで指示してください。",
                }
            # unclear → LLM推論へ回すため受理世代・入力番号・受付期限を確保
            g = r.game
            g["input_seq"] = int(g.get("input_seq") or 0) + 1
            ticket = g["input_seq"]
            gen = g["epoch"]
            g["input_deadline"] = time.time() + LLM_TICKET_TTL_S
            # L1 shadow評価（R5-06）— 制御経路へは接続しない。
            # parserのunclear理由・世代・受理番号を記録し、採用時の
            # 世代照合設計を評価データへ残す。
            decision_client().submit_shadow(
                text,
                candidates_for_unclear(text),
                {
                    "session_id": session_id,
                    "round_id": r.round_id,
                    "run_id": r.run_id,
                    "control_epoch": gen,
                    "input_seq": ticket,
                    "parser_why": verdict.get("why"),
                    "mode": "game",
                },
            )
        # ロック外 — 推論待ちは他の指令・停止・resetの受付を塞がない
        try:
            sem = interpret_game(text)
        except LlmError:
            sem = None
        cmd = gameparse.from_intent(sem) if sem is not None else None
        with self._run_mu:
            # 世代再検査: round/run失効・STOP/END(epoch進行)・より新しい
            # 入力・明示禁止/取消しの受理で、この応答は古い — 書き込まず破棄。
            if (
                r.phase != "RUNNING" or r.game is not g or g.get("done")
                or g["epoch"] != gen or g.get("input_seq") != ticket
            ):
                return {
                    "accepted": False,
                    "reason": "stale",
                    "message": "別の指示が先に受理されたため、この解釈は破棄しました。",
                }
            # 受付時に記録した期限の再検査 — 遅着回答へ新たな実行期限を与えない
            if time.time() > float(g.get("input_deadline") or 0):
                return {
                    "accepted": False,
                    "reason": "expired",
                    "message": "解釈の受理期限を過ぎたため、この解釈は破棄しました。",
                }
            if cmd is None:
                return {
                    "accepted": False,
                    "reason": "unclear",
                    "message": "聞き取れませんでした。「前」「少し左」「止まって」「割って」などで指示してください。",
                }
            if cmd["type"] == "move":
                gated = _pace_gate(cmd)
                if gated:
                    return {
                        "accepted": False,
                        "reason": f"unsupported_{gated}",
                        "message": _PACE_MESSAGES[gated],
                    }
            return self._game_write(r, cmd, "llm")

    def _game_write(self, r, cmd: dict, via: str) -> dict:
        """RUNNING中のgame ctxへ有界指令を書き込む。_run_mu保持下で呼ぶ。"""
        g = r.game
        spec = r.round_spec
        if cmd["type"] == "strike":
            st = self._game_controller_state(g)
            self._strike_pending_resolve(g, st)
            if g.get("strike_pending") is not None:
                return {
                    "accepted": False,
                    "reason": "strike_busy",
                    "message": "今の一打が終わってから指示してください。",
                }
            if (st.get("strike") or {}).get("phase") in (
                "pre", "dance", "swing", "recover"
            ):
                return {
                    "accepted": False,
                    "reason": "strike_busy",
                    "message": "今の一打が終わってから指示してください。",
                }
            started = int(st.get("strikes_started") or 0)
            if spec is not None and started >= spec.max_swings:
                return {
                    "accepted": False,
                    "reason": "out_of_swings",
                    "message": "振れる回数を使い切りました。",
                }
        if cmd["type"] in ("stop", "end"):
            g["epoch"] += 1
        g["seq"] += 1
        g["input_seq"] = int(g.get("input_seq") or 0) + 1
        seq = g["seq"]
        cmd["via"] = via  # 解釈経路の監査記録（controllerは未知keyを無視）
        try:
            if cmd["type"] == "strike":
                payload = write_strike_event(
                    g["strike_path"], g["lease_id"], seq, via=via
                )
                g["strike_pending"] = {
                    "strike_id": payload["cmd"]["strike_id"],
                    "seq": seq,
                    "expires_wall": payload["cmd"]["expires_wall"],
                }
            else:
                write_command(g["control_path"], g["lease_id"], seq, cmd)
        except OSError:
            return {"accepted": False, "reason": "control_path_unavailable"}
        label = gameparse.describe(cmd)
        r.message = f"指示「{label}」を送りました"
        out = {
            "accepted": True, "seq": seq, "cmd": cmd,
            "label": label, "via": via,
        }
        if cmd["type"] == "strike":
            out["strike_id"] = g["strike_pending"]["strike_id"]
            out["pending"] = True
        return out

    def state(self, session_id: str) -> dict:
        r = self.rounds[session_id]
        return {
            "session_id": session_id,
            "round_id": r.round_id,
            "phase": r.phase,
            "message": r.message,
            "last_intent": r.last_intent,
            "pending_plan_id": r.pending_plan_id,
            "run_id": r.run_id,
            "result": r.result,
            "course": {
                "targets": [
                    {"id": t.target_id, "label": t.label, "distance_m": t.distance_m}
                    for t in COURSE["targets"]
                ]
            },
            "live_enabled": self._live_accepted,
            "selftest": self._selftest,
            "capabilities": dict(CAPABILITIES),
        }

    # ---- intent ---------------------------------------------------------
    def interpret(self, session_id: str, text: str) -> dict:
        r = self.rounds[session_id]
        if self.run_lock:
            return {
                "decision": "busy",
                "message": "実行中です。停止してから入力してください。",
            }
        r.phase = "INTERPRETING"
        r.message = "指示を読んでいます…"
        # 決定論parser優先 — 動作指示として確定できる入力はLLMを経由しない。
        # 曖昧な入力だけがmarker選択用のLLM解釈へ進む。
        verdict = gameparse.parse_detailed(text)
        if verdict["kind"] == "cmd":
            cmd = verdict["cmd"]
            if cmd["type"] == "move":
                gated = _pace_gate(cmd)
                if gated:
                    r.phase = "READY"
                    r.message = _PACE_MESSAGES[gated]
                    return {
                        "decision": "reject",
                        "reason": f"unsupported_{gated}",
                        "message": r.message,
                    }
                return self._interpret_motion(r, session_id, cmd)
            r.phase = "READY"
            if cmd["type"] == "strike":
                r.message = "打撃はスイカ割りモードでのみ使えます。"
            elif cmd["type"] == "end":
                r.message = "終了するラウンドはありません。"
            else:
                r.message = "実行中の動作はありません。"
            return {
                "decision": "reject",
                "reason": "unsupported_action",
                "message": r.message,
            }
        if verdict["kind"] == "reject":
            r.phase = "READY"
            r.message = (
                "承知しました。動きの指示は出しません。"
                if verdict["why"] == "prohibit"
                else "その内容は動作の指示ではありません。"
            )
            return {
                "decision": "reject",
                "reason": verdict["why"],
                "message": r.message,
            }
        # unclear — marker参照だけをmarker解釈へ回す（R5-02）。
        # 「手前/奥 + マーカー」は一意に決まるためLLMを待たず決定的に解決。
        # 運動表現のunclearはmarker用途のLLMへ回さず、具体的な聞き返しで返す。
        why = verdict.get("why") or ""
        if why == "marker_word" and r.clarify_context is None:
            directive = gameparse.marker_directive(text)
            if directive in ("near", "far"):
                ts = sorted(COURSE["targets"], key=lambda t: t.distance_m)
                target_id = (ts[0] if directive == "near" else ts[-1]).target_id
                intent = {
                    "schema_version": "1.0",
                    "decision": "execute",
                    "target_ids": [target_id],
                    "avoid_ids": [],
                    "explanation": "明示されたマーカーへ進みます",
                }
                return self._respond_marker_intent(
                    r, session_id, intent
                )
        if why != "marker_word" and r.clarify_context is None:
            # L1 shadow評価（R5-06）— 制御経路へは接続しない。
            # 辞書外表現の分類品質を記録し、dict-onlyとの比較に使う。
            cands = candidates_for_unclear(text)
            decision_client().submit_shadow(
                text,
                cands,
                {
                    "session_id": session_id,
                    "round_id": r.round_id,
                    "parser_why": why,
                    "mode": "normal",
                },
            )
            r.phase = "CLARIFY"
            r.message = _UNCLEAR_MESSAGES.get(why, _UNCLEAR_DEFAULT)
            return {"decision": "clarify", "message": r.message}
        try:
            intent = interpret(
                text, COURSE["targets"], clarify_context=r.clarify_context
            )
        except LlmError as exc:
            r.phase = "FAULT"
            r.message = f"推論に失敗しました（{exc}）。もう一度お願いします。"
            return {"decision": "error", "message": r.message}
        r.clarify_context = None
        r.last_intent = intent
        decision = intent["decision"]
        if decision == "clarify":
            r.phase = "CLARIFY"
            r.clarify_context = intent["question"]
            r.message = intent["question"]
            return {
                "decision": "clarify",
                "question": intent["question"],
                "candidates": intent.get("candidate_target_ids", []),
            }
        if decision == "reject":
            r.phase = "READY"
            r.message = f"実行できません: {intent['explanation']}"
            return {
                "decision": "reject",
                "reason": intent["reason_code"],
                "message": intent["explanation"],
            }
        # execute → 計画を構築してREVIEWへ
        return self._respond_marker_intent(r, session_id, intent)

    def _respond_marker_intent(
        self, r: RoundState, session_id: str, intent: dict
    ) -> dict:
        """execute intent（LLM/決定的marker解決の両方）→ plan → REVIEW。"""
        r.clarify_context = None
        r.last_intent = intent
        world = _virtual_world(r.round_id)
        plan = build_plan(
            parse_intent(intent),
            world,
            session_id=session_id,
            round_id=r.round_id,
            plan_id=uuid.uuid4().hex,
            profile=PRODUCT_PROFILE,
            created_monotonic=time.monotonic(),
        )
        r.pending_plan_id = plan.plan_id
        r.plan_obj = plan
        r.phase = "REVIEW"
        target_label = _target_label(intent["target_ids"][0])
        r.message = target_label
        return {
            "decision": "execute",
            "plan_id": plan.plan_id,
            "target_label": target_label,
            "explanation": intent["explanation"],
            "review": {
                "action": "前進歩行",
                "target": target_label,
                "distance_m": _target_distance(intent["target_ids"][0]),
                "stop": "減速して停止し、立位を保持します",
            },
        }

    def _interpret_motion(self, r: RoundState, session_id: str, cmd: dict) -> dict:
        """決定論parserのmove cmd → motion_program plan → REVIEW。

        LLMを経由しない経路でも承認・plan hash・単回消費の契約はmarker経路と同一。
        """
        world = _virtual_world(r.round_id)
        try:
            plan = build_motion_plan(
                cmd.get("steps") or [],
                world,
                session_id=session_id,
                round_id=r.round_id,
                plan_id=uuid.uuid4().hex,
                profile=PRODUCT_PROFILE,
                created_monotonic=time.monotonic(),
            )
        except PlanRejected as exc:
            r.phase = "READY"
            r.message = f"実行できません: {exc}"
            return {"decision": "reject", "reason": str(exc), "message": r.message}
        r.pending_plan_id = plan.plan_id
        r.plan_obj = plan
        r.phase = "REVIEW"
        r.last_intent = {
            "decision": "execute",
            "program": True,
            "explanation": cmd.get("label") or "",
        }
        labels = [s.label or s.action_key for s in plan.steps]
        has_jog = any(s.action == "jog" for s in plan.steps)
        total_m = round(
            sum(s.target for s in plan.steps if s.action == "translate"), 3
        )
        r.message = " → ".join(labels)
        return {
            "decision": "execute",
            "plan_id": plan.plan_id,
            "program": True,
            "explanation": r.message,
            "review": {
                "action": "動作プログラム",
                "steps": labels,
                "distance_m": total_m,
                "stop": (
                    "「止まって」の指示・操縦範囲の境界・継続上限のいずれかで"
                    "減速して停止し、立位を保持します"
                    if has_jog
                    else "各動作の完了を観測で確認し、最後に減速して立位を保持します"
                ),
            },
        }

    # ---- approval -------------------------------------------------------
    def approve(self, session_id: str, plan_id: str) -> dict:
        r = self.rounds[session_id]
        if r.pending_plan_id != plan_id or r.phase != "REVIEW":
            return {"approved": False, "reason": "stale_plan"}
        plan = _rebuild_plan(r, session_id, plan_id)
        nonce = (self._boot_nonce_fn() if self._boot_nonce_fn else None) or self.boot_nonce
        try:
            record = _approval_record(plan, r, nonce)
        except ApprovalError as exc:
            return {"approved": False, "reason": exc.reason}
        approval_id = self.store.issue(record)
        return {"approved": True, "approval_id": approval_id}

    # ---- run ------------------------------------------------------------
    def start_run(
        self, session_id: str, plan_id: str, approval_id: str, sim_boot_nonce: str
    ) -> dict:
        """承認を原子的に消費し、runner起動権限を返す。

        run_lock のcheck-and-setと承認消費は同一lock区間で行う（同時HTTP要求対策）。
        """
        current_nonce = (self._boot_nonce_fn() if self._boot_nonce_fn else None) or sim_boot_nonce
        with self._run_mu:
            # 世代・受付状態・plan検査を lock 内で行う（lock取得までの間に
            # reset/pause が世代を失効させている可能性がある — R2-A）。
            # 受付閉鎖はplan検査より先に報告する（利用者へ正しい理由を返す）。
            r = self.rounds[session_id]
            if not self._admission_open:
                return {"started": False, "reason": "reset_in_progress"}
            if time.monotonic() < self._kiosk_exit_until:
                return {"started": False, "reason": "kiosk_exiting"}
            if r.phase != "REVIEW" or r.pending_plan_id != plan_id:
                return {"started": False, "reason": "stale_plan"}
            if self.run_lock:
                return {"started": False, "reason": "run_in_progress"}
            plan = _rebuild_plan(r, session_id, plan_id)
            try:
                grant = self.store.verify_and_consume(
                    approval_id,
                    plan_sha256=canonical_plan_sha256(plan),
                    world_version=plan.world_version,
                    session_id=session_id,
                    round_id=r.round_id,
                    sim_boot_id=current_nonce,
                    now=_now_utc(),
                )
            except ApprovalError as exc:
                return {"started": False, "reason": exc.reason}
            self.run_lock = True
            r.phase = "RUNNING"
            r.run_id = uuid.uuid4().hex
            r.message = "実行中"
        return {
            "started": True,
            "run_id": r.run_id,
            "grant": grant.model_dump(),
            "plan": plan.model_dump(),
        }

    def finish_run(self, session_id: str, run_id: str, result: dict) -> bool:
        """終了通知は実行世代（round+run_id）に結び付ける。

        受理時の全副作用（lock解除・RESULT遷移・result格納・boot反映）を
        同一 _run_mu 区間で世代検査と不可分に行う。reset/pauseで失効した
        旧runの遅着結果は現在の状態を一切変更せず履歴へ隔離する（R2-B:
        bootも旧値へ戻さない — boot正本はライフサイクル側が所有）。
        """
        with self._run_mu:
            r = self.rounds[session_id]
            if r.phase == "RUNNING" and r.run_id is not None and r.run_id == run_id:
                self.run_lock = False
                r.phase = "RESULT"
                r.result = result
                if result.get("boot_nonce"):
                    # controllerのresult.jsonはnonceをintで返す — ApprovalRecord
                    # (sim_boot_id: str)へ渡る前にstr正規化する（int混入で
                    # approveがValidationError 500になる事故の対策）。
                    self.boot_nonce = str(result["boot_nonce"])
                verdict = result.get("verdict", "UNKNOWN")
                if r.game is not None:
                    # ゲームround: outcome（勝敗）と制御health・verdictを分離。
                    # 異常（obs_lost等）を普通のtimeout/endedへ落とさない。
                    r.game["done"] = True
                    ctrl = result.get("control") or {}
                    health = ctrl.get("health", "ok")
                    round_end = None
                    for ev in result.get("events") or []:
                        if ev.get("event") == "round_end":
                            round_end = ev.get("reason", "timeout")
                    if health != "ok" or str(verdict).startswith("FAIL"):
                        # 制御異常・verdict失敗はゲーム勝敗と区別してfault
                        outcome = "fault"
                    elif str(verdict).startswith("UNKNOWN"):
                        # 状態不明（終了後立位の実観測不能等）は推測で
                        # ended/timeoutにしない — 要確認
                        outcome = "fault"
                    elif round_end is not None:
                        outcome = round_end
                    elif ctrl.get("end_received"):
                        outcome = "ended"
                    else:
                        outcome = "fault"
                    r.game["outcome"] = outcome
                    # 終端の転倒状態はcontrol_stateの最終書込に無い場合が
                    # ある（tail期のlatchは結果ファイルのみに残る）— 終端
                    # 正本としてresult側から引き継ぐ
                    r.game["fallen"] = bool(ctrl.get("fallen"))
                    r.message = {
                        "hit": "スイカを割りました！",
                        "out_of_swings": "振れる回数を使い切りました",
                        "timeout": "時間切れです",
                        "ended": "終了しました",
                        "arena_exit": "操縦範囲から出たため終了しました",
                        "fault": f"異常終了: {ctrl.get('shutdown_reason') or verdict}",
                    }.get(outcome, f"ラウンド終了（{outcome}）")
                else:
                    r.message = (
                        f"結果: {verdict} — 停止位置誤差 {result.get('err_m', '?')}m"
                        if verdict == "PASS"
                        else f"結果: {verdict}。{result.get('reasons') or ''}"
                    )
                return True
            # 世代不一致の遅着終了: 現在状態・所有権を変えず履歴に残す
            r.history.append(
                {"orphaned_run_id": run_id, "round_id": r.round_id, "result": result}
            )
            return False

    def release_run_lock(self, session_id: str) -> None:
        with self._run_mu:
            self.run_lock = False

    def owns_run(self, session_id: str, run_id: str) -> bool:
        """run_id が現在の実行世代の所有物かを _run_mu 内で判定する。

        runner launch の直前・直後検査に使う（C13: launch中のpause/resetによる
        世代失効を検出する）。外部I/Oは行わない。
        """
        with self._run_mu:
            r = self.rounds.get(session_id)
            return (
                r is not None
                and r.phase == "RUNNING"
                and r.run_id == run_id
            )

    def abort_run(self, session_id: str, message: str) -> None:
        """pause等の計画的中断: 実行中世代を失効させ実行受付を閉じる。

        RUNNING判定・世代失効・受付閉鎖・lock解放を同一 _run_mu 区間で行う
        （直前のfinishと競合してRESULTを上書きしない）。実行中でなければ
        lockの解放だけを行い、plan/roundは保持する（従来のpause契約）。
        runner killなどのI/Oは呼出し側が先に行う。
        """
        with self._run_mu:
            r = self.rounds[session_id]
            if r.phase != "RUNNING":
                self.run_lock = False
                return
            self._admission_open = False
            r.run_id = None
            r.pending_plan_id = None
            r.plan_obj = None
            r.game = None
            self.run_lock = False
            r.phase = "FAULT"
            r.message = message

    # ---- pause / reset ---------------------------------------------------
    def pause(self, session_id: str) -> dict:
        if self._sim_ctl is None:
            return {"paused": False, "reason": "sim_control_unavailable"}
        ok = self._sim_ctl.pause()
        r = self.rounds[session_id]
        if ok:
            r.message = "一時停止中（シミュレーション停止）"
        return {"paused": ok}

    def resume(self, session_id: str) -> dict:
        if self._sim_ctl is None:
            return {"resumed": False}
        ok = self._sim_ctl.resume()
        r = self.rounds[session_id]
        if ok:
            r.message = "実行を再開しました"
        return {"resumed": ok}

    def reset(
        self, session_id: str, restart_sim: bool = True, pre_restart=None
    ) -> dict:
        # C14: reset全体を _reset_mu で直列化する。並行resetは完了を待ち、
        # 直前のresetが新roundを確立済みなら重複再起動せず併合して返す。
        if not self._reset_mu.acquire(timeout=180):
            return {"reset": False, "reason": "reset_in_progress"}
        try:
            with self._run_mu:
                r = self.rounds[session_id]
                # 世界とrunnerは全sessionで共有する。別sessionの実行を
                # メニュー切替や古いタブからのresetで中断させない。
                if any(
                    other_id != session_id and other.phase == "RUNNING"
                    for other_id, other in self.rounds.items()
                ) or (self.run_lock and r.phase != "RUNNING"):
                    return {"reset": False, "reason": "other_session_running"}
                if time.monotonic() < self._kiosk_exit_until:
                    return {"reset": False, "reason": "kiosk_exiting"}
                if (
                    self._admission_open
                    and self._last_reset_session == session_id
                    and time.monotonic() - self._last_reset_mono < 2.0
                ):
                    return {"reset": True, "note": "coalesced"}
                # 1) 実行受付を閉じてから旧世代を失効させる（同一 _run_mu 区間で
                #    不可分に）。これ以後のstart_runは承認消費・run登録に進まない。
                self._admission_open = False
                # 別sessionのresetでも共有sim世界の旧targetは描画・開始不可。
                self._latest_spec_session = session_id
                # resetで失効する実行中runは、監督側で終端理由を確定記録する
                # （controllerがresult.jsonを書く前に終了したrunは「未観測」
                # と記録 — 存在しない打撃・保持結果を推測で生成しない）。
                interrupted = None
                if r.run_id is not None:
                    interrupted = {
                        "orphaned_run_id": r.run_id,
                        "round_id": r.round_id,
                        "terminated_by": "reset",
                        "result": "incomplete",
                    }
                    r.history.append(interrupted)
                r.run_id = None
                r.pending_plan_id = None
                r.plan_obj = None
                self.run_lock = False
                r.phase = "RESETTING"
                r.message = "初期化中…"
            # 2) 所有runnerの終了（kill）は世代失効の後・物理再起動の前に行う。
            #    外部I/O中は _run_mu を保持しない（停止操作を塞がない）。
            if pre_restart is not None:
                pre_restart()
            if restart_sim and self._sim_ctl is not None:
                ok, nonce = self._sim_ctl.restart()
                if not ok:
                    with self._run_mu:
                        r.phase = "FAULT"
                        r.message = (
                            "シミュレーションの再起動に失敗しました。運営を呼んでください。"
                        )
                    # 受付は閉じたまま（利用者は再度resetで復帰する）
                    return {"reset": False}
                self.boot_nonce = str(nonce) if nonce is not None else None
            with self._run_mu:
                self._new_round(session_id)
                if interrupted is not None:
                    # round置換で消えないよう、中断runの記録は新roundの履歴へ
                    # 引き継ぐ（旧round_id/run_idを名指しした監査証跡）。
                    self.rounds[session_id].history.append(interrupted)
                self._latest_spec_session = session_id
                self._admission_open = True
                self._last_reset_mono = time.monotonic()
                self._last_reset_session = session_id
            return {"reset": True}
        finally:
            self._reset_mu.release()


# ---- helpers -------------------------------------------------------------
_virtual_seq = {"version": 1}


def _virtual_world(round_id: str):
    """基本版の世界: 目的地上の仮想world（座標はランナー側のheading較正で実現）。"""
    from kotoba_contracts.world import ForbiddenRegion, World, WorldTarget

    targets = [
        WorldTarget(
            id=t.target_id,
            label=t.label,
            position_m=[t.distance_m, 0.0, 0.0],
            radius_m=0.15,
        )
        for t in COURSE["targets"]
    ]
    return World(
        world_version=_virtual_seq["version"],
        scene_sha256="0" * 64,
        targets=targets,
        forbidden_regions=[
            ForbiddenRegion(
                id="none", label="なし", polygon_xy_m=[[0, 0], [0, 0], [0, 0]]
            )
        ],
        capability_profile_sha256="1" * 64,
    )


def _target_label(target_id: str) -> str:
    from kotoba_api.world import target_by_id

    try:
        return target_by_id(target_id).label
    except KeyError:
        return target_id


def _target_distance(target_id: str) -> float:
    from kotoba_api.world import target_by_id

    try:
        return target_by_id(target_id).distance_m
    except KeyError:
        return 0.0


def _rebuild_plan(r: RoundState, session_id: str, plan_id: str):
    """approve/run は同一のplanオブジェクトを使う（canonical hash一致の保証）。"""
    if r.plan_obj is not None and r.pending_plan_id == plan_id:
        return r.plan_obj
    intent = parse_intent(r.last_intent)
    world = _virtual_world(r.round_id)
    return build_plan(
        intent,
        world,
        session_id=session_id,
        round_id=r.round_id,
        plan_id=plan_id,
        profile=PRODUCT_PROFILE,
        created_monotonic=time.monotonic(),
    )


def _approval_record(plan, r: RoundState, boot_nonce: Optional[str]):
    # run result由来のboot_nonceはintになり得る — ApprovalRecord(sim_boot_id: str)
    # はintを受理しないため、ここでstrへ正規化する（pydantic ValidationError=500
    # → worker側で502 backend_unreachableに正規化された事故の対策）。
    nonce = str(boot_nonce) if boot_nonce is not None else None
    if not nonce or nonce in ("boot-pending", "pending", "boot-unknown"):
        raise ApprovalError("boot_not_bound")
    from datetime import datetime, timedelta, timezone

    from kotoba_contracts.approval import ApprovalRecord

    return ApprovalRecord(
        session_id=plan.session_id,
        round_id=plan.round_id,
        plan_id=plan.plan_id,
        canonical_plan_sha256=canonical_plan_sha256(plan),
        world_version=plan.world_version,
        controller_profile_sha256="a"
        * 64,  # PRODUCT_PROFILEのsha256（デプロイ時に固定）
        sim_boot_id=nonce or "boot-pending",
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=90),
    )


def _now_utc():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)
