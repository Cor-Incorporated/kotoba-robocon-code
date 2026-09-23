"""MotionExecutor — 観測閉ループの有界移動/旋回（C1: 通常runnerとgame controllerの共通コア）。

従来の `issued_wall + dur_s` 固定時間nudgeは、LPF残動と輸送遅延で実移動量が
読めなかった。本モジュールは各step開始時の観測姿勢で基準を固定し、
実観測の変位・yaw差で進捗を追い、目標到達→残動減衰(settling)→静止確認までを
一つの有限状態として扱う。送信を止めた瞬間をcompletedとは呼ばない。

物理指令の所有者は呼出し側のrun/controller一つのみ — 本モジュールは
観測値を受け取り stickベクトルと状態を返す純粋計算（LCM送信はしない）。

座標規約（B1実測）: fwd+ 前進 / lat+ 左 / yaw+ 左旋回。
yaw はquaternionから atan2(2(xy+wz), 1-2(y²+z²)) で導出し、±πをunwrapする。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# 方向→stickベクトル（SDK gamepad規約 — gameparse/controlと同一値）
STICK = {
    "forward": (0.6, 0.0, 0.0),
    "back": (-0.85, 0.0, 0.0),
    # 横移動はpolicyの非線形域が深く ±0.8 では摩擦圏内でほぼ進まない
    # （9/18 direction-probe実測: ±1.0で0.3–0.4m/指令の実績）。
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
    "turn_left": (0.0, 0.0, 0.8),
    "turn_right": (0.0, 0.0, -0.8),
}
# 速歩/走行の前進stick — pm01_edu rl_walking_example の command_scale_pos[0]
# 上限は1.0（これ以上はpolicyの指令範囲外）。横移動・旋回・後退は増幅しない
# （前進のみのpace profile — R5 F1）。実効速度は実測較正で検証する。
STICK_FAST_FORWARD = (1.0, 0.0, 0.0)
PACED_FORWARD_STICK = {
    "walk": STICK["forward"],
    "fast_walk": STICK_FAST_FORWARD,
    "run": STICK_FAST_FORWARD,  # 走行policy未検収 — walk policyの上限速度まで
}
VALID_PACES = ("walk", "fast_walk", "run")
# translate のローカル方向ベクトル（base座標系: x+前, y+左）
_DIR_LOCAL = {
    "forward": (1.0, 0.0),
    "back": (-1.0, 0.0),
    "left": (0.0, 1.0),
    "right": (0.0, -1.0),
}
_TURN_SIGN = {"left": 1.0, "right": -1.0, "around": -1.0}  # aroundの既定方向=右
TURN_DIRECTIONS = ("left", "right", "around")
TRANSLATE_DIRECTIONS = tuple(_DIR_LOCAL)

# 状態: accepted → applying → moving|turning → settling → completed/failed
ST_ACCEPTED = "accepted"
ST_APPLYING = "applying"
ST_MOVING = "moving"
ST_TURNING = "turning"
ST_SETTLING = "settling"
ST_JOGGING = "jogging"          # 継続移動（停止指示・境界・期限で終了）
ST_JOG_STOPPING = "jog_stopping"  # jog終了後の減速・静止確認
ST_COMPLETED = "completed"
ST_FAILED = "failed"
ST_ABORTED = "aborted"


def yaw_from_quat(quat) -> float:
    """w,x,y,z quaternion → yaw[rad]（runnerと同一式 — 機体+Xをworld XYへ射影）。"""
    w, x, y, z = quat
    return math.atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


@dataclass(frozen=True)
class MotionStep:
    """1つの運動step — translate=基準方向への距離移動 / turn=その場旋回。

    target: translateなら距離[m]、turnなら角度[rad]。
    stick: 適用するgamepadベクトル（dirからSTICK表で一意に決まる）。
    """

    action: str          # "translate" | "turn" | "jog"
    direction: str       # forward/back/left/right または left/right/around
    target: float        # m または rad（jogは0 — 目標を持たない継続動作）
    action_key: str = ""  # 監査・UI用の意味キー（例 "turn_right_90"）
    label: str = ""       # canonical表示（例 "右に90°向く"）
    pace: str = "walk"    # walk|fast_walk|run — 非walkはforwardのみ

    def __post_init__(self):
        if self.pace not in VALID_PACES:
            raise ValueError(f"bad_pace:{self.pace}")
        if self.pace != "walk" and not (
            self.action in ("translate", "jog")
            and self.direction == "forward"
        ):
            raise ValueError(f"pace_dir:{self.pace}:{self.direction}")
        if self.action == "translate":
            if self.direction not in _DIR_LOCAL:
                raise ValueError(f"bad_translate_dir:{self.direction}")
            if not (0.01 < self.target <= 2.0):
                raise ValueError(f"bad_translate_target:{self.target}")
        elif self.action == "turn":
            if self.direction not in _TURN_SIGN:
                raise ValueError(f"bad_turn_dir:{self.direction}")
            # deg→radの丸め（round(x,6)）でπを僅かに超える生成値を
            # 拒否せずπへ飽和させる（「後ろを向いて」→3.141593の実害）。
            if not (0.01 < self.target <= math.pi + 1e-3):
                raise ValueError(f"bad_turn_target:{self.target}")
            if self.target > math.pi:
                object.__setattr__(self, "target", math.pi)
        elif self.action == "jog":
            if self.direction not in _DIR_LOCAL:
                raise ValueError(f"bad_jog_dir:{self.direction}")
            if abs(self.target) > 1e-9:
                raise ValueError(f"bad_jog_target:{self.target}")
        else:
            raise ValueError(f"bad_action:{self.action}")

    @property
    def stick(self):
        if self.action in ("translate", "jog"):
            if self.direction == "forward":
                return PACED_FORWARD_STICK[self.pace]
            return STICK[self.direction]
        d = "around" if self.direction == "around" else self.direction
        return STICK["turn_left" if _TURN_SIGN[d] > 0 else "turn_right"]


@dataclass
class ExecProgress:
    """現在stepの進捗 — command_state DTOへそのまま写せる形状。"""

    step_index: int = 0
    step_count: int = 0
    action_key: str = ""
    label: str = ""
    status: str = ST_ACCEPTED
    reason: str | None = None
    # 進捗の実測値（表示・判定用）
    dist_m: float = 0.0
    target_m: float = 0.0
    angle_deg: float = 0.0
    target_deg: float = 0.0
    lateral_dev_m: float = 0.0
    yaw_drift_deg: float = 0.0

    def to_dict(self):
        return {
            "step_index": self.step_index,
            "step_count": self.step_count,
            "action_key": self.action_key,
            "label": self.label,
            "status": self.status,
            "reason": self.reason,
            "progress": {
                "dist_m": round(self.dist_m, 3),
                "target_m": round(self.target_m, 3),
                "angle_deg": round(self.angle_deg, 1),
                "target_deg": round(self.target_deg, 1),
                "lateral_dev_m": round(self.lateral_dev_m, 3),
                "yaw_drift_deg": round(self.yaw_drift_deg, 1),
            },
        }


@dataclass
class ExecProfile:
    """実測校正で固定する有界値。目標値ではなく運動profile（サーバー由来）。

    residual_k は「stick解除後の残動量 ≈ 解除時速度 × k」の係数。
    Thor実測（2026-09-21, pm01_edu 25_1009 policy）: 並進 k≈0.73・旋回
    k≈0.57。過大だと早期解除で恒常不足、過小は超過方向 — 不足側に寄せて
    静止後の補正バーストで収束させる（真の閉ループ）。"""

    settle_speed_mps: float = 0.10     # 静止確認の水平速度上限（translate）
    settle_yaw_rps: float = 0.12       # 静止確認の旋回速度上限（rad/s・EMA）
    settle_hold_s: float = 0.6         # 静止の継続要件（sim観測時間でなくwall）
    step_deadline_s: float = 25.0      # 1stepの絶対上限（目標・残動を含む）
    no_progress_s: float = 6.0         # 進捗が伸びないままの上限
    residual_k_lin: float = 0.8        # 残動推定: 距離 ≈ speed × k（実測0.73）
    residual_k_yaw: float = 0.6        # 旋回残動推定: 角度 ≈ yawrate × k（実測0.57）
    max_lateral_dev_m: float = 0.6     # translate中の垂直逸脱の異常上限
    max_yaw_drift_deg: float = 35.0    # translate中の回頭異常上限
    apply_ticks: int = 3               # applying→moving の最初の送出tick数
    min_progress_eps: float = 0.005    # 進捗判定の有意増分（m または rad）
    under_frac: float = 0.18           # 不足許容 = target×frac（下限はabs）
    under_abs_m: float = 0.06          # translate不足許容の下限
    under_abs_deg: float = 4.0         # turn不足許容の下限
    # 超過の合格許容（収束制御ではなく利用者目標の受入値 — min(abs,frac)）:
    #   turn: 15°±5 / 90°±10 / 180°±10 → min(10°, target×0.33)
    #   translate: 0.25m±0.10 / 1m±0.15 → min(0.15m, target×0.4)
    # 超過は不可逆（逆方向へ戻れない）ため補正せず failed にする。
    over_frac: float = 0.4
    over_abs_m: float = 0.15
    over_abs_deg: float = 10.0
    max_corrections: int = 5           # 静止後の補正バースト上限
    correction_ticks: int = 4          # 1補正あたりの固定送出tick数
    correction_scale: float = 0.6      # 補正時のstick倍率（低速接近で残留を抑える）
    yaw_rate_ema: float = 0.3          # 旋回速度EMAの新観測重み
    jog_deadline_s: float = 30.0       # 継続jog 1指令の絶対上限（fail-safe。UI表示値ではない）
    boundary_margin_m: float = 0.35    # arena境界手前で止まる最低余裕


class MotionExecutor:
    """順序付きstep列の観測閉ループ実行。

    使い方（呼出し側の20Hzループから）:
        ex = MotionExecutor(steps, profile)
        ex.begin(pos, quat)
        stick = ex.tick(pos, quat, vel)   # None=idle送信 / 完了・失敗は ex.done
    STOPは呼出し側が優先経路で処理し ex.abort() を呼ぶ（本モジュール自身は
    送信を持たないため「止める」＝tickがNoneを返すだけ）。
    """

    def __init__(self, steps, profile: ExecProfile | None = None,
                 bound_center=None, bound_r=None):
        if not steps:
            raise ValueError("empty_steps")
        self.steps = list(steps)
        self.prof = profile or ExecProfile()
        self.prog = ExecProgress(step_count=len(self.steps))
        self.events = []
        # jogの事前境界停止（centerからbound_rに触れる前に減速へ入る）。
        # 呼出し側がround/runの開始位置とarena半径を渡す。
        self._bound_center = bound_center
        self._bound_r = bound_r
        self._i = -1
        self._phase = ST_ACCEPTED
        self._start_xy = None
        self._start_yaw = 0.0
        self._dir_world = None
        self._prev_yaw = None
        self._prev_t = None
        self._unwrapped = 0.0
        self._t0 = None
        self._last_prog = 0.0
        self._last_prog_t = None
        self._settle_since = None
        self._apply_left = 0
        self._corrections = 0
        self._correcting = False
        self._yaw_rate = 0.0
        self._dir_vel = 0.0        # 方向整合した進捗速度EMA（符号付き・位置差分由来）
        self._prev_prog = None
        self._jog_stop_req = None  # 外部からのjog停止要求（理由）
        self._jog_end_reason = None
        self.done = False
        self.failed = False
        self.abort_reason = None

    # -- 内部 --------------------------------------------------------
    def _emit(self, **ev):
        self.events.append(ev)

    def _fail(self, reason, **extra):
        self._phase = ST_FAILED
        self.prog.status = ST_FAILED
        self.prog.reason = reason
        self.failed = True
        self.done = True
        self._emit(event="motion_step_failed", reason=reason,
                   step_index=self._i, **extra)

    def _begin_step(self, pos, quat):
        s = self.steps[self._i]
        self._start_xy = (pos[0], pos[1])
        self._start_yaw = yaw_from_quat(quat)
        self._prev_yaw = self._start_yaw
        self._prev_t = None
        self._unwrapped = 0.0
        self._t0 = None
        self._last_prog = 0.0
        self._last_prog_t = None
        self._settle_since = None
        self._apply_left = self.prof.apply_ticks
        self._corrections = 0
        self._correcting = False
        self._yaw_rate = 0.0
        self._dir_vel = 0.0
        self._prev_prog = None
        self._jog_stop_req = None
        self._jog_end_reason = None
        if s.action in ("translate", "jog"):
            lx, ly = _DIR_LOCAL[s.direction]
            c, sn = math.cos(self._start_yaw), math.sin(self._start_yaw)
            self._dir_world = (lx * c - ly * sn, lx * sn + ly * c)
        else:
            self._dir_world = None
        self._phase = ST_APPLYING
        self.prog.step_index = self._i
        self.prog.action_key = s.action_key
        self.prog.label = s.label
        self.prog.status = ST_APPLYING
        self.prog.reason = None
        self.prog.dist_m = 0.0
        self.prog.angle_deg = 0.0
        self.prog.lateral_dev_m = 0.0
        self.prog.yaw_drift_deg = 0.0
        if s.action == "translate":
            self.prog.target_m = s.target
            self.prog.target_deg = 0.0
        elif s.action == "jog":
            self.prog.target_m = 0.0   # 継続動作 — 既定の終端到達表示をしない
            self.prog.target_deg = 0.0
        else:
            self.prog.target_deg = math.degrees(s.target)
            self.prog.target_m = 0.0
        self._emit(
            event="motion_step_begin",
            step_index=self._i,
            action_key=s.action_key,
            action=s.action,
            direction=s.direction,
            target=s.target,
        )

    def _complete_step(self):
        s = self.steps[self._i]
        self._emit(
            event="motion_step_done",
            step_index=self._i,
            action_key=s.action_key,
            dist_m=round(self.prog.dist_m, 3),
            angle_deg=round(self.prog.angle_deg, 1),
            lateral_dev_m=round(self.prog.lateral_dev_m, 3),
            yaw_drift_deg=round(self.prog.yaw_drift_deg, 1),
        )
        if self._i + 1 >= len(self.steps):
            self._phase = ST_COMPLETED
            self.prog.status = ST_COMPLETED
            self.done = True
            return
        self._i += 1
        # 次stepの基準は呼出し側の次tick観測で固定（古いheadingを使い回さない）
        self._phase = ST_ACCEPTED
        self.prog.status = ST_ACCEPTED

    # -- 外部 API ----------------------------------------------------
    def begin(self, pos, quat):
        """実行開始 — 最初のstep基準を現観測で固定。"""
        self._i = 0
        self._begin_step(pos, quat)

    def abort(self, reason="stop"):
        """STOP/END等による中断。未達stepは実行しない。"""
        if not self.done:
            self._phase = ST_ABORTED
            self.prog.status = ST_ABORTED
            self.prog.reason = reason
            self.abort_reason = reason
            self.done = True
            self._emit(event="motion_aborted", reason=reason, step_index=self._i)

    def request_stop(self, reason="stop"):
        """jog中の外部停止要求（STOP指令・heartbeat欠落・境界等）。

        finite stepには適用しない — それらの中断は abort() を使う。
        jogは「止まる」が正常終了であり abort（失敗側）とは区別する。
        """
        if self._phase == ST_JOGGING and self._jog_stop_req is None:
            self._jog_stop_req = reason

    @property
    def jogging(self):
        """継続移動が進行中（heartbeat等の生存条件を呼出し側が監視する対象）。"""
        return (
            not self.done
            and self._i >= 0
            and self.steps[self._i].action == "jog"
        )

    def tick(self, pos, quat, vel, now=None):
        """1tick分の評価。戻り値: stickベクトル または None（idle送信）。

        pos=(x,y,z), quat=(w,x,y,z), vel=(vx,vy,vz)。
        """
        if self.done:
            return None
        import time as _time

        now = _time.monotonic() if now is None else now
        if self._phase == ST_ACCEPTED:
            # begin()未呼出し経路（controllerは次tickの新鮮な観測で
            # アンカーする設計）: _i=-1のままだと steps[-1]（末尾）を
            # 実行し、完了後にstep0を再実行してしまう（実害: 180°旋回が
            # 2回走り実質350°回転）。先頭stepへ初期化してから基準固定。
            if self._i < 0:
                self._i = 0
            self._begin_step(pos, quat)
        s = self.steps[self._i]
        if self._t0 is None:
            self._t0 = now
        elapsed = now - self._t0
        if s.action != "jog" and elapsed > self.prof.step_deadline_s:
            self._fail("step_timeout", elapsed_s=round(elapsed, 2))
            return None

        yaw = yaw_from_quat(quat)
        speed = math.hypot(vel[0], vel[1])
        dt = now - self._prev_t if self._prev_t else 0.0

        if s.action in ("translate", "jog"):
            dx = pos[0] - self._start_xy[0]
            dy = pos[1] - self._start_xy[1]
            prog = dx * self._dir_world[0] + dy * self._dir_world[1]
            lat = abs(dx * -self._dir_world[1] + dy * self._dir_world[0])
            yaw_drift = abs(_wrap_pi(yaw - self._start_yaw))
            self.prog.dist_m = max(0.0, prog)
            self.prog.lateral_dev_m = lat
            self.prog.yaw_drift_deg = math.degrees(yaw_drift)
            if lat > self.prof.max_lateral_dev_m:
                self._fail("lateral_drift", dev_m=round(lat, 3))
                return None
            if math.degrees(yaw_drift) > self.prof.max_yaw_drift_deg:
                self._fail("yaw_drift", deg=round(math.degrees(yaw_drift), 1))
                return None
            # 残動推定は「目標方向へ符号付きの進捗速度」（位置差分EMA）。
            # velの絶対値を使うと後退・側滑中も前向き残動と誤認して
            # 早期解除する（実測不具合）。逆方向へ動いていれば残動0。
            if dt > 1e-3 and self._prev_prog is not None:
                inst_v = (prog - self._prev_prog) / dt
                a = self.prof.yaw_rate_ema
                self._dir_vel = a * inst_v + (1 - a) * self._dir_vel
            self._prev_prog = prog
            residual = max(0.0, self._dir_vel) * self.prof.residual_k_lin
            reached = (
                s.action == "translate" and prog + residual >= s.target
            )
            prog_val = prog
        else:  # turn
            dyaw = _wrap_pi(yaw - self._prev_yaw)
            self._prev_yaw = yaw
            self._unwrapped += dyaw
            sign = _TURN_SIGN[s.direction]
            prog_val = self._unwrapped * sign
            self.prog.angle_deg = math.degrees(max(0.0, prog_val))
            # 角速度は観測差分のEMA（velのyaw成分は観測に無い）。
            # settle判定もこの値を使う — その場旋回は水平速度≈0なので
            # 並進速度を見ると残旋回中に「静止」誤判定する（実測不具合）。
            inst = abs(dyaw / dt) if dt > 1e-3 else 0.0
            a = self.prof.yaw_rate_ema
            self._yaw_rate = a * inst + (1 - a) * self._yaw_rate
            residual = self._yaw_rate * self.prof.residual_k_yaw
            reached = prog_val + residual >= s.target
        self._prev_t = now

        # 進捗停滞の検出
        if prog_val > self._last_prog + self.prof.min_progress_eps:
            self._last_prog = prog_val
            self._last_prog_t = now
        elif self._last_prog_t is None:
            self._last_prog_t = now
        elif now - self._last_prog_t > self.prof.no_progress_s and self._phase in (
            ST_MOVING, ST_TURNING, ST_APPLYING, ST_JOGGING,
        ):
            self._fail("no_progress", progress=round(prog_val, 3))
            return None

        if self._phase == ST_APPLYING:
            # 開始時点で既に目標を満たす（複合stepの残動等）はstickを出さず
            # settlingへ。そうでなければ最小適用tick数だけstickを送る。
            # 補正burstは残動推定を再信用せず固定tick数だけ低速送出し、
            # 直後のsettleで実測し直す（推定誤差を再帰しない閉ループ）。
            if self._apply_left > 0 and (self._correcting or not reached):
                self._apply_left -= 1
                self.prog.status = ST_APPLYING
                if self._correcting:
                    k = self.prof.correction_scale
                    return (s.stick[0] * k, s.stick[1] * k, s.stick[2] * k)
                return s.stick
            if self._correcting:
                self._correcting = False
                self._phase = ST_SETTLING
                self._settle_since = None
            elif not reached:
                if s.action == "jog":
                    self._phase = ST_JOGGING
                else:
                    self._phase = (
                        ST_MOVING if s.action == "translate" else ST_TURNING
                    )
            else:
                self._phase = ST_SETTLING
                self._settle_since = None

        if self._phase == ST_JOGGING:
            self.prog.status = ST_JOGGING
            # 終了条件: 外部停止要求（STOP/heartbeat欠落）・絶対上限・境界。
            # 「止まる」は正常終了 — 距離未達のfailedとは別の記録を残す。
            stop_reason = self._jog_stop_req
            if stop_reason is None and elapsed > self.prof.jog_deadline_s:
                stop_reason = "jog_timeout"
            if (
                stop_reason is None
                and self._bound_center is not None
                and self._bound_r is not None
            ):
                dist_c = math.hypot(
                    pos[0] - self._bound_center[0],
                    pos[1] - self._bound_center[1],
                )
                margin = (
                    self.prof.boundary_margin_m
                    + max(0.0, self._dir_vel) * self.prof.residual_k_lin
                )
                if dist_c + margin >= self._bound_r:
                    stop_reason = "boundary"
            if stop_reason is not None:
                self._jog_stop_req = None
                self._jog_end_reason = stop_reason
                self._phase = ST_JOG_STOPPING
                self._settle_since = None
                self._emit(
                    event="motion_jog_stop",
                    reason=stop_reason,
                    step_index=self._i,
                    dist_m=round(self.prog.dist_m, 3),
                )
                return None
            return s.stick

        if self._phase == ST_JOG_STOPPING:
            self.prog.status = ST_JOG_STOPPING
            if speed < self.prof.settle_speed_mps:
                if self._settle_since is None:
                    self._settle_since = now
                elif now - self._settle_since >= self.prof.settle_hold_s:
                    reason = self._jog_end_reason or "stop"
                    self.prog.reason = reason
                    self._emit(
                        event="motion_step_done",
                        step_index=self._i,
                        action_key=s.action_key,
                        dist_m=round(self.prog.dist_m, 3),
                        reason=reason,
                    )
                    # jogは単独・または列の末尾でのみ生成される
                    self._phase = ST_COMPLETED
                    self.prog.status = ST_COMPLETED
                    self.done = True
            else:
                self._settle_since = None
            return None

        if self._phase in (ST_MOVING, ST_TURNING):
            if reached:
                # 目標到達（残動込み推定）→ stick解除して減衰待ちへ
                self._phase = ST_SETTLING
                self._settle_since = None
                self._emit(event="motion_reached", step_index=self._i,
                           progress=round(prog_val, 3))
            else:
                self.prog.status = self._phase
                return s.stick

        if self._phase == ST_SETTLING:
            self.prog.status = ST_SETTLING
            # 静止確認はstep種別の指標で見る — その場旋回は水平速度≈0
            # のため translate 指標を使うと残旋回中に誤完了する。
            if s.action == "translate":
                still = speed < self.prof.settle_speed_mps
            else:
                still = self._yaw_rate < self.prof.settle_yaw_rps
            if still:
                if self._settle_since is None:
                    self._settle_since = now
                elif now - self._settle_since >= self.prof.settle_hold_s:
                    # 静止して初めて不足/超過を確定する。不足は補正
                    # バーストで収束（残動推定の誤差を真の閉ループで解消）。
                    if s.action == "translate":
                        tol = max(
                            self.prof.under_abs_m,
                            s.target * self.prof.under_frac,
                        )
                    else:
                        tol = max(
                            math.radians(self.prof.under_abs_deg),
                            s.target * self.prof.under_frac,
                        )
                    shortfall = s.target - prog_val
                    if (
                        shortfall > tol
                        and self._corrections < self.prof.max_corrections
                    ):
                        self._corrections += 1
                        self._correcting = True
                        self._phase = ST_APPLYING
                        self._apply_left = self.prof.correction_ticks
                        self._settle_since = None
                        # 補正burstに進捗猶予を与える（停滞時計を継続させない）
                        self._last_prog_t = now
                        self._emit(
                            event="motion_correct",
                            step_index=self._i,
                            correction=self._corrections,
                            shortfall=round(shortfall, 3),
                        )
                        return None
                    if shortfall > tol:
                        self._fail(
                            "undershoot",
                            progress=round(prog_val, 3),
                            target=round(s.target, 3),
                        )
                        return None
                    # 超過は受入許容値（制御収束値ではなく利用者目標）で判定。
                    # 大きく超過したまま次stepへ進ませない（目標未達=failed）。
                    if s.action == "translate":
                        tol_over = min(
                            self.prof.over_abs_m,
                            s.target * self.prof.over_frac,
                        )
                    else:
                        tol_over = min(
                            math.radians(self.prof.over_abs_deg),
                            s.target * self.prof.over_frac,
                        )
                    over = prog_val - s.target
                    if over > tol_over:
                        self._emit(
                            event="motion_overshoot",
                            step_index=self._i,
                            over=round(over, 3),
                        )
                        self._fail(
                            "overshoot",
                            progress=round(prog_val, 3),
                            target=round(s.target, 3),
                        )
                        return None
                    if over > tol:
                        self._emit(
                            event="motion_overshoot",
                            step_index=self._i,
                            over=round(over, 3),
                        )
                    self._complete_step()
            else:
                self._settle_since = None
            return None
        return s.stick


__all__ = [
    "STICK",
    "MotionStep",
    "MotionExecutor",
    "ExecProfile",
    "ExecProgress",
    "yaw_from_quat",
    "TURN_DIRECTIONS",
    "TRANSLATE_DIRECTIONS",
]
