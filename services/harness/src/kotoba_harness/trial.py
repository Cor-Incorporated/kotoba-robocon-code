"""Trial状態機械・立位ready gate・転倒ラッチ・通常停止・採点。

旧スクリプトの欠陥（characterization test_09/10/12/13）の正しい仕様:
- 転立位（低height）から歩行を開始しない（ready gate）
- 終了時の姿勢・高さを判定せず転倒tailがPASSしない（posture考慮スコア）
- 減速なしのwalk→pd_stand直接遷移を通常停止と呼ばない
- 校正と評価を分離し、評価目標を移動前に固定する
"""

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from kotoba_harness.errors import HarnessError

# PM01 校正値（8/28実測 evidence: 立位 z=0.820, walk中 z_min=0.798, うつ伏せ z=0.14-0.20）
STAND_HEIGHT_MIN_M = 0.75
STAND_HEIGHT_MAX_M = 0.90
FALL_HEIGHT_M = 0.50
FALL_TILT_DEG = 45.0
READY_TILT_DEG = 15.0
READY_SPEED_MPS = 0.05
READY_HOLD_S = 2.0
TAIL_HOLD_S = 5.0


class Phase(str, Enum):
    STARTUP = "startup"
    READY_WAIT = "ready_wait"
    WALK = "walk"
    DECELERATING = "decelerating"
    SETTLING = "settling"
    TAIL = "tail"
    ABORT = "abort"
    DONE = "done"


def up_vector_tilt_deg(quat_wxyz) -> float:
    """四元数（wxyz）で機体上方向をworld回転し、world upからの傾き角を返す。

    A10: q と -q は同一回転。符号単独判定はしない。純yawはtilt 0。
    """
    w, x, y, z = quat_wxyz
    # world up (0,0,1) を quaternion で回転した z 成分 = 1 - 2*(x^2 + y^2)
    z_component = 1.0 - 2.0 * (x * x + y * y)
    z_component = max(-1.0, min(1.0, z_component))
    return math.degrees(math.acos(z_component))


@dataclass
class FallLatch:
    """一度成立した転倒は phases を記録して二度と解除しない。"""

    fallen: bool = False
    phases: List[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def observe(self, phase: str, sample) -> None:
        if self.fallen:
            return
        height = sample.position[2]
        tilt = up_vector_tilt_deg(sample.quaternion_wxyz)
        if height < FALL_HEIGHT_M or tilt > FALL_TILT_DEG:
            self.fallen = True
            self.phases.append(phase)
            self.detail = {"height": round(height, 4), "tilt_deg": round(tilt, 2)}

    @property
    def category(self) -> Optional[str]:
        if not self.fallen:
            return None
        phase = self.phases[0]
        if phase in (Phase.STARTUP.value, Phase.READY_WAIT.value):
            return "prep_fall"
        if phase == Phase.TAIL.value:
            return "tail_fall"
        return "walk_fall"


@dataclass
class ReadyGate:
    """立位readyの連続成立を確認する。固定秒待ちを証明にしない。"""

    hold_s: float = READY_HOLD_S
    _ok_since: Optional[float] = None
    ready_latched: bool = False

    def observe(self, sample, now: float, clock) -> bool:
        """READY_WAIT中のみ呼ぶこと。latchは立位が続く限り有効。"""
        height = sample.position[2]
        tilt = up_vector_tilt_deg(sample.quaternion_wxyz)
        speed = math.hypot(sample.velocity[0], sample.velocity[1])
        standing = (
            STAND_HEIGHT_MIN_M <= height <= STAND_HEIGHT_MAX_M
            and tilt <= READY_TILT_DEG
            and speed <= READY_SPEED_MPS
        )
        if not standing:
            self._ok_since = None
            self.ready_latched = False  # ready待ち中の姿勢崩れは無効化
            return False
        if self._ok_since is None:
            self._ok_since = now
        if now - self._ok_since >= self.hold_s:
            self.ready_latched = True
        return self.ready_latched

    def invalidate(self) -> None:
        self._ok_since = None
        self.ready_latched = False


@dataclass
class TiltMonitor:
    """姿勢の逸脱（転倒ではない範囲）を記録する。"""

    max_tilt_deg: float = 0.0

    def observe(self, sample) -> float:
        tilt = up_vector_tilt_deg(sample.quaternion_wxyz)
        self.max_tilt_deg = max(self.max_tilt_deg, tilt)
        return tilt


@dataclass
class NormalStopPolicy:
    """通常停止: 減速→静止確認→合法なpd_stand遷移→立位保持。

    稼働中のFSMを直接切り替えただけで静止とはしない。
    """

    stop_speed_mps: float = 0.05
    decel_timeout_s: float = 6.0
    still_confirm_s: float = 1.0
    _still_since: Optional[float] = None

    def decelerate_done(self, sample, now: float, started: float) -> bool:
        speed = math.hypot(sample.velocity[0], sample.velocity[1])
        if speed <= self.stop_speed_mps:
            if self._still_since is None:
                self._still_since = now
            if now - self._still_since >= self.still_confirm_s:
                return True
        else:
            self._still_since = None
        if now - started > self.decel_timeout_s:
            raise HarnessError("deceleration_timeout")
        return False


@dataclass
class Scorer:
    """終了判定。XY誤差・速度に 姿勢・高さ・保持 を必ず含める。"""

    tolerance_m: float = 0.15
    stop_speed_mps: float = 0.05

    def verdict(
        self,
        *,
        ready_latched: bool,
        fall_category: Optional[str],
        tail_ok: bool,
        tail_held_s: float,
        final_sample,
        target_xy: tuple,
    ) -> dict:
        pos = final_sample.position
        err = math.hypot(pos[0] - target_xy[0], pos[1] - target_xy[1])
        speed = math.hypot(final_sample.velocity[0], final_sample.velocity[1])
        height = pos[2]
        tilt = up_vector_tilt_deg(final_sample.quaternion_wxyz)
        posture_ok = (
            STAND_HEIGHT_MIN_M <= height <= STAND_HEIGHT_MAX_M
            and tilt <= READY_TILT_DEG
        )
        passed = (
            ready_latched
            and fall_category is None
            and tail_ok
            and tail_held_s >= TAIL_HOLD_S
            and err <= self.tolerance_m
            and speed <= self.stop_speed_mps
            and posture_ok
        )
        reasons = []
        if not ready_latched:
            reasons.append("never_ready")
        if fall_category:
            reasons.append(f"fall:{fall_category}")
        if not tail_ok:
            reasons.append("tail_not_ok")
        if tail_held_s < TAIL_HOLD_S:
            reasons.append(f"tail_hold_short:{tail_held_s:.2f}s")
        if err > self.tolerance_m:
            reasons.append(f"err:{err:.3f}m")
        if speed > self.stop_speed_mps:
            reasons.append(f"speed:{speed:.3f}m/s")
        if not posture_ok:
            reasons.append(f"posture:h={height:.3f} tilt={tilt:.1f}deg")
        return {
            "verdict": "PASS" if passed else "FAIL",
            "reasons": reasons,
            "err_m": round(err, 4),
            "speed_mps": round(speed, 4),
            "height_m": round(height, 4),
            "tilt_deg": round(tilt, 2),
        }


class TrialFSM:
    """STARTUP→READY_WAIT→WALK→DECELERATING→SETTLING→TAIL→DONE（ABORT例外系）。"""

    LEGAL = {
        Phase.STARTUP: {Phase.READY_WAIT, Phase.ABORT},
        Phase.READY_WAIT: {Phase.WALK, Phase.ABORT},
        Phase.WALK: {Phase.DECELERATING, Phase.ABORT},
        Phase.DECELERATING: {Phase.SETTLING, Phase.WALK, Phase.ABORT},
        Phase.SETTLING: {Phase.TAIL, Phase.ABORT},
        Phase.TAIL: {Phase.DONE, Phase.ABORT},
        Phase.ABORT: {Phase.DONE},
        Phase.DONE: set(),
    }

    def __init__(self) -> None:
        self.phase = Phase.STARTUP

    def transition(self, to: Phase) -> None:
        if to not in self.LEGAL[self.phase]:
            raise HarnessError(f"illegal_transition:{self.phase.value}->{to.value}")
        self.phase = to
