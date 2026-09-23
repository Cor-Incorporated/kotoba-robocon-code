"""観測の検証とlatest-state受信。

旧スクリプトの欠陥（characterization test_04/05/06/07）の正しい仕様:
- callbackに渡された時刻だけで旧パケットをfreshと呼ばない
  （stream速度の検証をfresh条件に含む）
- fingerprint・finite・quaternion normを検証する
- fingerprintがboot途中で変わったら拒否する
"""

import math
import struct
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from kotoba_harness.errors import HarnessError, InvalidPacket

# data::SimState hand-decode layout (legacy record_sim_state.py と同一)
_HEADER = struct.Struct(">qdi")
_MIN_PAYLOAD = 20 + 3 * 8  # header + joint配列最少（num_ranges>=0）
_QUAT_NORM_TOL = 0.2


@dataclass(frozen=True)
class SimSample:
    arrival_seq: int
    recv_monotonic: float
    source_timestamp: float
    position: tuple
    velocity: tuple
    quaternion_wxyz: tuple


class PacketValidator:
    """単一boot内で同一fingerprint・有限値・正常quaternionを要求する。

    step_seqの逆順・重複は stream故障ではなく UDP再順序/遅着として
    「ドロップ」扱いする（§3.6: latest-state受信・遅着拒否）。
    """

    DROP_WINDOW = 256  # max_seen より古い到着は遅着として破棄

    def __init__(self) -> None:
        self._boot_fingerprint: Optional[int] = None
        self._arrival_counter = 0
        self._max_seq = -1
        self.dropped_late = 0

    def reset_boot(self) -> None:
        self._boot_fingerprint = None
        self._max_seq = -1
        self.dropped_late = 0

    def _check_step_seq(self, step_seq: int) -> bool:
        """step_seqを持つchannel用。遅着(<= max_seen)はドロップ、Falseを返す。"""
        if self._max_seq >= 0:
            if step_seq <= self._max_seq:
                self.dropped_late += 1
                return False
            if step_seq > self._max_seq + self.DROP_WINDOW * 1000:
                raise InvalidPacket("seq_gap_too_large")
        self._max_seq = max(self._max_seq, step_seq)
        return True

    def validate(self, payload: bytes, recv_monotonic: float) -> SimSample:
        if len(payload) < _MIN_PAYLOAD:
            raise InvalidPacket("bad_size")
        fingerprint, source_ts, num_ranges = _HEADER.unpack_from(payload, 0)
        if self._boot_fingerprint is None:
            self._boot_fingerprint = fingerprint
        elif fingerprint != self._boot_fingerprint:
            raise InvalidPacket("fingerprint_changed")
        off = _HEADER.size + 3 * num_ranges * 8
        if num_ranges < 0 or off + 24 + 24 + 32 > len(payload):
            raise InvalidPacket("num_ranges_invalid")
        pos = struct.unpack_from(">3d", payload, off)
        off += 24
        vel = struct.unpack_from(">3d", payload, off)
        off += 24
        quat = struct.unpack_from(">4d", payload, off)
        values = (*pos, *vel, *quat)
        if any(not math.isfinite(v) for v in values):
            raise InvalidPacket("nonfinite")
        norm = math.sqrt(sum(q * q for q in quat))
        if abs(norm - 1.0) > _QUAT_NORM_TOL:
            raise InvalidPacket("bad_quat_norm")
        self._arrival_counter += 1
        return SimSample(
            arrival_seq=self._arrival_counter,
            recv_monotonic=recv_monotonic,
            source_timestamp=source_ts,
            position=pos,
            velocity=vel,
            quaternion_wxyz=quat,
        )


class ClockGate:
    """kotoba_sim_clock 受付判定。制御・採点・表示で共通して使う。

    同一boot内では seq と sim時刻の両方の有限性・単調増加を要求する:
    - seq増・sim時刻凍結（publisherがstepせず再配信し続ける異常）は受理しない
    - sim時刻の逆行・非有限値も受理しない
    - boot(nonce)変更時にseq/sim_t基準をresetする
    - 一度退役したboot(nonce)への遅着再配信は「新しいboot」として受理しない
      （nonceは boot ごとの乱数 — 再出現は旧パケットの遅着であり得る）
    """

    _MAX_RETIRED = 16  # 退役nonce履歴は小さく有界に保つ

    def __init__(self) -> None:
        self.nonce: Optional[int] = None
        self._max_seq = -1
        self._max_sim_t = float("-inf")
        self._retired: List[int] = []
        self.boot_changes = 0
        self.dropped_late = 0
        self.dropped_frozen = 0
        self.dropped_retired = 0

    def accept(self, nonce: int, seq: int, sim_t: float) -> tuple[bool, bool]:
        """(受理したか, bootが変わったか) を返す。"""
        if not math.isfinite(sim_t):
            self.dropped_frozen += 1
            return False, False
        if self.nonce is None:
            # 最初の標本: baselinesを初期化
            self.nonce = nonce
            self._max_seq = seq
            self._max_sim_t = sim_t
            return True, False
        if nonce != self.nonce:
            if nonce in self._retired:
                # 退役bootの遅着再配信 — 新しいresetとして採用しない
                self.dropped_retired += 1
                return False, False
            # 正当なboot遷移: 旧nonceを退役登録し基準をreset
            self._retired.append(self.nonce)
            if len(self._retired) > self._MAX_RETIRED:
                self._retired.pop(0)
            self.nonce = nonce
            self._max_seq = seq
            self._max_sim_t = sim_t
            self.boot_changes += 1
            return True, True
        if seq <= self._max_seq:
            # 遅着・再順序・凍結seqの再配信
            self.dropped_late += 1
            return False, False
        if sim_t <= self._max_sim_t:
            # seqだけ進んで物理時刻が進まない/逆行する入力は制御へ供給しない
            self.dropped_frozen += 1
            return False, False
        self._max_seq = seq
        self._max_sim_t = sim_t
        return True, False


class LatestStateObserver:
    """最新状態受信。fresh判定は 受信時刻の新しさ と stream速度 の両方。

    source timestamp（現SDKでは常に0 — A09）だけに依存せず、
    単位時間あたりの有効サンプル数でstreamの生存を確認する。
    """

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        expected_rate_hz: float = 500.0,
    ) -> None:
        self.validator = PacketValidator()
        self.clock = clock
        self.expected_rate_hz = expected_rate_hz
        self.samples: List[SimSample] = []

    def on_packet(self, payload: bytes) -> None:
        sample = self.validator.validate(payload, self.clock())
        self.samples.append(sample)

    def newest(self) -> Optional[SimSample]:
        return self.samples[-1] if self.samples else None

    def _recent_count(self, window_s: float) -> int:
        now = self.clock()
        cutoff = now - window_s
        return sum(1 for s in self.samples if s.recv_monotonic >= cutoff)

    def fresh(
        self,
        max_age_s: float = 0.1,
        min_rate_hz: float | None = None,
    ) -> SimSample:
        """新鮮な最新サンプルを返す。streamが死んでいれば HarnessError。"""
        sample = self.newest()
        if sample is None:
            raise HarnessError("no_observation")
        now = self.clock()
        age = now - sample.recv_monotonic
        if age > max_age_s:
            raise HarnessError(f"stale_observation:age={age:.3f}s")
        rate = min_rate_hz if min_rate_hz is not None else self.expected_rate_hz
        window = 0.2
        recent = self._recent_count(window)
        if recent < max(1, int(rate * window * 0.5)):
            raise HarnessError(
                f"observation_stream_dead:recent={recent} window={window}s"
            )
        return sample

    def backlog_pending(self) -> int:
        """未処理の到着分（テスト用の概念カウンタ。実LCMではhandleが持つ）。"""
        return 0
