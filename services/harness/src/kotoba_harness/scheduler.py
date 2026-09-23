"""絶対deadline scheduler。受信処理が送信周期を前倒ししない。

旧スクリプトの欠陥（characterization test_02/03）の正しい仕様:
- handle_timeoutを受信待機のsleep代わりに使うと、着信のたびに早期returnして
  20Hzが崩壊していた。本schedulerは time.sleep でdeadlineまで待ち、受信workerと
  送信を完全に分離する。
- deadline逸脱はcatch-up密集送信ではなく ScheduleOverrun で中断する。
"""

import time
from typing import Callable, List, Sequence, Tuple

from kotoba_harness.errors import ScheduleOverrun

DEFAULT_LATE_TOLERANCE_S = 0.02


class DeadlineScheduler:
    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.clock = clock
        self.sleep = sleep
        self.actual_send_times: List[float] = []

    def run(
        self,
        send: Callable[[int], None],
        frame_count: int,
        hz: float,
        *,
        late_tolerance_s: float = DEFAULT_LATE_TOLERANCE_S,
    ) -> List[float]:
        """frame i を t0 + i/hz に送る。遅延は中断（catch-up禁止）。"""
        if hz <= 0:
            raise ValueError("hz_must_be_positive")
        self.actual_send_times = []
        t0 = self.clock()
        for index in range(frame_count):
            due = t0 + index / hz
            while True:
                now = self.clock()
                if now >= due:
                    break
                remaining = due - now
                self.sleep(min(remaining, 0.005))
            now = self.clock()
            if now > due + late_tolerance_s:
                raise ScheduleOverrun(
                    f"late_send:index={index} due={due:.4f} now={now:.4f}"
                )
            send(index)
            self.actual_send_times.append(self.clock())
        return self.actual_send_times


def frame_periods(actual_times: Sequence[float]) -> List[float]:
    return [b - a for a, b in zip(actual_times, actual_times[1:])]
