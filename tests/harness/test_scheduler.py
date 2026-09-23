"""絶対deadline scheduler（characterization test_02/03 の green側）。

旧実装は handle_timeout を待機に使ったため、着信のたびに早期returnし
20Hz×20フレームが0.04秒で送り切れるバーストになっていた。
"""

import pytest

from kotoba_harness.errors import ScheduleOverrun
from kotoba_harness.scheduler import DeadlineScheduler, frame_periods


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.t

    def sleep(self, dt):
        assert dt >= 0
        self.sleeps.append(dt)
        self.t += dt


def test_period_holds_even_when_receiver_is_chatty():
    """受信が高頻度で来ても送信周期は前倒しされない（test_02 の対）。"""
    clock = FakeClock()
    sched = DeadlineScheduler(clock=clock.monotonic, sleep=clock.sleep)
    sent_at = []
    # chatty receiver相当: clock.t を微小に進める割り込みが大量に来る
    original_sleep = clock.sleep

    def chatty_sleep(dt):
        original_sleep(min(dt, 0.002))  # 着信で早期returnする状況を模擬
        clock.t += 0.0005  # 受信処理による時計進行

    sched.sleep = chatty_sleep
    sched.run(lambda i: sent_at.append(clock.monotonic()), 20, 20.0)
    total = sent_at[-1] - sent_at[0]
    assert total == pytest.approx(19 / 20.0, abs=0.02), (
        f"20Hzが崩壊: total={total:.3f}s"
    )
    periods = frame_periods(sent_at)
    assert max(periods) <= 0.051 and min(periods) >= 0.049


def test_no_catch_up_after_long_pause():
    """遅延後のcatch-up密集送信はせず、ScheduleOverrunで中断する。"""
    clock = FakeClock()
    sched = DeadlineScheduler(clock=clock.monotonic, sleep=clock.sleep)
    sent = []

    def stalling_sleep(dt):
        clock.t += 0.5  # 大幅な遅延を注入

    sched.sleep = stalling_sleep
    with pytest.raises(ScheduleOverrun):
        sched.run(lambda i: sent.append(i), 10, 20.0)
    assert len(sent) <= 1  # 追い上げ送信は発生しない


def test_actual_send_times_are_recorded():
    clock = FakeClock()
    sched = DeadlineScheduler(clock=clock.monotonic, sleep=clock.sleep)
    times = sched.run(lambda i: None, 5, 20.0)
    assert len(times) == 5
    assert times[0] == 0.0 and times[4] == pytest.approx(0.2)
