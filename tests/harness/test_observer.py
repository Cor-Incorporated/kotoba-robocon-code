"""観測検証（characterization test_04/05/06/07 の green側）。"""

import math
import struct

import pytest

from kotoba_harness.errors import HarnessError, InvalidPacket
from kotoba_harness.observer import LatestStateObserver, PacketValidator


def packet(
    *,
    stamp=0.0,
    fp=123,
    pos=(0.0, 0.0, 0.30),
    vel=(0.0, 0.0, 0.0),
    quat=(1.0, 0.0, 0.0, 0.0),
    num_ranges=0,
):
    return struct.pack(">qdi3d3d4d", fp, stamp, num_ranges, *pos, *vel, *quat)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


# --- test_06 の対: fingerprint検証 ---
def test_fingerprint_change_midboot_is_rejected():
    v = PacketValidator()
    v.validate(packet(fp=123), 0.0)
    with pytest.raises(InvalidPacket) as err:
        v.validate(packet(fp=999), 0.002)
    assert err.value.reason == "fingerprint_changed"


# --- test_07 の対: 非有限の拒否 ---
def test_nonfinite_position_is_rejected():
    v = PacketValidator()
    with pytest.raises(InvalidPacket) as err:
        v.validate(packet(pos=(math.nan, 0.0, 0.3)), 0.0)
    assert err.value.reason == "nonfinite"


def test_bad_quat_norm_is_rejected():
    v = PacketValidator()
    with pytest.raises(InvalidPacket):
        v.validate(packet(quat=(2.0, 0.0, 0.0, 0.0)), 0.0)


def test_minus_quat_is_accepted_same_rotation():
    """A10: q と -q は同一回転。符号だけの拒否をしない。"""
    v = PacketValidator()
    v.validate(packet(fp=1, quat=(1.0, 0.0, 0.0, 0.0)), 0.0)
    v.reset_boot()
    s = v.validate(packet(fp=1, quat=(-1.0, 0.0, 0.0, 0.0)), 0.002)
    assert s.quaternion_wxyz == (-1.0, 0.0, 0.0, 0.0)


# --- test_04 の対: 旧パケット再配信はfreshにならない ---
def test_replayed_old_packet_does_not_become_fresh():
    clock = FakeClock()
    obs = LatestStateObserver(clock=clock, expected_rate_hz=500.0)
    old_payload = packet(stamp=0.0)
    # t=10で1通だけ届く（500Hz streamなら0.2秒窓に~100通必要）
    clock.t = 10.0
    obs.on_packet(old_payload)
    with pytest.raises(HarnessError) as err:
        obs.fresh()
    assert "observation_stream_dead" in err.value.reason


def test_live_stream_is_fresh():
    clock = FakeClock()
    obs = LatestStateObserver(clock=clock, expected_rate_hz=500.0)
    for i in range(200):
        clock.t = i * 0.002
        obs.on_packet(packet(stamp=0.0))
    sample = obs.fresh()
    assert sample.position == (0.0, 0.0, 0.30)


def test_stale_by_age_is_refused():
    clock = FakeClock()
    obs = LatestStateObserver(clock=clock, expected_rate_hz=500.0)
    for i in range(200):
        clock.t = i * 0.002
        obs.on_packet(packet())
    clock.t += 0.5  # stream停止
    with pytest.raises(HarnessError) as err:
        obs.fresh()
    assert "stale_observation" in err.value.reason


def test_source_timestamp_is_carried_not_invented():
    """source時刻（現SDKは0）を勝手に現在時刻へ置き換えない（PR-B前提の契約）。"""
    clock = FakeClock()
    clock.t = 42.0
    obs = LatestStateObserver(clock=clock, expected_rate_hz=1.0)
    obs.on_packet(packet(stamp=0.0))
    sample = obs.newest()
    assert sample.source_timestamp == 0.0  # source時刻はsourceの値のまま
    assert sample.recv_monotonic == 42.0  # 受信時刻は別field
