"""build_move_command の仕様固定 — 符号・軸配置・bounds検証。

ことばでスイカ割りの方向指令（B1）の出口。SDK実装契約（Thor配備ソース照合）:
- analog[2] (LeftStick_X)  → vx 前後
- analog[3] → -LeftStick_Y → vy 左右（adapterが -analog[3] を代入）
- analog[5] → -RightStick_Y → yaw速度（adapterが -analog[5] を代入）
pm01_edu rl_walking_example default.yaml: command_scale=[1.0, 0.4, 1.0]
"""

import math
import struct

import pytest

from kotoba_harness.auth import SIM_PROFILE, build_move_command
from kotoba_harness.errors import AuthorizationRefused

from test_auth import CHANNEL, FakeHandle, _gateway


def _unpack(frame: bytes):
    assert len(frame) == 112
    digital = struct.unpack_from(">12i", frame, 16)
    analog = struct.unpack_from(">6d", frame, 16 + 48)
    return digital, analog


def test_forward_places_stick_on_left_x():
    _, analog = _unpack(build_move_command(0.6, 0.0, 0.0, SIM_PROFILE))
    assert analog[2] == pytest.approx(0.6)  # LeftStick_X → vx+
    assert analog[3] == pytest.approx(0.0)
    assert analog[5] == pytest.approx(0.0)


def test_backward_negates_left_x():
    _, analog = _unpack(build_move_command(-0.4, 0.0, 0.0, SIM_PROFILE))
    assert analog[2] == pytest.approx(-0.4)


def test_lateral_is_negated_into_left_y():
    # adapter: LeftStick_Y = -analog[3] → lat+ を送るには analog[3] が負
    _, analog = _unpack(build_move_command(0.0, 0.8, 0.0, SIM_PROFILE))
    assert analog[3] == pytest.approx(-0.8)
    assert analog[2] == pytest.approx(0.0)


def test_yaw_is_negated_into_right_y():
    # adapter: RightStick_Y = -analog[5] → yaw+ を送るには analog[5] が負
    _, analog = _unpack(build_move_command(0.0, 0.0, 0.6, SIM_PROFILE))
    assert analog[5] == pytest.approx(-0.6)
    assert analog[4] == pytest.approx(0.0)  # RightStick_X は歩行に未使用


def test_zero_command_is_idle_frame():
    digital, analog = _unpack(build_move_command(0.0, 0.0, 0.0, SIM_PROFILE))
    assert all(d == 0 for d in digital)
    assert all(a == 0.0 for a in analog)


def test_combined_axes_are_independent():
    _, analog = _unpack(build_move_command(0.3, -0.2, 0.5, SIM_PROFILE))
    assert analog[2] == pytest.approx(0.3)
    assert analog[3] == pytest.approx(0.2)  # -lat
    assert analog[5] == pytest.approx(-0.5)  # -yaw


# --- bounds: implied速度がprofile上限を超える入力は拒否 ---
def test_fwd_beyond_linear_bound_refused():
    with pytest.raises(AuthorizationRefused) as err:
        build_move_command(0.9, 0.0, 0.0, SIM_PROFILE)  # 0.9*1.0 > 0.85
    assert err.value.reason == "analog_out_of_bounds"


def test_lat_beyond_lateral_bound_refused():
    import dataclasses

    # vy_scale=0.4・|stick|<=1 では SIM_PROFILE の 0.4 上限に到達しないため、
    # implied検査の実効を厳しいprofileで検証する
    tight = dataclasses.replace(SIM_PROFILE, max_lateral_mps=0.1)
    with pytest.raises(AuthorizationRefused):
        build_move_command(0.0, 0.5, 0.0, tight)  # 0.5*0.4=0.2 > 0.1


def test_yaw_beyond_yaw_bound_refused():
    with pytest.raises(AuthorizationRefused):
        build_move_command(0.0, 0.0, 0.9, SIM_PROFILE)  # 0.9*1.0 > 0.8


def test_stick_out_of_unit_range_refused():
    for bad in (1.01, -1.5):
        with pytest.raises(AuthorizationRefused):
            build_move_command(bad, 0.0, 0.0, SIM_PROFILE)


def test_nonfinite_inputs_refused():
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(AuthorizationRefused) as err:
            build_move_command(bad, 0.0, 0.0, SIM_PROFILE)
        assert err.value.reason == "nonfinite_command"


def test_boundary_values_accepted():
    # implied == max ちょうどは受理（上限超過のみ拒否）
    build_move_command(0.85, 0.0, 0.0, SIM_PROFILE)
    build_move_command(0.0, 1.0, 0.0, SIM_PROFILE)  # 1.0*0.4=0.4 == max
    build_move_command(0.0, 0.0, 0.8, SIM_PROFILE)


# --- gateway経路: manifestのprofileで検証してからissue ---
def test_gateway_prepare_move_validates_then_issues(monkeypatch):
    handle = FakeHandle()
    gw = _gateway(monkeypatch, handle=handle)
    try:
        frame = gw.prepare_move(0.5, 0.0, 0.0)
        gw.issue(frame, now_monotonic=1.0, command_name="move_fwd")
        assert handle.sent == [(CHANNEL, frame)]
        assert gw.sent_log == [(1.0, "move_fwd")]
        with pytest.raises(AuthorizationRefused):
            gw.prepare_move(2.0, 0.0, 0.0)  # bounds外は構築時点で拒否
        assert len(handle.sent) == 1  # 拒否分は送信されていない
    finally:
        gw.close()
