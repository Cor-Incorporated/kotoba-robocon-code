"""送信境界の正しい仕様（characterization test_08/11 の green側）。"""

import pytest

from kotoba_harness.auth import SIM_PROFILE, RunManifest, SendGateway, build_command
from kotoba_harness.errors import AuthorizationRefused

CHANNEL = "virtual_gamepad/gamepad_keys"


class FakeHandle:
    def __init__(self):
        self.sent = []

    def publish(self, channel, payload):
        self.sent.append((channel, payload))


def _manifest(expires=1e9):
    return RunManifest(
        run_id="run-test",
        purpose="calibration",
        profile=SIM_PROFILE,
        sim_boot_id="boot1",
        expires_monotonic=expires,
    )


def _gateway(monkeypatch, arming="1", sim_mode=True, handle=None):
    monkeypatch.setenv("KOTOBA_PUBLISH", arming)
    return SendGateway(
        _manifest(),
        handle or FakeHandle(),
        CHANNEL,
        now_monotonic=0.0,
        sim_mode_confirmed=sim_mode,
    )


# --- test_08 の対: arming無しでは生成すらできない ---
def test_gateway_refuses_without_arming(monkeypatch):
    with pytest.raises(AuthorizationRefused) as err:
        _gateway(monkeypatch, arming="0")
    assert err.value.reason == "no_arming"


def test_gateway_refuses_with_invalid_arming_value(monkeypatch):
    with pytest.raises(AuthorizationRefused):
        _gateway(monkeypatch, arming="yes-but-not-1")


def test_gateway_refuses_unconfirmed_sim_mode(monkeypatch):
    with pytest.raises(AuthorizationRefused) as err:
        _gateway(monkeypatch, sim_mode=False)
    assert err.value.reason == "no_sim_mode"


def test_second_gateway_is_refused_until_closed(monkeypatch):
    g1 = _gateway(monkeypatch)
    try:
        with pytest.raises(AuthorizationRefused) as err:
            _gateway(monkeypatch)
        assert err.value.reason == "second_sender"
    finally:
        g1.close()
    _gateway(monkeypatch).close()  # close後は再度生成できる


# --- test_11 の対: 未知コマンドは既定変換されず明示拒否 ---
def test_unknown_command_is_rejected_not_defaulted(monkeypatch):
    with pytest.raises(AuthorizationRefused) as err:
        build_command("not_a_valid_motion", SIM_PROFILE)
    assert err.value.reason == "unknown_command"
    with pytest.raises(AuthorizationRefused):
        build_command("combo_rl_lab", SIM_PROFILE)  # 禁止コンボも未知として拒否


def test_known_commands_build_distinct_frames(monkeypatch):
    stand = build_command("combo_pd_stand", SIM_PROFILE)
    walk = build_command("combo_walk", SIM_PROFILE)
    dance = build_command("combo_dance", SIM_PROFILE)
    idle = build_command("idle", SIM_PROFILE)
    stick = build_command("walk_stick", SIM_PROFILE)
    assert len({stand, walk, dance, idle, stick}) == 5
    assert len(stand) == 112


def test_combo_dance_uses_rb_b_buttons():
    # dance = RB(1)+B(3) — pm01_edu task_motion key割当と一致
    import struct

    digital = struct.unpack_from(
        ">12i", build_command("combo_dance", SIM_PROFILE), 16
    )
    assert digital[1] == 1 and digital[3] == 1
    assert sum(digital) == 2


def test_expired_manifest_refuses_send(monkeypatch):
    handle = FakeHandle()
    gw = _gateway(monkeypatch, handle=handle)
    try:
        gw.manifest = RunManifest(
            run_id="r",
            purpose="calibration",
            profile=SIM_PROFILE,
            sim_boot_id="boot1",
            expires_monotonic=10.0,
        )
        with pytest.raises(AuthorizationRefused) as err:
            gw.send("idle", now_monotonic=11.0)
        assert err.value.reason == "expired"
        assert handle.sent == []
    finally:
        gw.close()


def test_send_records_actual_time_and_payload(monkeypatch):
    handle = FakeHandle()
    gw = _gateway(monkeypatch, handle=handle)
    try:
        payload = gw.send("idle", now_monotonic=1.5)
        assert handle.sent == [(CHANNEL, payload)]
        assert gw.sent_log == [(1.5, "idle")]
    finally:
        gw.close()
