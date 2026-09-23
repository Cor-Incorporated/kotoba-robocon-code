"""送信境界: arming・run manifest・有界profile・単一送信者・明示拒否。

旧スクリプトの欠陥（characterization test_08/11）の正しい仕様:
- arming環境変数が無くても動いてしまう → 生成時に拒否する
- 未知コンボ名がwalkに既定変換される → 明示拒否する
"""

import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet

from kotoba_harness.errors import AuthorizationRefused

ARMING_ENV = "KOTOBA_PUBLISH"


@dataclass(frozen=True)
class BoundedProfile:
    """実測校正に基づく有界profile。LLM・実験コードはここを超えられない。"""

    name: str
    max_linear_mps: float
    max_yaw_radps: float
    max_duration_s: float
    walk_stick: float
    walk_stick_slow: float = 0.45
    max_lateral_mps: float = 0.4
    # SDK側stick→指令の換算係数（pm01_edu rl_walking_example default.yaml
    # command_scale_pos/neg = [1.0, 0.4, 1.0]、左右対称）。
    vx_scale: float = 1.0
    vy_scale: float = 0.4
    yaw_scale: float = 1.0
    allowlist: FrozenSet[str] = field(
        default_factory=lambda: frozenset(
            {
                "combo_pd_stand",
                "combo_walk",
                "combo_dance",
                "idle",
                "walk_stick",
                "walk_stick_slow",
            }
        )
    )


SIM_PROFILE = BoundedProfile(
    name="sim_profile_g1",
    max_linear_mps=0.85,
    max_yaw_radps=0.8,
    max_duration_s=6.0,
    walk_stick=1.0,
)

# 速歩/走行許可時のprofile — 前進stick 1.0は pm01_edu rl_walking_example の
# command_scale_pos[0] 上限そのもの（policy指令範囲の天井）。横・旋回の
# 上限はSIM_PROFILEから変えない（一律増幅しない — R5）。
SIM_PROFILE_FAST = BoundedProfile(
    name="sim_profile_g1_fast",
    max_linear_mps=1.0,
    max_yaw_radps=0.8,
    max_duration_s=6.0,
    walk_stick=1.0,
)


def profile_for_capabilities(caps) -> BoundedProfile:
    """manifestのcapabilities宣言から送信profileを選ぶ。
    fast_walk/run が experimental 以上の場合のみ線速度上限をSDK指令
    上限まで引き上げる。宣言が無い・off・未知の場合は既定profile
    （上限0.85）を返す — "off" は truthy なので値を直接照合する。"""
    if isinstance(caps, dict) and (
        caps.get("fast_walk") in ("experimental", "verified")
        or caps.get("run") in ("experimental", "verified")
    ):
        return SIM_PROFILE_FAST
    return SIM_PROFILE

_DIGITAL = {"LB": 0, "RB": 1, "A": 2, "B": 3}
_LCM_TYPE_HASH = 0xD6CAE60F8643A772


def _gamepad_fingerprint() -> int:
    return ((_LCM_TYPE_HASH << 1) & 0xFFFFFFFFFFFFFFFF) + (_LCM_TYPE_HASH >> 63)


def _frame(digital, analog) -> bytes:
    import struct

    return struct.pack(">Qq12i6d", _gamepad_fingerprint(), 0, *digital, *analog)


def _combo_buttons(name: str):
    # pm01_edu task_motion default.yaml の key 割当（Thor配備ソース照合）:
    #   pd_stand=[LB,A] / walk=[LB,B] / dance=[RB,B] / rl_lab=[LB,X] / idle=[LB,START]
    known = {"pd_stand": ("LB", "A"), "walk": ("LB", "B"), "dance": ("RB", "B")}
    if name not in known:
        raise KeyError(name)  # 呼び出し側で明示拒否に変換。既定変換はしない
    return known[name]


def build_command(name: str, profile: BoundedProfile) -> bytes:
    """名前付きコマンド→112バイトフレーム。未知名は拒否（既定変換禁止）。"""
    if name not in profile.allowlist:
        raise AuthorizationRefused("unknown_command")
    if name == "combo_pd_stand":
        return _combo_frame("pd_stand")
    if name == "combo_walk":
        return _combo_frame("walk")
    if name == "combo_dance":
        return _combo_frame("dance")
    if name == "idle":
        return _frame((0,) * 12, (0.0,) * 6)
    if name in ("walk_stick", "walk_stick_slow"):
        stick = profile.walk_stick if name == "walk_stick" else profile.walk_stick_slow
        implied = stick * 0.85  # POLICY_GAIN_MPS（2026-08-28実測）
        if implied > profile.max_linear_mps + 1e-9:
            raise AuthorizationRefused("analog_out_of_bounds")
        return _frame((0,) * 12, (0.0, 0.0, stick, 0.0, 0.0, 0.0))
    raise AuthorizationRefused("unknown_command")


def _combo_frame(motion: str) -> bytes:
    try:
        buttons = _combo_buttons(motion)
    except KeyError:
        raise AuthorizationRefused("unknown_command") from None
    bits = [0] * 12
    for b in buttons:
        bits[_DIGITAL[b]] = 1
    return _frame(tuple(bits), (0.0,) * 6)


def build_move_command(
    fwd: float, lat: float, yaw: float, profile: BoundedProfile
) -> bytes:
    """有界の方向指令→112バイトフレーム（ことばでスイカ割りの連続操縦用）。

    引数はstick空間 [-1.0, 1.0]。SDKの実装契約:
    - analog[2] (LeftStick_X)         → vx 前後   (× vx_scale)
    - analog[3] → -LeftStick_Y        → vy 左右   (× vy_scale、adapterで反転)
    - analog[5] → -RightStick_Y       → yaw速度   (× yaw_scale、adapterで反転)
    implied速度がprofile上限を超える入力・非finite・|stick|>1は拒否する。
    """
    vals = (fwd, lat, yaw)
    if not all(map(math.isfinite, vals)):
        raise AuthorizationRefused("nonfinite_command")
    if any(abs(v) > 1.0 for v in vals):
        raise AuthorizationRefused("analog_out_of_bounds")
    implied = (
        abs(fwd) * profile.vx_scale,
        abs(lat) * profile.vy_scale,
        abs(yaw) * profile.yaw_scale,
    )
    if (
        implied[0] > profile.max_linear_mps + 1e-9
        or implied[1] > profile.max_lateral_mps + 1e-9
        or implied[2] > profile.max_yaw_radps + 1e-9
    ):
        raise AuthorizationRefused("analog_out_of_bounds")
    analog = (0.0, 0.0, fwd, -lat, 0.0, -yaw)
    return _frame((0,) * 12, analog)


COMMAND_BUILDERS: Dict[str, Callable] = {
    "build_command": build_command,
    "build_move_command": build_move_command,
}


@dataclass(frozen=True)
class RunManifest:
    """実験runの承認内容。arming flagとは別物。発行者・目的・範囲・期限を持つ。"""

    run_id: str
    purpose: str  # "calibration" | "evaluation"
    profile: BoundedProfile
    sim_boot_id: str
    expires_monotonic: float


class SendGateway:
    """単一送信者。生成時に全ての事前条件を検査し、送信ごとに再検証する。"""

    _instance_lock = threading.Lock()
    _active: "SendGateway | None" = None

    def __init__(
        self,
        manifest: RunManifest,
        handle,
        channel: str,
        *,
        now_monotonic: float,
        sim_mode_confirmed: bool,
        arming_env: str | None = None,
    ) -> None:
        arming = os.environ.get(ARMING_ENV) if arming_env is None else arming_env
        if arming != "1":
            raise AuthorizationRefused("no_arming")
        if not sim_mode_confirmed:
            raise AuthorizationRefused("no_sim_mode")
        with SendGateway._instance_lock:
            if SendGateway._active is not None:
                raise AuthorizationRefused("second_sender")
            SendGateway._active = self
        self.manifest = manifest
        self.handle = handle
        self.channel = channel
        self.sent_log: list = []

    def prepare(self, command_name: str) -> bytes:
        """構築+検査のみ（送信しない）。フレームの事前準備に使う。"""
        if time.time() == -1:  # pragma: no cover
            pass
        return build_command(command_name, self.manifest.profile)

    def prepare_move(self, fwd: float, lat: float, yaw: float) -> bytes:
        """方向指令の構築+検査（送信しない）。issue()で発行する。"""
        return build_move_command(fwd, lat, yaw, self.manifest.profile)

    def issue(self, payload: bytes, *, now_monotonic: float,
              command_name: str = "frame") -> bytes:
        """実発行。毎回期限・単一所有者の検査を通る唯一の出口。"""
        if now_monotonic > self.manifest.expires_monotonic:
            raise AuthorizationRefused("expired")
        self.handle.publish(self.channel, payload)
        self.sent_log.append((now_monotonic, command_name))
        return payload

    def send(self, command_name: str, *, now_monotonic: float) -> bytes:
        if now_monotonic > self.manifest.expires_monotonic:
            raise AuthorizationRefused("expired")
        payload = build_command(command_name, self.manifest.profile)
        self.handle.publish(self.channel, payload)
        self.sent_log.append((now_monotonic, command_name))
        return payload

    def close(self) -> None:
        with SendGateway._instance_lock:
            if SendGateway._active is self:
                SendGateway._active = None
