"""観測snapshot — docs/handoff/contracts/state-snapshot.schema.json の mirror。

四元数は wxyz を明示 (A10: q と -q は同一回転。符号単独で転倒判定しない)。
sim_time_s が常に0の時計は sim_clock_status="unverified" として扱う (A09)。
"""

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field

from kotoba_contracts.intent import IdStr, SCHEMA_VERSION
from kotoba_contracts.world import Sha256Hex

Mode = Literal["initializing", "pd_stand", "walk", "passive", "paused", "fault"]


class BodyPose(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    body_name: IdStr
    position_m: List[float] = Field(min_length=3, max_length=3)
    rotation_matrix_row_major: List[float] = Field(min_length=9, max_length=9)


class StateSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    sim_boot_id: IdStr
    world_version: int = Field(ge=1)
    sequence: int = Field(ge=0)
    physics_step_index: int = Field(ge=0)
    sim_time_s: float = Field(ge=0)
    sim_clock_status: Literal["valid", "unverified"]
    capture_monotonic_ns: int = Field(ge=0)
    clock_domain: IdStr
    model_sha256: Sha256Hex
    mode: Mode
    state_source: Literal["native_mujoco_truth"] = "native_mujoco_truth"
    base_position_m: List[float] = Field(min_length=3, max_length=3)
    base_quaternion_wxyz: List[float] = Field(min_length=4, max_length=4)
    base_linear_velocity_mps: List[float] = Field(min_length=3, max_length=3)
    body_poses: List[BodyPose] = Field(min_length=1, max_length=512)


__all__ = ["StateSnapshot", "BodyPose", "Mode"]
