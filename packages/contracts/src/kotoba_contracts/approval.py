"""承認binding — docs/handoff/contracts/approval-binding.schema.json の mirror。

SERVER INTERNAL。参加者からこのオブジェクトを受け取ってはならない。
権限はサーバーstoreに置き、原子的な単回消費で行使する（A02）。
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kotoba_contracts.intent import IdStr, SCHEMA_VERSION
from kotoba_contracts.world import Sha256Hex


class ApprovalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    session_id: IdStr
    round_id: IdStr
    plan_id: IdStr
    canonical_plan_sha256: Sha256Hex
    world_version: int = Field(ge=1)
    controller_profile_sha256: Sha256Hex
    sim_boot_id: IdStr
    expires_at: datetime
    consumed: bool = False


class ExecutionGrant(BaseModel):
    """verify_and_consume が成功したときのみ発行される一回限りの実行許可。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: IdStr
    plan_id: IdStr
    session_id: IdStr
    round_id: IdStr
    sim_boot_id: IdStr


__all__ = ["ApprovalRecord", "ExecutionGrant"]
