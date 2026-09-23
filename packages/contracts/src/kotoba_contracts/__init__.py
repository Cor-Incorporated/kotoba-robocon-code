"""kotoba_contracts — WP-02境界契約の実装（docs/handoff/contracts の mirror）。

設計原則:
- LLM意味エンベロープ(Intent*)は速度・秒数・座標・承認情報の数値fieldを一切持たない (A01)
- extra="forbid" で特権field混入を構造的に拒否する（旧envelope防御の継承）
- allow_inf_nan=False で非有限数を拒否
- frozen で不変。承認・world・snapshotはサーバー管理の正本データ
"""

from kotoba_contracts.approval import ApprovalRecord, ExecutionGrant
from kotoba_contracts.canonical import canonical_json_bytes, canonical_plan_sha256
from kotoba_contracts.intent import (
    IntentClarify,
    IntentEnvelope,
    IntentExecute,
    IntentReject,
    parse_intent,
)
from kotoba_contracts.plan import ControllerProfile, ExecutionPlan, PlanKind
from kotoba_contracts.snapshot import StateSnapshot
from kotoba_contracts.world import ForbiddenRegion, World, WorldTarget

__all__ = [
    "ApprovalRecord",
    "ExecutionGrant",
    "ControllerProfile",
    "ExecutionPlan",
    "PlanKind",
    "StateSnapshot",
    "World",
    "WorldTarget",
    "ForbiddenRegion",
    "IntentEnvelope",
    "IntentExecute",
    "IntentClarify",
    "IntentReject",
    "parse_intent",
    "canonical_json_bytes",
    "canonical_plan_sha256",
]
