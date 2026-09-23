"""サーバー管理の承認store。単回消費は同一lock区間で原子的に行う (A02)。"""

import threading
import uuid
from datetime import datetime
from typing import Dict

from kotoba_contracts.approval import ApprovalRecord, ExecutionGrant

from kotoba_orchestrator.errors import ApprovalError


class ApprovalStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: Dict[str, Tuple[ApprovalRecord, bool]] = {}

    def issue(self, record: ApprovalRecord) -> str:
        """承認レコードを登録し、不透明なapproval_idを返す。参加者はIDを确认するだけ。"""
        approval_id = uuid.uuid4().hex
        with self._lock:
            self._records[approval_id] = record
        return approval_id

    def _peek(self, approval_id: str) -> ApprovalRecord:
        try:
            return self._records[approval_id]
        except KeyError:
            raise ApprovalError("unknown_id") from None

    def verify_and_consume(
        self,
        approval_id: str,
        *,
        plan_sha256: str,
        world_version: int,
        session_id: str,
        round_id: str,
        sim_boot_id: str,
        now: datetime,
    ) -> ExecutionGrant:
        """全binding条件を検証し、成功時のみ原子的に消費する。

        検証順序: unknown_id -> consumed -> expired -> plan_sha -> world -> session -> round -> boot
        二重クリック・旧world・旧session・reset後の遅延応答はすべてここで拒否される。
        """
        with self._lock:
            record = self._peek(approval_id)
            if record.consumed:
                raise ApprovalError("consumed")
            if now > record.expires_at:
                raise ApprovalError("expired")
            if record.canonical_plan_sha256 != plan_sha256:
                raise ApprovalError("plan_sha_mismatch")
            if record.world_version != world_version:
                raise ApprovalError("world_version_mismatch")
            if record.session_id != session_id:
                raise ApprovalError("session_mismatch")
            if record.round_id != round_id:
                raise ApprovalError("round_mismatch")
            if record.sim_boot_id != sim_boot_id:
                raise ApprovalError("boot_mismatch")
            stored = record.model_copy(update={"consumed": True})
            self._records[approval_id] = stored
            return ExecutionGrant(
                approval_id=approval_id,
                plan_id=record.plan_id,
                session_id=record.session_id,
                round_id=record.round_id,
                sim_boot_id=record.sim_boot_id,
            )

    def is_consumed(self, approval_id: str) -> bool:
        with self._lock:
            return self._peek(approval_id).consumed

    def size(self) -> int:
        with self._lock:
            return len(self._records)
