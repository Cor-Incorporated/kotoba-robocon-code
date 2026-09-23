"""エラーの理由コードはテスト・監査ログで反証可能な文字列として使う。"""


class OrchestratorError(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class PlanRejected(OrchestratorError):
    """意図・計画が受入条件を満たさない。数値の黙って補完はしない。"""


class ApprovalError(OrchestratorError):
    """承認の検証・消費に失敗。reasonで失敗種別を区別する。"""
