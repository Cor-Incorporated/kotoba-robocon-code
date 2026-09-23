"""セッション・ラウンド・リクエストIDの分離 (A07)。IDは全てサーバー生成。"""

import time
import uuid
from dataclasses import dataclass
from typing import Dict

from kotoba_orchestrator.errors import ApprovalError


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class Session:
    session_id: str
    created_monotonic: float


@dataclass(frozen=True)
class Round:
    session_id: str
    round_id: str
    request_id: str


class SessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}

    def create_session(self) -> Session:
        session = Session(session_id=_new_id(), created_monotonic=time.monotonic())
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise ApprovalError("unknown_session") from None

    def begin_round(self, session_id: str) -> Round:
        session = self.get(session_id)
        return Round(
            session_id=session.session_id,
            round_id=_new_id(),
            request_id=_new_id(),
        )
