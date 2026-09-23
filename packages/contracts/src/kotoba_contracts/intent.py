"""LLM出力は意味レベルのみ。docs/handoff/contracts/intent.schema.json の mirror。

速度・秒数・座標・承認トークン等の数値/特権fieldは意図的に存在しない (A01)。
"""

from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from typing_extensions import TypeAlias

SCHEMA_VERSION = "1.0"

ID_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$"
IdStr = Annotated[str, Field(pattern=ID_PATTERN)]

ReasonCode = Literal[
    "unknown_target",
    "unsupported_action",
    "contradictory_constraints",
    "untrusted_instruction",
    "invalid_input",
]


class _EnvelopeBase:
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)
    schema_version: Literal["1.0"] = SCHEMA_VERSION


class IntentExecute(_EnvelopeBase, BaseModel):
    decision: Literal["execute"] = "execute"
    target_ids: List[IdStr] = Field(min_length=1, max_length=1)
    avoid_ids: List[IdStr] = Field(default_factory=list, max_length=8)
    explanation: str = Field(min_length=1, max_length=240)


class IntentClarify(_EnvelopeBase, BaseModel):
    decision: Literal["clarify"] = "clarify"
    question: str = Field(min_length=1, max_length=240)
    candidate_target_ids: List[IdStr] = Field(default_factory=list, max_length=8)


class IntentReject(_EnvelopeBase, BaseModel):
    decision: Literal["reject"] = "reject"
    reason_code: ReasonCode
    explanation: str = Field(min_length=1, max_length=240)


IntentEnvelope: TypeAlias = Union[IntentExecute, IntentClarify, IntentReject]

_adapter = TypeAdapter(IntentEnvelope)


def parse_intent(raw: dict) -> Union[IntentExecute, IntentClarify, IntentReject]:
    """dict から意味エンベロープを検証付きで構築する。余計なfieldは全て拒否。"""
    return _adapter.validate_python(raw)


__all__ = [
    "SCHEMA_VERSION",
    "ID_PATTERN",
    "IdStr",
    "ReasonCode",
    "IntentEnvelope",
    "IntentExecute",
    "IntentClarify",
    "IntentReject",
    "parse_intent",
    "ValidationError",
]
