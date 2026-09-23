"""正規JSON生成と計画hash。言語間golden vectorsで検査される。"""

import hashlib
import json
import math
from typing import Any

from kotoba_contracts.plan import ExecutionPlan


def _reject_nonfinite(obj: Any) -> None:
    if isinstance(obj, float) and not math.isfinite(obj):
        raise ValueError("non-finite number in canonical JSON")
    if isinstance(obj, dict):
        for v in obj.values():
            _reject_nonfinite(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _reject_nonfinite(v)


def canonical_json_bytes(obj: Any) -> bytes:
    """UTF-8 / key sort / 有限numberのみ / 許可fieldは呼び出し側(plan)が保証。"""
    _reject_nonfinite(obj)
    text = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return text.encode("utf-8")


def canonical_plan_sha256(plan: ExecutionPlan) -> str:
    return hashlib.sha256(canonical_json_bytes(plan.to_canonical_dict())).hexdigest()


__all__ = ["canonical_json_bytes", "canonical_plan_sha256"]
