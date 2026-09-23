"""live観測の鮮度判定（表示層が共有する唯一の実装）。

観測の受信wall時刻（obs_wall）だけを根拠にする。
ファイル書出し時刻（wall）は監査用で、fresh判定には使わない —
古い観測を新しい時刻で書き直しても新鮮にはならない（PR22 R3）。
"""

from __future__ import annotations

from typing import Optional, Tuple


def live_freshness(
    payload: dict, now: float, max_age_s: float = 1.0
) -> Tuple[bool, Optional[float]]:
    """(freshか, 観測age秒) を返す。obs_wall欠落は明示的にstale。"""
    obs_wall = payload.get("obs_wall")
    if obs_wall is None:
        return False, None
    age = now - obs_wall
    return (0.0 <= age < max_age_s), age
