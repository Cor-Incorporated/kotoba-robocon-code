"""ことばでロボコン — コース定義（1ロボット・1コース・前進1軸・2目的地）。

距離は READY 成立後の機体前進方向に対する相対距離（メートル）。
値は G1 閉ループで検証済みの可動域内の暫定値であり、評価前に固定される。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    target_id: str
    label: str
    distance_m: float


COURSE = {
    "schema_version": "1.0",
    "coordinate_convention": "right_handed_z_up_meters",
    "targets": [
        Target(target_id="goal_near", label="手前のマーカー", distance_m=0.45),
        Target(target_id="goal_far", label="奥のマーカー", distance_m=1.0),
    ],
    # 基本版に禁止領域は無い（avoid_ids は常に空。将来拡張用に契約だけ用意）
    "forbidden_regions": [],
}


def target_by_id(target_id: str) -> Target:
    for t in COURSE["targets"]:
        if t.target_id == target_id:
            return t
    raise KeyError(target_id)
