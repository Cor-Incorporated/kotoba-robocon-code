"""RoundWorld v2 — ことばでスイカ割りの世界正本（砂浜・地上スイカ）。

seed+履歴から再現可能なスイカworld位置を生成し、ラウンド中は不変。
配置は annulus [MIN_SPAWN_M, MAX_SPAWN_M] × bearing範囲:
  - bearing既定は前方 ±75°（turn-aroundのThor実測受入後に360°へ拡張する
    ため full_circle フラグで明示的に切替 — 未実測の暗黙360°にしない）
  - 直近の通常spawnとの間隔 ≥ MIN_SPACING_M（同じ場所の連続を避ける）
  - 候補はarena円の内側（ARENA_R - MARGIN）に限定
  - 有限試行で条件を満たさなければ spawn_failed（古い位置を返さない）
高さは砂浜の地表 — 台は廃止（地上スイカ: 中心z = 半径）。
座標系は右-handed z-up meters（world.py COURSE と同一規約）。

hit判定そのものは hitbox.py（FK軌跡×hit volume）が担当し、
本モジュールは world配置の正本と整合性検証のみを担う。
再現性は seed + spawnアルゴリズム版 + 履歴コンテキストで定義される
（seed単独ではない — 履歴が変われば結果も変わる）。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# ---- シーン/arena正本（API・controller・renderer・UIで共有する唯一の源）
SCENE_MODE = "beach"
GEOMETRY_VERSION = "beach-v1"
SPAWN_ALGO_VERSION = "annulus-v2"
ARENA_CENTER = (0.0, 0.0)          # 既定値 — 実際の中心はround開始位置
                                   # （spawn時のロボットworld位置）を使う。
                                   # spec.scene.arena_center が正本。
ARENA_R_M = 5.0                    # 円形半径（正方形ではない）
ARENA_SPAWN_MARGIN_M = 0.4         # spawn候補は境界からこれだけ内側

# ---- 地上スイカ（砂浜 — 台なし）
WATERMELON_R = 0.15                # 見た目半径
GROUND_SURFACE_M = 0.0             # 砂浜地表の高さ
WATERMELON_Z = WATERMELON_R        # 中心 = 半径（地表に接地）
HIT_VOLUME_R = 0.20                # 判定半径（スイカ0.15 + 工具先端≈0.05）

# ---- 打撃到達帯（地上打撃: 固定工具先端 — B4実測dance軌跡由来の候補）
# 旧台上面帯（z≈1.05）は廃止。地上打撃の実測到達域は
# capture(run記録)の棒先端軌跡から校正する — ここでは打撃可否の粗い
# ゲートとして前足域を使う（厳密なhit判定はhitboxの実観測軌跡）。
SWEEP_X = (-0.15, 0.90)   # base前方+ （工具込みの前方到達候補）
SWEEP_Y = (-0.90, 0.90)   # base左+ / 右-
SWEEP_Z = (-0.90, 0.30)   # 地上目標: base下へ届く（base_z≈0.78 → z≈0.15）

# ---- スイカ配置制約
MIN_SPAWN_M = 1.0     # 近すぎると誘導の意味がない
MAX_SPAWN_M = 3.5     # 歩行で到達できる範囲（60sラウンド想定の候補値）
SPAWN_BEARING_RAD = math.radians(75)      # 既定: 前方±75°
SPAWN_BEARING_FULL_RAD = math.pi          # turn_around受入後の360°
MIN_SPACING_M = 0.75  # 直近通常spawnとの最小間隔

MAX_SPAWN_TRIES = 64


@dataclass(frozen=True)
class Watermelon:
    pos: tuple        # world (x,y,z) — 中心（地上: z=半径）
    radius: float     # 見た目半径
    hit_radius: float # 判定半径
    surface_m: float  # 接地した面の高さ（砂浜=0.0 — 台は廃止）


@dataclass(frozen=True)
class RoundSpec:
    round_id: str
    seed: int
    watermelon: Watermelon
    robot_start: tuple  # spawn時のロボットworld位置（到達制約の基準）
    robot_yaw: float    # spawn時のロボットyaw
    time_limit_s: float
    max_swings: int
    scene: dict = field(default_factory=dict)  # 世界メタデータの正本


def default_scene(full_circle: bool = False, center=None) -> dict:
    """世界メタデータ — spec/API/manifest/rendererで共有する正本。

    center: arena中心 = round開始位置（spawn時のロボットworld位置のXY）。
    「開始位置から半径5mの砂浜」という世界仕様を全層で一致させる。"""
    return {
        "mode": SCENE_MODE,
        "arena_center": list(center) if center is not None else list(ARENA_CENTER),
        "arena_r_m": ARENA_R_M,
        "geometry_version": GEOMETRY_VERSION,
        "spawn_algo_version": SPAWN_ALGO_VERSION,
        "bearing_deg": round(
            math.degrees(
                SPAWN_BEARING_FULL_RAD if full_circle else SPAWN_BEARING_RAD
            ),
            1,
        ),
        "min_spacing_m": MIN_SPACING_M,
    }


def _in_sweep(rel) -> bool:
    """base座標系の相対位置が打撃到達候補帯内か（粗いゲート）。"""
    x, y, z = rel
    return (
        SWEEP_X[0] <= x <= SWEEP_X[1]
        and SWEEP_Y[0] <= y <= SWEEP_Y[1]
        and SWEEP_Z[0] <= z <= SWEEP_Z[1]
    )


def reachable_for_strike(robot_pos, robot_yaw, target_pos, base_z=0.82) -> bool:
    """robot位置・向きから見て target が打撃到達候補帯内か（world→base変換）。"""
    dx = target_pos[0] - robot_pos[0]
    dy = target_pos[1] - robot_pos[1]
    c, s = math.cos(-robot_yaw), math.sin(-robot_yaw)
    rel = (dx * c - dy * s, dx * s + dy * c, target_pos[2] - base_z)
    return _in_sweep(rel)


def spawn_watermelon(
    seed,
    robot_pos,
    robot_yaw,
    history=(),
    full_circle=False,
) -> tuple:
    """seed+履歴から再現可能なスイカworld位置を生成する（純粋関数）。

    annulus [MIN_SPAWN_M, MAX_SPAWN_M] × bearing（既定±75°、full_circleで
    360°）。直近spawnとの間隔・arena内側を検査し、有限試行で見つから
    なければ spawn_failed（古い位置へ縮退しない）。
    """
    rng = random.Random(seed)
    bearing_lim = (
        SPAWN_BEARING_FULL_RAD if full_circle else SPAWN_BEARING_RAD
    )
    hist = [tuple(h) for h in (history or ())]
    # arena中心 = round開始位置（ロボットのspawn位置）— spawn候補は
    # その円の内側に限定する。world原点の固定円ではない（§4.3の一致）。
    center = (robot_pos[0], robot_pos[1])
    for _ in range(MAX_SPAWN_TRIES):
        dist = rng.uniform(MIN_SPAWN_M, MAX_SPAWN_M)
        bearing = rng.uniform(-bearing_lim, bearing_lim)
        ang = robot_yaw + bearing
        pos = (
            robot_pos[0] + dist * math.cos(ang),
            robot_pos[1] + dist * math.sin(ang),
            WATERMELON_Z,
        )
        # arena円の内側に限定（境界への貼り付きを避ける）
        if (
            math.hypot(pos[0] - center[0], pos[1] - center[1])
            > ARENA_R_M - ARENA_SPAWN_MARGIN_M
        ):
            continue
        # 直近spawnとの最小間隔（同一位置の連続を避ける）
        if any(
            math.hypot(pos[0] - h[0], pos[1] - h[1]) < MIN_SPACING_M
            for h in hist
        ):
            continue
        return pos
    raise ValueError("spawn_failed")


def make_round(
    round_id,
    seed,
    robot_pos,
    robot_yaw,
    time_limit_s=60.0,
    max_swings=3,
    target=None,
    history=(),
    full_circle=False,
) -> RoundSpec:
    """ラウンド仕様を生成（seed+履歴で決定的・ラウンド中不変）。

    target が指定された場合は fixture 対照用の明示world位置（operator専用・
    spec.seed=-1 で監査可能に標識）。通常のゲーム経路は常に seed 生成。
    """
    pos = (
        tuple(target)
        if target is not None
        else spawn_watermelon(
            seed, robot_pos, robot_yaw,
            history=history, full_circle=full_circle,
        )
    )
    wm = Watermelon(
        pos=pos,
        radius=WATERMELON_R,
        hit_radius=HIT_VOLUME_R,
        surface_m=GROUND_SURFACE_M,
    )
    scene = default_scene(
        full_circle=full_circle, center=(robot_pos[0], robot_pos[1])
    )
    return RoundSpec(
        round_id=round_id,
        # fixture標識: 明示target時はseed=-1（乱数生成ではないことを監査可能に）
        seed=seed if target is None else -1,
        watermelon=wm,
        robot_start=tuple(robot_pos),
        robot_yaw=robot_yaw,
        time_limit_s=time_limit_s,
        max_swings=max_swings,
        scene=scene,
    )


def strike_alignment(robot_pos, robot_yaw, target_pos, base_z=0.82):
    """ロボット姿勢→スイカのbase座標系相対位置を返す（UIの誘導ヒント用）。

    戻り値: (rel_xyz, in_sweep)
    """
    dx = target_pos[0] - robot_pos[0]
    dy = target_pos[1] - robot_pos[1]
    c, s = math.cos(-robot_yaw), math.sin(-robot_yaw)
    rel = (dx * c - dy * s, dx * s + dy * c, target_pos[2] - base_z)
    return rel, _in_sweep(rel)
