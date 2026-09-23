"""FK・hitbox判定の仕様固定 — URDF実値チェーンの先端位置・hit/miss境界。"""

import math

import pytest

from kotoba_harness.hitbox import HitRecorder
from kotoba_harness.kinematics import (
    ARM_L,
    ARM_R,
    dist,
    hand_world,
    tip_in_base,
)

J = [0.0] * 25


def test_zero_pose_left_tip():
    # 全関節0: 先端は原点+全originの和（回転無し）
    p = tip_in_base(ARM_L, J)
    expect = (
        0.01216 - 0.027105 - 0.0371 + 0.0371 + 0.0 + 0.013817 + 0.03,
        0.0 + 0.12916 + 0.066941 + 0.017645 + 0.0065994 + 0.0097723 - 0.02,
        0.0809 + 0.21549 - 0.020838 - 0.070132 - 0.10487 - 0.1547 - 0.14,
    )
    assert dist(p, expect) < 1e-9


def test_zero_pose_right_mirrors_left():
    pl, pr = tip_in_base(ARM_L, J), tip_in_base(ARM_R, J)
    # URDF実値は厳密対称ではない(J16/J21等が~1e-6ずれる)ため1e-4許容
    assert abs(pl[0] - pr[0]) < 1e-4
    assert abs(pl[1] + pr[1]) < 1e-4  # yが鏡像
    assert abs(pl[2] - pr[2]) < 1e-4


def test_shoulder_pitch_moves_tip():
    j0 = list(J)
    p0 = tip_in_base(ARM_L, j0)
    j1 = list(J)
    j1[13] = 1.0  # J13_SHOULDER_PITCH_L
    p1 = tip_in_base(ARM_L, j1)
    assert dist(p0, p1) > 0.05  # 腕が動けば先端も動く


def test_waist_affects_both_arms():
    j1 = list(J)
    j1[12] = 0.5  # J12_WAIST_YAW
    pl, pr = tip_in_base(ARM_L, j1), tip_in_base(ARM_R, j1)
    pl0, pr0 = tip_in_base(ARM_L, J), tip_in_base(ARM_R, J)
    assert dist(pl, pl0) > 0.01 and dist(pr, pr0) > 0.01


def test_hand_world_translation():
    p = hand_world((1.0, 2.0, 0.8), (1.0, 0.0, 0.0, 0.0), J, "L")
    p_base = tip_in_base(ARM_L, J)
    assert dist(p, tuple(p_base[i] + (1.0, 2.0, 0.8)[i] for i in range(3))) < 1e-9


def test_hand_world_yaw_rotation():
    # base yaw=90°(z軸): base-frameの+xがworld+yになる
    yaw90 = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
    p = hand_world((0.0, 0.0, 0.8), yaw90, J, "L")
    p_base = tip_in_base(ARM_L, J)
    # world = Rot90 · p_base + (0,0,0.8)
    expect = (-p_base[1], p_base[0], p_base[2] + 0.8)
    assert dist(p, expect) < 1e-6


def _rec(points):
    """hand L/R の軌跡を直接入れた recorder を作る。"""
    r = HitRecorder()
    for t, hl, hr in points:
        r.samples.append((t, hl, hr))
    return r


def test_judge_hit_inside_radius():
    r = _rec([(0.0, (0.2, 0.0, 0.15), (0.0, 0.0, 0.0))])
    out = r.judge((0.25, 0.0, 0.15), 0.20)
    assert out["hit"] and out["hand"] == "L" and abs(out["min_dist_m"] - 0.05) < 1e-3


def test_judge_miss_outside_radius():
    r = _rec([(0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))])
    out = r.judge((0.5, 0.0, 0.0), 0.20)
    assert not out["hit"] and abs(out["min_dist_m"] - 0.5) < 1e-3


def test_judge_boundary_is_hit():
    r = _rec([(0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))])
    assert r.judge((0.20, 0.0, 0.0), 0.20)["hit"]  # 境界はhit
    assert not r.judge((0.21, 0.0, 0.0), 0.20)["hit"]


def test_judge_picks_closest_hand():
    r = _rec([(0.0, (0.0, 0.0, 0.0), (0.9, 0.0, 0.0))])
    out = r.judge((1.0, 0.0, 0.0), 0.20)
    assert out["hand"] == "R" and out["hit"]


def test_judge_no_samples():
    assert HitRecorder().judge((0, 0, 0), 0.2) == {"hit": False, "reason": "no_samples"}


# --- swept segment判定（C3: サンプル間を横切る当たりを拾う） ---
def test_swept_segment_crossing_is_hit():
    """C3レビュー反例T13: 線分が球を横切るが両端は球外 → 点判定ならMISS、
    swept判定ならHIT。サンプル間の軌跡を見落とさない。"""
    r = _rec([
        (0.0, (-0.03, 0.198, 0.0), (5.0, 5.0, 5.0)),
        (0.1, (0.03, 0.198, 0.0), (5.0, 5.0, 5.0)),
    ])
    out = r.judge((0.0, 0.0, 0.0), 0.2)
    # 両端は 0.2003 > 0.2 だが、線分は target から 0.198 に達する
    assert out["hit"] is True
    assert out["min_dist_m"] <= 0.2
    assert out["judge_method"] == "swept_segment"


def test_swept_segment_gap_does_not_connect():
    """欠測（時刻gap超過）の両側は線分で結ばない — 飛び越えHITを防ぐ。"""
    r = _rec([
        (0.0, (-0.03, 0.198, 0.0), (5.0, 5.0, 5.0)),
        # 0.4s欠測（MAX_SEGMENT_GAP_S=0.25超）→ 線分不成立、点のみ評価
        (0.5, (0.03, 0.198, 0.0), (5.0, 5.0, 5.0)),
    ])
    out = r.judge((0.0, 0.0, 0.0), 0.2)
    assert out["hit"] is False
    assert abs(out["min_dist_m"] - 0.2003) < 1e-3


def test_swept_jump_does_not_connect():
    """不連続jump（1サンプル0.75m超の移動）は線分で結ばない。"""
    r = _rec([
        (0.0, (-1.0, 0.0, 0.0), (5.0, 5.0, 5.0)),
        (0.1, (1.0, 0.0, 0.0), (5.0, 5.0, 5.0)),  # 2.0m jump
    ])
    out = r.judge((0.0, 0.0, 0.0), 0.2)
    # jumpは結ばず点のみ — 両端は半径外
    assert out["hit"] is False and out["min_dist_m"] == 1.0


def test_swept_time_rewind_does_not_connect():
    """時刻巻戻りのサンプル対は線分で結ばない。"""
    r = _rec([
        (0.2, (-0.03, 0.198, 0.0), (5.0, 5.0, 5.0)),
        (0.1, (0.03, 0.198, 0.0), (5.0, 5.0, 5.0)),  # t巻戻り
    ])
    out = r.judge((0.0, 0.0, 0.0), 0.2)
    assert out["hit"] is False


def test_swept_real_trajectory_hit():
    """実軌跡様の連続サンプル: 球の近傍を通過する軌跡はHIT。"""
    pts = []
    for i in range(11):
        t = i * 0.05
        x = -0.25 + i * 0.05  # -0.25 → +0.25 へ 0.05step
        pts.append((t, (x, 0.05, 0.0), (9.0, 9.0, 9.0)))
    out = _rec(pts).judge((0.0, 0.0, 0.0), 0.2)
    assert out["hit"] is True and out["hand"] == "L"


def test_path_summary_envelope():
    r = _rec(
        [
            (0.0, (0.1, 0.2, 0.3), (-0.1, 0.0, 0.5)),
            (0.1, (0.4, -0.2, 0.1), (0.0, 0.1, 0.2)),
        ]
    )
    s = r.path_summary()
    assert s["min"] == [-0.1, -0.2, 0.1] and s["max"] == [0.4, 0.2, 0.5]
