"""PM01 順運動学 — URDF由来の腕チェーンで手先(LINK_ELBOW_END_*)world位置を計算。

sim_state は base pos/quat + 関節角のみ(リンクworld位置非含有)のため、
serial_pm01_edu.urdf (SDK assets) の origin/axis 実値からFKを組み手先位置を導出する。

sim_state joints配列はMJCF順(freejoint除く):
  12 = J12_WAIST_YAW, 13-17 = 左腕J13-J17, 18-22 = 右腕J18-J22
チェーン: LINK_BASE → J12 → LINK_TORSO_YAW → J13..J17 → LINK_ELBOW_END_L
                                          → J18..J22 → LINK_ELBOW_END_R
各関節の変換は translate(origin)·axisangle(axis,q)。rpyは全て0(URDF実値)。
"""

from __future__ import annotations

import math

# (origin_xyz, axis_xyz, joint_index or None for fixed) — URDF実値
WAIST = ((0.01216, 0.0, 0.0809), (0.0, 0.0, 1.0), 12)

ARM_L = (
    ((-0.027105, 0.12916, 0.21549), (0.0, 0.99803, 0.062791), 13),
    ((-0.0371, 0.066941, -0.020838), (1.0, 0.0, 0.0), 14),
    ((0.0371, 0.017645, -0.070132), (0.0, -0.062803, 0.99803), 15),
    ((0.0, 0.0065994, -0.10487), (0.0027243, 0.99802, 0.062803), 16),
    ((0.013817, 0.0097723, -0.1547), (-0.21479, -0.061921, 0.9747), 17),
    ((0.03, -0.02, -0.14), None, None),  # J_FIXED_ELBOW_END_L
)

ARM_R = (
    ((-0.027105, -0.12916, 0.21549), (0.0, 0.99803, -0.062791), 18),
    ((-0.0371, -0.066941, -0.020838), (1.0, 0.0, 0.0), 19),
    ((0.0371, -0.017644, -0.070132), (0.0, 0.062791, 0.99803), 20),
    ((0.0, -0.006598, -0.10487), (0.0027243, 0.99802, -0.06279), 21),
    ((0.013817, -0.0097704, -0.1547), (-0.21479, 0.061909, 0.9747), 22),
    ((0.03, 0.02, -0.14), None, None),  # J_FIXED_ELBOW_END_R
)


def _axis_angle_rot(axis, q):
    """axis(単位)まわりの回転 q[rad] → 3x3行列(row-major)。"""
    x, y, z = axis
    c, s = math.cos(q), math.sin(q)
    C = 1.0 - c
    return (
        (x * x * C + c, x * y * C - z * s, x * z * C + y * s),
        (y * x * C + z * s, y * y * C + c, y * z * C - x * s),
        (z * x * C - y * s, z * y * C + x * s, z * z * C + c),
    )


def _matmul(A, B):
    return tuple(
        tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _matvec(R, v):
    return tuple(sum(R[i][k] * v[k] for k in range(3)) for i in range(3))


def _quat_to_rot(quat):
    """w,x,y,z quaternion → 3x3行列。"""
    w, x, y, z = quat
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def frame_in_base(chain, joints):
    """Return (position, rotation) of chain end in LINK_BASE frame.

    Same map as tip_in_base but also returns the terminal link orientation,
    needed to project a virtual stick rigidly attached to the hand link.
    joints=sim_state array."""
    R = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    t = (0.0, 0.0, 0.0)
    for origin, axis, idx in (WAIST,) + chain:
        # p' = R * origin + t — move to joint position in parent frame
        t = tuple(t[i] + sum(R[i][k] * origin[k] for k in range(3)) for i in range(3))
        if axis is not None and idx is not None:
            R = _matmul(R, _axis_angle_rot(axis, joints[idx]))
    return t, R


def tip_in_base(chain, joints):
    """Chain end position in base (LINK_BASE) frame. joints=sim_state array."""
    return frame_in_base(chain, joints)[0]


# Virtual stick local direction on the elbow-end link. FK analysis of the
# 2026-09-21 swing capture showed the link's -z axis reaches the ground
# most stably during dance.
STICK_LOCAL_DIR = (0.0, 0.0, -1.0)
STICK_LEN_M = 0.7  # virtual stick length — tip reaches ground z≈0 (measured)


def stick_tip_world(base_pos, base_quat, joints, side, stick_m=STICK_LEN_M):
    """World position of a virtual stick tip fixed to the hand link.

    The stick extends stick_m along STICK_LOCAL_DIR of the elbow-end frame
    and follows hand orientation during dance, reaching ground watermelons.
    """
    chain = ARM_L if side.upper() == "L" else ARM_R
    p_base, R_hand = frame_in_base(chain, joints)
    d_local = tuple(v * stick_m for v in STICK_LOCAL_DIR)
    d_base = _matvec(R_hand, d_local)
    tip_base = tuple(p_base[i] + d_base[i] for i in range(3))
    R_base = _quat_to_rot(base_quat)
    tip_rot = _matvec(R_base, tip_base)
    return tuple(base_pos[i] + tip_rot[i] for i in range(3))


def stick_tips_world(base_pos, base_quat, joints, stick_m=STICK_LEN_M):
    """(left stick tip, right stick tip) in world coordinates."""
    return (
        stick_tip_world(base_pos, base_quat, joints, "L", stick_m),
        stick_tip_world(base_pos, base_quat, joints, "R", stick_m),
    )


def hand_world(base_pos, base_quat, joints, side):
    """手先(LINK_ELBOW_END_L/R)のworld座標。side='L'|'R'。"""
    chain = ARM_L if side.upper() == "L" else ARM_R
    p_base = tip_in_base(chain, joints)
    R_base = _quat_to_rot(base_quat)
    p_rot = _matvec(R_base, p_base)
    return tuple(base_pos[i] + p_rot[i] for i in range(3))


def hands_world(base_pos, base_quat, joints):
    """(左手, 右手) world座標。"""
    return (
        hand_world(base_pos, base_quat, joints, "L"),
        hand_world(base_pos, base_quat, joints, "R"),
    )


def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))
