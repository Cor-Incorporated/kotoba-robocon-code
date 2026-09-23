"""RoundWorld v2 の仕様固定 — seed+履歴決定性・到達制約・座標整合・
砂浜scene正本・地上スイカ・履歴分離・fixture分離。"""

import math

import pytest

from kotoba_harness.roundworld import (
    ARENA_CENTER,
    ARENA_R_M,
    ARENA_SPAWN_MARGIN_M,
    HIT_VOLUME_R,
    MAX_SPAWN_M,
    MIN_SPAWN_M,
    MIN_SPACING_M,
    SWEEP_X,
    SWEEP_Y,
    SWEEP_Z,
    WATERMELON_R,
    WATERMELON_Z,
    make_round,
    reachable_for_strike,
    spawn_watermelon,
    strike_alignment,
)

ORIGIN = (0.0, 0.0, 0.82)


def test_spawn_deterministic_same_seed():
    a = spawn_watermelon(42, ORIGIN, 0.0)
    b = spawn_watermelon(42, ORIGIN, 0.0)
    assert a == b


def test_spawn_differs_by_seed():
    positions = {spawn_watermelon(s, ORIGIN, 0.0) for s in range(20)}
    assert len(positions) > 15  # 衝突ほぼ無し


def test_spawn_respects_distance_band():
    for s in range(100):
        pos = spawn_watermelon(s, ORIGIN, 0.0)
        d = math.hypot(pos[0], pos[1])
        assert MIN_SPAWN_M <= d <= MAX_SPAWN_M


def test_spawn_respects_forward_bearing_default():
    # 既定は前方±75°（turn-aroundの実測受入後にのみ360°へ拡張する）
    for s in range(100):
        pos = spawn_watermelon(s, ORIGIN, 0.0)
        ang = math.atan2(pos[1], pos[0])
        assert abs(ang) <= math.radians(75) + 1e-9


def test_spawn_full_circle_only_when_flagged():
    # full_circle=True で360°環状（後方にも出るseedが存在する）
    behind = 0
    for s in range(40):
        pos = spawn_watermelon(s, ORIGIN, 0.0, full_circle=True)
        ang = math.atan2(pos[1], pos[0])
        if abs(ang) > math.radians(120):
            behind += 1
        # 距離帯とarena内側は維持
        d = math.hypot(pos[0], pos[1])
        assert MIN_SPAWN_M <= d <= MAX_SPAWN_M
        assert d <= ARENA_R_M - ARENA_SPAWN_MARGIN_M + 1e-9
    assert behind > 0


def test_spawn_inside_arena_circle():
    # 5m円の内側（正方形ではなく円 — 境界からmargin内側）
    for s in range(200):
        pos = spawn_watermelon(s, ORIGIN, 0.0)
        d = math.hypot(
            pos[0] - ARENA_CENTER[0], pos[1] - ARENA_CENTER[1]
        )
        assert d <= ARENA_R_M - ARENA_SPAWN_MARGIN_M + 1e-9


def test_spawn_height_is_ground():
    # 地上スイカ — 台上面ではなく中心=半径（地表に接地）
    for s in range(20):
        assert spawn_watermelon(s, ORIGIN, 0.0)[2] == WATERMELON_Z


def test_ground_z_inside_sweep_z():
    # 地上目標は打撃到達候補帯内（base_z=0.82想定で下方へ届く）
    rel_z = WATERMELON_Z - 0.82
    assert SWEEP_Z[0] <= rel_z <= SWEEP_Z[1]


def test_spawn_history_spacing():
    # 履歴に近い場所には出ない（連続して同じ位置を返さない）
    hist = [(1.5, 0.0), (-1.5, 0.0)]
    for s in range(50):
        pos = spawn_watermelon(s, ORIGIN, 0.0, history=hist)
        for h in hist:
            assert math.hypot(pos[0] - h[0], pos[1] - h[1]) >= MIN_SPACING_M


def test_spawn_history_changes_result():
    # 再現性はseed+履歴 — 履歴が変われば結果も変わる（seed単独ではない）
    blocked = spawn_watermelon(3, ORIGIN, 0.0)
    moved = spawn_watermelon(3, ORIGIN, 0.0, history=[blocked])
    assert moved != blocked


def test_spawn_finite_failure():
    # 全域を履歴で埋めた場合は有限試行で失敗（古い位置を返さない）
    hist = [
        (math.cos(a) * d, math.sin(a) * d)
        for a in [i * 0.05 for i in range(126)]
        for d in (1.2, 1.8, 2.4, 3.0)
    ]
    with pytest.raises(ValueError, match="spawn_failed"):
        spawn_watermelon(1, ORIGIN, 0.0, history=hist)


def test_make_round_immutable_spec():
    r1 = make_round("r1", 5, ORIGIN, 0.0)
    r2 = make_round("r1", 5, ORIGIN, 0.0)
    assert r1 == r2  # 同一seed+履歴で完全同一
    assert r1.watermelon.hit_radius == HIT_VOLUME_R
    assert r1.watermelon.surface_m == 0.0  # 砂浜 — 台なし


def test_scene_metadata_is_authoritative():
    r = make_round("r1", 5, ORIGIN, 0.0)
    sc = r.scene
    assert sc["mode"] == "beach"
    assert sc["arena_r_m"] == 5.0
    assert sc["arena_center"] == [0.0, 0.0]
    assert sc["spawn_algo_version"] == "annulus-v2"
    assert sc["geometry_version"] == "beach-v1"
    assert sc["bearing_deg"] == 75.0


def test_fixture_marked_and_isolated():
    # 明示targetはfixture — seed=-1標識・履歴外・通常母集団に混ぜない
    r = make_round("fx", 123, ORIGIN, 0.0, target=(0.5, 0.3, WATERMELON_Z))
    assert r.seed == -1
    assert r.watermelon.pos == (0.5, 0.3, WATERMELON_Z)


def test_reachable_for_strike_front_ground():
    # ロボット前方0.55mの地上スイカは到達候補帯内
    assert reachable_for_strike(
        (0, 0, 0.82), 0.0, (0.55, 0.0, WATERMELON_Z))


def test_reachable_for_strike_yaw_rotated():
    assert reachable_for_strike(
        (0, 0, 0.82), math.pi / 2, (0.0, 0.55, WATERMELON_Z))
    assert not reachable_for_strike(
        (0, 0, 0.82), math.pi / 2, (0.0, 1.2, WATERMELON_Z))


def test_reachable_rejects_far():
    assert not reachable_for_strike(
        (0, 0, 0.82), 0.0, (2.0, 0.0, WATERMELON_Z))


def test_strike_alignment_reports_rel():
    rel, ok = strike_alignment(
        (0, 0, 0.82), 0.0, (0.4, 0.3, WATERMELON_Z))
    assert abs(rel[0] - 0.4) < 1e-9 and abs(rel[1] - 0.3) < 1e-9
    assert ok == (
        SWEEP_X[0] <= rel[0] <= SWEEP_X[1]
        and SWEEP_Y[0] <= rel[1] <= SWEEP_Y[1]
        and SWEEP_Z[0] <= rel[2] <= SWEEP_Z[1]
    )


def test_round_trip_spawn_then_align():
    # spawn位置がUI誘導の整合性を持つ: スイカへ近づき正面を向けば候補帯へ
    pos = spawn_watermelon(11, ORIGIN, 0.0)
    yaw = math.atan2(pos[1], pos[0])
    # 地上打撃の到達候補は前足域 — スイカの0.5m手前に立つ
    rx, ry = pos[0] - 0.5 * math.cos(yaw), pos[1] - 0.5 * math.sin(yaw)
    rel, ok = strike_alignment((rx, ry, 0.82), yaw, pos)
    assert ok, f"rel={rel} should be in sweep"


def test_spawn_distribution_10k():
    # 10k seed健全性 — 全域が距離帯・arena・有限時間に収まる
    import random as _r

    rng = _r.Random(0)
    seeds = [rng.randrange(2**31) for _ in range(10000)]
    for s in seeds:
        pos = spawn_watermelon(s, ORIGIN, 0.0)
        d = math.hypot(pos[0], pos[1])
        assert MIN_SPAWN_M <= d <= MAX_SPAWN_M
        assert d <= ARENA_R_M - ARENA_SPAWN_MARGIN_M + 1e-9
        assert pos[2] == WATERMELON_Z
