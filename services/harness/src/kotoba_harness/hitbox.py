"""hitbox判定（C3改訂）— 手先world軌跡 × スイカhit volume。

RoundWorld(正本)が与えるスイカworld位置に対し、strike中の手先軌跡が
hit_volume半径以内を通過すれば HIT。判定は観測(FK導出の実手先位置)のみに
基づく — 送信指令や期待値は証拠にならない。

C3: 点サンプル最小値ではなく、連続する有効サンプル間の「線分↔球」の
swept判定を正本とする（サンプル間を線分が横切る当たりを見落とさない）。
線分を結べるのは同一strike内の時間的に連続したサンプル間だけ —
長い欠測・時刻の巻戻り・不連続jumpの両側は結ばない。
"""

from __future__ import annotations

import math

from kotoba_harness.kinematics import dist, hands_world, stick_tips_world

# 線分を結んでもよいサンプル間隔の上限（秒）— これを超える欠測は分断
MAX_SEGMENT_GAP_S = 0.25
# 線分を結んでもよい手先の1サンプル移動量上限（m）— 不連続jumpは分断
MAX_SEGMENT_STEP_M = 0.75


def _seg_point_dist(a, b, p):
    """線分abと点pの最短距離（closest point on segment）。"""
    ax, ay, az = a
    bx, by, bz = b
    px, py, pz = p
    abx, aby, abz = bx - ax, by - ay, bz - az
    apx, apy, apz = px - ax, py - ay, pz - az
    ab2 = abx * abx + aby * aby + abz * abz
    if ab2 <= 1e-12:
        return math.dist(a, p), 0.0
    t = max(0.0, min(1.0, (apx * abx + apy * aby + apz * abz) / ab2))
    cx, cy, cz = ax + abx * t, ay + aby * t, az + abz * t
    return math.dist((cx, cy, cz), p), t


def _seg_closest_point(a, b, p):
    ax, ay, az = a
    abx, aby, abz = b[0] - ax, b[1] - ay, b[2] - az
    ab2 = abx * abx + aby * aby + abz * abz
    if ab2 <= 1e-12:
        return a
    t = max(
        0.0,
        min(
            1.0,
            ((p[0] - ax) * abx + (p[1] - ay) * aby + (p[2] - az) * abz) / ab2,
        ),
    )
    return (ax + abx * t, ay + aby * t, az + abz * t)


class HitRecorder:
    """strike中の手先world軌跡を記録する（同一strike内の連続サンプル）。"""

    def __init__(self, stick_m: float = 0.0) -> None:
        # stick_m>0: 手先に固定した仮想棒の先端軌跡を記録する
        # （地上スイカは素手では届かない — 台なし配置の実効エフェクタ）。
        # samples の L/R は end effector 位置（stick有り=棒先端、無し=手先）。
        self.stick_m = float(stick_m)
        self.samples = []  # (t, eff_L, eff_R)

    def record(self, t, base_pos, base_quat, joints):
        if self.stick_m > 0:
            hl, hr = stick_tips_world(base_pos, base_quat, joints, self.stick_m)
        else:
            hl, hr = hands_world(base_pos, base_quat, joints)
        self.samples.append((t, hl, hr))

    def _continuous_pairs(self):
        """連続とみなせる隣接サンプル対をyieldする。

        時刻巻戻り・長い欠測・不連続jumpで分断し、それらを線で結ばない。
        """
        for i in range(1, len(self.samples)):
            t0, l0, r0 = self.samples[i - 1]
            t1, l1, r1 = self.samples[i]
            dt = t1 - t0
            if not (0 < dt <= MAX_SEGMENT_GAP_S):
                continue
            for hand, a, b in (("L", l0, l1), ("R", r0, r1)):
                if math.dist(a, b) <= MAX_SEGMENT_STEP_M:
                    yield hand, a, b, t0, t1

    def judge(self, target, hit_radius):
        """swept判定: 観測点と連続線分の球への最短距離 ≤ hit_radius なら HIT。

        - 点は常に実観測として評価する
        - 線分は連続と確認できた隣接サンプル間だけに作る
          （欠測・巻戻り・不連続jumpを結んでHITにしない）
        """
        best = None  # (dist, t, hand, point)
        # 実観測点の評価
        for t, hl, hr in self.samples:
            for hand, p in (("L", hl), ("R", hr)):
                d = dist(p, target)
                if best is None or d < best[0]:
                    best = (d, t, hand, p)
        # 連続区間の線分評価（サンプル間を横切る当たりを拾う）
        for hand, a, b, t0, t1 in self._continuous_pairs():
            d, _t = _seg_point_dist(a, b, target)
            if best is None or d < best[0]:
                best = (d, (t0 + t1) / 2, hand, _seg_closest_point(a, b, target))
        if best is None:
            return {"hit": False, "reason": "no_samples"}
        d, t, hand, p = best
        return {
            "hit": d <= hit_radius,
            "min_dist_m": round(d, 4),
            "min_dist_t_s": round(t, 2),
            "hand": hand,
            "effector": ("stick_tip" if self.stick_m > 0 else "hand"),
            "stick_m": self.stick_m,
            "closest_point": [round(v, 4) for v in p],
            "target": [round(v, 4) for v in target],
            "hit_radius_m": hit_radius,
            "samples": len(self.samples),
            "judge_method": "swept_segment",
        }

    def path_summary(self):
        """軌跡の包絡(各軸min/max) — 狙い位置の設計用。

        AABBは候補領域であり、箱の全点に手が到達する証拠ではない
        （実到達は連続軌跡との交差で判定する）。
        """
        if not self.samples:
            return {}
        pts = [p for _, hl, hr in self.samples for p in (hl, hr)]
        lo = [min(p[i] for p in pts) for i in range(3)]
        hi = [max(p[i] for p in pts) for i in range(3)]
        return {"min": [round(v, 3) for v in lo], "max": [round(v, 3) for v in hi]}
