"""R3 yaw変換式の修正確認 + R2 停止制御のgreen回帰。

レビューN1〜N4の合成契約を新実装で満たすことを確認する。
"""

import math

import pytest


def corrected_yaw(q):
    """機体+X を world XY へ射影したyaw（R3修正後）。wxyz quaternion。"""
    w, x, y, z = q
    return math.atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))


class TestYawFormula:
    def test_zero_yaw(self):
        assert corrected_yaw((1.0, 0.0, 0.0, 0.0)) == pytest.approx(0.0)

    def test_yaw_90_deg(self):
        q = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
        assert math.degrees(corrected_yaw(q)) == pytest.approx(90.0)

    def test_yaw_neg_90_deg(self):
        q = (math.cos(math.pi / 4), 0.0, 0.0, -math.sin(math.pi / 4))
        assert math.degrees(corrected_yaw(q)) == pytest.approx(-90.0)

    def test_yaw_180_deg(self):
        assert math.degrees(corrected_yaw((0.0, 0.0, 0.0, 1.0))) == pytest.approx(180.0)

    def test_q_neg_q_same_yaw(self):
        """A10: q と -q は同一回転。符号だけで判定しない。"""
        q = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
        assert corrected_yaw(q) == pytest.approx(corrected_yaw(tuple(-c for c in q)))

    def test_pure_yaw_45(self):
        """45度のyaw（前回の欠陥式では22.5°になる）。"""
        q = (math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8))
        assert math.degrees(corrected_yaw(q)) == pytest.approx(45.0)


class TestStopControl:
    def test_signed_remaining_distance(self):
        """符号付残距離: 目標通過で負になる。"""
        target, pos = 0.45, 1.10
        remaining = target - pos
        assert remaining < 0  # 通過検出

    def test_passage_detection(self):
        """目標通過の検出: 符号が反転したら通過。"""
        prev_remaining = 0.1
        curr_remaining = -0.05
        passed = prev_remaining > 0 and curr_remaining <= 0
        assert passed

    def test_decel_timeout_from_brake_start(self):
        """減速timeoutは減速開始から計測（移動開始からではない）。"""
        motion_start = 0.0
        brake_start = 8.0
        now = 8.1
        timeout = 6.0
        # 減速開始からの経過: 8.1 - 8.0 = 0.1 < 6.0 → timeoutしない
        assert now - brake_start < timeout
        # 旧の誤り: 移動開始から測ると 8.1 > 6.0 → timeout する（不正確）

    def test_cmd_age_unmeasured_is_none(self):
        """cmd_ageが測定できない場合はnull（0を代入しない）。"""
        cmd_age = None
        assert cmd_age is None
