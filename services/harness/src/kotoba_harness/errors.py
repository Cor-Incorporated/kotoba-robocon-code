"""Harness例外。reasonは反証可能な文字列。"""


class HarnessError(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class AuthorizationRefused(HarnessError):
    """送信拒否。理由: no_arming / no_sim_mode / second_sender / expired /
    unknown_command / command_not_in_manifest / analog_out_of_bounds。"""


class ScheduleOverrun(HarnessError):
    """送信deadlineの逸脱。catch-up密集送信の代わりに中断する。"""


class InvalidPacket(HarnessError):
    """観測パケットの検証失敗。bad_size / bad_fingerprint / nonfinite /
    bad_quat_norm / num_ranges_invalid / fingerprint_changed。"""
