"""kotoba_harness — G1実験用の試験基盤（PR-A）。

設計原則（NEXT_INSTRUCTIONS_G1_RECOVERY.md §2）:
- 送信は承認されたgatewayだけが行う。arming環境変数は起動条件であり承認そのものではない
- 未知コマンドの既定変換は禁止（明示拒否）
- 送信周期は絶対monotonic deadlineで管理。受信処理が送信を前倒ししない
- 観測は型・finite・fingerprint・stream速度を検証してからfreshと呼ぶ
- 立位readyの成立を観測で確認してから歩行へ入る
- 転倒は準備〜tailまで監視・ラッチし、XY誤差だけではPASSしない
"""

from kotoba_harness.auth import (
    COMMAND_BUILDERS,
    BoundedProfile,
    RunManifest,
    SendGateway,
)
from kotoba_harness.errors import (
    AuthorizationRefused,
    HarnessError,
    InvalidPacket,
    ScheduleOverrun,
)
from kotoba_harness.observer import LatestStateObserver, PacketValidator, SimSample
from kotoba_harness.scheduler import DeadlineScheduler
from kotoba_harness.trial import (
    FALL_HEIGHT_M,
    FallLatch,
    NormalStopPolicy,
    ReadyGate,
    Scorer,
    TrialFSM,
    TiltMonitor,
)

__all__ = [
    "COMMAND_BUILDERS",
    "BoundedProfile",
    "RunManifest",
    "SendGateway",
    "AuthorizationRefused",
    "HarnessError",
    "InvalidPacket",
    "ScheduleOverrun",
    "LatestStateObserver",
    "PacketValidator",
    "SimSample",
    "DeadlineScheduler",
    "FALL_HEIGHT_M",
    "FallLatch",
    "NormalStopPolicy",
    "ReadyGate",
    "Scorer",
    "TrialFSM",
    "TiltMonitor",
]
