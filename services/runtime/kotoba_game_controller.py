#!/usr/bin/env python3
"""ゲーム制御controller（SDK container内）— C3改訂版。

「ことばでスイカ割り」の実行中、参加者の方向指示を 20Hz の連続stick指令へ
変換する。2つの指令経路:

- /kotoba/run/control.json（latest-wins）: move/stop/end。moveは
  issued_wall+dur_s の有界nudgeとして適用し、期限後は自動でidleへ戻す。
  issued_wall が CMD_STALE_S より古い指令は受理しない。
- /kotoba/run/strike.json（単発event mailbox）: strike_id+expires_wall。
  各strike_idは最大1回だけ実行（冪等）。打撃開始（dance遷移観測）を
  もって「実消費」とし、開始前の失敗・期限切れは消費しない。

打撃は有限状態機械（pre→dance→swing→recover）として制御ループの各tickで
駆動する — do_strikeの同期ブロックは廃止。打撃中も STOP/END・期限・
観測・state更新を通常どおり処理する（約5秒以上入力を放置しない）。
STOPで打撃を管理中断（recoverへ）、ENDでrecover後に終了する。

結果の分離（3.4）:
- verdict: 制御経路の健全性評価（PASSは正常終了＋終了後立位保持の実観測）
- control.health: ok / obs_lost / manifest_expired / control_lost
- round_end: ゲーム規則による終了理由（hit/out_of_swings/timeout/ended）
obs_lost等の異常をPASSや普通のtimeoutへ落とさない。

入力: <result.json> <manifest.json>
manifest: run_id / lease_id(必須) / arming("1") / expires_in_s / boot_expect
          target=[x,y,z]? / hit_radius? / max_swings? / time_limit_s? /
          end_on_hit?
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "/kotoba/harness_src")

from kotoba_harness.auth import RunManifest, SendGateway
from kotoba_harness.control import (
    CMD_STALE_S,
    END,
    MOVE,
    STOP,
    ControlRejected,
    LatestWinsGate,
    check_fresh,
    parse_command,
    parse_strike_event,
)
from kotoba_harness.errors import AuthorizationRefused, HarnessError
from kotoba_harness.hitbox import HitRecorder
from kotoba_harness.motion import (
    ExecProfile,
    MotionExecutor,
    MotionStep as ExecMotionStep,
)
from kotoba_harness.trial import (
    FallLatch,
    Phase,
    ReadyGate,
    TrialFSM,
    up_vector_tilt_deg,
)

from kotoba_runner import Sample
from kotoba_strike_probe import StrikeObserver

HZ = 20.0
CONTROL_PATH = Path("/kotoba/run/control.json")
STRIKE_PATH = Path("/kotoba/run/strike.json")
STATE_PATH = Path("/kotoba/run/control_state.json")
# 継続jog中の生存確認 — UIが操作者の存続を定期通知する（POST /api/game/heartbeat）。
# 欠落は操作者消失・通信断とみなし jog を減速停止させる（安全側）。
HEARTBEAT_PATH = Path("/kotoba/run/heartbeat.json")
HB_STALE_S = 2.5    # この秒数heartbeatが無ければjog停止
HB_GRACE_S = 3.5    # jog開始〜最初のheartbeat到着の猶予
STATE_WRITE_HZ = 2.0
OBS_LOST_ABORT_S = 1.0
# 打撃FSM（B2実測値ベース）
STRIKE_PRE_S = 4.5        # 静止・支持姿勢の確認猶予（nudge後の実測減衰~4sを許容・TTL内）
STRIKE_PRE_VEL = 0.12     # 静止判定の水平速度上限 (m/s) — walk-idleの微動~0.1を許容
STRIKE_PRE_HOLD_S = 0.3   # 静止の持続要件
DANCE_WAIT_S = 4.0        # dance遷移の観測猶予
SWING_WINDOW_S = 4.5      # 有効打撃区間（軌跡記録）
RECOVER_HOLD_S = 0.6      # walk復帰送信
RECOVER_VERIFY_S = 2.0    # walk task復帰の観測猶予
# 終了後の立位保持確認（正常終了の実観測証拠）
# 複合条件: 高さ・姿勢・位置逸脱・軌跡長・sim時刻の持続 — 瞬間速度や
# 純変位だけでは停止・安定を代替しない（0.1m/s等速・45度静止傾きは失格）
END_HOLD_S = 2.5           # 保持に要する継続時間（sim時刻で計測）
END_HOLD_Z_MIN = 0.6       # 立位帯0.75-0.90と倒伏<0.50の中間
END_HOLD_VEL = 0.25        # 診断用 — walk-idle微振動~0.16-0.25を含む
END_HOLD_TILT_DEG = 20.0   # 姿勢上限 — idle平衡は10度未満、45度傾きは失格
END_HOLD_DISPL_M = 0.20    # アンカーからの水平逸脱上限 — 0.1m/s等速を2sで検出
END_HOLD_PATH_M = 0.5      # 保持中の累積軌跡長上限 — 円周・往復歩行を検出
END_SETTLE_STEP_M = 0.06   # 0.5s(sim時刻)窓の純変位上限 — 実歩行は0.2m超/0.5s
END_SETTLE_WIN_S = 0.5     # 変位測定窓（sim時刻 — pause/凍結は保持に算入しない）
# 窓制御の不感帯対策: popは「次点がまだ窓を覆う」時のみ — 粗いsim_t刻み
# (~0.2s)でpop直後スパンが0.45s充填条件を永遠に下回る事故を防ぐ。
# 終了指令後の減衰猶予 — LPF残動(~4s)が収まるまで保持計測を開始しない
END_SETTLE_S = 8.0
END_SETTLE_STABLE_S = 0.8  # 安定条件の継続時間（sim時刻）
# 操縦範囲（試用profileの明示的限定）— 開始位置からの水平距離上限。
# 実測では全成功roundが<3mに収まり、誘導不收束の漂流は14mで転倒した。
# 範囲外への歩き続けは安全側でラウンド終了（arena_exit）にする。
ARENA_R_M = 5.0


class StrikeFSM:
    """一振りの有限状態機械 — 制御tickごとに駆動（非ブロッキング）。

    phases: pre → dance → swing → recover → done。
    consumed は dance遷移を実観測した時点で確定（実開始=消費正本）。
    それ以前の失敗・期限切れ・中断は消費しない（返却扱い）。
    """

    def __init__(self, strike_id, seq, target, hit_radius, stick_m=0.0):
        self.stick_m = float(stick_m)
        self.strike_id = strike_id
        self.seq = seq
        self.phase = "pre"
        self.t0 = time.monotonic()
        self.still_since = None
        self.consumed = False
        self.recorder = (
            HitRecorder(stick_m=self.stick_m) if target is not None else None
        )
        self.target = target
        self.hit_radius = hit_radius
        self.entry = {"event": "strike", "seq": seq, "strike_id": strike_id}
        self.abort_reason = None  # "stop" | "end" — 管理中断
        self.pending_end = False

    def _phase_to(self, phase):
        self.phase = phase
        self.t0 = time.monotonic()
        self.still_since = None

    def abort(self, reason):
        """STOP/ENDによる管理中断 — swing/dance/preのどこでも受け付け、
        recover（walk復帰）へ移す。物理的瞬間停止ではなく検証済み回復。
        recover中のENDも記録し、完了時に終了へ反映する。"""
        if reason == "end":
            self.pending_end = True
        if self.phase in ("done", "recover"):
            return
        self.abort_reason = reason
        self.entry["interrupted"] = reason
        self._phase_to("recover")

    def tick(self, issue, frames, obs_sample, obs_ok=True):
        """1制御tick分の処理。返り値: (frame, done, done_info)。"""
        frame_dance, frame_walk, frame_idle = frames
        p, v, q, j, tsk = obs_sample
        now = time.monotonic()
        elapsed = now - self.t0
        # 静止判定は水平速度のみ（walk-idleの上下動で誤って非静止扱いしない）
        speed = (v[0] ** 2 + v[1] ** 2) ** 0.5

        if self.phase == "pre":
            # 打撃前の支持姿勢/速度条件を実測（送信しただけでは静止成立にしない）
            if not obs_ok:
                self.still_since = None
            elif tsk == "walk" and speed < STRIKE_PRE_VEL:
                if self.still_since is None:
                    self.still_since = now
                if now - self.still_since >= STRIKE_PRE_HOLD_S:
                    self.entry["sent_dance"] = True
                    self._phase_to("dance")
                    return frame_dance, False, None
            else:
                self.still_since = None
            if elapsed > STRIKE_PRE_S:
                self.entry["status"] = "not_started"
                self.entry["aborted"] = "precondition_timeout"
                self.entry["stillness_vel"] = round(speed, 3)
                return frame_idle, True, self.entry
            return frame_idle, False, None

        if self.phase == "dance":
            if tsk == "dance":
                # 打撃動作の実開始を観測 — ここで消費確定
                self.consumed = True
                self.entry["started"] = True
                self._phase_to("swing")
                return frame_idle, False, None
            if elapsed > DANCE_WAIT_S:
                self.entry["status"] = "not_started"
                self.entry["aborted"] = "dance_not_started"
                return frame_idle, True, self.entry
            return frame_dance, False, None

        if self.phase == "swing":
            # 観測欠測サンプルは軌跡へ記録しない（偽の手先位置で誤判定しない）
            if self.recorder is not None and obs_ok:
                self.recorder.record(elapsed, p, q, j)
            if elapsed > SWING_WINDOW_S:
                self._phase_to("recover")
                return frame_walk, False, None
            return frame_idle, False, None

        # recover: walk復帰を送信し、task復帰または猶予切れで完了
        if self.phase == "recover":
            if (tsk == "walk" and elapsed > RECOVER_HOLD_S) or (
                elapsed > RECOVER_VERIFY_S
            ):
                self.entry["done"] = True
                if self.consumed:
                    self.entry["status"] = "done"
                elif self.entry.get("sent_dance"):
                    # dance指令は送出済みだが遷移を観測できなかった —
                    # 物理的に振ったか不明 = 要確認（推測で非消費にも消費にもしない）
                    self.entry["status"] = "unverified"
                else:
                    self.entry["status"] = "not_started"
                if self.recorder is not None:
                    self.entry["judge"] = self.recorder.judge(
                        self.target, self.hit_radius
                    )
                return frame_walk, True, self.entry
            return frame_walk, False, None
        return frame_idle, True, self.entry


def _verify_end_hold(obs, latch, events) -> bool:
    """終了後の立位保持を実観測で検証する。保持=「転ばず・立ったまま・
    歩き去らない」の複合条件 — 高さ・姿勢・アンカー逸脱・累積軌跡を
    全サンプルで検査し、sim時刻の継続観測を要求する（pause/欠測・
    wall-clock経過は保持に算入しない）。変位だけでは0.1m/s等速や
    45度静止傾きを通すため複合化し、全観測はfall latchへも流して
    転倒を終了判定と整合させる。
    契約上の観測断（stale例外）: その区間は保持時間へ加算せず、復帰
    サンプルから連続保持を積み直す。空白前後の端点差はdev/pathの
    下限として評価し続けるが、空白中を正常とは推定しない。
    """
    settled = False
    p = (0, 0, 0)
    q = (1, 0, 0, 0)
    v = (0, 0, 0)
    speed = 0.0
    tilt = 0.0
    step = 0.0
    sim_t = 0.0
    stable_since = None
    win = []
    settle_deadline = time.monotonic() + END_SETTLE_S
    while time.monotonic() < settle_deadline:
        try:
            p, v, q, j, sim_t, _q2, _n, tsk = obs.latest()
            latch.observe(Phase.TAIL.value, Sample(p, v, q))
            speed = (v[0] ** 2 + v[1] ** 2) ** 0.5
            tilt = up_vector_tilt_deg(q)
            win.append((sim_t, p[0], p[1]))
            # 窓制御の不感帯対策: 「次点がまだ0.5s超を覆う」時だけpopする。
            # 0.5s超を無条件popすると粗いsim_t刻み(~0.2s)でpop直後スパンが
            # 0.45s充填条件を永遠に下回り、静止していてもsettleしない
            # （実機で確認）。この条件ならpop後スパンは常に>=0.5sを維持する。
            while len(win) > 1 and sim_t - win[1][0] > END_SETTLE_WIN_S:
                win.pop(0)
            step = (
                ((p[0] - win[0][1]) ** 2 + (p[1] - win[0][2]) ** 2) ** 0.5
                if len(win) > 1
                else 0.0
            )
            win_full = (
                bool(win) and sim_t - win[0][0] >= END_SETTLE_WIN_S * 0.9
            )
            if (
                not latch.fallen
                and p[2] >= END_HOLD_Z_MIN
                and tilt <= END_HOLD_TILT_DEG
                and win_full
                and step <= END_SETTLE_STEP_M
            ):
                if stable_since is None:
                    stable_since = sim_t
                if sim_t - stable_since >= END_SETTLE_STABLE_S:
                    settled = True
                    break
            else:
                stable_since = None
        except HarnessError:
            # 契約上の観測断: 安定継続と変位窓を失効し、復帰後に新規の
            # 0.5s窓から積み直す（欠測前の古いサンプルを窓へ残さない）。
            stable_since = None
            win.clear()
        time.sleep(0.1)
    if not settled:
        events.append(
            {
                "event": "end_hold_unsettled",
                "z": round(p[2], 3),
                "speed": round(speed, 3),
                "tilt": round(tilt, 1),
                "step": round(step, 3),
                "fallen": latch.fallen,
            }
        )
        return False
    ax, ay = p[0], p[1]  # settle完了点をアンカーとする
    hold_start_sim = sim_t
    prev = (p[0], p[1])
    path = 0.0
    max_dev = 0.0
    dev = 0.0
    gap = False   # 契約上の観測断（stale例外）が進行中か
    gap_count = 0
    # sim時刻でEND_HOLD_S保持 + wall上限で観測断を検出
    wall_deadline = time.monotonic() + END_HOLD_S * 4
    while True:
        if time.monotonic() > wall_deadline:
            events.append({"event": "end_hold_observation_timeout"})
            return False
        try:
            p, v, q, j, sim_t, _q2, _n, tsk = obs.latest()
            latch.observe(Phase.TAIL.value, Sample(p, v, q))
            if gap:
                # 空白区間は連続保持へ算入しない — 復帰サンプルから保持を
                # 積み直す（両端が正常でも空白中を正常とは推定しない）。
                # アンカー・累積軌跡・最大逸脱は継続評価: 空白中の移動は
                # 復元できないが、端点差はdev/pathの下限として残る。
                hold_start_sim = sim_t
                gap_count += 1
                events.append(
                    {
                        "event": "end_hold_observation_gap",
                        "sim_t": round(sim_t, 2),
                        "gaps": gap_count,
                    }
                )
                gap = False
            speed = (v[0] ** 2 + v[1] ** 2) ** 0.5
            tilt = up_vector_tilt_deg(q)
            dev = ((p[0] - ax) ** 2 + (p[1] - ay) ** 2) ** 0.5
            path += ((p[0] - prev[0]) ** 2 + (p[1] - prev[1]) ** 2) ** 0.5
            prev = (p[0], p[1])
            max_dev = max(max_dev, dev)
            if (
                latch.fallen
                or p[2] < END_HOLD_Z_MIN
                or tilt > END_HOLD_TILT_DEG
                or dev > END_HOLD_DISPL_M
                or path > END_HOLD_PATH_M
            ):
                events.append(
                    {
                        "event": "end_hold_broken",
                        "z": round(p[2], 3),
                        "tilt": round(tilt, 1),
                        "dev": round(dev, 3),
                        "path": round(path, 3),
                        "speed": round(speed, 3),
                        "fallen": latch.fallen,
                    }
                )
                return False
            if sim_t - hold_start_sim >= END_HOLD_S:
                break
        except HarnessError:
            # 契約上の観測断（observerが明示するstale・未受信・worker死）—
            # 通常の間引き受信とは区別され、ここに来た時点で実欠測と確定。
            # 空白を保持時間へ加算しないため、復帰サンプルで積み直す。
            gap = True
        time.sleep(0.1)
    events.append(
        {
            "event": "end_hold_detail",
            "max_dev": round(max_dev, 3),
            "path": round(path, 3),
            "sim_s": round(sim_t - hold_start_sim, 2),
            "gaps": gap_count,
        }
    )
    return True


def main() -> int:
    result_path = sys.argv[1]
    manifest = json.loads(Path(sys.argv[2]).read_text())
    lease_id = manifest.get("lease_id")
    result = {"run_id": manifest.get("run_id", "unknown"), "events": []}
    events = result["events"]
    target = manifest.get("target")  # スイカworld位置（RoundWorld正本から）
    hit_radius = float(manifest.get("hit_radius", 0.20))
    # 地上スイカ用の仮想棒（0=素手先端）。FK実測で0.7m棒が地表へ届く。
    stick_m = float(manifest.get("stick_m", 0.0))
    max_swings = manifest.get("max_swings")
    time_limit_s = float(manifest.get("time_limit_s", 0) or 0)
    end_on_hit = bool(manifest.get("end_on_hit", target is not None))
    arena_anchor = manifest.get("robot_start")
    arena_r = float(manifest.get("arena_r_m") or ARENA_R_M)

    expect = manifest.get("boot_expect")
    obs = None
    gateway = None
    try:
        if not lease_id:
            raise HarnessError("lease_not_bound")
        expected = int(expect) if expect and str(expect).isdigit() else None
        obs = StrikeObserver(expected_nonce=expected)
        deadline = time.monotonic() + 15
        while True:
            try:
                pos, vel, quat, joints, sim_t, seq0, nonce, task = obs.latest()
                break
            except HarnessError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.1)
        if not expect or str(expect) in ("boot-pending", "pending"):
            raise HarnessError("boot_not_bound")
        if str(nonce) != str(expect):
            raise HarnessError(f"boot_mismatch:{nonce}!={expect}")
        result["boot_nonce"] = nonce

        rm = RunManifest(
            run_id=manifest["run_id"],
            purpose=manifest.get("purpose", "evaluation"),
            profile=__import__(
                "kotoba_harness.auth", fromlist=["profile_for_capabilities"]
            ).profile_for_capabilities(manifest.get("capabilities")),
            sim_boot_id=nonce,
            expires_monotonic=time.monotonic()
            + float(manifest.get("expires_in_s", 240)),
        )
        gateway = SendGateway(
            rm,
            obs.handle,
            "virtual_gamepad/gamepad_keys",
            now_monotonic=time.monotonic(),
            sim_mode_confirmed=True,
            arming_env=manifest.get("arming", ""),
        )

        fsm = TrialFSM()
        latch = FallLatch()
        gate = ReadyGate(hold_s=2.0)
        seq_gate = LatestWinsGate()
        fsm.transition(Phase.READY_WAIT)

        ready_deadline = time.monotonic() + 15.0
        while time.monotonic() < ready_deadline:
            pos, vel, quat, joints, *_ = obs.latest()
            sample = Sample(pos, vel, quat)
            latch.observe(Phase.READY_WAIT.value, sample)
            if latch.fallen:
                raise HarnessError("prep_fall")
            if gate.observe(sample, time.monotonic(), None):
                break
        else:
            raise HarnessError("invalid_start_never_ready")
        events.append({"event": "ready", "z": round(pos[2], 3)})

        frame_idle = gateway.prepare("idle")
        frame_walk = gateway.prepare("combo_walk")
        frame_dance = gateway.prepare("combo_dance")
        frames = (frame_dance, frame_walk, frame_idle)

        def issue(frame, name):
            gateway.issue(
                frame, now_monotonic=time.monotonic(), command_name=name
            )

        # walkモードへ入る（方向指令はwalk taskでのみ有効）
        fsm.transition(Phase.WALK)
        for _ in range(int(0.6 * HZ)):
            issue(frame_walk, "combo_walk")
            time.sleep(1 / HZ)
        events.append({"event": "walk_mode"})

        # ---- 制御ループ状態 ----
        move_frame = None          # 現行の有界nudge（旧形式互換）
        move_until_wall = 0.0      # issued_wall + dur_s
        motion_ex = None           # steps付きmoveの観測閉ループexecutor
        motion_seq = 0             # executorが担うcmdのseq
        jog_since_mono = None      # 現行jog指令の開始（heartbeat猶予の起点）
        last_cmd_state = None      # 直近の実行状態（完了・中断後も保持）
        motion_ev_len = 0          # executor eventsの排出済み数
        current_type = "idle"
        applied_seq = 0
        strikes_started = 0        # 実消費正本（dance遷移観測数）
        strikes_done = 0
        last_strike = None
        last_strike_attempt = None  # 未開始で終わった直近打撃の理由（消費なし）
        last_strike_done = None    # 完了したstrike_id（予約解消用）
        strike = None              # StrikeFSM | None
        strike_ids_seen = set()
        strike_expired = []        # 期限切れ等で開始しなかった strike_id
        strike_unverified = []     # dance送出後に開始未確認で中断した strike_id
        rejects = []
        file_was_present = CONTROL_PATH.exists()
        strike_file_seen = STRIKE_PATH.exists()
        obs_dead_since = None
        last_state_write = 0.0
        end_received = False
        arena_exited = False       # 操縦範囲外のラウンド終了は1回のみ
        shutdown_reason = None     # manifest_expired / obs_lost / control_lost
        round_end_reason = None    # hit / out_of_swings / timeout / arena_exit
        health = "ok"
        round_deadline_wall = (
            time.time() + time_limit_s if time_limit_s > 0 else None
        )

        # ---- 音声トリガー用の有界event delta（R5-04） --------------------
        # HTTP受理・ASR transcriptではなく、controllerが観測した実状態遷移
        # だけを出す。control_state.json経由でUIのAudioManagerへ配信し、
        # client側はrun毎のhigh-water markで新規分だけ再生する。
        audio_seq = 0
        events_tail = []  # 直近32件 — reload時のsnapshot用に有界化

        def emit_audio(kind, **fields):
            nonlocal audio_seq
            audio_seq += 1
            events_tail.append(
                {
                    "event_seq": audio_seq,
                    "kind": kind,
                    "t_wall": round(time.time(), 3),
                    **fields,
                }
            )
            del events_tail[:-32]

        # 実プレイ時計（round_deadline_wall）が起動した = round開始の実イベント
        emit_audio("round_started")

        def _cmd_state_dict():
            """実行状態DTO — steps付きmoveの進捗をUI/APIへ露出。
            完了・中断後も直近状態を残す（次指令で上書き）。"""
            if motion_ex is None:
                return last_cmd_state
            d = motion_ex.prog.to_dict()
            d["seq"] = motion_seq
            d["mode"] = "closed_loop"
            return d

        def _drain_motion_events():
            nonlocal motion_ev_len
            if motion_ex is None:
                return
            new = motion_ex.events[motion_ev_len:]
            motion_ev_len = len(motion_ex.events)
            for ev in new:
                ev["seq"] = motion_seq
                events.append(ev)

        def write_state(t_mono, force=False):
            nonlocal last_state_write
            if not force and t_mono - last_state_write < 1 / STATE_WRITE_HZ:
                return
            last_state_write = t_mono
            try:
                STATE_PATH.write_text(
                    json.dumps(
                        {
                            "lease_id": lease_id,
                            "applied_seq": applied_seq,
                            "current_type": current_type,
                            "move_until_wall": (
                                move_until_wall if move_frame else None
                            ),
                            "strike": (
                                {
                                    "strike_id": strike.strike_id,
                                    "phase": strike.phase,
                                    "consumed": strike.consumed,
                                }
                                if strike is not None
                                else None
                            ),
                            "strike_expired": strike_expired[-5:],
                            "strike_unverified": strike_unverified[-5:],
                            # 未開始で終わった直近打撃の理由（消費なし —
                            # last_strikeは消費済み判定専用で混ぜない）
                            "last_strike_attempt": last_strike_attempt,
                            "last_strike_done": last_strike_done,
                            "strikes_started": strikes_started,
                            "strikes_done": strikes_done,
                            "rejects": rejects[-5:],
                            "last_strike": last_strike,
                            "round_deadline_wall": round_deadline_wall,
                            "command_state": _cmd_state_dict(),
                            "max_swings": max_swings,
                            "health": health,
                            "fallen": latch.fallen,
                            "events_tail": list(events_tail),
                            "t_wall": time.time(),
                        }
                    )
                )
            except OSError:
                pass

        while True:
            t0 = time.monotonic()
            now_wall = time.time()
            # 0) ラウンド時限 — 制御下で終了処理へ
            if (
                round_deadline_wall is not None
                and now_wall > round_deadline_wall
            ):
                round_end_reason = "timeout"
                events.append({"event": "round_end", "reason": "timeout"})
                emit_audio("round_failed", reason="timeout")
                break
            # 1) latest-wins指令（move/stop/end）
            try:
                data = json.loads(CONTROL_PATH.read_text())
                file_was_present = True
                cmd = parse_command(data, lease_id)
                check_fresh(cmd, now_wall)
                if seq_gate.accept(cmd):
                    applied_seq = cmd.seq
                    if cmd.type == MOVE:
                        if cmd.steps:
                            # 観測閉ループ実行 — 基準姿勢は次tickの観測で固定
                            # （指令受理時点の古い観測でアンカーしない）
                            msteps = [
                                ExecMotionStep(
                                    action=st.action,
                                    direction=st.direction,
                                    target=(
                                        st.target_m
                                        if st.action == "translate"
                                        else math.radians(st.target_deg)
                                    ),
                                    action_key=st.action_key,
                                    label=st.label,
                                    pace=st.pace,
                                )
                                for st in cmd.steps
                            ]
                            # 置換される旧指令はsupersededとして記録する
                            # （完了/失敗と同じく「なぜ終わったか」を残す）
                            if motion_ex is not None and not motion_ex.done:
                                motion_ex.abort("superseded")
                                # abortで追加された末尾イベントも監査へ排出
                                for ev in motion_ex.events[motion_ev_len:]:
                                    ev["seq"] = motion_seq
                                    events.append(ev)
                                last_cmd_state = _cmd_state_dict()
                                events.append(
                                    {"event": "cmd_superseded",
                                     "seq": motion_seq,
                                     "by_seq": cmd.seq}
                                )
                            bound_center = (
                                (arena_anchor[0], arena_anchor[1])
                                if arena_anchor is not None
                                else None
                            )
                            motion_ex = MotionExecutor(
                                msteps,
                                bound_center=bound_center,
                                bound_r=arena_r,
                            )
                            motion_seq = cmd.seq
                            move_frame = None
                            current_type = MOVE
                            # 新executorのeventsは0から排出（旧indexの
                            # 引継ぎで先頭イベントを落とさない）
                            motion_ev_len = 0
                            # jog開始時刻 — heartbeat猶予の起点
                            jog_since_mono = (
                                time.monotonic()
                                if any(st.action == "jog" for st in cmd.steps)
                                else None
                            )
                            events.append(
                                {
                                    "event": "cmd",
                                    "seq": cmd.seq,
                                    "type": MOVE,
                                    "steps": [
                                        {
                                            "action": st.action,
                                            "dir": st.direction,
                                            "m": st.target_m,
                                            "deg": st.target_deg,
                                            "pace": st.pace,
                                        }
                                        for st in cmd.steps
                                    ],
                                    "action_key": cmd.action_key,
                                    "stop_after": cmd.stop_after,
                                    "via": (data.get("cmd") or {}).get("via"),
                                }
                            )
                        else:
                            try:
                                move_frame = gateway.prepare_move(
                                    cmd.fwd, cmd.lat, cmd.yaw
                                )
                                move_until_wall = cmd.issued_wall + cmd.dur_s
                                motion_ex = None
                                current_type = MOVE
                                events.append(
                                    {
                                        "event": "cmd",
                                        "seq": cmd.seq,
                                        "type": MOVE,
                                        "v": [cmd.fwd, cmd.lat, cmd.yaw],
                                        "dur_s": cmd.dur_s,
                                        "via": (data.get("cmd") or {}).get("via"),
                                    }
                                )
                            except AuthorizationRefused as exc:
                                rejects.append(
                                    {"seq": cmd.seq, "reason": str(exc)}
                                )
                    elif cmd.type == STOP:
                        move_frame = None
                        if motion_ex is not None and not motion_ex.done:
                            if motion_ex.jogging:
                                # 継続jogの停止は正常終了 — 減速→静止確認まで
                                # 観測して completed(reason=stop) とする。
                                # finite stepへのSTOPはabort（中断）のまま。
                                motion_ex.request_stop("stop")
                            else:
                                motion_ex.abort("stop")
                        current_type = STOP
                        if strike is not None:
                            strike.abort("stop")
                        events.append(
                            {
                                "event": "cmd",
                                "seq": cmd.seq,
                                "type": STOP,
                                "via": (data.get("cmd") or {}).get("via"),
                            }
                        )
                    elif cmd.type == END:
                        end_received = True
                        move_frame = None
                        emit_audio("round_aborted", reason="end")
                        if motion_ex is not None and not motion_ex.done:
                            motion_ex.abort("end")
                        events.append(
                            {
                                "event": "cmd",
                                "seq": cmd.seq,
                                "type": END,
                                "via": (data.get("cmd") or {}).get("via"),
                            }
                        )
                        if strike is not None:
                            # 打撃中のEND: 管理中断→recover完了後に終了
                            strike.abort("end")
                        else:
                            break
            except FileNotFoundError:
                if file_was_present:
                    file_was_present = False
                    move_frame = None
                    if motion_ex is not None and not motion_ex.done:
                        motion_ex.abort("control_lost")
                    current_type = STOP
                    events.append({"event": "control_file_lost"})
            except (json.JSONDecodeError, ControlRejected) as exc:
                reason = getattr(exc, "reason", "bad_json")
                rejects.append({"reason": reason})
            # 1b) strike mailbox（単発event — latest-winsに混ぜない）
            try:
                sdata = json.loads(STRIKE_PATH.read_text())
                strike_file_seen = True
                scmd = parse_strike_event(sdata, lease_id)
                if scmd.strike_id not in strike_ids_seen:
                    strike_ids_seen.add(scmd.strike_id)
                    if strike is not None:
                        # 打撃中の追加分は実行しない（API側でbusy拒否が正規）
                        strike_expired.append(scmd.strike_id)
                        events.append(
                            {
                                "event": "strike_skipped",
                                "strike_id": scmd.strike_id,
                                "reason": "busy",
                            }
                        )
                    elif now_wall >= scmd.expires_wall:
                        strike_expired.append(scmd.strike_id)
                        events.append(
                            {
                                "event": "strike_expired",
                                "strike_id": scmd.strike_id,
                            }
                        )
                    else:
                        strike = StrikeFSM(
                            scmd.strike_id,
                            scmd.seq,
                            target,
                            hit_radius,
                            stick_m=stick_m,
                        )
                        # 新規受理で前回の未開始理由は陳腐化（直近試行はこれ）
                        last_strike_attempt = None
                        events.append(
                            {
                                "event": "strike_accepted",
                                "strike_id": scmd.strike_id,
                                "seq": scmd.seq,
                                "via": (sdata.get("cmd") or {}).get("via"),
                            }
                        )
            except FileNotFoundError:
                if strike_file_seen:
                    strike_file_seen = False
                    events.append({"event": "strike_file_lost"})
            except (json.JSONDecodeError, ControlRejected) as exc:
                reason = getattr(exc, "reason", "bad_json")
                rejects.append({"reason": f"strike_{reason}"})
            # 2) 観測（1tick 1回 — 指令・FSM・健全性で共有）
            try:
                p, v, q, j, _st, _sq, _nn, tsk = obs.latest()
                latch.observe(Phase.WALK.value, Sample(p, v, q))
                if latch.fallen:
                    emit_audio("control_fault", reason="fallen")
                    raise HarnessError("walk_fall")
                obs_dead_since = None
                obs_ok = True
            except HarnessError as exc:
                if "walk_fall" in str(exc):
                    raise
                obs_ok = False
                p, v, q, j, tsk = (0, 0, 0), (0, 0, 0), (0, 0, 0, 1), {}, ""
                if obs_dead_since is None:
                    obs_dead_since = time.monotonic()
                elif time.monotonic() - obs_dead_since > OBS_LOST_ABORT_S:
                    shutdown_reason = "obs_lost"
                    health = "obs_lost"
                    emit_audio("control_fault", reason="obs_lost")
                    break
            # 2b) 操縦範囲の実観測 — 開始位置からの水平逸脱が上限を超えたら
            #     移動を止めラウンド終了（seed15反例: 誘導不收束のまま14m
            #     漂流して転倒した。範囲外への歩き続けを安全側で止める）
            if (
                obs_ok
                and arena_anchor is not None
                and not arena_exited
                and math.hypot(
                    p[0] - arena_anchor[0], p[1] - arena_anchor[1]
                )
                > arena_r
            ):
                arena_exited = True
                move_frame = None
                if motion_ex is not None and not motion_ex.done:
                    motion_ex.abort("arena_exit")
                current_type = STOP
                round_end_reason = "arena_exit"
                # 安全境界によるラウンド終了 — ゲーム規則の失敗ではなく
                # 中断として扱い、失敗音で安全イベントを装飾しない
                emit_audio("round_aborted", reason="arena_exit")
                events.append(
                    {
                        "event": "round_end",
                        "reason": "arena_exit",
                        "excursion_m": round(
                            math.hypot(
                                p[0] - arena_anchor[0],
                                p[1] - arena_anchor[1],
                            ),
                            3,
                        ),
                        "arena_r_m": arena_r,
                    }
                )
                if strike is not None:
                    # 打撃中は管理中断 → recover完了後に終了処理へ
                    strike.abort("end")
                else:
                    break
            # 3) 打撃FSM駆動（非ブロッキング — tick毎にframeを得る）
            if strike is not None:
                # 観測欠測中は軌跡を記録せず、task遷移も検出されないため
                # FSMは自然に進まない。欠測がOBS_LOST_ABORT_Sを超えれば
                # 上位のobs_lost breakが制御全体を止める。
                frame, done, done_info = strike.tick(
                    issue, frames, (p, v, q, j, tsk), obs_ok=obs_ok
                )
                # dance遷移の実観測で消費確定（1回のみ計上 — 実消費正本）
                if strike.consumed and not getattr(strike, "counted", False):
                    strike.counted = True
                    strikes_started += 1
                    # 棒を実際に振り始めた実観測 — requestedではなくstarted
                    emit_audio("strike_started", strike_id=strike.strike_id)
                if done:
                    events.append(done_info)
                    last_strike_done = strike.strike_id
                    if strike.consumed:
                        strikes_done += 1
                        last_strike = done_info.get("judge")
                        if not (last_strike or {}).get("hit"):
                            # 実振り済みで命中しなかった空振り（判定正本より）
                            emit_audio(
                                "strike_missed", strike_id=strike.strike_id
                            )
                    else:
                        last_strike_attempt = {
                            "strike_id": strike.strike_id,
                            "status": done_info.get("status"),
                            "aborted": done_info.get("aborted"),
                            "interrupted": done_info.get("interrupted"),
                            "stillness_vel": done_info.get("stillness_vel"),
                        }
                        if done_info.get("status") == "unverified":
                            strike_unverified.append(strike.strike_id)
                        else:
                            strike_expired.append(strike.strike_id)
                    pending_end = strike.pending_end or end_received
                    strike = None
                    if pending_end:
                        break
                    judge = last_strike
                    if end_on_hit and judge and judge.get("hit"):
                        round_end_reason = "hit"
                        events.append(
                            {
                                "event": "round_end",
                                "reason": "hit",
                                "judge": judge,
                            }
                        )
                        emit_audio("round_succeeded", outcome=judge)
                        break
                    if (
                        max_swings is not None
                        and strikes_started >= int(max_swings)
                    ):
                        round_end_reason = "out_of_swings"
                        events.append(
                            {
                                "event": "round_end",
                                "reason": "out_of_swings",
                                "strikes": strikes_started,
                            }
                        )
                        emit_audio(
                            "round_failed",
                            reason="out_of_swings",
                            strikes=strikes_started,
                        )
                        break
            # 打撃中でなければmoveを適用
            if strike is None:
                if motion_ex is not None:
                    _drain_motion_events()
                    if motion_ex.done:
                        # 完了/失敗/中断 — 結果を残して解除
                        last_cmd_state = _cmd_state_dict()
                        motion_ex = None
                        move_frame = None
                        if current_type == MOVE:
                            current_type = "idle"
                        frame = frame_idle
                    elif not obs_ok:
                        # 観測断中はstickを出さずexecutorの時計も進めない
                        frame = frame_idle
                    else:
                        # 継続jogの生存確認 — 操作者heartbeatの欠落は
                        # 通信断・放置とみなし減速停止へ（猶予期間あり）。
                        if motion_ex.jogging and jog_since_mono is not None:
                            hb_fresh = False
                            try:
                                hb = json.loads(HEARTBEAT_PATH.read_text())
                                hb_t = float(hb.get("t_wall") or 0.0)
                                hb_fresh = (
                                    hb.get("lease_id") == lease_id
                                    and 0.0 <= now_wall - hb_t <= HB_STALE_S
                                )
                            except (
                                FileNotFoundError,
                                ValueError,
                                TypeError,
                                json.JSONDecodeError,
                            ):
                                hb_fresh = False
                            if (
                                not hb_fresh
                                and t0 - jog_since_mono > HB_GRACE_S
                            ):
                                motion_ex.request_stop("heartbeat_lost")
                                events.append(
                                    {"event": "heartbeat_lost",
                                     "seq": motion_seq}
                                )
                        stick = motion_ex.tick(p, q, v)
                        frame = (
                            frame_idle
                            if stick is None
                            else gateway.prepare_move(*stick)
                        )
                elif move_frame is not None and now_wall < move_until_wall:
                    frame = move_frame
                else:
                    if move_frame is not None:
                        move_frame = None
                        current_type = "idle"
                    frame = frame_idle
            try:
                issue(frame, f"ctrl_{current_type}")
            except AuthorizationRefused:
                shutdown_reason = "manifest_expired"
                health = "manifest_expired"
                emit_audio("control_fault", reason="manifest_expired")
                break
            write_state(t0)
            due = t0 + 1 / HZ
            while time.monotonic() < due:
                time.sleep(0.001)

        # ループをbreakした終端経路（END/timeout/fault）では最後の
        # write_stateを踏まないため、round_aborted等の終端audio eventが
        # events_tailに残らない — kioskがBGM停止契機を取りこぼす。
        # 終了処理の前に必ずflushする。
        try:
            write_state(time.monotonic(), force=True)
        except Exception:
            pass

        # ---- 終了処理: idle → pd_stand → 立位保持を実観測で確認 ----
        end_hold_verified = False
        if strike is not None:
            # 打撃中に終了した場合もrecover相当としてwalkを送る
            for _ in range(int(RECOVER_HOLD_S * HZ)):
                try:
                    issue(frame_walk, "final_walk")
                except AuthorizationRefused:
                    break
                time.sleep(1 / HZ)
            strike = None
        for _ in range(10):
            try:
                issue(frame_idle, "final_idle")
            except AuthorizationRefused:
                break
            time.sleep(1 / HZ)
        clean_end = (
            end_received or round_end_reason is not None
        ) and health == "ok"
        if clean_end:
            try:
                frame_stand = gateway.prepare("combo_pd_stand")
                for _ in range(10):
                    issue(frame_stand, "combo_pd_stand")
                    time.sleep(1 / HZ)
            except AuthorizationRefused:
                health = "manifest_expired"
                clean_end = False
            fsm.transition(Phase.DECELERATING)
            fsm.transition(Phase.SETTLING)
            fsm.transition(Phase.TAIL)
            # 立位保持の継続観測（送信だけでPASSにしない）。
            # 実基準は「転ばず・立ったまま・歩き去らない」— 高さ・姿勢・
            # アンカー逸脱・累積軌跡の複合条件を全サンプルで検査し、
            # sim時刻の継続観測を要求する。保持中の全観測はfall
            # latchへも流して終了判定と整合させる。
            if clean_end:
                end_hold_verified = _verify_end_hold(obs, latch, events)
                events.append(
                    {"event": "end_hold", "verified": end_hold_verified}
                )
        fsm.transition(Phase.DONE)

        result["control"] = {
            "applied_seq": applied_seq,
            "strikes_started": strikes_started,
            "strikes_done": strikes_done,
            "rejects": rejects[-20:],
            "end_received": end_received,
            "round_end": round_end_reason,
            "shutdown_reason": shutdown_reason,
            "health": health,
            "end_hold_verified": end_hold_verified,
            "fallen": latch.fallen,
        }
        # verdict: 正常終了＋立位保持の実観測でのみPASS。
        # 確認不能はUNKNOWN、異常終了はFAIL系。
        if latch.fallen:
            result["verdict"] = "FAIL_FALLEN"
        elif clean_end and end_hold_verified:
            result["verdict"] = "PASS"
        elif clean_end:
            result["verdict"] = "UNKNOWN_END_HOLD"
        else:
            result["verdict"] = "FAIL_" + (shutdown_reason or "no_clean_end").upper()
    except (HarnessError, AuthorizationRefused) as exc:
        result["verdict"] = "FAIL_" + getattr(exc, "reason", "error").upper()
        result["reasons"] = [str(exc)]
    except Exception as exc:
        result["verdict"] = "FAIL_INTERNAL"
        result["reasons"] = [f"{type(exc).__name__}: {exc}"]
    finally:
        if gateway is not None:
            try:
                gateway.issue(
                    gateway.prepare("idle"),
                    now_monotonic=time.monotonic(),
                    command_name="final_idle",
                )
                gateway.close()
            except Exception:
                pass
        if obs is not None:
            try:
                obs.stop()
            except Exception:
                pass
        Path(result_path).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return 0 if result.get("verdict") == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
