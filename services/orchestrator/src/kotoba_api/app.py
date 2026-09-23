"""ことばでロボコン API — 単一origin配信（UI静的 + REST）。

起動: uvicorn kotoba_api.app:app --host 0.0.0.0 --port 8700
前提: 展示専用Ollama(11435)、SDK sim（SimControlが管理）
"""

import asyncio
import json
import math
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from kotoba_api.service import CAPABILITIES, ProductSession, SpawnRejected
from kotoba_api.world import COURSE
from kotoba_orchestrator.freshness import live_freshness

KIOSK_DIST = Path(
    os.environ.get(
        "KOTOBA_KIOSK_DIST",
        Path(__file__).resolve().parents[4] / "apps" / "kiosk" / "dist",
    )
)
RUNNER = (
    Path(__file__).resolve().parents[3] / "services" / "runtime" / "kotoba_runner.py"
)
KOTOBA_HOME = os.environ.get("KOTOBA_HOME", "/home/terisuke/kotoba")


def _resolve_harness_src() -> Path:
    # Thor配備は app/ 直下フラットと app/services/... nested の二重構造で、
    # `cd app && python -m uvicorn` ではフラット側が sys.path[0] で勝つ。
    # parents[3] 固定だとフラット側からは /home/cloudia/harness/src を指して
    # docker -v が空dirを自動生成し runner が即死する（2026-09-21 障害）。
    # 明示 env > 実在する候補の順で解決する。
    env = os.environ.get("KOTOBA_HARNESS_SRC")
    candidates = [Path(env)] if env else []
    here = Path(__file__).resolve()
    candidates.append(here.parents[3] / "harness" / "src")
    candidates.append(Path(KOTOBA_HOME) / "app" / "services" / "harness" / "src")
    for c in candidates:
        if (c / "kotoba_harness").is_dir():
            return c
    return candidates[0]


HARNESS_SRC = _resolve_harness_src()
RUNTIME_DIR = Path(os.environ.get("KOTOBA_RUNTIME_DIR", f"{KOTOBA_HOME}/runtime"))
SIM_IMAGE = os.environ.get("KOTOBA_SIM_IMAGE", "engineai_robotics_env:humble-desktop-local")
RUNNER_IMAGE = os.environ.get("KOTOBA_RUNNER_IMAGE", SIM_IMAGE)
SIM_SDK = os.environ.get("KOTOBA_SDK_DIR", f"{KOTOBA_HOME}/sdk")
SIM_XVFB_DISPLAY = os.environ.get("KOTOBA_XVFB_DISPLAY", ":99")
ROS_DISTRO = os.environ.get("KOTOBA_ROS_DISTRO", "humble")
SIM_NETWORK = os.environ.get("KOTOBA_SIM_NETWORK", "host")
# 追加bind mount（hostごとの third_party/hardware/mujoco 等。空白区切りの -v 列）
KOTOBA_EXTRA_MOUNTS = os.environ.get("KOTOBA_EXTRA_MOUNTS", "").strip()
# ctl/mujoco の起動コマンドはホスト環境ごとに固定する（既定はEVOのrun.sh系）。
SIM_CTL_CMD = os.environ.get(
    "KOTOBA_CTL_CMD",
    f"source /opt/ros/{ROS_DISTRO}/setup.bash; export ROS_LOCALHOST_ONLY=1; ./run.sh pm01_edu",
)
SIM_MUJOCO_CMD = os.environ.get(
    "KOTOBA_MUJOCO_CMD",
    f"source /opt/ros/{ROS_DISTRO}/setup.bash; ./scripts/run_mujoco.sh pm01_edu",
)

app = FastAPI(title="kotoba-robocon", version="0.1.0-trial")

# オフラインモード: SDK非接続でもUIの起動・確認・拒否・resetを試験できる。
# LIVE_EXECUTED=False の間は物理ランナーを起動しない（合成状態のみ）。
OFFLINE_MODE = os.environ.get("KOTOBA_OFFLINE", "1") != "0"
LIVE_EXECUTED = not OFFLINE_MODE

# 運営者の自己検証モード。参加者LIVE受入（LIVE_ACCEPTED）とは別のゲート。
# 実SDKを動かす検証を「参加者開放」と混同しないため明示的なenvが必要。
SELFTEST_MODE = os.environ.get("KOTOBA_SELFTEST", "0") == "1"

# LIVE未受入の間、run開始は操作者トークンを要求する（SELFTESTの全体開放を防ぐ）。
# 値はenvのみに置き、URL/UI/repoへ埋め込まない。未設定ならSELFTESTでも拒否する。
OPERATOR_TOKEN = os.environ.get("KOTOBA_OPERATOR_TOKEN", "")

# LIVE操作の受入フラグ。G1物理の正常系10連続が受入されるまでサーバー側で閉じる。
LIVE_ACCEPTED = False


class SimControl:
    """自分たちのnamed containerだけを管理する（業務containerには触れない）。"""

    MUJOCO = "kotoba-mujoco"
    CTL = "kotoba-ctl"
    NETNS = "kotoba-net0"  # LCM専用の共有netns anchor（host processから隔離）

    def _sh(self, cmd: list) -> tuple[int, str]:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return r.returncode, (r.stdout + r.stderr)[-400:]

    def running(self, name: str) -> bool:
        rc, out = self._sh(["docker", "ps", "--format", "{{.Names}}"])
        return name in out.split()

    def pause(self) -> bool:
        ok = True
        for name in (self.MUJOCO, self.CTL):
            if self.running(name):
                rc, _ = self._sh(["docker", "pause", name])
                ok = ok and rc == 0
        return ok

    def resume(self) -> bool:
        ok = True
        for name in (self.MUJOCO, self.CTL):
            rc, _ = self._sh(["docker", "unpause", name])
            ok = ok and rc == 0
        return ok

    def stop(self) -> None:
        # live observer も anchor のnetnsを共有する — anchor除去前に畳まないと
        # netnsが消滅して受信不能のまま残る（実測: reset後に観測凍結）
        for name in (self.MUJOCO, self.CTL, "kotoba-stand", "kotoba-live"):
            self._sh(["docker", "rm", "-f", name])
        # netns anchorは使用コンテナ停止後に畳む（生存中は維持）
        if SIM_NETWORK.startswith("container:"):
            self._sh(["docker", "rm", "-f", self.NETNS])

    def _ensure_netns(self) -> bool:
        """LCM専用の共有netns anchorを確保（専用bridge/namespaceの代替）。

        ttl=0のLCMはdocker bridgeを越えない（実測済み）ため、
        全kotobaコンテナが anchor のnetnsを共有する構成にする。
        """
        if not SIM_NETWORK.startswith("container:"):
            return True
        if self.running(self.NETNS):
            return True
        rc, _ = self._sh(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                self.NETNS,
                "--entrypoint",
                "sleep",
                RUNNER_IMAGE,
                "infinity",
            ]
        )
        return rc == 0

    def _xvfb_running(self) -> bool:
        rc, out = self._sh(["pgrep", "-c", "Xvfb"])
        return out.strip() not in ("0", "")

    def restart(self) -> tuple[bool, str | None]:
        """起動ライフサイクル v3（ctl先起動+PD先行+物理後起動）:
        1. Xvfb起動
        2. ctl起動 → passive入場待ち（mujoco無しでもFSMは動く）
        3. pd_stand burst送信（stand manager container）→ FSM pd_stand遷移
        4. mujoco起動 → 物理開始時点で既にPD制御が活性
        順序により「無制御落下→捕獲」のレースを構造的に排除する。"""
        import shlex

        runtime_dir = RUNTIME_DIR
        self.stop()
        if not self._ensure_netns():
            return False, None
        for marker in ("boot-ready.json", "passive-detected", "boot-lost.json"):
            (runtime_dir / marker).unlink(missing_ok=True)
        if not self._xvfb_running():
            Path(KOTOBA_HOME, "logs").mkdir(parents=True, exist_ok=True)
            self._sh(
                [
                    "bash",
                    "-c",
                    f"Xvfb {SIM_XVFB_DISPLAY} -screen 0 1280x720x24 -ac +extension GLX "
                    f"+render -noreset >{KOTOBA_HOME}/logs/xvfb.log 2>&1 & "
                    f"echo $! > /tmp/kotoba-xvfb.pid; sleep 0.5",
                ]
            )
        mounts = (
            "-v /tmp/.X11-unix:/tmp/.X11-unix "
            f"-v {SIM_SDK}:{SIM_SDK} "
            f"-v {KOTOBA_HOME}:/kotoba "
            f"-v {HARNESS_SRC}:/kotoba/harness_src "
            + (KOTOBA_EXTRA_MOUNTS + " " if KOTOBA_EXTRA_MOUNTS else "")
        )
        sim_base = (
            f"docker run -d --rm --init --name {{name}} --user $(id -u):$(id -g) "
            f"--network {SIM_NETWORK} -e DISPLAY={SIM_XVFB_DISPLAY} -e ROS_LOCALHOST_ONLY=1 "
            f"{mounts}-w {SIM_SDK} --entrypoint bash {SIM_IMAGE} -lc {{cmd}}"
        )
        # 1) ctl起動（mujocoより先）
        rc1, _ = self._sh(
            [
                "bash",
                "-c",
                sim_base.format(
                    name=self.CTL,
                    cmd=shlex.quote(SIM_CTL_CMD),
                ),
            ]
        )
        # 2) passive入場を待つ（docker logs -f）
        watcher = (
            "docker logs -f kotoba-ctl 2>&1 | grep --line-buffered -m1 -q "
            "'Entered motion \\[ passive \\]' && "
            f"touch {RUNTIME_DIR}/passive-detected"
        )
        self._sh(
            [
                "bash",
                "-c",
                f"setsid bash -c {shlex.quote(watcher)} < /dev/null > /dev/null 2>&1 &",
            ]
        )
        # passive入場待ち（最大30秒）
        passive_deadline = time.time() + 30
        while time.time() < passive_deadline:
            if (runtime_dir / "passive-detected").exists():
                break
            time.sleep(0.5)
        else:
            return False, None
        # 3) mujoco起動（物理開始 — ロボットが崩落開始する）
        rc2, _ = self._sh(
            [
                "bash",
                "-c",
                sim_base.format(
                    name=self.MUJOCO,
                    cmd=shlex.quote(SIM_MUJOCO_CMD),
                ),
            ]
        )
        # 4) stand manager起動（即時pd_stand burstで崩落を捕獲 — 8/28レシピ）
        stand_cmd = (
            "docker run -d --rm --init --name kotoba-stand --user $(id -u):$(id -g) "
            f"--network {SIM_NETWORK} -e KOTOBA_PUBLISH=1 -e DISPLAY="
            + SIM_XVFB_DISPLAY
            + " -e ROS_LOCALHOST_ONLY=1 "
            + mounts
            + "-w /kotoba --entrypoint python3 "
            + RUNNER_IMAGE
            + " -u /kotoba/runner/kotoba_stand_manager.py"
        )
        rc0, _ = self._sh(["bash", "-c", stand_cmd])
        # 5) 常駐観測（read-only）を新netnsで再起動 — live.json→/api/obs/live→UI
        live_cmd = (
            "docker run -d --rm --init --name kotoba-live --user $(id -u):$(id -g) "
            f"--network {SIM_NETWORK} "
            f"-v {KOTOBA_HOME}:/kotoba "
            f"-v {HARNESS_SRC}:/kotoba/harness_src "
            "-w /kotoba --entrypoint python3 "
            + RUNNER_IMAGE
            + " -u /kotoba/runner/kotoba_live_observer.py"
        )
        self._sh(["bash", "-c", live_cmd])
        # ready待ち（stand managerが直立を検証するまで）
        ready_file = runtime_dir / "boot-ready.json"
        deadline = time.time() + 30
        nonce = None
        while time.time() < deadline:
            if ready_file.exists():
                try:
                    data = json.loads(ready_file.read_text())
                    if data.get("standing") and data.get("boot_nonce"):
                        nonce = str(data["boot_nonce"])
                        break
                except json.JSONDecodeError:
                    pass
            time.sleep(0.5)
        ok = rc1 == 0 and rc2 == 0 and nonce is not None
        return ok, nonce

    def restart_with_retry(self, attempts: int = 6) -> tuple[bool, str | None]:
        """捕獲は boot 毎に確率的（崩落との競争）。失敗したら boot をやり直す。"""
        nonce = None
        ok = False
        for attempt in range(1, attempts + 1):
            ok, nonce = self.restart()
            print(f"[stand] attempt {attempt}: ok={ok} nonce={nonce}", flush=True)
            if ok and nonce:
                return ok, nonce
        return ok, nonce


sim = SimControl()


def _boot_nonce_provider():
    return _boot_ready()[1]


session = ProductSession(
    sim_ctl=sim,
    boot_nonce_fn=_boot_nonce_provider,
    live_accepted=LIVE_ACCEPTED,
    selftest=SELFTEST_MODE,
)
_pending: dict = {}  # plan_id -> {"distance_m": float, "session_id", "anchor", "heading_yaw", "target_id"}
# run_id -> 実行開始時に凍結した表示用anchor/heading（runnerのtarget計算と同じ式）。
# 表示専用。採点・制御には使わない（controller/scorer targetはrunner側の計算）。
_run_display: dict = {}


# ---- models --------------------------------------------------------------
class IntentIn(BaseModel):
    session_id: str
    text: str


class ApproveIn(BaseModel):
    session_id: str
    plan_id: str


class RunIn(BaseModel):
    session_id: str
    plan_id: str
    approval_id: str


class SessionIn(BaseModel):
    session_id: str


# ---- REST ----------------------------------------------------------------
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """全応答に X-Request-Id — 409等の失敗をapi.logの行と対応付ける
    相関ID（認証資格ではない診断用）。"""
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    response = await call_next(request)
    response.headers["X-Request-Id"] = rid
    return response


@app.post("/api/sessions")
def create_session():
    sid = session.create_session()
    return {"session_id": sid}


@app.get("/api/state/{session_id}")
def get_state(session_id: str):
    if session_id not in session.rounds:
        raise HTTPException(404, "unknown_session")
    st = session.state(session_id)
    # RUNNING中の動作プログラム進捗 — steps runnerが run_state.json を書く。
    # 認識・送信・適用・実動作の区別表示用（読み取り専用・制御には使わない）。
    r = session.rounds[session_id]
    if r.phase == "RUNNING" and r.run_id:
        f = RUNTIME_DIR / r.run_id / "run_state.json"
        try:
            m = json.loads(f.read_text())
            # UIは dist_m/target_m 等をトップレベルで読む — progressを平坦化
            if isinstance(m.get("progress"), dict):
                m.update(m["progress"])
            st["motion"] = m
        except Exception:
            st["motion"] = None
    return st


@app.post("/api/intents")
def post_intent(body: IntentIn):
    if body.session_id not in session.rounds:
        raise HTTPException(404, "unknown_session")
    if len(body.text) > 300:
        raise HTTPException(422, "text_too_long")
    result = session.interpret(body.session_id, body.text)
    if result.get("decision") == "execute":
        # 表示用anchor: 解釈成立時点のlive姿勢を凍結する（markerはrunnerと同じ
        # anchor+heading*距離の式で置く。実行中のmarker滑走を防ぐ）。
        anchor, yaw = _live_anchor()
        target_id = None
        li = session.rounds[body.session_id].last_intent
        if li and li.get("target_ids"):
            target_id = li["target_ids"][0]
        plan_obj = session.rounds[body.session_id].plan_obj
        _pending[result["plan_id"]] = {
            "distance_m": result["review"]["distance_m"],
            "session_id": body.session_id,
            "anchor": anchor,
            "heading_yaw": yaw,
            "target_id": target_id,
            # motion_program: 実行するstep列をmanifestへ引き渡す。
            # marker経路では空（runnerは従来どおり単一distance）。
            "steps": (
                [s.model_dump() for s in plan_obj.steps]
                if plan_obj is not None and plan_obj.kind == "motion_program"
                else None
            ),
        }
    return result


@app.post("/api/plans/{plan_id}/approve")
def approve(plan_id: str, body: ApproveIn):
    result = session.approve(body.session_id, plan_id)
    if not result.get("approved"):
        raise HTTPException(409, result.get("reason", "not_approved"))
    return result


def _boot_ready() -> tuple[bool, str | None]:
    f = RUNTIME_DIR / "boot-ready.json"
    lost = RUNTIME_DIR / "boot-lost.json"
    if lost.exists():
        return False, None
    if not f.exists():
        return False, None
    try:
        data = json.loads(f.read_text())
        if data.get("standing") and data.get("boot_nonce"):
            if time.time() - data.get("verified_at", 0) < 3600:
                return True, str(data["boot_nonce"])
    except json.JSONDecodeError:
        pass
    return False, None


@app.post("/api/runs")
def start_run(body: RunIn, request: Request):
    if OFFLINE_MODE:
        raise HTTPException(409, "live_not_accepted_offline_mode")
    if not LIVE_ACCEPTED:
        # 自己試験経路は操作者限定: 同じ実行サービスを通すがアクセス権を区別する
        if not SELFTEST_MODE:
            raise HTTPException(409, "live_not_accepted")
        if not OPERATOR_TOKEN or request.headers.get("x-kotoba-operator") != OPERATOR_TOKEN:
            raise HTTPException(403, "operator_required")
    pending = _pending.get(body.plan_id)
    if pending is None or pending["session_id"] != body.session_id:
        raise HTTPException(409, "stale_plan")
    ready, nonce = _boot_ready()
    if not ready:
        raise HTTPException(409, "sim_not_ready")
    started = session.start_run(body.session_id, body.plan_id, body.approval_id, nonce)
    if not started.get("started"):
        raise HTTPException(409, started.get("reason", "not_started"))
    run_id = started["run_id"]
    # 実行中のmarker表示用に plan の anchor/heading を凍結して持ち越す
    _run_display[run_id] = {
        "anchor": pending.get("anchor"),
        "heading_yaw": pending.get("heading_yaw"),
        "distance_m": pending.get("distance_m"),
        "target_id": pending.get("target_id"),
    }
    threading.Thread(
        target=_execute_run,
        args=(
            body.session_id,
            run_id,
            pending,
            started["grant"].get("sim_boot_id"),
        ),
        daemon=True,
    ).start()
    return {"run_id": run_id}


@app.post("/api/run/pause")
def pause_run(body: SessionIn):
    if OFFLINE_MODE:
        session.release_run_lock(body.session_id)
        return {"paused": True, "note": "offline mode: run aborted"}
    ok = sim.pause()
    _kill_runners()
    # RUNNING世代のみ失効（判定と失効・受付閉鎖・lock解放を同一run_mu区間で —
    # 直前のfinish競合でRESULTをFAULT上書きしない）。非実行中のpauseは
    # 従来契約（sim停止のみ・plan/round保持）を維持する。
    session.abort_run(
        body.session_id,
        "一時停止しました（実行を中断）。resetして最初からやり直せます。",
    )
    return {"paused": ok, "note": "run aborted"}


@app.post("/api/kiosk/exit-ready")
def kiosk_exit_ready(request: Request):
    """全画面を閉じる前に、安全状態と短期受付停止を不可分に確保する。"""
    if not OPERATOR_TOKEN or request.headers.get("x-kotoba-operator") != OPERATOR_TOKEN:
        raise HTTPException(403, "operator_required")
    if not session.reserve_kiosk_exit():
        raise HTTPException(409, "run_or_reset_in_progress")
    return {"safe": True}


@app.post("/api/run/resume")
def resume_run(body: SessionIn):
    if OFFLINE_MODE:
        return {"resumed": False, "note": "offline mode"}
    return session.resume(body.session_id)


@app.post("/api/round/reset")
def reset_round(body: SessionIn):
    if OFFLINE_MODE:
        out = session.reset(body.session_id, restart_sim=False)
        if out.get("reason") in {"other_session_running", "kiosk_exiting"}:
            raise HTTPException(409, out["reason"])
        return {"reset": True, "note": "offline mode: sim restart skipped"}
    # 世代失効→runner終了→sim再起動→新受付の順序を service.reset 内で保証する
    out = session.reset(body.session_id, restart_sim=True, pre_restart=_kill_runners)
    if out.get("reason") in {"other_session_running", "kiosk_exiting"}:
        raise HTTPException(409, out["reason"])
    if out.get("reset"):
        # 表示用・受理待ちの残存contextは新roundでは無効（旧plan/旧run targetを
        # 引きずらない — retry後に旧markerが残らない）
        _pending.clear()
        _run_display.clear()
    return out


@app.get("/api/obs/live")
def obs_live():
    """ライブ観測のサマリ（UIの2D map用）。runner稼働中はその値、無い場合は診断値。"""
    return _live_summary()


@app.get("/api/health")
def health():
    if OFFLINE_MODE:
        # オフライン: 実Docker/Ollamaに接続しない。合成状態を返すのみ。
        return {
            "mode": "offline_synthetic",
            "live_executed": False,
            "ollama": None,
            "sim_mujoco": None,
            "sim_ctl": None,
            "stand_ready": None,
            "run_lock": False,
        }
    rc = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True
    )
    names = rc.stdout.split()
    ready, _ = _boot_ready()
    return {
        "mode": "live",
        "live_executed": True,
        "selftest": SELFTEST_MODE,
        "ollama": _ollama_ok(),
        "sim_mujoco": sim.MUJOCO in names,
        "sim_ctl": sim.CTL in names,
        "stand_ready": ready,
        "run_lock": session.run_lock,
    }


def _kill_runners() -> None:
    """実行中のrunnerコンテナを終了する（名前prefix限定、pkill不使用）。"""
    subprocess.run(
        [
            "bash",
            "-c",
            "docker rm -f $(docker ps -aq --filter name=kotoba-runner-) 2>/dev/null",
        ],
        capture_output=True,
        timeout=30,
    )


# ---- run execution ---------------------------------------------------------
def _current_boot_nonce() -> str:
    """runnerを短く叩いてboot nonceだけ取得するのは重いため、
    前回runで確認済みのnonceが無ければ起動直後の観測を要求する方式をとる。
    基本版ではnonce未取得の初回は 'boot-pending' を使い、runner側で実nonce確認する。
    """
    return getattr(session, "boot_nonce", None) or "boot-pending"


def _execute_run(
    session_id: str, run_id: str, pending: dict, expected_boot: str | None = None
) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RUNTIME_DIR / run_id
    run_dir.mkdir(exist_ok=True)
    manifest = {
        "run_id": run_id,
        "purpose": "evaluation",
        "arming": "1",
        "expires_in_s": 240,
        # runnerは承認時に検証済みのbootへ固定する（launch時点の現bootではなく —
        # 承認後のboot変更を実行runnerへ持ち込まない R3-B）
        "boot_expect": expected_boot or _current_boot_nonce(),
        "tolerance_m": 0.15,
        "stop_trigger_m": 0.35,
        # pace→profile選択の能力台帳（受理側と同一の台帳を実行側へ渡す — R5）
        "capabilities": dict(CAPABILITIES),
    }
    motion_steps = pending.get("steps")
    if motion_steps:
        # motion_program: 順序付きstep列をmanifest経由でrunnerへ渡す。
        # 数値はplan検証済みの実値のみ（承認hashに結合済み）。
        manifest["motion_steps"] = motion_steps
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    result_path = run_dir / "result.json"
    harness_src = str(HARNESS_SRC)
    if motion_steps:
        runner_args = (
            "/kotoba/runner/kotoba_steps_runner.py /kotoba/run/result.json "
            "/kotoba/run/manifest.json"
        )
    else:
        runner_args = (
            "/kotoba/runner/kotoba_runner.py /kotoba/run/result.json "
            f"{pending['distance_m']} /kotoba/run/manifest.json"
        )
    cmd = (
        # -d detached: 起動直後に世代再検査してから wait/rm する（C13）。
        # --rm は付けない（logs取得とpost-checkが終わるまで残骸を残す必要がある）。
        f"docker run -d --init --name kotoba-runner-{run_id[:8]} "
        f"--user $(id -u):$(id -g) --network {SIM_NETWORK} "
        f"-e KOTOBA_PUBLISH=1 "
        f"-v {KOTOBA_HOME}:/kotoba "
        f"-v {harness_src}:/kotoba/harness_src "
        f"-v {run_dir}:/kotoba/run "
        f"-w /kotoba --entrypoint python3 {RUNNER_IMAGE} "
        f"{runner_args}"
    )
    # 起動直前に世代を再検査: start_run受理後にreset/pauseが世代を失効させた
    # 遅延スレッドはコンテナを起動しない（新bootへの誤爆防止 — R2）
    if not session.owns_run(session_id, run_id):
        return
    # runner.py とコース実行に必要なsrcを実機側の /kotoba/runner へ置く（配備時に配置済み）
    cname = f"kotoba-runner-{run_id[:8]}"
    result = {}
    try:
        # detached起動 → 直後に世代を再検査する（C13: 世代検査〜コンテナ生成の窓で
        # pause/resetが挟まっても、kill対象に載らない遅延launchを残さない）。
        launch = subprocess.run(
            ["bash", "-c", cmd], capture_output=True, text=True, timeout=60
        )
        if launch.returncode != 0:
            result = {
                "verdict": "RUNNER_ERROR",
                "reasons": [(launch.stderr or "docker run failed")[-300:]],
            }
        elif not session.owns_run(session_id, run_id):
            # launch中に世代失効: 自分が起こしたコンテナを直ちに終了する
            subprocess.run(
                ["docker", "rm", "-f", cname], capture_output=True, timeout=30
            )
            result = {
                "verdict": "ABORTED_AT_LAUNCH",
                "reasons": ["generation invalidated during runner launch"],
            }
        else:
            try:
                subprocess.run(
                    ["docker", "wait", cname],
                    capture_output=True, text=True, timeout=300,
                )
            except Exception as exc:
                result = {
                    "verdict": "RUNNER_ERROR",
                    "reasons": [f"wait:{str(exc)[-200:]}"],
                }
            if not result:
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                else:
                    logs = subprocess.run(
                        ["docker", "logs", "--tail", "40", cname],
                        capture_output=True, text=True, timeout=15,
                    )
                    tail = ((logs.stdout or "") + (logs.stderr or ""))[-300:]
                    result = {
                        "verdict": "RUNNER_ERROR",
                        "reasons": [tail or "runner produced no result.json"],
                    }
            subprocess.run(
                ["docker", "rm", "-f", cname], capture_output=True, timeout=30
            )
    except Exception as exc:
        # timeout・起動例外でも所有runだけを終端化する（finallyでfinishする）
        result = {"verdict": "RUNNER_ERROR", "reasons": [str(exc)[-300:]]}
    finally:
        result["run_id"] = run_id
        # 世代一致なら現在roundへ結果・lock解除・boot反映（全副作用が世代検査の
        # 内側）。reset/pause済みなら履歴へ隔離されbootも旧値へ戻らない（R2-B）。
        session.finish_run(session_id, run_id, result)


def _ollama_ok() -> bool:
    import urllib.request

    host = __import__("os").environ.get("OLLAMA_HOST", "http://127.0.0.1:11435")
    if not host.startswith("http"):
        host = f"http://{host}"
    try:
        with urllib.request.urlopen(f"{host}/api/version", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def _live_summary() -> dict:
    try:
        return _read_latest_state_file()
    except Exception:
        return {"fresh": False}


def _live_anchor() -> tuple:
    """live観測のpos/quat → (anchor_xy, heading_yaw)。

    runnerのtarget計算と同じ式（quatのyaw成分→機体+Xのworld XY射影）。
    表示専用 — 制御・採点には使わない。
    """
    try:
        data = json.loads((RUNTIME_DIR / "live.json").read_text())
        pos = data["pos"]
        w, x, y, z = data["quat_wxyz"]
        yaw = math.atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))
        return [round(pos[0], 4), round(pos[1], 4)], yaw
    except Exception:
        return None, None


# ---- 3D表示用の目標marker（表示専用 — 制御/採点とは別系統） -------------------
@app.get("/api/display/targets")
def display_targets():
    """3D rendererが読む目標marker状態。

    - idle: 現在のlive姿勢をanchorに両markerを表示（考え中から見える）
    - REVIEW: 解釈成立時のanchor + 選ばれたtargetをhighlight
    - RUNNING: runnerが実際に追うtarget（target.json）をhighlight
    - 計算式はrunnerと同じ（anchor + heading*距離、yawはquat由来）
    """
    live = _live_summary()
    fresh = bool(live.get("fresh"))
    # アクティブなround（RUNNING > RESULT > REVIEW）を探す
    active = None
    for sid, r in session.rounds.items():
        if r.phase in ("RUNNING", "RESULT", "REVIEW"):
            if active is None or r.phase == "RUNNING":
                active = (sid, r)
    anchor, yaw = _live_anchor()
    highlight_xy = None
    highlight_id = None
    if active is not None:
        sid, r = active
        if r.phase in ("RUNNING", "RESULT") and r.run_id:
            ctx = _run_display.get(r.run_id) or {}
            anchor = ctx.get("anchor") or anchor
            yaw = ctx.get("heading_yaw") if ctx.get("heading_yaw") is not None else yaw
            # runnerが書いた実target（あれば表示targetをそれへ合わせる）
            tgt = _active_target()
            if tgt is not None:
                highlight_xy = tgt
            else:
                # motion_program はmarker目標を持たない — 推測highlightは出さない
                d = ctx.get("distance_m") if not ctx.get("steps") else None
                if anchor and yaw is not None and d:
                    highlight_xy = [
                        round(anchor[0] + math.cos(yaw) * d, 4),
                        round(anchor[1] + math.sin(yaw) * d, 4),
                    ]
            highlight_id = ctx.get("target_id")
        elif r.phase == "REVIEW" and r.pending_plan_id:
            ctx = _pending.get(r.pending_plan_id) or {}
            anchor = ctx.get("anchor") or anchor
            yaw = ctx.get("heading_yaw") if ctx.get("heading_yaw") is not None else yaw
            highlight_id = ctx.get("target_id")
            # motion_program はmarker目標を持たない — 距離からの推測highlightは出さない
            d = ctx.get("distance_m") if not ctx.get("steps") else None
            if anchor and yaw is not None and d:
                highlight_xy = [
                    round(anchor[0] + math.cos(yaw) * d, 4),
                    round(anchor[1] + math.sin(yaw) * d, 4),
                ]
    markers = {}
    if anchor and yaw is not None:
        for t in COURSE["targets"]:
            markers[t.target_id] = [
                round(anchor[0] + math.cos(yaw) * t.distance_m, 4),
                round(anchor[1] + math.sin(yaw) * t.distance_m, 4),
            ]
    return {
        "fresh": fresh,
        "anchor": anchor,
        "heading_yaw": yaw,
        "markers": markers,
        "highlight_id": highlight_id,
        "highlight_xy": highlight_xy,
        "phase": active[1].phase if active else "IDLE",
        "boot_nonce": live.get("boot_nonce"),
    }


# ---- ことばでスイカ割り: RoundWorld spawn + 表示 ---------------------------
class SpawnIn(BaseModel):
    session_id: str
    expected_round_id: str
    expected_boot_nonce: str
    seed: int | None = None  # 省略時はサーバーが採番（検証時は明示）
    # fixture対照用の明示world位置 [x,y,z]（operator専用）。指定時は
    # spec.seed=-1 で監査可能に標識される。通常ゲーム経路では使わない。
    target: list[float] | None = None


def _spec_dict(spec) -> dict:
    wm = spec.watermelon
    return {
        "round_id": spec.round_id,
        "seed": spec.seed,
        "watermelon": {
            "pos": [round(v, 4) for v in wm.pos],
            "radius": wm.radius,
            "hit_radius": wm.hit_radius,
            "surface_m": wm.surface_m,
        },
        "scene": spec.scene,
        "robot_start": [round(v, 4) for v in spec.robot_start],
        "robot_yaw": round(spec.robot_yaw, 4),
        "time_limit_s": spec.time_limit_s,
        "max_swings": spec.max_swings,
    }


@app.post("/api/round/spawn")
def spawn_round(body: SpawnIn, request: Request):
    """RoundWorldでスイカ位置を生成しラウンドに固定する（operator専用）。

    robot位置・向きは現在のlive観測から取る（world正本の座標系と一致）。
    """
    if not OPERATOR_TOKEN or request.headers.get("x-kotoba-operator") != OPERATOR_TOKEN:
        raise HTTPException(403, "operator_required")
    if body.session_id not in session.rounds:
        raise HTTPException(404, "unknown_session")
    # reset は _reset_mu → _run_mu の順で世界世代を入れ替える。観測の取得から
    # RoundSpec確定まで同じ順で直列化し、旧bootの座標を新roundへ固定しない。
    if not session._reset_mu.acquire(blocking=False):
        raise HTTPException(409, "reset_in_progress")
    try:
        ready, boot_nonce = _boot_ready()
        if not ready or not boot_nonce:
            raise HTTPException(409, "live_not_ready")
        try:
            live = _read_latest_state_file()
            if not live.get("fresh"):
                raise HTTPException(409, "live_not_ready")
            live_boot = str(live.get("boot_nonce") or "")
            if live_boot != boot_nonce or (
                session.boot_nonce is not None and live_boot != session.boot_nonce
            ):
                raise HTTPException(409, "live_boot_mismatch")
            pos = live["pos"]
            w, x, y, z = live["quat_wxyz"]
            anchor = [round(pos[0], 4), round(pos[1], 4)]
            yaw = math.atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(409, "live_not_ready") from exc
        seed = body.seed if body.seed is not None else uuid.uuid4().int % (2**31)
        target = tuple(body.target) if body.target is not None else None
        spec = session.spawn_round(
            body.session_id, seed, (anchor[0], anchor[1], 0.82), yaw,
            target=target, boot_nonce=boot_nonce,
            expected_round_id=body.expected_round_id,
            expected_boot_nonce=body.expected_boot_nonce,
        )
    except SpawnRejected as exc:
        raise HTTPException(409, exc.reason) from exc
    finally:
        session._reset_mu.release()
    return {"spawned": True, **_spec_dict(spec)}


@app.get("/api/display/round")
def display_round():
    """3D rendererが読むラウンド世界（watermelon world位置の正本）。

    sim worldのスイカは常に1つ — 最後にspawnしたroundのspecが正本。
    最後のspawnまたはreset対象sessionを正本とする。reset後は旧sessionの
    fixture specへフォールバックせず、スイカを消す。
    """
    _ready, boot_nonce = _boot_ready()
    spec = session.display_spec(boot_nonce)
    return {"active": True, **_spec_dict(spec)} if spec else {"active": False}


# ---- ことばでスイカ割り: ゲーム実行・指令・状態 ------------------------------
class GameCommandIn(BaseModel):
    session_id: str
    text: str


@app.post("/api/game/start")
def game_start(body: SessionIn, request: Request):
    """スイカ割りラウンド開始（operator専用 — LIVE未受入の間）。

    spawn済みRoundSpecに対し game controller container を起動する。
    指令経路は lease+seq+bounds+期限 で fail-closed 検査（B3機構）。
    """
    if OFFLINE_MODE:
        raise HTTPException(409, "live_not_accepted_offline_mode")
    if not OPERATOR_TOKEN or request.headers.get("x-kotoba-operator") != OPERATOR_TOKEN:
        raise HTTPException(403, "operator_required")
    r = session.rounds.get(body.session_id)
    if r is None:
        raise HTTPException(404, "unknown_session")
    if r.round_spec is None:
        raise HTTPException(409, "no_round_spawned")
    ready, nonce = _boot_ready()
    if not ready:
        raise HTTPException(409, "sim_not_ready")
    run_id = uuid.uuid4().hex
    run_dir = RUNTIME_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started = session.start_game(body.session_id, nonce, run_id, run_dir)
    if not started.get("started"):
        raise HTTPException(409, started.get("reason", "not_started"))
    threading.Thread(
        target=_execute_game,
        args=(body.session_id, run_id, started["lease_id"], nonce),
        daemon=True,
    ).start()
    return {"run_id": run_id, "started": True}


@app.post("/api/game/command")
def game_command(body: GameCommandIn):
    """参加者のことば→有界指令→control.json。

    参加者向け（token不要 — 指令はlease+seq+有界値でfail-closed検査され、
    roundを開始したoperatorのlease無しには届かない）。解釈不能は聞き返し。
    """
    if OFFLINE_MODE:
        raise HTTPException(409, "offline_mode")
    if len(body.text) > 100:
        raise HTTPException(422, "text_too_long")
    result = session.game_command(body.session_id, body.text)
    if result.get("reason") == "unknown_session":
        raise HTTPException(404, "unknown_session")
    return result


class RunCommandIn(BaseModel):
    session_id: str
    type: str  # "stop" — 通常runの正常停止（管理中断のpauseとは別経路）


@app.post("/api/game/heartbeat")
def game_heartbeat(body: SessionIn):
    """継続jog中の生存確認 — control_dir/heartbeat.jsonへlease結合で書込。

    参加者向け（token不要 — jogを開始・延長する権能は持たず、
    実行中jogの「操作者がまだ居る」確認のみ。lease不一致や
    古いheartbeatはcontroller側が無効とする）。
    """
    if OFFLINE_MODE:
        raise HTTPException(409, "offline_mode")
    res = session.game_heartbeat(body.session_id)
    if res is None:
        return {"ok": False, "reason": "not_running"}
    lease_id, control_dir = res
    try:
        (Path(control_dir) / "heartbeat.json").write_text(
            json.dumps({"lease_id": lease_id, "t_wall": time.time()})
        )
    except OSError:
        return {"ok": False, "reason": "control_path_unavailable"}
    return {"ok": True}


@app.post("/api/run/command")
def run_command(body: RunCommandIn):
    """通常runへの停止指令 — run_dir/run_cmd.json（runnerが閉ループで処理）。

    継続jog中の「止まって」をrunへ届ける正常停止経路。
    管理中断（物理・全run停止）の /api/run/pause と混同しない。
    """
    if OFFLINE_MODE:
        raise HTTPException(409, "offline_mode")
    if body.type != "stop":
        raise HTTPException(422, "unknown_cmd_type")
    run_id = session.run_command_accept(body.session_id)
    if run_id is None:
        return {"accepted": False, "reason": "not_running"}
    try:
        (RUNTIME_DIR / run_id / "run_cmd.json").write_text(
            json.dumps(
                {"type": "stop", "run_id": run_id, "t_wall": time.time()}
            )
        )
    except OSError:
        return {"accepted": False, "reason": "control_path_unavailable"}
    return {"accepted": True}


@app.post("/api/run/heartbeat")
def run_heartbeat(body: SessionIn):
    """通常runの継続jog用 heartbeat — run_dir/heartbeat.jsonへ。"""
    if OFFLINE_MODE:
        raise HTTPException(409, "offline_mode")
    run_id = session.run_command_accept(body.session_id)
    if run_id is None:
        return {"ok": False, "reason": "not_running"}
    try:
        (RUNTIME_DIR / run_id / "heartbeat.json").write_text(
            json.dumps({"run_id": run_id, "t_wall": time.time()})
        )
    except OSError:
        return {"ok": False, "reason": "control_path_unavailable"}
    return {"ok": True}


@app.get("/api/game/state/{session_id}")
def game_state(session_id: str):
    """ゲーム画面用の状態: round位相・残時間・振数・現行指令・直近打撃判定。"""
    r = session.rounds.get(session_id)
    if r is None:
        raise HTTPException(404, "unknown_session")
    out = {
        "phase": r.phase,
        "message": r.message,
        "round_id": r.round_id,
        "spec": _spec_dict(r.round_spec) if r.round_spec else None,
        "spec_boot_nonce": r.spec_boot_nonce if r.round_spec else None,
    }
    # 開始可否のtyped DTO — UIはボタンdisabled理由をここから表示する。
    # 実判定は /api/game/start が必ずサーバー側で再検査する（ここは表示用）。
    sb = session.startable(session_id)
    ready, _nonce = _boot_ready()
    if not ready:
        sb["reasons"].append("sim_not_ready")
    if OFFLINE_MODE:
        sb["reasons"].append("offline_mode")
    sb["ok"] = sb["ok"] and ready and not OFFLINE_MODE
    out["startable"] = sb
    if r.game is not None:
        g = r.game
        game = {
            "run_id": r.run_id,
            "seq": g["seq"],
            "done": g.get("done", False),
            "outcome": g.get("outcome"),
            "strike_pending": bool(g.get("strike_pending")),
            "strike_unconfirmed": g.get("strike_unconfirmed", False),
        }
        if r.round_spec is not None:
            game["max_swings"] = r.round_spec.max_swings
            game["time_limit_s"] = r.round_spec.time_limit_s
        # controllerが書く実状態（applied_seq/現指令/実消費振数/打撃判定/残時間正本）
        try:
            st = json.loads(
                (Path(g["control_path"]).parent / "control_state.json").read_text()
            )
            game["applied_seq"] = st.get("applied_seq")
            game["command_state"] = st.get("command_state")
            game["current_type"] = st.get("current_type")
            game["move_until_wall"] = st.get("move_until_wall")
            # 実消費正本: 打撃動作を実際に開始した回数
            game["strikes_started"] = st.get("strikes_started")
            game["strikes_done"] = st.get("strikes_done")
            game["strike_phase"] = (st.get("strike") or {}).get("phase")
            game["last_strike"] = st.get("last_strike")
            # 未開始で終わった直近打撃の理由（姿勢未安定等 — 消費なしの証跡）
            game["last_strike_attempt"] = st.get("last_strike_attempt")
            # 直近に完了した打撃のid（判定・試行の新旧を照合するための同定子）
            game["last_strike_done"] = st.get("last_strike_done")
            game["strike_expired"] = st.get("strike_expired")
            game["strike_unverified"] = st.get("strike_unverified")
            game["health"] = st.get("health", "ok")
            # 音声トリガー用の実イベントdelta（R5-04）— run結合のseqで
            # clientがhigh-water管理し、snapshot由来の過去SFXは再生しない
            game["events"] = [
                {
                    "id": f"{r.run_id}:{ev.get('event_seq')}",
                    **ev,
                }
                for ev in (st.get("events_tail") or [])
            ]
            # 終端のfallenはfinish_runがresultから引き継いだ正本を優先
            # （tail期の転倒はcontrol_state最終書込に間に合わない）
            game["fallen"] = bool(g.get("fallen") or st.get("fallen", False))
            dl = st.get("round_deadline_wall")
            game["remaining_s"] = (
                round(max(0.0, dl - time.time()), 1) if dl else None
            )
            game["controller_alive"] = (time.time() - st.get("t_wall", 0)) < 3.0
        except (OSError, json.JSONDecodeError):
            game["controller_alive"] = False
        # 残り振数: 実消費正本から算出（未確認は減らさない）
        if r.round_spec is not None:
            started = game.get("strikes_started")
            if started is not None:
                game["swings_left"] = max(
                    0, r.round_spec.max_swings - int(started)
                )
        out["game"] = game
    return out


def _execute_game(session_id: str, run_id: str, lease_id: str, expected_boot: str) -> None:
    """ゲームcontroller containerを起動・監視し、終了結果をroundへ反映する。

    _execute_run のゲーム版: manifestに lease_id・スイカtarget・hit半径・
    ラウンド規則（time_limit/max_swings）を載せる。指令は run dir の
    control.json 経由で participants から届く（latest-wins）。
    """
    r = session.rounds.get(session_id)
    spec = r.round_spec if r else None
    run_dir = RUNTIME_DIR / run_id
    manifest = {
        "run_id": run_id,
        "purpose": "game_round",
        "arming": "1",
        "lease_id": lease_id,
        # manifest期限は外郭（game時計+起動余裕）。プレイ時計は time_limit_s
        "expires_in_s": (spec.time_limit_s + 60) if spec else 300,
        "boot_expect": expected_boot,
        # pace→profile選択の能力台帳（受理側と同一の台帳を実行側へ渡す — R5）
        "capabilities": dict(CAPABILITIES),
    }
    if spec is not None:
        manifest["target"] = list(spec.watermelon.pos)
        manifest["hit_radius"] = spec.watermelon.hit_radius
        # 地上スイカ（surface_m=0）は素手が届かない — 手先に固定した
        # 仮想棒の先端を実効エフェクタとして判定に使う（FK実測校正済み）。
        if spec.watermelon.surface_m <= 0.05:
            from kotoba_harness.kinematics import STICK_LEN_M

            manifest["stick_m"] = STICK_LEN_M
        manifest["max_swings"] = spec.max_swings
        manifest["time_limit_s"] = spec.time_limit_s
        manifest["end_on_hit"] = True
        # 試用範囲の明示的限定（seed15対策）: 開始位置からの操縦範囲。
        # 実測で全成功roundは<3mに収まり、不收束の漂流は14mで転倒した。
        # 範囲外への歩き続けは安全側でラウンド終了（arena_exit）にする。
        manifest["robot_start"] = list(spec.robot_start)
        # 操縦範囲はRoundWorld正本から — ハードコードしない
        manifest["arena_r_m"] = (spec.scene or {}).get("arena_r_m", 5.0)
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    result_path = run_dir / "result.json"
    harness_src = str(HARNESS_SRC)
    cname = f"kotoba-runner-{run_id[:8]}"
    cmd = (
        f"docker run -d --init --name {cname} "
        f"--user $(id -u):$(id -g) --network {SIM_NETWORK} "
        f"-e KOTOBA_PUBLISH=1 "
        f"-v {KOTOBA_HOME}:/kotoba "
        f"-v {harness_src}:/kotoba/harness_src "
        f"-v {run_dir}:/kotoba/run "
        f"-w /kotoba --entrypoint python3 {RUNNER_IMAGE} "
        f"/kotoba/runner/kotoba_game_controller.py /kotoba/run/result.json "
        f"/kotoba/run/manifest.json"
    )
    if not session.owns_run(session_id, run_id):
        return
    result = {}
    try:
        launch = subprocess.run(
            ["bash", "-c", cmd], capture_output=True, text=True, timeout=60
        )
        if launch.returncode != 0:
            result = {
                "verdict": "RUNNER_ERROR",
                "reasons": [(launch.stderr or "docker run failed")[-300:]],
            }
        elif not session.owns_run(session_id, run_id):
            subprocess.run(
                ["docker", "rm", "-f", cname], capture_output=True, timeout=30
            )
            result = {
                "verdict": "ABORTED_AT_LAUNCH",
                "reasons": ["generation invalidated during runner launch"],
            }
        else:
            try:
                subprocess.run(
                    ["docker", "wait", cname],
                    capture_output=True, text=True, timeout=420,
                )
            except Exception as exc:
                result = {
                    "verdict": "RUNNER_ERROR",
                    "reasons": [f"wait:{str(exc)[-200:]}"],
                }
            if not result:
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                else:
                    logs = subprocess.run(
                        ["docker", "logs", "--tail", "40", cname],
                        capture_output=True, text=True, timeout=15,
                    )
                    tail = ((logs.stdout or "") + (logs.stderr or ""))[-300:]
                    result = {
                        "verdict": "RUNNER_ERROR",
                        "reasons": [tail or "controller produced no result.json"],
                    }
            subprocess.run(
                ["docker", "rm", "-f", cname], capture_output=True, timeout=30
            )
    except Exception as exc:
        result = {"verdict": "RUNNER_ERROR", "reasons": [str(exc)[-300:]]}
    finally:
        result["run_id"] = run_id
        session.finish_run(session_id, run_id, result)


# ---- ローカルASR転写プロキシ（R6 P1-D） ------------------------------------
# ブラウザのAudioWorkletが切り出した発話区間（PCM16/16kHz/mono）を受け、
# 専用ASRサービス（Qwen3-ASR、KOTOBA_ASR_URL）へ転送してテキストを返す。
# 転写のみ — 指令化・実行は既存の /api/game/command・/api/intents 経路の
# validatorを必ず通る。ASR経路が直接motor指令を発行しない。
ASR_URL = os.environ.get("KOTOBA_ASR_URL", "http://127.0.0.1:8710")
ASR_MAX_BYTES = 600 * 1024  # 16kHz×2byte×18s上限 + pre-roll余裕
ASR_TIMEOUT_S = float(os.environ.get("KOTOBA_ASR_TIMEOUT", "6"))
_asr_locks: dict[str, asyncio.Lock] = {}
_asr_locks_mu = threading.Lock()


@app.post("/api/asr/recognize")
async def asr_recognize(request: Request, session_id: str = ""):
    """発話区間PCM16 → テキスト。sessionあたりin-flight=1（超過は429）。

    参加者向け（token不要 — 転写のみで権能を持たない）。録音の永続化は
    しない: bodyはメモリ内でASRへ渡し、応答後に破棄する。
    """
    try:
        session.sessions.get(session_id)
    except Exception:
        raise HTTPException(404, "unknown_session")
    clen = request.headers.get("content-length")
    if clen and int(clen) > ASR_MAX_BYTES:
        raise HTTPException(413, "audio_too_large")
    body = await request.body()
    if not body or len(body) > ASR_MAX_BYTES or len(body) % 2:
        raise HTTPException(422, "bad_pcm16")
    with _asr_locks_mu:
        lock = _asr_locks.setdefault(session_id, asyncio.Lock())
    if lock.locked():
        raise HTTPException(429, "asr_busy")

    def _call() -> dict:
        import urllib.request as _u

        req = _u.Request(
            f"{ASR_URL}/recognize",
            data=body,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        with _u.urlopen(req, timeout=ASR_TIMEOUT_S) as r:
            return json.loads(r.read().decode("utf-8"))

    async with lock:
        try:
            out = await asyncio.to_thread(_call)
        except HTTPException:
            raise
        except Exception as exc:
            # ASR未稼働・timeout・応答異常を区別できるtyped失敗
            raise HTTPException(503, f"asr_unavailable:{type(exc).__name__}")
    if not isinstance(out, dict):
        raise HTTPException(502, "asr_bad_response")
    return {
        "text": str(out.get("text") or ""),
        "latency_ms": out.get("latency_ms"),
        "engine": out.get("engine"),
    }


@app.get("/api/asr/status")
def asr_status():
    """ASRサービスの稼働面（UIの音声engine選択とhealth表示用）。

    reachable/engine/model/queueを返す。未配備時はreachable=False —
    UIはローカルASR不可時にon-device Web Speechへ明示フォールバックする
    （黙ってcloud経路へ流さない）。
    """
    import urllib.request as _u

    out = {"reachable": False, "engine": "qwen3-asr", "model": None}
    try:
        with _u.urlopen(f"{ASR_URL}/health", timeout=1.5) as r:
            st = json.loads(r.read().decode("utf-8"))
        out.update(st)
        out["reachable"] = True
    except Exception:
        pass
    return out


@app.get("/api/render/status")
def render_status():
    """rendererプロセスの実状態（UIのREADY判定用）。

    Connected表示やcanvas有無は判定材料にしない。rendererが書く
    render_status.json（model/mesh読込・初回frame・現bootのpose反映）を
    返し、live bootとの一致を boot_match で示す。未生成・staleは
    reachable=False で返す（偽ready禁止）。
    """
    live = _live_summary()
    out = {"reachable": False, "state": "unreachable",
           "boot_match": False, "boot_nonce": live.get("boot_nonce")}
    try:
        st = json.loads((RUNTIME_DIR / "render_status.json").read_text())
    except (OSError, json.JSONDecodeError):
        return out
    age = time.time() - float(st.get("ts", 0) or 0)
    out.update(st)
    out["reachable"] = age < 3.0
    out["status_age_s"] = round(age, 3)
    if st.get("boot_nonce") is not None and live.get("boot_nonce") is not None:
        out["boot_match"] = st["boot_nonce"] == live["boot_nonce"]
    if not out["reachable"]:
        out["state"] = "stale"
    return out


# ---- 3D renderer WS橋渡し（/render/* → localhost:8080 のviser） --------------
# ブラウザのWSを既存tunnel経路（8700）で受け、描画専用viserへ中継する。
# tunnel ingress変更を避けるためAPI内で橋渡し。描画は読み取り専用。
RENDER_WS_UPSTREAM = os.environ.get("KOTOBA_RENDER_WS", "ws://127.0.0.1:8080")


@app.websocket("/render/{path:path}")
async def render_ws(websocket: WebSocket, path: str):
    import asyncio
    import secrets
    import time as _time

    import websockets

    # 接続ごとの診断ID（認証資格ではない短い識別子 — R3 §3.1）。
    # 最初に切れた側・例外・close code・転送量を構造化してapi.logへ残す。
    cid = secrets.token_hex(4)
    t0 = _time.monotonic()
    counts = {"up_n": 0, "up_b": 0, "up_max": 0, "dn_n": 0, "dn_b": 0, "dn_max": 0}

    def _log(ev: str, **kw) -> None:
        body = " ".join(f"{k}={v}" for k, v in kw.items())
        print(f"[render-ws {cid}] {ev} {body}", flush=True)

    upstream_url = f"{RENDER_WS_UPSTREAM}/render/{path}"
    if websocket.url.query:
        upstream_url += f"?{websocket.url.query}"
    # viserはversionをWS subprotocol（viser-vX.Y.Z）で検査する。
    # ブラウザ提示のsubprotocolをそのままupstreamへ渡し、選択結果を
    # accept時にブラウザへ返さないと "Client: 'unknown'" で拒否される。
    client_subprotocols = list(websocket.scope.get("subprotocols") or [])
    try:
        async with websockets.connect(
            upstream_url,
            subprotocols=client_subprotocols or None,
            # シーン初期状態は1メッセージ数十MBになり得る（PM01 mesh群）。
            # localhost信頼経路のため上限は設けない。
            max_size=None,
            open_timeout=15,
            # upstream keepalive: 既定20s/20s。初期scene転送（実測〜60MB）中に
            # event loopが転送へ占有されてもping応答が遅れて落ちないよう、
            # localhost信頼区間のみ timeout を60sへ拡大する（変更記録: R3 §4.3）。
            # interval自体は既定のまま — 障害検知を消す設定（None）にはしない。
            ping_interval=20,
            ping_timeout=60,
            close_timeout=10,
        ) as upstream:
            await websocket.accept(subprotocol=upstream.subprotocol)
            _log(
                "open",
                handshake_ms=round((_time.monotonic() - t0) * 1000),
                proto=upstream.subprotocol,
                path=path,
            )

            async def client_to_upstream() -> str:
                while True:
                    msg = await websocket.receive()
                    if msg.get("bytes") is not None:
                        data = msg["bytes"]
                        counts["up_n"] += 1
                        counts["up_b"] += len(data)
                        counts["up_max"] = max(counts["up_max"], len(data))
                        await upstream.send(data)
                    elif msg.get("text") is not None:
                        data = msg["text"]
                        counts["up_n"] += 1
                        counts["up_b"] += len(data)
                        counts["up_max"] = max(counts["up_max"], len(data))
                        await upstream.send(data)
                    elif msg.get("type") == "websocket.disconnect":
                        return "client_disconnect"

            async def upstream_to_client() -> str:
                async for frame in upstream:
                    n = len(frame)
                    counts["dn_n"] += 1
                    counts["dn_b"] += n
                    counts["dn_max"] = max(counts["dn_max"], n)
                    if isinstance(frame, bytes):
                        await websocket.send_bytes(frame)
                    else:
                        await websocket.send_text(frame)
                return "upstream_eof"

            # どちらかが先に終わったら sibling をcancelして両socketを有界close。
            # gather両待ちでは片側終了後に他方が残留し、原因も記録されない。
            tasks = [
                asyncio.create_task(client_to_upstream()),
                asyncio.create_task(upstream_to_client()),
            ]
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            first_side, first_exc = "unknown", None
            for t in done:
                if t.cancelled():
                    continue
                exc = t.exception()
                if exc is not None:
                    first_exc = f"{type(exc).__name__}:{str(exc)[:160]}"
                    first_side = "pump_exc"
                else:
                    first_side = t.result()
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

            up_code = getattr(upstream, "close_code", None)
            up_reason = (getattr(upstream, "close_reason", None) or "")[:120]
            _log(
                "closing",
                first=first_side,
                exc=first_exc,
                up_code=up_code,
                up_reason=up_reason,
                dur_s=round(_time.monotonic() - t0, 2),
                **counts,
            )
            # 予約code（1005/1006/1015）は送れない — 有効codeへ写像する。
            try:
                await asyncio.wait_for(upstream.close(), 3)
            except Exception:
                pass
            try:
                if up_code and up_code not in (1005, 1006, 1015):
                    await websocket.close(code=up_code)
                else:
                    await websocket.close(code=1000)
            except Exception:
                pass
    except Exception as e:
        # 最初の故障理由を二次例外で上書きしない — ここでは記録だけ。
        _log(
            "error",
            exc=f"{type(e).__name__}:{str(e)[:160]}",
            code=getattr(e, "code", None),
            dur_s=round(_time.monotonic() - t0, 2),
            **counts,
        )
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


def _read_latest_state_file() -> dict:
    """常駐observerが書き出す live.json（あれば）を返す。

    鮮度は「観測の受信wall時刻」（obs_wall）から判定する。ファイル書出し時刻
    （wall）をfresh条件に使わない — 古い観測を書き直しても新鮮にはならない。
    """
    latest = RUNTIME_DIR / "live.json"
    if not latest.exists():
        return {"fresh": False, "note": "observer inactive"}
    data = json.loads(latest.read_text())
    fresh, age = live_freshness(data, time.time())
    data["fresh"] = fresh
    data["age_s"] = round(age, 2) if age is not None else None
    if age is None:
        data["note"] = "obs_wall missing (legacy format)"
    data["file_age_s"] = round(time.time() - data.get("wall", 0), 2)
    # 実行中runの目標を同一planへ結び付けてUIへ返す（runnerが計算した真のtarget）
    tgt = _active_target()
    if tgt is not None:
        data["target"] = tgt
    return data


def _active_target() -> list | None:
    """現在RUNNING中のrunが追っている目標xyを返す（runnerが run dir に記録）。"""
    for r in session.rounds.values():
        if r.phase == "RUNNING" and r.run_id:
            p = RUNTIME_DIR / r.run_id / "target.json"
            if p.exists():
                try:
                    d = json.loads(p.read_text())
                    if d.get("run_id") == r.run_id and d.get("target_xy"):
                        return d["target_xy"]
                except (json.JSONDecodeError, KeyError):
                    pass
    return None


# ---- static UI ------------------------------------------------------------
@app.get("/")
def index():
    index_html = KIOSK_DIST / "index.html"
    if index_html.exists():
        return FileResponse(index_html)
    return JSONResponse({"error": "ui_not_built"}, status_code=503)


@app.get("/{path:path}")
def static_files(path: str):
    target = (KIOSK_DIST / path).resolve()
    if str(target).startswith(str(KIOSK_DIST.resolve())) and target.is_file():
        return FileResponse(target)
    raise HTTPException(404, "not_found")
