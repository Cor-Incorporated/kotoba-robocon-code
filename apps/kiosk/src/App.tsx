import { useCallback, useEffect, useRef, useState } from "react";
import { inspectModeReturn } from "./mode_return";
import { classifySpawnSnapshot, mayOfferManualRetry, nextAbsentReads, readSpawnSnapshot, spawnRequestPayload } from "./spawn_reconcile";
import type { SpawnIdentity, SpawnSnapshot } from "./spawn_reconcile";
import { AudioManager } from "./audio";
import type { AssetsState, OutputState } from "./audio";
import { VoiceController, probeVoice } from "./voice";
import type { VoiceHealth, VoiceProbe } from "./voice";
import { LocalAsrController } from "./localasr";
import type { AsrHealth, AsrInputInfo, AsrTiming } from "./localasr";
import "./App.css";

// R7 1.2: ASRサービス稼働面の型分け — HTTP status・requestId・engine/model/
// deviceの実値を保持し、403を「pack不足」「モデル未配備」へ変換しない。
// 到達できても model読込/warm-up 未完は ready と別状態にする。
type AsrServiceState =
  | "unknown"        // status未取得
  | "ready"          // reachable + model loaded
  | "model_loading"  // reachable・model warm-up中
  | "model_error"    // reachable・model load失敗
  | "forbidden"      // HTTP 403（operator許可外 — 権限の問題）
  | "unauthorized"   // HTTP 401（Access未認証）
  | "unreachable"    // 経路/サービス不達・5xx・timeout
  | "degraded_cpu";  // 採用GPU構成からCPUへ低下（性能状態を明示）

interface AsrService {
  state: AsrServiceState;
  httpStatus: number | null;
  requestId: string | null;
  engine: string | null;
  model: string | null;
  device: string | null;      // "cuda:0" / "cpu"
  deviceName: string | null;
  detail: string;
}

const ASR_INIT: AsrService = {
  state: "unknown", httpStatus: null, requestId: null,
  engine: null, model: null, device: null, deviceName: null, detail: "",
};

// R7 1.2: 展示はローカルASR専用（LOCAL_ASR_ONLY相当の明示設定）。
// Web Speechへの自動選択をしない — ASR不調時は実際の原因を表示する。
// 非展示用途でWeb Speechを使う場合のみ false にする。
const LOCAL_ASR_ONLY = true;

// R7 1.4: 実行中commandの由来（UI側所有権）。voice由来の継続動作は
// mic/ASR故障・disarm・非表示で正常STOPを送り、heartbeat延命を止める。
type CmdOrigin = "text" | "button" | "voice";
interface CmdOwner { origin: CmdOrigin; epoch: number }

type Phase =
  | "READY" | "INTERPRETING" | "CLARIFY" | "REVIEW" | "RUNNING"
  | "RESULT" | "FAULT" | "RESETTING";

interface CourseTarget { id: string; label: string; distance_m: number }
interface StateDto {
  session_id: string; round_id: string; phase: Phase; message: string;
  pending_plan_id?: string; run_id?: string; result?: any;
  course: { targets: CourseTarget[] }; live_enabled: boolean; selftest?: boolean;
  // 動作プログラム実行中の進捗（steps runnerのrun_state.json写像）
  motion?: {
    status?: string; reason?: string | null;
    step_index?: number; step_count?: number;
    action_key?: string; label?: string;
    dist_m?: number; target_m?: number;
    angle_deg?: number; target_deg?: number;
  } | null;
}
interface LiveDto {
  fresh?: boolean; pos?: number[]; vel?: number[]; sim_time_s?: number;
  age_s?: number; target?: number[]; task?: string;
  boot_nonce?: string | number | null; step_seq?: number;
}
interface MetaDto { mode?: string; email?: string; ui_commit?: string; deployed_at?: string }
interface RenderStatusDto {
  reachable?: boolean; state?: string; boot_match?: boolean;
  boot_nonce?: number; model_nq?: number; pose_age_s?: number;
  markers_stale?: boolean; instance_id?: string;
}
// ことばでスイカ割り（/api/game/state の写像）
interface GameSpec {
  round_id: string; seed: number;
  watermelon: { pos: number[]; radius: number; hit_radius: number;
    surface_m?: number };
  robot_start: number[]; robot_yaw: number;
  time_limit_s: number; max_swings: number;
  scene?: { mode?: string; arena_r_m?: number; arena_center?: number[];
    geometry_version?: string; spawn_algo_version?: string;
    bearing_deg?: number } | null;
}
interface GameAudioEventDto {
  id: string; event_seq: number; kind: string; t_wall: number;
  strike_id?: string; reason?: string; outcome?: unknown;
}
interface GameRun {
  run_id?: string; seq: number;
  done: boolean; outcome?: string | null;
  events?: GameAudioEventDto[];
  max_swings?: number; time_limit_s?: number;
  applied_seq?: number; current_type?: string;
  // 実消費正本（controllerが打撃開始を実観測した回数）
  strikes_started?: number; strikes_done?: number;
  swings_left?: number;
  strike_pending?: boolean; strike_phase?: string | null;
  strike_unconfirmed?: boolean;
  move_until_wall?: number | null;
  // 実行状態DTO — 認識/適用/実動作の区別に使う（HTTP受理≠実移動）
  command_state?: {
    seq?: number; status?: string; reason?: string | null;
    action_key?: string; label?: string;
    step_index?: number; step_count?: number;
    progress?: { dist_m?: number; target_m?: number;
      angle_deg?: number; target_deg?: number;
      lateral_dev_m?: number; yaw_drift_deg?: number };
  } | null;
  health?: string;
  last_strike?: { hit: boolean; min_dist_m?: number; hand?: string } | null;
  // 未開始で終わった直近打撃の理由（姿勢未安定等 — 振数は消費されない）
  last_strike_attempt?: { status?: string; aborted?: string;
    interrupted?: string; stillness_vel?: number } | null;
  remaining_s?: number | null; controller_alive?: boolean;
}
interface GameStateDto {
  phase: Phase; message: string; round_id?: string;
  spec?: GameSpec | null; spec_boot_nonce?: string | number | null;
  game?: GameRun;
  // 開始可否のtyped DTO（表示専用 — 実判定は /api/game/start が再検査）
  startable?: { ok: boolean; reasons: string[] };
}
interface SpawnAttempt extends SpawnIdentity {
  requestedAt: number;
  absentReads: number;
  requestSettled: boolean;
  retryReady: boolean;
}
type RendererPhase =
  | "connecting" | "loading" | "syncing" | "ready" | "stale" | "error"
  | "disconnected";

// 入力例 — 押すと入力欄へ入るだけ（実行はしない）。通常モードの必須選択肢:
// 前1m・左右90°・振り向き180°・少し左右・複合(右90°→1m)・旧marker回帰。
const EXAMPLES = [
  "前に1m進んで",
  "右に90°旋回して",
  "左に90°旋回して",
  "後ろを向いて",
  "少し右に進んで",
  "少し左に進んで",
  "右に90°旋回して1m歩いて",
  "手前のマーカーまで進んで",
];

// 聞き返し時はマウスだけでも代表的な回答を選べる短いラベルを並べる。
const CLARIFY_CHOICES = [
  { label: "前", value: "前" },
  { label: "後ろ", value: "後ろ" },
  { label: "右90°", value: "右に90°向いて" },
  { label: "左90°", value: "左に90°向いて" },
  { label: "少し右", value: "少し右" },
  { label: "少し左", value: "少し左" },
  { label: "手前", value: "手前" },
  { label: "奥", value: "奥" },
];

const TARGET_COLORS: Record<string, string> = {
  goal_near: "#32e65a", // 3D上の手前markerと同色
  goal_far: "#f07332",  // 3D上の奥markerと同色
};

export default function App() {
  const [sessionId, setSessionId] = useState<string>("");
  const [state, setState] = useState<StateDto | null>(null);
  const [live, setLive] = useState<LiveDto>({});
  const [text, setText] = useState("");
  const [composing, setComposing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [review, setReview] = useState<any>(null);
  const [clarifyAnswer, setClarifyAnswer] = useState("");
  const [error, setError] = useState("");
  // mode切替の応答断だけを識別し、後続pollが正常roundを観測した時に消す。
  const modeReturnError = useRef<string | null>(null);
  const [meta, setMeta] = useState<MetaDto | null>(null);
  // 描画実状態 — R6: client計測はtransport/転送の診断のみ。
  // ready合意の正本はserver側 render_status.json（state=ready・boot一致・
  // pose反映・status鮮度）。sceneBytes/streamAlive/connStableは
  // 「転送中/同期中」の進捗表示と接続健全性の補助証拠として使う。
  const [sceneBytes, setSceneBytes] = useState(0); // 転送進捗（診断のみ）
  const [sceneReadyS, setSceneReadyS] = useState<number | null>(null);
  const [streamAlive, setStreamAlive] = useState(false); // 現接続でstream到達
  const [connStable, setConnStable] = useState(false); // openが3s継続
  const [clientSceneReady, setClientSceneReady] = useState(false); // 現socketでPM01 nodeを復号
  const [renderStatus, setRenderStatus] = useState<RenderStatusDto | null>(null);
  const [statusFail, setStatusFail] = useState(false);
  // iframe内WSの実接続状態（注入clientのBroadcastChannel報告が正本。
  // DOMの"Connected"文字は診断材料であって根拠にしない）
  const [conn, setConn] = useState<"init" | "open" | "closed" | "error">("init");
  const [connDetail, setConnDetail] = useState("");
  // R7 1.6: client側描画ACK — 現socketでscene受信済みかつmsgが流れ続けて
  // いる実測。server readyだけでなくclientのdecode生存もready条件にする
  // （clientのdecode停止・別bootの古いack・注入未稼働の偽readyを防ぐ）。
  const lastAckAt = useRef(0);
  const [clientAckOk, setClientAckOk] = useState(false);
  // API到達性（業務エラー表示とは分離 — reject文を次のpollで消さない）
  const [apiDown, setApiDown] = useState(false);
  const [renderTick, setRenderTick] = useState(0); // 再接続操作でのiframe再mount用
  const [sidebarOpen, setSidebarOpen] = useState(true); // 右サイドバー（折りたたみ可 — iframeは再mountしない）
  // スイカ割り状態（specが立つとゲームモード）
  const [game, setGame] = useState<GameStateDto | null>(null);
  // spawn成功からspecの読み戻しまで再生成を塞ぐ。HTTP応答だけでは
  // 画面とサーバーのworldが一致した証拠にならない。
  const [spawnPending, setSpawnPending] = useState(false);
  const [spawnRetryAvailable, setSpawnRetryAvailable] = useState(false);
  const [spawnMessage, setSpawnMessage] = useState("");
  const [spawnGenerationChanged, setSpawnGenerationChanged] = useState(false);
  const spawnAttempt = useRef<SpawnAttempt | null>(null);
  // 通常runのPOST受理から次の状態読戻しまで診断遷移・モード切替を塞ぐ。
  const [runStartPending, setRunStartPending] = useState(false);
  const runStartPendingSession = useRef<string | null>(null);
  const [gameText, setGameText] = useState("");
  const [lastCmd, setLastCmd] = useState("");
  const [lastCmdSeq, setLastCmdSeq] = useState<number | null>(null);
  // 音声入力（明示arm — 自動起動しない・号令即commit・STOP先行）
  const [voiceArmed, setVoiceArmed] = useState(false);
  const [voiceInterim, setVoiceInterim] = useState("");
  const [voiceHealth, setVoiceHealth] = useState<VoiceHealth | AsrHealth>("off");
  const [voiceProbe, setVoiceProbe] = useState<VoiceProbe | null>(null);
  // R7 1.2: typed ASR稼働面（boolean到達可否ではなく原因を保持）
  const [asrService, setAsrService] = useState<AsrService>(ASR_INIT);
  const [voiceEngine, setVoiceEngine] = useState<"local-asr" | "webspeech" | null>(null);
  const voiceRef = useRef<VoiceController | null>(null);
  const asrRef = useRef<LocalAsrController | null>(null);
  // arm時に結んだrun owner — run終了/失効でdisarmする（通常modeはnull）
  const voiceRunRef = useRef<string | null>(null);
  // R7 1.4: 実行中commandの由来とepoch — voice由来jogの健全性を
  // heartbeat条件へ結び付ける。UIのarm状態だけで由来を失わない。
  const jogOwnerRef = useRef<CmdOwner | null>(null);
  const intentOriginRef = useRef<CmdOwner>({ origin: "text", epoch: 0 });
  const jogActiveRef = useRef(false);
  // R7 2.3: 実音声デバイス面 — 選択・許可・実sampleRate・レベル・ASR時刻
  const [micDevices, setMicDevices] = useState<MediaDeviceInfo[]>([]);
  const [micDeviceId, setMicDeviceId] = useState("");
  const [micPerm, setMicPerm] = useState<string>("unknown");
  const [micLevel, setMicLevel] = useState(0);
  const [asrInputInfo, setAsrInputInfo] = useState<AsrInputInfo | null>(null);
  const [lastAsrTiming, setLastAsrTiming] = useState<AsrTiming | null>(null);
  // R7 1.5/2.1: local entryが存在する場合のoperator session状態
  // （/local/session が応答した時点でlocal入口経路と判定する）
  const [localEntry, setLocalEntry] = useState<boolean | null>(null);
  const [localOperator, setLocalOperator] = useState(false);
  // 音声出力（実イベント駆動BGM/SFX — App所有singleton。iframe/サイドバーの
  // mountに結び付けない。assets/output/muteは分離stateで表示）
  const audioRef = useRef<AudioManager | null>(null);
  const [audioAssets, setAudioAssets] = useState<AssetsState>("idle");
  const [audioOutput, setAudioOutput] = useState<OutputState>("locked");
  const [audioMuted, setAudioMuted] = useState(false);
  // 開始単一飛行 — 二重POSTをUI層で防ぐ（サーバーのrun_in_progressは最終防衛線）
  const startInFlight = useRef(false);
  // POST受理〜controller実起動（round_started観測）までの「起動中」区間
  const [gameStarting, setGameStarting] = useState(false);
  const trailRef = useRef<{ x: number; y: number }[]>([]);
  const anchorRef = useRef<{ x: number; y: number } | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const renderFrameRef = useRef<HTMLIFrameElement>(null);
  const lastSessionTry = useRef(0);
  const sessionGeneration = useRef(0);
  const navigationInFlight = useRef(false);

  // 操作可否のUI反映はmeta基準。metaが無いローカル配信ではUIを開くが、
  // 実行可否は常にサーバー側ゲート（operator token/approval/世代）が判定する。
  const readOnly = meta !== null ? meta.mode !== "operator" : false;

  useEffect(() => {
    // preview配信時のみ存在するメタ情報（ローカル配信では404）。
    // mode==="operator" はAccess email→Worker注入のサーバー側許可の写像。
    fetch("/preview-meta.json")
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => setMeta(j))
      .catch(() => {});
    const generation = sessionGeneration.current;
    (async () => {
      try {
        const r = await fetch("/api/sessions", { method: "POST" });
        if (!r.ok) throw new Error(String(r.status));
        const j = await r.json();
        if (generation === sessionGeneration.current) setSessionId(j.session_id);
      } catch {
        if (generation === sessionGeneration.current) {
          setApiDown(true);
          setError("バックエンドに接続できません（閲覧のみ可能な場合があります）");
        }
      }
    })();
  }, []);

  // poll単一飛行 — 600ms周期で前回pollが未完なら今回は捨てる
  // （応答遅延時にGETが積み上がり、古いsnapshotが新しいsnapshotを
  // 上書きする競合を防ぐ）
  const pollBusy = useRef(false);
  const lastGameRunId = useRef<string | null>(null);
  const clearSpawnAttempt = useCallback((message = "") => {
    spawnAttempt.current = null;
    setSpawnPending(false);
    setSpawnRetryAvailable(false);
    setSpawnMessage(message);
  }, []);
  const reconcileSpawn = useCallback((
    attempt: SpawnAttempt,
    snapshot: SpawnSnapshot<StateDto, GameStateDto>,
  ) => {
    if (spawnAttempt.current !== attempt) return;
    const outcome = classifySpawnSnapshot(attempt, snapshot);
    if (outcome === "spawned") {
      setGame(snapshot.game);
      clearSpawnAttempt();
    } else if (outcome === "changed") {
      setSpawnGenerationChanged(true);
      clearSpawnAttempt("ラウンドまたは3Dの起動世代が変わりました。現在の状態から選び直してください");
    } else {
      attempt.absentReads = nextAbsentReads(
        attempt.absentReads, outcome, attempt.requestSettled,
      );
      // GETの一瞬の不在だけで再送を許可しない。以後のPOSTは明示クリックのみ。
      if (mayOfferManualRetry(attempt.absentReads, attempt.requestedAt, Date.now())) {
        attempt.retryReady = true;
        setSpawnPending(false);
        setSpawnRetryAvailable(true);
        setSpawnMessage("同じラウンドにスイカはまだありません。下のボタンで生成を再試行できます");
      }
    }
  }, [clearSpawnAttempt]);
  const poll = useCallback(async () => {
    if (document.hidden || pollBusy.current) return;
    pollBusy.current = true;
    const generation = sessionGeneration.current;
    try {
      const l = await fetch("/api/obs/live");
      if (l.ok) setLive(await l.json());
      const rs = await fetch("/api/render/status");
      if (rs.ok) { setRenderStatus(await rs.json()); setStatusFail(false); }
      else { setRenderStatus(null); setStatusFail(true); }
      let s: Response | null = null;
      if (sessionId) {
        s = await fetch(`/api/state/${sessionId}`);
        if (generation !== sessionGeneration.current) return;
        if (s.ok) {
          const snapshot = await s.json();
          if (generation !== sessionGeneration.current) return;
          setState(snapshot);
          if (runStartPendingSession.current === sessionId &&
              ["RUNNING", "RESULT", "FAULT", "RESETTING"].includes(snapshot.phase)) {
            runStartPendingSession.current = null;
            setRunStartPending(false);
          }
          const g = await fetch(`/api/game/state/${sessionId}`);
          if (generation !== sessionGeneration.current) return;
          if (g.ok) {
            const j = await g.json();
            if (generation !== sessionGeneration.current) return;
            const sameRound = j?.round_id === snapshot.round_id &&
              (j?.spec == null || j.spec.round_id === snapshot.round_id);
            setGame(sameRound ? j : null);
            if (snapshot.phase === "READY" && j?.spec === null &&
                typeof snapshot.round_id === "string" && snapshot.round_id.length > 0 &&
                j?.round_id === snapshot.round_id &&
                modeReturnError.current) {
              const stale = modeReturnError.current;
              modeReturnError.current = null;
              setError((message) => message === stale ? "" : message);
            }
            const attempt = spawnAttempt.current;
            if (attempt?.sessionId === sessionId) {
              // 通常表示用liveはstate/gameより古い。判定用bootは最後に再取得。
              const bootResponse = await fetch("/api/obs/live", { cache: "no-store" });
              if (bootResponse.ok) {
                const currentLive: LiveDto = await bootResponse.json();
                if (generation !== sessionGeneration.current) return;
                reconcileSpawn(attempt, { state: snapshot, game: j, live: currentLive });
              }
            }
            lastGameRunId.current = j?.game?.run_id ?? null;
            // 実状態の唯一の制御面 — snapshotのrun結合・終端・鮮度で
            // BGM/SFXをreconcile（新eventのみSFX、過去snapshotは鳴らさない）
            audio().reconcile({
              runId: j?.game?.run_id ?? null,
              events: j?.game?.events ?? [],
              running: sameRound && j?.phase === "RUNNING" && !(j?.game?.done ?? true),
              fresh: sameRound,
            });
          } else {
            // game状態が取れない — snapshot鮮度切れとしてBGMを止める
            audio().reconcile({
              runId: lastGameRunId.current, events: [],
              running: false, fresh: false,
            });
          }
        } else if (s.status === 404) {
          // サーバー再起動等でsessionが消えた → 再取得へ（旧操作は再送しない）
          if (spawnAttempt.current?.sessionId === sessionId) clearSpawnAttempt();
          if (runStartPendingSession.current === sessionId) {
            runStartPendingSession.current = null;
            setRunStartPending(false);
          }
          setSessionId("");
          setState(null);
          setReview(null);
          setGame(null);
          lastGameRunId.current = null;
          audio().reconcile({ runId: null, events: [], running: false, fresh: true });
        }
      } else if (Date.now() - lastSessionTry.current > 5000) {
        // セッション未取得時のみ低頻度で再試行（制御POSTの自動再送ではない）
        lastSessionTry.current = Date.now();
        const r = await fetch("/api/sessions", { method: "POST" });
        if (r.ok) {
          const j = await r.json();
          if (generation === sessionGeneration.current) setSessionId(j.session_id);
        }
      }
      // API到達性と業務エラーは別state — ここでは接続可否だけを更新する
      // （reject/聞き返し等の業務messageを次のpollで消さない）
      setApiDown(!l.ok || (s !== null && !s.ok && s.status !== 404));
    } catch {
      setApiDown(true);
      // API断 = snapshot鮮度切れ — BGMは止める（SFX履歴はrun結合で温存）
      audio().reconcile({
        runId: lastGameRunId.current, events: [],
        running: false, fresh: false,
      });
    } finally {
      pollBusy.current = false;
    }
  }, [sessionId, clearSpawnAttempt, reconcileSpawn]);

  useEffect(() => {
    const t = setInterval(poll, 600);
    return () => clearInterval(t);
  }, [poll]);

  // ---- 3D renderer の実接続状態 -------------------------------------------
  // READYは現socketのopen+stable、PM01 mesh nodeとreplay完了の復号、
  // Worker定期ACK、server側の現boot/新鮮なposeを全て要求する。
  // byte数や静止中のWS受信間隔、DOMの「Connected」単独ではreadyにしない。
  const sceneLoadStart = useRef(Date.now());
  // iframe世代token — postMessage照合用（古いiframeのreadyを再利用しない）
  const renderGen = useRef(Math.random().toString(36).slice(2, 12));

  // vendored clientのWorkerがBroadcastChannelで現socket世代の
  // lifecycle・復号済みscene node・ACKを報告する。bytes/scene/liveは診断用。
  // 新接続（begin）・切断・世代変更で stream/安定フラグは即時失効する。
  const connCid = useRef(0);      // 現socket世代（注入clientが採番）
  const sawReport = useRef(false); // 注入client生存の実測（DOM副経路の可否）
  useEffect(() => {
    let bc: BroadcastChannel | null = null;
    try {
      bc = new BroadcastChannel("kotoba-render");
    } catch {
      return; // 非対応ブラウザ → DOM副経路のみ
    }
    bc.onmessage = (ev) => {
      const m = ev.data?.kotobaRender;
      if (!m || m.rg !== renderGen.current) return;
      sawReport.current = true;
      // 新しいsocket世代の開始 — 旧世代の状態を全て失効させる
      if (m.ev === "begin") {
        if (typeof m.cid === "number" && m.cid > connCid.current) {
          connCid.current = m.cid;
          setConn("init");
          setConnDetail("");
          setConnStable(false);
          setStreamAlive(false);
          setSceneBytes(0);
          lastAckAt.current = 0;
          setClientSceneReady(false);
          setClientAckOk(false);
        }
        return;
      }
      // begin以外は現世代のsocketからのみ受理（旧socket遅着を破棄）
      if (typeof m.cid === "number" && m.cid !== connCid.current) return;
      if (m.ev === "open") {
        setConn("open");
        setConnDetail("");
      } else if (m.ev === "stable") {
        // openが3秒継続 — 瞬間openではreadyにしない
        setConnStable(true);
      } else if (m.ev === "close") {
        setConn("closed");
        setConnDetail(`close ${m.code ?? "?"}${m.reason ? ` ${m.reason}` : ""}`);
        setConnStable(false);
        setStreamAlive(false);
        setClientAckOk(false);
      } else if (m.ev === "error") {
        setConn("error");
        setConnDetail("socket error");
        setConnStable(false);
        setClientAckOk(false);
      } else if (m.ev === "progress") {
        // 転送進捗 — 「転送中 X MB」表示用（ready条件ではない）
        setSceneBytes(m.bytes ?? 0);
      } else if (m.ev === "scene") {
        // 固定バイト閾値の到達 — 進捗記録のみ（ready条件にしない）
        setSceneBytes(m.bytes ?? 0);
        setSceneReadyS(m.elapsed_s ?? null);
        console.info(`[render] シーン受信(進捗): ${m.bytes}B ${m.elapsed_s}s cid=${m.cid}`);
      } else if (m.ev === "scene_node") {
        // 復号済みmessage_batchにPM01の実mesh nodeがあった現socketのみ。
        setClientSceneReady(true);
      } else if (m.ev === "live") {
        // 現接続でpose streamが流れている実測（診断 — server正本と併用）
        setStreamAlive(true);
      } else if (m.ev === "ack") {
        // ACKは現socketのWorker生存証拠。静止姿勢ではViserが同値更新を
        // 送らないため、最終WS messageの経過時間をready条件にしない。
        // scene成立は復号済みPM01 nodeで別途確かめる。
        lastAckAt.current = Date.now();
        if (m.scene === true) setClientSceneReady(true);
        setClientAckOk(typeof m.bytes === "number" && m.bytes > 0);
      }
      // retry_drop / retry_backoff / retry_allow は診断用 —
      // 状態は変えない（consoleのみ、秘密情報は含まない）
      else if (typeof m.ev === "string" && m.ev.startsWith("retry")) {
        console.info(`[render] ${m.ev} cid=${m.cid} st=${m.st ?? "-"} wait=${m.wait ?? "-"}`);
      }
    };
    return () => bc?.close();
  }, []);

  // 旧client向けDOM診断。注入clientの報告が届けばDOMはready根拠にしない。
  useEffect(() => {
    const check = () => {
      const doc = renderFrameRef.current?.contentDocument;
      if (!doc) { if (!sawReport.current) { setConn("init"); setStreamAlive(false); setClientSceneReady(false); } return; }
      const txt = doc.body?.innerText ?? "";
      const hasCanvas = doc.querySelectorAll("canvas").length > 0;
      const domConn = hasCanvas && /\bConnected\b/.test(txt);
      // scene treeにPM01のbodyノードが現れれば構築済みの補助証拠。
      // WorldAxes/fixed_bodiesのみの状態は未構築。LINK_BASEはPM01の
      // root body名で、期待モデルが実際に読み込まれた実証にもなる。
      const built = /LINK_BASE|\/bodies\//i.test(txt);
      if (sawReport.current) return; // 接続状態は注入clientの実イベントが正本
      setConn((c) => (c === "init" && domConn ? "open" : c));
      if (built) {
        setClientSceneReady(domConn);
        setStreamAlive(true);
        setSceneReadyS((prev) =>
          prev === null
            ? Math.round(((Date.now() - sceneLoadStart.current) / 1000) * 10) / 10
            : prev,
        );
      }
    };
    const t = setInterval(check, 1000);
    return () => clearInterval(t);
  }, []);

  // R7 1.6: client ACKの鮮度減衰 — worker自体が死ぬとack自体が止まるため、
  // 最終ackから3s超でclient側生存を失効する（server readyのままclientが
  // 止まった反例をreadyにしない）。
  useEffect(() => {
    const t = setInterval(() => {
      if (sawReport.current && Date.now() - lastAckAt.current > 3000) {
        setClientAckOk(false);
      }
    }, 1000);
    return () => clearInterval(t);
  }, []);

  // 派生phase — 現socketのsemantic scene＋ACKとserver側の現boot・
  // 新鮮なposeを別々に照合。bytes/scene/liveは進捗診断だけに使う。
  //   disconnected: WS close/errorを検出（自動再接続はclientが可視時に継続）
  //   connecting:   WS未確立 or open直後（3s安定待ち）
  //   loading:      WS open安定・serverがmodel/mesh読込中
  //   syncing:      serverが現boot poseへ追従中 / status未到達
  //   ready/stale:  server stateとboot一致を反映
  const renderer: RendererPhase = (() => {
    const st = renderStatus;
    if (conn === "closed" || conn === "error") return "disconnected";
    if (conn !== "open") {
      // WS未確立。server側がloading_modelならその表示、
      // API観測自体が死んでいれば error、それ以外は transport 接続待ち。
      if (st?.state === "loading_model") return "loading";
      return statusFail || (st && !st.reachable) ? "error" : "connecting";
    }
    if (!connStable) return "connecting"; // open直後 — 安定化中
    if (!st || !st.reachable) return "syncing"; // server status未到達/鮮度切れ
    if (st.state === "ready" && st.boot_match) {
      // pose反映の鮮度 — 古いposeをreadyと偽らない（更新が止まればsyncing）
      const poseFresh =
        typeof st.pose_age_s === "number" ? st.pose_age_s < 2.0 : true;
      if (!poseFresh) return "syncing";
      // PM01 mesh nodeを現socketのWorkerが復号したことを確認。50MB固定閾値は
      // シーン圧縮率で変わるためreadyの証拠にしない。
      if (!clientSceneReady) return "syncing";
      // 注入clientが稼働中なら現socketの定期ACKも必須。
      // ACK自体が止まれば3秒以内にreadyを失効する。
      if (sawReport.current && !clientAckOk) return "syncing";
      return "ready";
    }
    if (st.state === "stale" || (st.state === "ready" && !st.boot_match))
      return "stale"; // 旧boot姿勢を新bootとして出さない
    if (st.state === "loading_model") return "loading";
    return "syncing";
  })();

  // 無限同期をしない: 準備未完が30sを越えたら未充足条件、60sで明示失敗＋再接続操作。
  const syncElapsed = (Date.now() - sceneLoadStart.current) / 1000;
  const rendererTimedOut = renderer !== "ready" && syncElapsed > 60;
  const retryRender = () => {
    // iframeを新世代で再mount（旧接続のscene/pose/ACKを失効）
    // rg変更で旧iframeの報告は照合落ち、connCidも初期化して
    // 旧socket世代の遅着イベントを拒否する。
    renderGen.current = Math.random().toString(36).slice(2, 12);
    connCid.current = 0;
    sawReport.current = false;
    sceneLoadStart.current = Date.now();
    setConn("init");
    setConnDetail("");
    setConnStable(false);
    setStreamAlive(false);
    setSceneBytes(0);
    setSceneReadyS(null);
    setClientSceneReady(false);
    setRenderTick((n) => n + 1);
  };

  // ---- 運営補助の2D map（details畳込み。主画面は3DのPM01） ------------------
  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d")!;
    const w = c.width, h = c.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#101418";
    ctx.fillRect(0, 0, w, h);
    if (live.pos && anchorRef.current === null) {
      anchorRef.current = { x: live.pos[0], y: live.pos[1] };
    }
    if (!live.pos) {
      ctx.fillStyle = "#5a6472";
      ctx.font = "16px sans-serif";
      ctx.fillText("観測待機中…", 20, 40);
      return;
    }
    const anchor = anchorRef.current ?? { x: live.pos[0], y: live.pos[1] };
    const scale = 90;
    const cx = w / 2, cy = h - 60;
    const px = (wx: number) => cx + (wx - anchor.x) * scale;
    const py = (wy: number) => cy - (wy - anchor.y) * scale;
    trailRef.current.push({ x: live.pos[0], y: live.pos[1] });
    if (trailRef.current.length > 600) trailRef.current.shift();
    ctx.strokeStyle = "#3d7bd9"; ctx.lineWidth = 2; ctx.beginPath();
    trailRef.current.forEach((p, i) => {
      const X = px(p.x), Y = py(p.y);
      if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
    });
    ctx.stroke();
    const t = live.target;
    if (t) {
      ctx.fillStyle = "#d9a53d";
      ctx.beginPath(); ctx.arc(px(t[0]), py(t[1]), 10, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#d9a53d"; ctx.font = "13px sans-serif";
      ctx.fillText("目的地", px(t[0]) + 14, py(t[1]) + 4);
    }
    ctx.fillStyle = "#69c46d";
    ctx.beginPath(); ctx.arc(px(live.pos[0]), py(live.pos[1]), 9, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#e8edf2"; ctx.font = "13px sans-serif";
    ctx.fillText("ロボット", px(live.pos[0]) + 13, py(live.pos[1]) - 8);
    ctx.strokeStyle = "#2a323c"; ctx.beginPath();
    for (let m = 0; m <= 2; m += 0.5) {
      ctx.moveTo(cx + m * scale, cy); ctx.lineTo(cx + m * scale, cy + 6);
    }
    ctx.stroke();
  }, [live]);

  // POST失敗の表示 — server detailを落とさず、api.log対応付け用の
  // X-Request-Idを併記する（409等の実原因を利用者/運営が追えるように）。
  const postError = (r: Response, j: any) => {
    const rid = r.headers.get("x-request-id");
    const detail = j?.detail;
    setError(
      (detail === "read_only_preview"
        ? "閲覧専用プレビューのため操作できません"
        : detail === "local_operator_required"
          ? "この端末での操作権がありません — 上の「この端末で操作を開始」を押してください"
          : detail === "operator_required"
            ? "操作者権限が必要です（この端末での操作開始を確認してください）"
            : typeof detail === "string"
              ? detail
              : `エラー(${r.status})`) + (rid ? ` ［req ${rid}］` : ""),
    );
  };

  // R7 1.5/2.1: local entryの明示操作でoperator sessionを発行/終了する。
  // sessionはserver側所有・cookieはHttpOnly — 秘密値はUIへ現れない。
  const localSessionStart = async () => {
    const r = await fetch("/local/session/start", { method: "POST" });
    setLocalOperator(r.ok);
    if (!r.ok) setError("この端末の操作権を取得できませんでした");
  };
  const localSessionEnd = async () => {
    await fetch("/local/session/end", { method: "POST" }).catch(() => {});
    setLocalOperator(false);
  };

  const post = async (url: string, body: any) => {
    setBusy(true);
    try {
      const r = await fetch(url, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, ...body }),
      });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) { postError(r, j); return null; }
      return j;
    } finally { setBusy(false); }
  };

  // busyを立てないPOST — ゲーム指令専用。解釈待ち中も停止系を
  // 塞がないため、busy表示・busyガードの対象外にする。
  const postQuiet = async (url: string, body: any) => {
    const r = await fetch(url, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, ...body }),
    });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) { postError(r, j); return null; }
    return j;
  };

  // 値を直接受け取る（clarify回答・chip入力をsetText経由で送らない — 旧state事故防止）
  const submitIntent = async (
    value?: string,
    origin: CmdOrigin = "text",
    epoch = 0,
  ) => {
    const t = (value ?? text).trim();
    if (!t || busy || composing || spawnAttempt.current || spawnGenerationChanged) return;
    if (origin === "voice" && voiceEngine === "local-asr") {
      if (!asrRef.current?.isArmed() || asrRef.current.epochNow() !== epoch)
        return; // 遅着ASRの再送防止（R7 1.4）
    }
    // このintentが生んだrunの由来を記録（approveAndRunで所有権へ移す）
    intentOriginRef.current = { origin, epoch };
    setError(""); setReview(null);
    const j = await post("/api/intents", { text: t });
    if (!j) return;
    if (j.decision === "execute") setReview(j);
    if (j.decision === "busy" || j.decision === "error") setError(j.message);
    setText("");
    setClarifyAnswer("");
  };

  const approveAndRun = async () => {
    if (!review || !state?.pending_plan_id) return;
    setBusy(true);
    try {
      const a = await post(`/api/plans/${state.pending_plan_id}/approve`, { plan_id: state.pending_plan_id });
      if (!a?.approved) { setError(`承認できませんでした: ${a?.reason ?? ""}`); return; }
      const r = await post("/api/runs", { plan_id: state.pending_plan_id, approval_id: a.approval_id });
      if (!r?.run_id) { setError(`実行を開始できませんでした: ${r?.reason ?? ""}`); return; }
      runStartPendingSession.current = sessionId;
      setRunStartPending(true);
      // 実行開始されたrunの由来を所有権へ移す（R7 1.4 — 音声由来のrunは
      // 音声健全性をheartbeat条件へ含める）
      jogOwnerRef.current = intentOriginRef.current;
      trailRef.current = [];
      anchorRef.current = null;
      setReview(null);
    } finally { setBusy(false); }
  };

  const clearRoundView = () => {
    setText(""); setReview(null); setClarifyAnswer("");
    setGameText(""); setLastCmd(""); setGame(null);
    trailRef.current = [];
    anchorRef.current = null;
  };
  const resetRound = async (): Promise<boolean> => {
    // 再挑戦: 新roundの受理後にローカル表示を初期化する。
    let j: { reset?: boolean } | null;
    try {
      j = await post("/api/round/reset", {});
    } catch {
      // 直接「再挑戦」を押した場合も未処理のPromise拒否にしない。
      // serverが適用済みかは不明なので、ここで自動再POSTしない。
      setError("初期化の応答が途切れました。状態を確認してください");
      return false;
    }
    if (!j?.reset) {
      if (j) setError("ラウンドを初期化できませんでした");
      return false;
    }
    // reset前に開始したpollの旧round snapshotを描画へ戻さない。
    sessionGeneration.current += 1;
    clearSpawnAttempt();
    setState(null);
    clearRoundView();
    setSpawnGenerationChanged(false);
    modeReturnError.current = null;
    setError("");
    return true;
  };

  // ---- ことばでスイカ割り -------------------------------------------------
  // spawn(operator): seed省略でサーバー採番。生成後specはラウンド中不変。
  // start(operator): spawn済みspecにcontrollerを起動（renderer ready必須）。
  const startGame = async () => {
    // 開始単一飛行 — 二重POSTをUI層で防ぐ。クリックの同期区間で
    // flagを立て、awaitの前に必ず確定させる（再入不可）。
    // gameStarting（POST受理〜controller実起動の区間）もガードに含める —
    // POST完了後・実開始前の逐次クリックも新たなstartを撃たない。
    if (startInFlight.current || gameStarting || busy ||
        spawnAttempt.current || spawnGenerationChanged) return;
    startInFlight.current = true;
    setGameStarting(true); // POST受理〜controller実起動の「起動中」区間
    modeReturnError.current = null;
    setError("");
    try {
      // AudioContextのresumeは操作の同期区間で行う（autoplay許可）が、
      // 音源の取得・decodeは開始を塞がない — 別々の非直列処理。
      // 拒否/失敗はoutput/assetsの別stateとして表示される。
      void audio().resumeFromGesture().then(setAudioOutput);
      void audio().preloadAssets().then(setAudioAssets);
      const st = await post("/api/game/start", {});
      // post()が失敗時にserver detail（run_in_progress等）を表示済み。
      // 成功200でも started:false は理由をそのまま出す（汎用文で潰さない）
      if (!st) { setGameStarting(false); return; }
      if (!st.started) {
        setError(`開始できませんでした（${st.reason ?? "unknown"}）`);
        setGameStarting(false);
        return;
      }
      trailRef.current = [];
      setLastCmd("");
      // 「起動中」はcontroller_alive/round_started観測で解消する —
      // ここでは終わらせない（受理≠実起動を区別する）
    } finally {
      startInFlight.current = false;
    }
  };

  const toggleMute = () => {
    const a = audio();
    const m = !audioMuted;
    a.setMuted(m);
    setAudioMuted(m);
  };

  // 参加者指令: ことば → 有界semantic指令（latest-winsで差し替え）。
  // 解釈不能ならサーバーmessageをそのまま表示（聞き返し）。
  // busyを立てない・見ない: 解釈待ち（LLM経路は最大数十秒）の間も
  // 「止まって」「終わって」「中断」は常に送れる必要がある。
  // 新旧の仲裁はサーバー側の受理順（input_seq/epoch）が行う。
  const sendGameCmd = async (
    value?: string,
    origin: CmdOrigin = "text",
    epoch = 0,
  ) => {
    const t = (value ?? gameText).trim();
    if (!t || composing) return;
    // 遅着ASRの再送防止（R7 1.4）— commit時epochと現在が一致しない
    // voice指令は送らない（STOP/disarm後の古い発話で動作を再開しない）。
    if (origin === "voice" && voiceEngine === "local-asr") {
      if (!asrRef.current?.isArmed() || asrRef.current.epochNow() !== epoch)
        return;
    }
    setError("");
    const j = await postQuiet("/api/game/command", { text: t });
    if (!j) return;
    if (j.accepted) {
      // 受理された指令の由来を所有権として記録 — heartbeat条件の判定に使う
      jogOwnerRef.current = { origin, epoch };
      setLastCmd(j.via === "llm" ? `${j.label || t}（AI解釈）` : j.label || t);
      setLastCmdSeq(typeof j.seq === "number" ? j.seq : null);
      setGameText("");
      setVoiceInterim("");
    } else if (j.message) {
      setError(j.message);
    }
  };

  // 常設停止 — IME変換中・busy・LLM待ち・renderer状態に関わらず届ける。
  // owner別dispatch: ゲームroundは最新勝ち指令「止まって」、通常runは
  // 正常停止経路（run_cmd.json → runnerが減速→立位保持→正常終了）へ。
  // 管理中断の /api/run/pause（runner強制終了）とは別のボタンとして分離。
  const stopNow = async () => {
    setError("");
    if (gameMode) {
      await postQuiet("/api/game/command", { text: "止まって" });
    } else {
      await postQuiet("/api/run/command", { type: "stop" });
    }
  };

  // ---- 音声アダプタ（R5-05 — 明示arm・号令即commit・STOP先行） ----
  // 経路はテキストと同一dispatch（/api/game/command・/api/intents）へ入る。
  // 既定はon-device Web Speech — available()の実結果をprobeして表示し、
  // 非対応・pack不在・locality未確認を区別する。クラウドASRへの黙った
  // fallbackはしない。STOPはinterimの時点で送信キューを迂回する。
  const voiceDispatch = useRef<(t: string) => void>(() => {});
  const voiceStopRef = useRef<() => void>(() => {});
  useEffect(() => {
    probeVoice().then(setVoiceProbe);
    // local entry経路かどうか（/local/sessionが応答するか）を判定 —
    // 展示loopback入口ではUIへ「この端末で操作を開始」を出す。
    fetch("/local/session")
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => {
        setLocalEntry(j !== null);
        setLocalOperator(!!j?.operator);
      })
      .catch(() => setLocalEntry(false));
    // 入力デバイス一覧・マイク許可状態（R7 2.3 — 実機のマイク/出力をUIへ出す）
    navigator.mediaDevices?.enumerateDevices?.()
      .then((devs) => setMicDevices(devs.filter((d) => d.kind === "audioinput")))
      .catch(() => {});
    (navigator as any).permissions?.query?.({ name: "microphone" })
      .then((p: any) => {
        setMicPerm(p.state);
        p.onchange = () => setMicPerm(p.state);
      })
      .catch(() => setMicPerm("unsupported-api"));
    // R7 1.2: ASRサービス稼働面を型分け — HTTP status・requestId・
    // engine/model/deviceを保持し、403を「pack不足」へ変換しない。
    const chk = () => {
      fetch("/api/asr/status")
        .then(async (r) => {
          const rid = r.headers.get("x-request-id");
          if (r.status === 403) {
            setAsrService({ ...ASR_INIT, state: "forbidden", httpStatus: 403,
              requestId: rid, detail: "ASR経路がこの接続では許可されていません（操作権限が必要です）" });
            return;
          }
          if (r.status === 401) {
            setAsrService({ ...ASR_INIT, state: "unauthorized", httpStatus: 401,
              requestId: rid, detail: "認証が必要です" });
            return;
          }
          if (!r.ok) {
            setAsrService({ ...ASR_INIT, state: "unreachable", httpStatus: r.status,
              requestId: rid, detail: `ASR状態取得に失敗（HTTP ${r.status}）` });
            return;
          }
          const j = await r.json().catch(() => ({}));
          if (!j.reachable) {
            setAsrService({ ...ASR_INIT, state: "unreachable", httpStatus: 200,
              requestId: rid, engine: j.engine ?? null,
              detail: "ASRサービスへ到達できません（サービス停止中）" });
            return;
          }
          const dev = typeof j.device === "string" ? j.device : null;
          const base = {
            httpStatus: 200, requestId: rid,
            engine: j.engine ?? null, model: j.model ?? null,
            device: dev, deviceName: j.device_name ?? null,
          };
          if (j.error) {
            setAsrService({ ...base, state: "model_error",
              detail: `ASRモデル読込失敗: ${j.error}` });
          } else if (!j.ok) {
            setAsrService({ ...base, state: "model_loading",
              detail: "ASRモデルを読み込んでいます…" });
          } else if (dev === "cpu" || (dev && !dev.startsWith("cuda"))) {
            // 採用GPU構成からCPUへ低下 — 性能状態を明示し、検収済み構成
            // と同じものとして黙って扱わない（R7 1.2）
            setAsrService({ ...base, state: "degraded_cpu",
              detail: "ASRがCPUで動作中（GPU構成ではありません — 応答が遅くなります）" });
          } else {
            setAsrService({ ...base, state: "ready", detail: "" });
          }
        })
        .catch(() =>
          setAsrService({ ...ASR_INIT, state: "unreachable",
            detail: "ASR状態取得に失敗（ネットワーク）" }));
    };
    chk();
    const t = setInterval(chk, 15000);
    return () => clearInterval(t);
  }, []);
  // R7 1.2: LOCAL_ASR_ONLY — Web Speechを自動選択しない。ASR経路が
  // arm不能なら実際の原因（403/503/model_loading等）をそのまま表示する。
  const asrArmable =
    asrService.state === "ready" || asrService.state === "degraded_cpu";

  // R7 1.4: 音声故障の共通処理 — voice由来の継続動作が残っているなら
  // 正常STOPを優先送信し、voice所有権を失効させる。再接続後に自動
  // re-armしない（利用者の明示操作のみ）。text/button由来の所有権には
  // 触れない（音声故障を口実に無関係な所有者へ引き継がない）。
  const onVoiceFailed = (detail?: string) => {
    setVoiceArmed(false);
    setVoiceEngine(null);
    setError(`音声認識が停止しました（${detail ?? "error"}）`);
    if (jogOwnerRef.current?.origin === "voice") {
      if (jogActiveRef.current) void stopNow();
      jogOwnerRef.current = null;
    }
  };

  const startVoice = () => {
    if (voiceArmed || readOnly || !sessionId) return;
    // LOCAL_ASR_ONLY: ローカルASR（Qwen3-ASR・Thor内完結）のみ。
    // 到達不能・権限不足・モデル未完のときはWeb Speechへ流さず、
    // 実際の原因を表示する（音声が外部へ送られる経路を作らない）。
    if (asrArmable) {
      if (!asrRef.current) {
        asrRef.current = new LocalAsrController({
          onCommit: (t, timing) => {
            if (timing) setLastAsrTiming(timing);
            voiceDispatch.current(t);
          },
          onStop: () => voiceStopRef.current(),
          onInterim: (t) => setVoiceInterim(t),
          onSpeech: (a) => audio()?.setSpeechActive(a),
          onInputInfo: (info) => setAsrInputInfo(info),
          onLevel: (rms) => setMicLevel(rms),
          onHealth: (h, detail) => {
            setVoiceHealth(h);
            if (h === "failed") onVoiceFailed(detail);
          },
        });
      }
      voiceRunRef.current = game?.game?.run_id ?? null;
      setVoiceEngine("local-asr");
      setVoiceArmed(true);
      audio()?.setVoiceArmed(true);
      void asrRef.current.arm(sessionId, micDeviceId || undefined).then((ok) => {
        if (!ok) {
          setVoiceArmed(false);
          setVoiceEngine(null);
          // arm拒否は実際の原因（mic拒否/worklet不調/権限）を
          // onHealthのfailed detailが既に示している — ASR稼働面を
          // 「使えない」へ下げる誤変換はしない（R7 1.2）。
        }
      });
      return;
    }
    if (LOCAL_ASR_ONLY) return; // ASR不可時の代替経路なし — 原因はUIに表示済み
    if (!voiceProbe?.hasSR) return;
    // 非展示向けのon-device Web Speech経路 — available()で確認できない
    // 限りarmしない（locality未確認の音声を外部ASRへ黙って送らない）。
    if (voiceProbe.locality !== "on-device") return;
    if (!voiceRef.current) {
      voiceRef.current = new VoiceController({
        onCommit: (t) => voiceDispatch.current(t),
        onStop: () => voiceStopRef.current(),
        onInterim: (t) => setVoiceInterim(t),
        onHealth: (h, detail) => {
          setVoiceHealth(h);
          if (h === "failed") onVoiceFailed(detail);
        },
      });
    }
    // arm時のrun ownerを結ぶ — そのrunの終了/失効でdisarmする
    voiceRunRef.current = game?.game?.run_id ?? null;
    voiceRef.current.arm(voiceProbe.locality === "on-device");
    setVoiceEngine("webspeech");
    setVoiceArmed(true);
    audio()?.setVoiceArmed(true);
  };
  const stopVoice = () => {
    voiceRef.current?.disarm();
    asrRef.current?.disarm();
    voiceRunRef.current = null;
    setVoiceEngine(null);
    setVoiceArmed(false);
    setVoiceInterim("");
    audio()?.setVoiceArmed(false);
    audio()?.setSpeechActive(false);
  };
  // ゲーム終了・reset・run owner変更・タブ非表示でdisarm —
  // 古い発話の自動実行を防ぐ（復旧後は明示的再arm）。
  // arm時に結んだrunを追跡し、そのrunが終了/失効したときだけ解く
  // （通常モードのarmはrun非所有のためゲーム状態では解かない）。
  const gameRunDone = game?.game?.done ?? false;
  const gameRunIdNow = game?.game?.run_id ?? null;
  useEffect(() => {
    if (!voiceArmed) return;
    const bound = voiceRunRef.current;
    if (bound && (gameRunIdNow !== bound || gameRunDone)) stopVoice();
  }, [voiceArmed, gameRunIdNow, gameRunDone]);
  useEffect(() => {
    // session自体の失効（reset/再起動）はどちらのmodeでもdisarm
    if (!voiceArmed) return;
    if (!sessionId) stopVoice();
  }, [voiceArmed, sessionId]);
  useEffect(() => {
    const onHide = () => { if (document.hidden) stopVoice(); };
    document.addEventListener("visibilitychange", onHide);
    window.addEventListener("pagehide", onHide);
    return () => {
      document.removeEventListener("visibilitychange", onHide);
      window.removeEventListener("pagehide", onHide);
    };
  }, []);
  const audio = () => (audioRef.current ??= new AudioManager());
  // 「起動中」区間の解消 — round_startedイベントの実観測で解除する
  // （controller_aliveはcontainer生存だけを示す。実開始の証拠は
  // round_started。位相がRUNNINGを離れたら失敗として解除）。
  const controllerAlive = game?.game?.controller_alive ?? false;
  const roundStartedSeen = (game?.game?.events ?? []).some(
    (e) => e.kind === "round_started",
  );
  useEffect(() => {
    if (!gameStarting) return;
    if (roundStartedSeen || controllerAlive) setGameStarting(false);
  }, [gameStarting, roundStartedSeen, controllerAlive]);
  useEffect(() => {
    // 失敗時のみ解除 — POST受理直後〜poll反映まで位相はまだREADYの
    // ままなので、RUNNING以外への一般遷移では解除しない（誤解除で
    // ボタンが再有効化され逐次クリックがPOSTを撃つ）。
    // 終端・fault・初期化中への遷移 = controller起動失敗の実証のみ解除。
    const p = state?.phase ?? "READY";
    if (gameStarting && (p === "RESULT" || p === "FAULT" || p === "RESETTING")) {
      setGameStarting(false);
    }
  }, [gameStarting, state?.phase]);
  // assets/outputのstate変化を表示へ反映（reconcile内の遅延状態変化も拾う）
  useEffect(() => {
    const t = setInterval(() => {
      const a = audioRef.current;
      if (!a) return;
      setAudioAssets((p) => (p === a.assetsState ? p : a.assetsState));
      setAudioOutput((p) => (p === a.outputState ? p : a.outputState));
    }, 800);
    return () => clearInterval(t);
  }, []);

  const phase = state?.phase ?? "READY";
  const result = state?.result;
  // 聞き返し・未承認planからの離脱は新しいserver sessionで始める。
  // 旧sessionの文脈を持ち込まず、simの再起動も起こさない。
  const newSession = async (): Promise<string | null> => {
    const r = await fetch("/api/sessions", { method: "POST" });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) { postError(r, j); return null; }
    if (typeof j.session_id !== "string" || !j.session_id) {
      setError("新しいラウンドを準備できませんでした");
      return null;
    }
    stopVoice();
    sessionGeneration.current += 1;
    clearSpawnAttempt();
    setSpawnGenerationChanged(false);
    setSessionId(j.session_id);
    setState(null);
    setGame(null);
    setReview(null);
    setText("");
    setClarifyAnswer("");
    setGameText("");
    setLastCmd("");
    setLastCmdSeq(null);
    setError("");
    trailRef.current = [];
    anchorRef.current = null;
    lastGameRunId.current = null;
    audioRef.current?.reconcile({ runId: null, events: [], running: false, fresh: true });
    return j.session_id;
  };
  const navigationBlocked = busy || phase === "RUNNING" || phase === "RESETTING" ||
    phase === "INTERPRETING" || gameStarting || spawnPending || runStartPending ||
    !!game?.startable?.reasons.includes("run_in_progress");
  const spawnUnresolved = spawnPending || spawnRetryAvailable;
  const diagnosticsBlocked = busy || phase === "RUNNING" || phase === "RESETTING" ||
    gameStarting || runStartPending;
  const newInstruction = async () => {
    if (navigationInFlight.current || navigationBlocked ||
        spawnAttempt.current || spawnGenerationChanged || readOnly) return;
    navigationInFlight.current = true;
    stopVoice();
    setBusy(true);
    try { await newSession(); }
    catch { setError("新しいラウンドを準備できませんでした"); }
    finally { navigationInFlight.current = false; setBusy(false); }
  };
  const switchToGame = async () => {
    if (gameMode || navigationInFlight.current || navigationBlocked || readOnly) return;
    navigationInFlight.current = true;
    modeReturnError.current = null;
    setError("");
    // awaitより前に旧ASR epochを失効させ、遅着音声を新modeへ送らない。
    stopVoice();
    setBusy(true);
    try {
      // READYなら同じroundを使い、過去のgame specを持つsessionを増やさない。
      // CLARIFY/REVIEW等では文脈を捨てた新sessionから生成する。
      const sid = phase === "READY" && sessionId && !state?.pending_plan_id
        ? sessionId : await newSession();
      if (!sid) return;
      const before = await readSpawnSnapshot<StateDto, GameStateDto>(fetch, sid);
      if (!before || before.state.session_id !== sid || !before.state.round_id ||
          before.state.round_id !== before.game.round_id ||
          before.live.boot_nonce == null) {
        setSpawnMessage("ラウンドの状態を確認できません。接続が戻ったらスイカ割りを押してください");
        return;
      }
      if (before.game.spec) {
        if (before.game.spec.round_id !== before.state.round_id ||
            before.game.spec_boot_nonce == null ||
            String(before.game.spec_boot_nonce) !== String(before.live.boot_nonce)) {
          clearSpawnAttempt();
          setSpawnGenerationChanged(true);
          setSpawnMessage("スイカの生成世代が現在の3Dと一致しません。通常モードに戻して初期化してください");
          return;
        }
        setSpawnGenerationChanged(false);
        setState(before.state);
        setGame(before.game);
        clearSpawnAttempt();
        return;
      }
      if (before.state.phase !== "READY" || before.game.phase !== "READY") {
        setSpawnMessage("ラウンドの準備が終わってから、もう一度スイカ割りを押してください");
        return;
      }
      const attempt: SpawnAttempt = {
        sessionId: sid, roundId: before.state.round_id,
        bootNonce: String(before.live.boot_nonce),
        requestedAt: Date.now(), absentReads: 0, requestSettled: false,
        retryReady: false,
      };
      spawnAttempt.current = attempt;
      setSpawnGenerationChanged(false);
      setSpawnPending(true);
      setSpawnRetryAvailable(false);
      setSpawnMessage("スイカの生成結果を確認中です…");
      const controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), 8000);
      let response: Response | null = null;
      try {
        response = await fetch("/api/round/spawn", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify(spawnRequestPayload(attempt)),
          signal: controller.signal,
        });
      } catch {
        // 通信断やタイムアウトでもserver側で適用済みの可能性がある。
      } finally {
        window.clearTimeout(timeout);
      }
      if (spawnAttempt.current !== attempt) return;
      attempt.requestSettled = true;
      // GET自体が応答しなくても画面操作を永久に塞がない。再クリック時は
      // 必ずGETで同世代を読み直してからPOST可否を決める。
      window.setTimeout(() => {
        if (spawnAttempt.current !== attempt || attempt.retryReady) return;
        attempt.retryReady = true;
        setSpawnPending(false);
        setSpawnRetryAvailable(true);
        setSpawnMessage("生成状態を取得できません。接続が戻ったら下のボタンで再確認してください");
      }, 12000);
      const after = await readSpawnSnapshot<StateDto, GameStateDto>(fetch, sid);
      if (spawnAttempt.current !== attempt) return;
      if (after) reconcileSpawn(attempt, after);
      if (spawnAttempt.current !== attempt) return;
      if (response && response.status < 500 && response.status !== 409 && !response.ok) {
        clearSpawnAttempt();
        postError(response, await response.json().catch(() => ({})));
      } else {
        // 409は既存specとの競合も含む。再POSTせず同世代のGETを続ける。
        setSpawnMessage("生成結果を照合中です。状態が確定するまでお待ちください");
      }
    } catch { setSpawnMessage("状態を確認できません。接続が戻ったらスイカ割りを押してください"); }
    finally { navigationInFlight.current = false; setBusy(false); }
  };
  const switchToNormal = async () => {
    if (!gameMode || navigationInFlight.current || navigationBlocked || readOnly) return;
    navigationInFlight.current = true;
    stopVoice();
    modeReturnError.current = null;
    setError("");
    const sid = sessionId;
    const generation = sessionGeneration.current;
    try {
      // 5xxや通信断ではPOSTが適用済みかもしれない。自動再送せずGETで確認する。
      try {
        if (await resetRound()) return;
      } catch {
        // HTTP応答前に接続が切れた場合も、同じsessionの観測で結果を判定する。
      }
      const observed = await inspectModeReturn<StateDto, GameStateDto>(fetch, sid);
      if (generation !== sessionGeneration.current) return;
      if (observed.kind === "normal") {
        sessionGeneration.current += 1;
        setState(observed.state);
        clearRoundView();
        setError("");
      } else if (observed.kind === "pending") {
        if (observed.state.phase === "RESETTING") {
          setState(observed.state);
          setGame(observed.game);
        }
        const notice = observed.state.phase === "RESETTING"
          ? "初期化中です。完了後に通常モードの状態を再確認します"
          : "ラウンド状態を照合中です。画面が更新されるまでお待ちください";
        modeReturnError.current = notice;
        setError(notice);
      } else if (observed.kind === "game") {
        setState(observed.state);
        setGame(observed.game);
        const notice = "通常モードへ戻れませんでした。上の「通常」をもう一度押してください";
        modeReturnError.current = notice;
        setError(notice);
      } else {
        const notice = "切替結果を確認できません。状態が表示されたら上の「通常」を再試行してください";
        modeReturnError.current = notice;
        setError(notice);
      }
    } finally { navigationInFlight.current = false; }
  };
  const speed = live.vel ? Math.hypot(live.vel[0], live.vel[1]) : 0;
  const task = (live.task || "").replace(/\0/g, "");

  // 実行中の細かい表示: 実観測のtask/速度から現在の動作を説明する
  const runningLabel =
    task.includes("walk") || speed > 0.08
      ? "歩いています"
      : speed > 0.01
        ? "停止位置を調整中"
        : "止まれたか確認中";

  const verdictText = (v?: string) =>
    v === "PASS" ? "成功!"
    : (v || "").includes("ABORT") || (v || "").includes("PAUSE") ? "中断しました"
    : `惜しい（${v || "判定なし"}）`;

  // 動作プログラムの実行状態を人が読める語へ（認識/送信/適用/実動作の区別）
  const motionStatusText = (s?: string) =>
    s === "accepted" ? "受理"
    : s === "applying" ? "送信開始"
    : s === "moving" ? "移動中（実測）"
    : s === "turning" ? "旋回中（実測）"
    : s === "jogging" ? "継続移動中（「止まって」で停止）"
    : s === "jog_stopping" ? "減速・静止確認中"
    : s === "settling" ? "静止確認中"
    : s === "completed" ? "完了"
    : s === "failed" ? "失敗"
    : s === "aborted" ? "中断"
    : s ?? "—";

  // jog/中断の終了理由を人が読める語へ（境界停止を距離達成と偽らない）
  const reasonText = (r?: string | null) =>
    r === "stop" ? "停止指示"
    : r === "boundary" ? "操縦範囲の境界"
    : r === "heartbeat_lost" ? "操作通信の途絶"
    : r === "jog_timeout" ? "継続時間の上限"
    : r === "superseded" ? "新しい指示へ差し替え"
    : r === "control_lost" ? "指令経路の消失"
    : r === "end" ? "終了指示"
    : r ?? "";

  // ---- ゲームモード派生値 --------------------------------------------------
  // specが立つとこのroundはスイカ割り（旧marker回帰flowは別round専用）。
  const gameMode = !!game?.spec;
  const gamePrestart = gameMode && !["RUNNING", "RESULT", "RESETTING", "FAULT"].includes(phase);
  const gameStartBlockers: string[] = [];
  if (gamePrestart) {
    if (spawnUnresolved) gameStartBlockers.push("ラウンドの生成結果を照合中です");
    if (spawnGenerationChanged) gameStartBlockers.push("起動世代が変わりました。通常モードに戻して初期化してください");
    if (readOnly) gameStartBlockers.push("閲覧専用です");
    if (localEntry === null) gameStartBlockers.push("この端末の操作権を確認中です");
    if (localEntry === true && !localOperator) gameStartBlockers.push("上部の「この端末で操作を開始」を押してください");
    if (renderer !== "ready") gameStartBlockers.push("3D表示の接続を待っています");
    if (!game?.startable) gameStartBlockers.push("開始条件を確認中です");
    const reasons = new Set(game?.startable?.reasons ?? []);
    if (reasons.has("sim_not_ready")) gameStartBlockers.push("シミュレーションの起動を待っています");
    if (reasons.has("reset_in_progress")) gameStartBlockers.push("初期化の完了を待っています");
    if (reasons.has("run_in_progress")) gameStartBlockers.push("実行中のラウンドがあります");
    if (reasons.has("no_round_spawned")) gameStartBlockers.push("ラウンド生成を待っています");
    if (reasons.has("offline_mode")) gameStartBlockers.push("実行モードの受入が必要です");
    if (game?.startable && !game.startable.ok && gameStartBlockers.length === 0)
      gameStartBlockers.push("開始条件を確認してください");
  }
  const gameStartDisabled = busy || gameStarting || spawnUnresolved ||
    spawnGenerationChanged || gameStartBlockers.length > 0;
  const gr = game?.game;
  // 残り振数はserverが実消費（strikes_started）から算出した正本を使う
  const swingsLeft = gr?.swings_left ?? null;
  const strikePhase = gr?.strike_phase;
  const strikeActive =
    !!gr?.strike_pending ||
    (strikePhase != null && ["pre", "dance", "swing", "recover"].includes(strikePhase));
  const outcome = gr?.outcome;
  const outcomeOk = outcome === "hit";

  // 実行状態の段階表示用の派生値（HTTP受理≠適用≠実移動を区別）
  const cs = gr?.command_state ?? null;
  const csSeq = typeof cs?.seq === "number" ? cs.seq : null;
  const csStatus = cs?.status || "";
  const csActive =
    csSeq !== null && csSeq === lastCmdSeq &&
    ["applying", "moving", "turning", "settling", "jogging", "jog_stopping"]
      .includes(csStatus);
  const csDone = csSeq !== null && csSeq === lastCmdSeq &&
    csStatus === "completed";
  const csFailed = csSeq !== null && csSeq === lastCmdSeq &&
    (csStatus === "failed" || csStatus === "aborted");

  // 継続jogが有効な間だけ生存確認を送る — 欠落するとrunner/controllerが
  // 減速停止する（heartbeat_lost）。UIを閉じた・通信が切れた＝操作者消失。
  const jogActive =
    (gameMode &&
      !!cs &&
      (cs.action_key || "").startsWith("jog") &&
      ["applying", "jogging", "jog_stopping"].includes(csStatus)) ||
    (!gameMode &&
      !!state?.motion &&
      (state.motion.action_key || "").startsWith("jog") &&
      ["applying", "jogging", "jog_stopping"].includes(
        state.motion.status || "",
      ));
  // 音声経路のdispatchはテキストと同一経路（別経路を作らない）。
  // VoiceControllerのhandlerから最新closureを参照するためのref。
  // voice_epoch結合（R7 1.4）— commitからdispatch間にdisarm/故障した
  // 遅着ASR結果を指令化しない。epochが進んでいれば破棄する。
  voiceDispatch.current = (t: string) => {
    const ep =
      voiceEngine === "local-asr" ? asrRef.current?.epochNow() ?? -1 : 0;
    if (gameMode) void sendGameCmd(t, "voice", ep);
    else void submitIntent(t, "voice", ep);
  };
  voiceStopRef.current = () => { void stopNow(); };
  jogActiveRef.current = jogActive; // 非effectコールバック用の同期mirror

  useEffect(() => {
    if (!jogActive || readOnly) return;
    // R7 1.4: voice由来の継続動作は、mic/ASRの健全性をlease条件に含める。
    // 実行中commandに結んだ由来（input_origin）で判定する — UIのarm
    // 状態だけでは由来を失う（失敗時 disarm しても判定は残る）。
    // listening/speech/transcribing/starting は健全（発話・認識中も
    // 操作は生きている）。text/button由来は回線健全だけで延命する。
    const owner = jogOwnerRef.current;
    if (owner?.origin === "voice") {
      const healthy =
        voiceArmed &&
        (voiceHealth === "starting" ||
          voiceHealth === "listening" ||
          voiceHealth === "speech" ||
          voiceHealth === "transcribing") &&
        (voiceEngine !== "local-asr" ||
          asrRef.current?.epochNow() === owner.epoch);
      if (!healthy) return; // heartbeat_lostでserver側が減速停止する
    }
    const path = gameMode ? "/api/game/heartbeat" : "/api/run/heartbeat";
    let cancelled = false;
    const beat = () => {
      if (cancelled) return;
      fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      }).catch(() => {});
    };
    beat();
    const t = setInterval(beat, 500);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [jogActive, gameMode, readOnly, sessionId, voiceArmed, voiceHealth, voiceEngine]);
  const appliedSeen =
    lastCmdSeq !== null &&
    ((gr?.applied_seq ?? 0) >= lastCmdSeq || csSeq === lastCmdSeq);

  return (
    <div className="app">
      <div className="stage">
      <div className="hud-top">
        <h1>ことばでロボコン</h1>
        <div className="badges">
          <span className={"badge live " + (state?.live_enabled ? "" : "warn")}>
            {state?.live_enabled
              ? "LIVE（実機シミュレーション）"
              : state?.selftest
                ? "SELFTEST（運営検証モード）"
                : "観察モード（実行は未受入）"}
          </span>
          <span className={"badge conn " + (apiDown ? "bad" : "ok")}>
            {apiDown ? "接続断" : "接続OK"}
          </span>
          <span className={"badge obs obs-age " + (live.fresh ? "ok" : "warn")}>
            観測: {live.fresh ? `新鮮 (${(live.age_s ?? 0).toFixed(2)}s)` : "停止中"}
          </span>
          <span className={"badge obs " + (renderer === "ready" ? "ok" : "warn")}
            title={sceneReadyS !== null ? `初回シーン受信・構築: ${sceneReadyS}s` : ""}>
            3D: {renderer === "ready" ? "描画中"
              : renderer === "disconnected" ? "接続が切れています"
              : renderer === "stale" ? "古い姿勢"
              : renderer === "syncing" ? "姿勢同期中"
              : renderer === "loading" ? "モデル読込中"
              : renderer === "error" ? "未接続"
              : "接続中"}
          </span>
          {meta?.mode === "read-only" && <span className="badge live warn">閲覧専用プレビュー</span>}
          {meta?.mode === "operator" && <span className="badge live">操作者: {meta.email}</span>}
          {/* R7 2.1: local entry経由時のoperator session — 明示操作でのみ発行。
              sessionなしのUI活性と操作権の有無を分離して表示する */}
          {localEntry === true && (
            localOperator ? (
              <button className="badge live ok" onClick={localSessionEnd}
                data-action-id="local-session-end"
                title="この端末の操作権を終了します">
                操作者: この端末
              </button>
            ) : (
              <button className="badge live warn" onClick={localSessionStart}
                data-action-id="local-session-start"
                title="この端末で操作を開始します（loopback限定・明示操作のみ）">
                この端末で操作を開始
              </button>
            )
          )}
        </div>
      </div>

          <iframe
            ref={renderFrameRef}
            key={renderTick}
            className="render3d"
            // viser clientは window.location からWS URLを導出するが、assetsの
            // 正規化で /render/ へ落ちると導出先が /render になり edge の307で
            // WSが死ぬ。?websocket= で接続先を実証済みの /render/ へ固定する。
            src={`/render/index.html?websocket=${encodeURIComponent(
              `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/render/`,
            )}&rg=${renderGen.current}`}
            title="PM01 3D表示（実観測の姿勢・関節）"
            allow="fullscreen"
          />
          {renderer !== "ready" && (
            <div className="map-note warn" style={{ position: "absolute", top: 48, left: 12, zIndex: 25 }}>
              {renderer === "disconnected"
                ? `3D接続が切れています — 再接続中${connDetail ? `（${connDetail}）` : ""}。停止などの操作は使えます。`
                : renderer === "stale"
                  ? "3D表示の姿勢が古いか別ブートのものです。実行はできませんが、中断・初期化などの復旧操作は使えます。"
                  : renderer === "error"
                    ? "3D表示が未接続です（復旧を自動で待っています）。実行はできませんが、中断・初期化などの復旧操作は使えます。"
                    : renderer === "loading"
                      ? "3Dモデルを読み込んでいます…"
                      : "3D表示を準備中… PM01とスイカが実際に同期するまで実行できません。"}
              {syncElapsed > 30 && !rendererTimedOut && (
                <div style={{ fontSize: 12, opacity: 0.85, marginTop: 4 }}>
                  未充足: {conn !== "open"
                    ? "3D通信の確立"
                    : !connStable
                      ? "接続の安定化"
                      : sceneBytes === 0
                        ? "モデルデータの転送開始"
                        : !renderStatus?.reachable
                          ? "描画サーバーの状態報告"
                          : !streamAlive
                            ? "姿勢ストリームの到達"
                            : "現ブート姿勢の反映"}
                  {sceneBytes > 0 ? `（受信 ${(sceneBytes / 1e6).toFixed(1)}MB）` : ""}
                  {connDetail ? ` / ${connDetail}` : ""}
                </div>
              )}
              {rendererTimedOut && (
                <div style={{ marginTop: 6 }}>
                  準備できませんでした。{" "}
                  <button className="example" onClick={retryRender}>3D表示を再接続</button>
                </div>
              )}
            </div>
          )}

        <aside className={"sidebar" + (sidebarOpen ? "" : " folded")}>
          <nav className="sidebar-nav" aria-label="体験と画面の操作">
            <div className="mode-switch" role="group" aria-label="体験モード">
              <button className={!gameMode ? "active" : ""} aria-pressed={!gameMode}
                onClick={switchToNormal} disabled={!gameMode || navigationBlocked || readOnly}
                data-action-id="mode-normal" title={gameMode && navigationBlocked ? "ラウンド終了後に切り替えられます" : "通常モードへ切り替える"}>
                通常
              </button>
              <button className={gameMode ? "active" : ""} aria-pressed={gameMode}
                onClick={switchToGame} disabled={gameMode || navigationBlocked || readOnly}
                data-action-id="mode-game" title={!gameMode && navigationBlocked ? "実行終了後に切り替えられます" : "スイカ割りの新しいラウンドを作る"}>
                スイカ割り
              </button>
            </div>
            <button className="sidebar-diag-link" type="button"
              onClick={() => window.location.assign("/diag")}
              disabled={diagnosticsBlocked}
              aria-label="診断・安全復旧を開く"
              title={diagnosticsBlocked ? "実行・初期化の終了後に開けます" : "診断・安全復旧を開く"}
              data-action-id="open-diagnostics">診断</button>
            <button className="fold-btn" onClick={() => setSidebarOpen(false)}
              aria-label="操作メニューを閉じる" data-action-id="sidebar-fold">
              閉じる
            </button>
          </nav>
          {/* 固定ヘッダ区画（R6-C: スクロールしない状態行 — 位相変化で
              操作部が動かないよう常時同じ高さを維持する） */}
          <div className="sidebar-head">
            {gameMode ? (
              <>
                <div className="head-line">
                  <span className="head-title">ことばでスイカ割り</span>
                  <span className="head-cell">{phase === "RUNNING" && gr?.remaining_s != null ? `残 ${gr.remaining_s.toFixed(0)}s` : phase === "RESULT" ? "結果" : "準備中"}</span>
                  <span className="head-cell">振り残 {swingsLeft ?? "—"}</span>
                </div>
                <div className="head-status">
                  {gameStarting
                    ? "起動中…（controller実開始を待っています）"
                    : phase === "RUNNING" && !controllerAlive
                      ? "controller応答なし"
                      : (lastCmd || game?.message || "—")}
                </div>
              </>
            ) : (
              <>
                <div className="head-line">
                  <span className="head-title">通常モード</span>
                  <span className="head-cell">{phase}</span>
                </div>
                <div className="head-status">{state?.message || "—"}</div>
              </>
            )}
          </div>
          {(spawnPending || spawnRetryAvailable || spawnGenerationChanged || spawnMessage) && !gameMode && (
            <div className="game-start-panel" role="status" aria-live="polite">
              <p className="game-start-status">{spawnMessage}</p>
              {spawnRetryAvailable && (
                <button className="primary game-start-button" type="button"
                  onClick={switchToGame} disabled={busy || readOnly}
                  data-action-id="retry-spawn">スイカの状態を確認・再試行</button>
              )}
              {spawnGenerationChanged && (
                <button className="primary game-start-button" type="button"
                  onClick={resetRound}
                  disabled={busy || readOnly}
                  data-action-id="reset-stale-spawn">現在のラウンドを初期化</button>
              )}
            </div>
          )}
          {gamePrestart && (
            <div className="game-start-panel">
              <button className="primary game-start-button" onClick={startGame}
                data-action-id="start-round"
                aria-label="スイカ割りのラウンドを開始"
                aria-describedby="game-start-status"
                disabled={gameStartDisabled}>
                {gameStarting ? "起動中…" : "ラウンド開始"}
              </button>
              <p id="game-start-status" className={"game-start-status" + (gameStartBlockers.length ? " warn-text" : "")}
                role="status" aria-live="polite">
                {gameStarting ? "開始を受理しました。ロボットの起動を待っています" :
                  busy ? "準備中です" :
                    gameStartBlockers.length ? `開始待ち: ${gameStartBlockers.join(" / ")}` :
                      "開始できます。クリックするとロボットが立ち上がります"}
              </p>
              {error && <div className="alert" role="alert">{error}</div>}
            </div>
          )}
          <section className="dock">
          {error && !gamePrestart && <div className="alert" role="alert">{error}</div>}

          {/* ============ ことばでスイカ割り ============ */}
          {gameMode && (
            <>
              <p className="msg">
                砂浜のスイカへ歩いて近付き、割ってください
                {game.spec && (
                  <span className="spec-line">
                    （制限 {game.spec.time_limit_s}秒 / 振り {game.spec.max_swings}回
                    {game.spec.seed >= 0 ? ` / seed ${game.spec.seed}` : ""}）
                  </span>
                )}
              </p>

              {/* 準備完了・未開始 */}
              {gamePrestart && (
                <details className="game-guide">
                  <summary>遊び方とルール</summary>
                  <table className="review">
                    <tbody>
                      <tr><th>制限時間</th><td>{game.spec?.time_limit_s} 秒</td></tr>
                      <tr><th>振れる回数</th><td>{game.spec?.max_swings} 回</td></tr>
                      <tr><th>操縦範囲</th><td>開始位置から約{game.spec?.scene?.arena_r_m ?? 5}mの円内（範囲外へ出ると安全停止して終了します）</td></tr>
                      <tr><th>遊び方</th><td>「前」「後ろ」は止まるまで動き続けます。「少し右」は0.25mだけ微調整。「右に90°向いて」で旋回、「止まって」で停止、近付いたら「割って」</td></tr>
                    </tbody>
                  </table>
                </details>
              )}

              {/* ラウンド中 HUD */}
              {phase === "RUNNING" && (
                <>
                  <div className="hud">
                    <div className="hud-cell">
                      <div className="hud-num">{gr?.remaining_s != null ? gr.remaining_s.toFixed(0) : "—"}</div>
                      <div className="hud-label">残り秒</div>
                    </div>
                    <div className="hud-cell">
                      <div className="hud-num">{swingsLeft ?? "—"}</div>
                      <div className="hud-label">残り振り</div>
                    </div>
                    <div className="hud-cell wide">
                      <div className="hud-num small">{lastCmd || game?.message || "—"}</div>
                      {/* 段階表示: 認識→送信受理→controller適用→実動作→完了。
                          HTTP受理だけでは実移動にならないことを可視化する */}
                      {lastCmdSeq !== null && (
                        <div className="exec-strip">
                          <span className="exec-stage on">認識</span>
                          <span className={"exec-stage " + (appliedSeen ? "on" : "doing")}>
                            適用{appliedSeen ? ` #${lastCmdSeq}` : "待ち"}
                          </span>
                          <span className={
                            "exec-stage " +
                            (csStatus === "settling" || csStatus === "jog_stopping" ? "doing"
                              : csActive ? "doing" : "")}>
                            {csStatus === "settling" || csStatus === "jog_stopping" ? "安定化"
                              : csStatus === "turning" ? "旋回中"
                              : csStatus === "moving" ? "移動中"
                              : csStatus === "jogging" ? "継続移動中"
                              : csStatus === "applying" ? "起動中"
                              : appliedSeen ? "実動作" : "—"}
                          </span>
                          <span className={
                            "exec-stage " +
                            (csDone ? "on" : csFailed ? "bad" : "")}>
                            {csDone ? "完了" : csFailed ? "中断/失敗" : "結果"}
                          </span>
                          {cs && csSeq === lastCmdSeq && cs.progress && (
                            <span className="exec-detail">
                              {(cs.step_count ?? 0) > 1
                                ? `step ${(cs.step_index ?? 0) + 1}/${cs.step_count} `
                                : ""}
                              {cs.label || cs.action_key || ""}
                              {csStatus === "jogging" || csStatus === "jog_stopping"
                                ? ` ${(cs.progress.dist_m ?? 0).toFixed(2)}m 進行中`
                                : cs.progress.target_m
                                  ? ` ${cs.progress.dist_m}/${cs.progress.target_m}m`
                                  : cs.progress.target_deg
                                    ? ` ${cs.progress.angle_deg}/${cs.progress.target_deg}°`
                                    : ""}
                              {/* 完了理由を明示（境界停止を距離達成と偽らない） */}
                              {(csStatus === "failed" || csStatus === "aborted" ||
                                (csStatus === "completed" && cs.reason)) && cs.reason
                                ? `（${reasonText(cs.reason) || cs.reason}）` : ""}
                            </span>
                          )}
                          {gr?.controller_alive === false && (
                            <span className="exec-stage bad">controller応答なし</span>
                          )}
                        </div>
                      )}
                      <div className="hud-label">
                        現在の指示
                        {gr?.applied_seq != null ? `（適用 #${gr.applied_seq}）` : ""}
                      </div>
                    </div>
                  </div>
                  {strikeActive && (
                    <p className="strike-note">
                      {strikePhase === "swing" || strikePhase === "dance"
                        ? "振っています…"
                        : strikePhase === "recover"
                          ? "振り終えています…"
                          : "振りの準備中…"}
                    </p>
                  )}
                  {gr?.strike_unconfirmed && (
                    <p className="strike-note warn-text">
                      直前の振りの結果を確認できませんでした（要確認）。
                    </p>
                  )}
                  {!strikeActive && gr?.last_strike && (
                    <p className={"strike-note " + (gr.last_strike.hit ? "ok-text" : "")}>
                      直近の振り: {gr.last_strike.hit ? "命中！" : `外れ（最短 ${gr.last_strike.min_dist_m}m）`}
                    </p>
                  )}
                  {!strikeActive && gr?.last_strike_attempt && (
                    <p className="strike-note warn-text">
                      直前の振りは開始できませんでした（
                      {gr.last_strike_attempt.interrupted
                        ? "中断されました"
                        : gr.last_strike_attempt.aborted === "precondition_timeout"
                          ? "姿勢が安定しませんでした — 止まってから再度"
                          : "未開始"}）。振数は消費されていません。
                    </p>
                  )}
                  {(voiceArmed || voiceInterim) && (
                    <p className={"voice-note" + (voiceArmed ? " rec" : "")}>
                      {voiceHealth === "transcribing"
                        ? "認識中…（ローカルASR）"
                        : voiceInterim
                          ? `聞き取り中: ${voiceInterim}`
                          : voiceHealth === "listening"
                            ? `話してください…（「前」などの号令で即送信${
                                voiceEngine === "local-asr" ? " / ローカルASR" : ""
                              }）`
                            : `音声認識 ${voiceHealth === "restarting" ? "再起動中" : "起動中"}…`}
                    </p>
                  )}
                  {!LOCAL_ASR_ONLY &&
                    voiceArmed &&
                    voiceEngine === "webspeech" &&
                    voiceProbe &&
                    voiceProbe.locality !== "on-device" && (
                    <p className="voice-note warn-text">
                      {voiceProbe.locality === "pack-required"
                        ? "日本語オンデバイスpackが未install — install後に再armしてください"
                        : voiceProbe.locality === "unverified"
                          ? "on-device処理を確認できません（音声が外部へ送られる可能性）"
                          : "on-device音声認識は利用できません"}
                    </p>
                  )}
                  {/* R7 1.2: ASR不調時は実際の原因を表示（403/pack未install等への
                      誤変換をしない）。LOCAL_ASR_ONLYでは代替engineを案内しない */}
                  {!voiceArmed && asrService.state !== "ready" && asrService.state !== "unknown" && (
                    <p className="voice-note warn-text">
                      {asrService.detail || `ローカルASR: ${asrService.state}`}
                      {asrService.requestId ? ` ［req ${asrService.requestId}］` : ""}
                    </p>
                  )}
                  <div className="row">
                    <button className="warn" onClick={() => post("/api/run/pause", {})} disabled={readOnly}
                      data-action-id="pause-run">
                      中断
                    </button>
                  </div>
                  {/* audio状態 — assets/output/muteは別stateで表示。
                      操作ボタンは固定パッド（.pad）側に常設する */}
                  {audioOutput === "denied" && (
                    <p className="voice-note warn-text">
                      音声出力がブロックされています — 下の音声ボタンで有効化できます
                    </p>
                  )}
                  {audioAssets === "failed" && (
                    <p className="voice-note warn-text">
                      音源を読み込めません（音なしでプレイできます）
                    </p>
                  )}
                  <p className="hint">
                    動いている間も指示を差し替えられます（新しい指示が優先）。
                    「割って」は腕を一振りします。
                  </p>
                </>
              )}

              {/* ラウンド結果 */}
              {phase === "RESULT" && (
                <>
                  <p className={"msg big " + (outcomeOk ? "ok" : "ng")}>
                    {game?.message || state?.message}
                  </p>
                  <table className="review">
                    <tbody>
                      {gr?.last_strike && (
                        <tr><th>最も近付いた振り</th><td>
                          {gr.last_strike.hit ? "命中" : "外れ"} —
                          最短距離 {gr.last_strike.min_dist_m}m（{gr.last_strike.hand === "R" ? "右手" : "左手"}）
                        </td></tr>
                      )}
                      <tr><th>振った回数</th><td>{gr?.strikes_started ?? "—"} 回</td></tr>
                    </tbody>
                  </table>
                  <div className="row">
                    <button className="primary" onClick={resetRound} disabled={busy || readOnly}
                      data-action-id="reset-round">
                      もう一度挑戦
                    </button>
                  </div>
                </>
              )}
            </>
          )}

          {/* ============ 旧markerコース（回帰flow。非ゲームround専用） ============ */}
          {!gameMode && (phase === "READY" || (phase === "REVIEW" && !review)) && (
            <>
              <p className="msg">{state?.message ?? "始めます"}</p>
              <p className="hint">
                下の入力例はクリックで入力欄に入り、「解釈する」で確認できます。
                行き先が3D上で光り、確認後に実行します。
                「前へ」「少し右」は継続・微調整の運動指示としても使えます。
                {readOnly ? "（閲覧専用のため操作はできません）" : ""}
              </p>
              <p className="hint">{spawnPending
                ? "スイカ割りの生成状態を確認中…"
                : "スイカ割りは上の「スイカ割り」から始められます。"}</p>
            </>
          )}

          {phase === "INTERPRETING" && <p className="msg big">指示を読んでいます…</p>}

          {phase === "CLARIFY" && (
            <>
              <p className="msg big">{state?.message}</p>
              <p className="hint">下で答えるか、「別の指示を出す」で最初から入力できます。</p>
            </>
          )}

          {phase === "REVIEW" && review && (
            <>
              <p className="msg big">実行内容の確認</p>
              <table className="review">
                <tbody>
                  <tr><th>動作</th><td>{review.review.action}</td></tr>
                  {review.review.steps ? (
                    <tr><th>手順</th><td>{review.review.steps.join(" → ")}</td></tr>
                  ) : (
                    <tr><th>目的地</th><td>{review.review.target}</td></tr>
                  )}
                  <tr><th>距離</th><td>
                    {review.review.steps && !(review.review.distance_m > 0)
                      ? "継続動作 — 停止指示・範囲境界・上限時間で止まります"
                      : `前進 約${review.review.distance_m}m`}
                  </td></tr>
                  <tr><th>停止</th><td>{review.review.stop}</td></tr>
                </tbody>
              </table>
              <p className="hint">あなたの指示: 「{review.explanation}」{renderer === "ready" && !review.review.steps ? " — 3D上で目的地が光っています。" : ""}</p>
              {renderer !== "ready" && (
                <p className="hint warn-text">3D表示が未接続のため実行できません（接続の復旧を待っています）。</p>
              )}
              <div className="row">
                <button className="primary" onClick={approveAndRun} disabled={busy || readOnly || renderer !== "ready"}
                  data-action-id="approve-run">
                  この内容で実行
                </button>
                <button onClick={() => setReview(null)} disabled={busy}
                  data-action-id="cancel-review">
                  やめる（別の指示を入力）
                </button>
              </div>
            </>
          )}

          {phase === "RUNNING" && !gameMode && (
            <>
              <p className="msg big">実行中 — {runningLabel}</p>
              {state?.motion && (
                <table className="review motion">
                  <tbody>
                    <tr><th>手順</th><td>{(state.motion.step_index ?? 0) + 1}/{state.motion.step_count} {state.motion.label || ""}</td></tr>
                    <tr><th>状態</th><td>{motionStatusText(state.motion.status)}</td></tr>
                    {(state.motion.target_m ?? 0) > 0 && (
                      <tr><th>進行</th><td>{(state.motion.dist_m ?? 0).toFixed(2)} / {state.motion.target_m} m</td></tr>
                    )}
                    {(state.motion.target_deg ?? 0) > 0 && (
                      <tr><th>旋回</th><td>{(state.motion.angle_deg ?? 0).toFixed(0)} / {(state.motion.target_deg ?? 0).toFixed(0)}°</td></tr>
                    )}
                  </tbody>
                </table>
              )}
              <div className="row">
                <button className="warn" onClick={() => post("/api/run/pause", {})} disabled={busy || readOnly}
                  data-action-id="pause-run">
                  一時停止（中断）
                </button>
              </div>
              <p className="hint">
                実行中は新しい指示を受け付けません。中断した場合はresetしてやり直してください。
              </p>
            </>
          )}

          {phase === "RESETTING" && (
            <p className="msg big">初期化中…（ロボットを立ち位置へ戻しています）</p>
          )}

          {phase === "RESULT" && result && !gameMode && (
            <>
              <p className={"msg big " + (result.verdict === "PASS" ? "ok" : "ng")}>
                {verdictText(result.verdict)}
              </p>
              <table className="review">
                <tbody>
                  {result.program ? (
                    <>
                      <tr><th>動作手順</th><td>{result.program.map((s: any) => s.label || s.action_key).join(" → ")}</td></tr>
                      <tr><th>完了step</th><td>{result.control?.steps_done ?? "?"}/{result.control?.steps_total ?? result.program.length}</td></tr>
                    </>
                  ) : (
                    <tr><th>停止位置の誤差</th><td>{result.err_m ?? result.min_err_m ?? "?"} m</td></tr>
                  )}
                  <tr><th>最終速度</th><td>{result.speed_mps ?? "?"} m/s</td></tr>
                  <tr><th>姿勢・高さ</th><td>{result.height_m ?? "?"} m / 傾き {result.tilt_deg ?? "?"}°</td></tr>
                  {result.reasons?.length > 0 && (
                    <tr><th>理由</th><td>{result.reasons.join("、")}</td></tr>
                  )}
                </tbody>
              </table>
              <div className="row">
                <button className="primary" onClick={resetRound} disabled={busy || readOnly}
                  data-action-id="reset-round">
                  もう一度挑戦
                </button>
              </div>
            </>
          )}

          {phase === "FAULT" && (
            <>
              <p className="msg big">{state?.message}</p>
              <button className="primary" onClick={resetRound} disabled={busy || readOnly}
                data-action-id="reset-round">
                初期状態へ戻す
              </button>
            </>
          )}

          {/* 設定・診断 — サイドバー内の折りたたみ（既定は閉） */}
          <details className="drawer-in">
            <summary>設定・診断</summary>
            <div className="drawer-body">
              <div className={"map-note " + (renderer === "ready" ? "" : "warn")}>
                {renderer === "ready"
                  ? "実PM01を表示中（描画専用・物理はSDK+MuJoCo）。視点は3D左上の「Track camera」で追従/固定を切替できます。カメラ設定は世界座標を変えません。"
                  : "3D表示が未接続のため実行できません。"}
              </div>
              {!gameMode && (
                <div className="targets">
                  {state?.course.targets.map((t) => (
                    <div key={t.id} className="target-chip">
                      <span className="dot" style={{ background: TARGET_COLORS[t.id] || "#d9a53d" }} />
                      {t.label}（前進 {t.distance_m}m）
                    </div>
                  ))}
                </div>
              )}
              <div>
                <canvas ref={canvasRef} width={560} height={300} aria-label="コースとロボットの2D位置" />
                <div className="opvals">
                  pos: {live.pos ? live.pos.map((v) => v.toFixed(3)).join(", ") : "—"} /
                  速度 {speed.toFixed(2)}m/s / task {task || "—"} /
                  boot {live.boot_nonce ?? "—"} / seq {live.step_seq ?? "—"}
                </div>
              </div>
            </div>
          </details>
        </section>

        {/* R7 2.3: 音声デバイス面 — 実機のマイク/出力/ASR状態を利用者が
            確認・選択できる診断ブロック。認識不調でもSTOPは別経路で常設。 */}
        <details className="audio-diag">
          <summary>音声デバイス・認識の状態</summary>
          <div className="diag-body">
            <div className="diag-row">
              <span className="diag-k">マイク</span>
              <select
                value={micDeviceId}
                onChange={(e) => setMicDeviceId(e.target.value)}
                disabled={voiceArmed}
                data-action-id="mic-select"
              >
                <option value="">OSの既定入力</option>
                {micDevices.map((d) => (
                  <option key={d.deviceId} value={d.deviceId}>
                    {d.label || `デバイス ${d.deviceId.slice(0, 8)}`}
                  </option>
                ))}
              </select>
            </div>
            <div className="diag-row">
              <span className="diag-k">マイク許可</span>
              <span>{micPerm === "granted" ? "許可済み" : micPerm === "denied" ? "拒否されています" : micPerm === "prompt" ? "未確認（操作時に確認）" : micPerm}</span>
            </div>
            {asrInputInfo && (
              <div className="diag-row">
                <span className="diag-k">入力デバイス</span>
                <span>
                  {asrInputInfo.deviceLabel || asrInputInfo.deviceId || "—"}・
                  {asrInputInfo.sampleRate}Hz{asrInputInfo.resampled ? "→16kHz変換" : ""}
                  {asrInputInfo.channelCount != null ? `・${asrInputInfo.channelCount}ch` : ""}
                  ・ctx:{asrInputInfo.ctxState}
                </span>
              </div>
            )}
            <div className="diag-row">
              <span className="diag-k">入力レベル</span>
              <span className="level-meter" aria-label="入力レベル">
                <span className="level-fill" style={{ width: `${Math.min(100, micLevel * 500)}%` }} />
              </span>
              <span>{voiceArmed ? (voiceHealth === "speech" ? "発話中" : voiceHealth === "transcribing" ? "認識中" : "待機") : "—"}</span>
            </div>
            <div className="diag-row">
              <span className="diag-k">認識エンジン</span>
              <span>
                {LOCAL_ASR_ONLY ? "ローカルASR固定" : "ローカルASR優先"}
                {asrService.model ? `（${asrService.model}` : ""}
                {asrService.device ? ` / ${asrService.device}${asrService.deviceName ? ` ${asrService.deviceName}` : ""}` : ""}
                {asrService.model ? "）" : ""}
                {asrService.state !== "ready" && asrService.state !== "unknown" ? ` — ${asrService.state}` : ""}
              </span>
            </div>
            {lastAsrTiming && (
              <div className="diag-row">
                <span className="diag-k">直近の認識</span>
                <span>
                  ASR {lastAsrTiming.serverLatencyMs != null ? `${Math.round(lastAsrTiming.serverLatencyMs)}ms` : "—"}
                  （往復 {lastAsrTiming.doneAtMs - lastAsrTiming.sentAtMs}ms
                  {lastAsrTiming.queueWaitMs > 0 ? `・待ち ${lastAsrTiming.queueWaitMs}ms` : ""}）
                </span>
              </div>
            )}
            <div className="diag-row">
              <span className="diag-k">音声出力</span>
              <span>
                {(() => {
                  const oi = audioRef.current?.outputInfo();
                  if (!oi || oi.state === "none") return "未初期化";
                  return `${oi.state}${oi.sinkId ? `・sink:${oi.sinkId}` : "・OS既定"}${!oi.setSinkId ? "（出力選択はOS側で行います）" : ""}`;
                })()}
              </span>
              <button
                className="example"
                onClick={() => void audio().testTone().then(setAudioOutput)}
                data-action-id="test-tone"
              >
                テスト音を出す
              </button>
            </div>
          </div>
        </details>

        {/* 固定操作盤（R6-C）— 状態遷移でunmountしない。使えない状態は
            disabled。各操作は安定した data-action-id を持つ */}
        <div className="pad">
          {gameMode ? (
            <div className="pad-grid">
              {/* ゲーム指令 — RUNNING中のみ有効、他位相はdisabledで固定 */}
              {(() => {
                const padOff = readOnly || phase !== "RUNNING" || !!gr?.done;
                return (
                  <>
                    <button className="pad-btn" data-action-id="dir-forward"
                      onClick={() => sendGameCmd("前")} disabled={padOff}>前</button>
                    <button className="pad-btn" data-action-id="dir-left-small"
                      onClick={() => sendGameCmd("少し左")} disabled={padOff}>少し左</button>
                    <button className="pad-btn" data-action-id="dir-right-small"
                      onClick={() => sendGameCmd("少し右")} disabled={padOff}>少し右</button>
                    <button className="pad-btn" data-action-id="dir-back"
                      onClick={() => sendGameCmd("後ろ")} disabled={padOff}>後ろ</button>
                    <button className="pad-btn" data-action-id="turn-left"
                      onClick={() => sendGameCmd("左に90°向いて")} disabled={padOff}>左90°</button>
                    <button className="pad-btn" data-action-id="turn-right"
                      onClick={() => sendGameCmd("右に90°向いて")} disabled={padOff}>右90°</button>
                    <button className="pad-btn" data-action-id="turn-around"
                      onClick={() => sendGameCmd("後ろを向いて")} disabled={padOff}>振り向き</button>
                    <button className="pad-btn" data-action-id="walk-1m"
                      onClick={() => sendGameCmd("1m歩いて")} disabled={padOff}>1m歩く</button>
                    <button className="pad-btn pad-strike" data-action-id="strike"
                      onClick={() => sendGameCmd("割って")} disabled={padOff}>割って</button>
                    <button className="pad-btn warn" data-action-id="end-round"
                      onClick={() => sendGameCmd("終わって")} disabled={padOff}>終わって</button>
                  </>
                );
              })()}
            </div>
          ) : (
            <div className={"pad-grid " + (phase === "CLARIFY" ? "pad-clarify" : "pad-examples")}>
              {/* 通常モード — 入力例を入力欄へ入れるだけ（実行はしない）。
                  実行中はdisabled（入力欄の内容を実行中に書き換えない） */}
              {(phase === "CLARIFY"
                ? CLARIFY_CHOICES
                : EXAMPLES.map((e) => ({ label: e, value: e }))).map((choice) => (
                <button key={choice.value} className="pad-btn"
                  onClick={() => phase === "CLARIFY" ? setClarifyAnswer(choice.value) : setText(choice.value)}
                  disabled={busy || readOnly || spawnUnresolved || spawnGenerationChanged || phase === "RUNNING"}
                  data-action-id={`ex-${choice.value}`}>
                  {choice.label}
                </button>
              ))}
            </div>
          )}
          {/* 共通制御行 — 音声arm/消音は両モード同一位置に固定。
              R7 §3: 音声buttonは常に場所を残し、サービス状態の変化で
              出現/消失しない（隣buttonを動かさない） */}
          <div className="pad-row">
            {(() => {
              // R7 1.2: 使えない理由は実際の原因を出す — 403をpack不足や
              // モデル未配備へ変換しない。LOCAL_ASR_ONLYではWeb Speechを
              // 代替として提示しない。
              const canArm = LOCAL_ASR_ONLY
                ? asrArmable
                : asrArmable || voiceProbe?.locality === "on-device";
              const reason = voiceArmed
                ? (voiceEngine === "local-asr" ? "ローカルASR（この機器内で認識）" : "on-device音声認識")
                : asrArmable
                  ? `ローカルASR（この機器内で認識）${
                      asrService.state === "degraded_cpu" ? " — CPU動作中" : ""
                    }${asrService.model ? ` / ${asrService.model}` : ""}${
                      asrService.device ? ` / ${asrService.device}` : ""}`
                  : asrService.detail ||
                    (asrService.state === "unknown"
                      ? "ASR稼働状態を確認中…"
                      : "ローカルASRが使えません");
              return (
                <button
                  className={"pad-btn mic" + (voiceArmed ? " rec" : "")}
                  data-action-id="voice-arm"
                  onClick={voiceArmed ? stopVoice : startVoice}
                  disabled={readOnly || (!voiceArmed && (spawnUnresolved || spawnGenerationChanged || !canArm))}
                  aria-label={voiceArmed ? "音声操縦を止める" : "音声操縦を始める"}
                  aria-pressed={voiceArmed}
                  title={reason}
                >
                  {voiceArmed
                    ? voiceEngine === "local-asr"
                      ? "■ 音声(局所)"
                      : "■ 音声"
                    : "🎤 音声"}
                </button>
              );
            })()}
            <button
              className="pad-btn"
              data-action-id="audio-mute"
              onClick={toggleMute}
              disabled={audioAssets !== "ready"}
              aria-label={audioMuted ? "音を出す" : "消音する"}
              aria-pressed={audioMuted}
              title={audioAssets === "ready" ? "" : "音源の準備がまだです"}
            >
              {audioMuted ? "🔇 消音中" : "🔊 音"}
            </button>
          </div>
        </div>

        {/* 下段固定 — 入力欄＋送信＋常設停止。スクロールで埋もれない */}
        <div className="sidebar-foot">
          {gameMode && (
            <div className="input-row">
              <input
                value={gameText}
                onChange={(e) => setGameText(e.target.value)}
                onCompositionStart={() => setComposing(true)}
                onCompositionEnd={() => setComposing(false)}
                onKeyDown={(e) => { if (e.key === "Enter" && !composing) sendGameCmd(); }}
                placeholder={phase === "RUNNING" ? "例: 前 / 少し左 / 右に90°向いて / 割って" : "開始後に指示できます"}
                disabled={readOnly || phase !== "RUNNING" || !!gr?.done}
                aria-label="ゲーム指示の入力"
                data-action-id="input-command"
              />
              <button onClick={() => sendGameCmd()} data-action-id="send"
                disabled={composing || !gameText.trim() || readOnly || phase !== "RUNNING" || !!gr?.done}>
                指示する
              </button>
            </div>
          )}
          {!gameMode && phase !== "CLARIFY" && (
            <div className="input-row">
              <input
                value={text}
                onChange={(e) => setText(e.target.value)}
                onCompositionStart={() => setComposing(true)}
                onCompositionEnd={() => setComposing(false)}
                onKeyDown={(e) => { if (e.key === "Enter" && !composing) submitIntent(); }}
                placeholder="例: 手前のマーカーまで進んで / 前へ / 少し右"
                disabled={busy || readOnly || spawnUnresolved || spawnGenerationChanged}
                aria-label="指示の入力"
                data-action-id="input-command"
              />
              <button onClick={() => submitIntent()} data-action-id="send"
                disabled={busy || composing || !text.trim() || readOnly || spawnUnresolved || spawnGenerationChanged}>
                解釈する
              </button>
            </div>
          )}
          {!gameMode && phase === "CLARIFY" && (
            <>
              <div className="input-row">
                <input
                  value={clarifyAnswer}
                  onChange={(e) => setClarifyAnswer(e.target.value)}
                  onCompositionStart={() => setComposing(true)}
                  onCompositionEnd={() => setComposing(false)}
                  onKeyDown={(e) => { if (e.key === "Enter" && !composing) submitIntent(clarifyAnswer); }}
                  placeholder="答えを入力（例: 手前）"
                  disabled={busy || readOnly || spawnUnresolved || spawnGenerationChanged}
                  aria-label="聞き返しへの回答"
                  data-action-id="input-command"
                />
                <button
                  disabled={busy || composing || !clarifyAnswer.trim() || readOnly || spawnUnresolved || spawnGenerationChanged}
                  onClick={() => submitIntent(clarifyAnswer)}
                  data-action-id="send"
                >
                  答える
                </button>
              </div>
              <button onClick={newInstruction} disabled={navigationBlocked || readOnly || spawnUnresolved || spawnGenerationChanged}
                data-action-id="new-instruction">別の指示を出す</button>
            </>
          )}
          {/* 常設停止 — 常時mount・無効時はdisabled。ゲームroundは最新勝ち
              指令「止まって」、通常runは正常停止経路（runnerが減速→立位保持）
              へ送る。管理中断のpause（コンテナ強制終了）とは別経路。 */}
          <button
            className="perm-stop"
            onClick={stopNow}
            disabled={!(phase === "RUNNING" || (gameMode && gr && !gr.done)) || readOnly}
            aria-label="今すぐ止まる"
            data-action-id="stop"
          >
            止まって
          </button>
        </div>
        </aside>

        {/* 折りたたみ時の細いrail — 再表示＋停止は常に届く。
            iframeは再mountしない（sidebarはtransformで隠すだけ）。 */}
        {!sidebarOpen && (
          <div className="rail">
            <button
              className="rail-btn"
              onClick={() => setSidebarOpen(true)}
              aria-label="操作メニューを開く"
              data-action-id="sidebar-open"
            >
              ◀ メニュー
            </button>
            <button
              className="rail-stop"
              onClick={stopNow}
              disabled={!(phase === "RUNNING" || (gameMode && gr && !gr.done)) || readOnly}
              aria-label="今すぐ止まる"
              data-action-id="stop-rail"
            >
              止
            </button>
          </div>
        )}
      </div>

      <footer>
        <span>ことばでロボコン 試用版（シミュレーション / Thor内ローカルAI）</span>
        <span>
          歩行policy: EngineAI公式 / 解釈・実行管理・UI: 本制作（LLLM.jp）
          {meta?.ui_commit ? ` / UI ${meta.ui_commit.slice(0, 8)}` : ""}
        </span>
      </footer>
    </div>
  );
}
