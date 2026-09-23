/**
 * AudioManager — ゲーム実イベント駆動のBGM/SFX再生（R6-B 状態機械）。
 *
 * 状態は4系統に分離する:
 * - assets:   idle / loading / ready / failed（assetごとの結果も持つ）
 * - output:   locked / running / suspended / denied / failed
 * - game:     run_id、実開始（round_started済み）、終端、snapshot鮮度、
 *             event cursor（high-water）
 * - capture:  voiceArmed / speechActive（ASR共存のduck用）
 *
 * 原則:
 * - 親UI(App)が1つだけ所有。sidebar/iframeのmountに結び付けない。
 * - HTTP受理・ボタンclick・ASR transcriptでは音を鳴らさない。
 *   再生根拠はcontroller観測の実イベント（control_state経由）のみ。
 * - resumeFromGesture()（操作の同期区間でAudioContextをresume）と
 *   preloadAssets()（音源取得・decode — 操縦を待たせない）は分離する。
 * - 制御面は reconcile(snapshot) のみ — React effectとpollが独立に
 *   resetしない。順序は旧run無効化→新snapshot登録→BGM同期→新eventのSFX。
 * - 一回限りSFXは購読開始以降の未消費eventのみ。再接続/snapshotの過去分は
 *   鳴らさない。BGMだけは現在状態（round_started済み・終端でない有効run）
 *   から再開してよい — 終端snapshot・鮮度切れ・API断でも停止する。
 * - SFXの再生sourceはrun別に管理し、run失効でcancelする。
 * - muteは再生失敗ではない。localStorageのmuteとUI初期表示を一致させる。
 * - 音量設定のみlocalStorageに保存可。録音/transcript/run資格は保存しない。
 */

export type AudioAssetId = "bgm_suika" | "sfx_swing" | "sfx_success" | "sfx_miss";

interface AudioAsset {
  file: string;
  sha256: string;
  bytes: number;
  duration_s: number;
  loop: boolean;
}

export interface GameAudioEvent {
  id: string;
  event_seq: number;
  kind: string;
  t_wall: number;
  strike_id?: string;
  reason?: string;
  outcome?: unknown;
}

export type AssetsState = "idle" | "loading" | "ready" | "failed";
export type OutputState =
  | "locked" // user gesture未投入（AudioContext未作成/suspended）
  | "running"
  | "suspended"
  | "denied" // autoplay拒否 — 独立した有効化操作で再試行可
  | "failed"; // AudioContext自体が使えない

/** reconcileへ渡す現在snapshot（pollの実データ — 推測で作らない） */
export interface AudioSnapshot {
  runId: string | null;
  events: GameAudioEvent[];
  /** controllerが実プレイ中（RUNNING && !done） */
  running: boolean;
  /** このsnapshot自体が新鮮か（API応答成功の実データか） */
  fresh: boolean;
}

const BGM_DUCK_DB = -15; // 音声操縦armed中のBGM減衰（12〜18dB帯の中央）
const SFX_DUCK_DB = -9;
const SPEECH_DUCK_DB = -10; // 実発話activeの追加減衰
const LS_KEY = "kotoba.audio.v1";
// roundのterminal event — これ以降BGMの存在条件は成立しない
const TERMINAL_KINDS = new Set([
  "round_succeeded",
  "round_failed",
  "round_aborted",
  "control_fault",
]);

function dbToGain(db: number): number {
  return Math.pow(10, db / 20);
}

export class AudioManager {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private bgmBus: GainNode | null = null;
  private sfxBus: GainNode | null = null;
  private buffers = new Map<AudioAssetId, AudioBuffer>();
  private bgmSrc: AudioBufferSourceNode | null = null;
  private bgmPlaying = false;
  private bgmFade: GainNode | null = null;
  // run所有のSFX source — run失効でcancelする（旧runの音を残さない）
  private sfxSources = new Set<AudioBufferSourceNode>();
  private preloadFlight: Promise<AssetsState> | null = null;
  private resumeFlight: Promise<OutputState> | null = null;

  // --- 分離state ---
  assetsState: AssetsState = "idle";
  outputState: OutputState = "locked";
  muted = false;
  voiceArmed = false;
  speechActive = false;
  assetErrors: Partial<Record<AudioAssetId, string>> = {};

  // --- game cursor（run結合） ---
  private runId: string | null = null;
  private hwSeq = 0; // 購読開始点/消費済みのevent_seq上限
  private sawRoundStarted = false;
  private sawTerminal = false;
  private lastEventKind: string | null = null;

  constructor() {
    try {
      const raw = localStorage.getItem(LS_KEY);
      if (raw) {
        const v = JSON.parse(raw);
        if (typeof v.muted === "boolean") this.muted = v.muted;
      }
    } catch {
      /* localStorage不可は音量保存のみ諦める */
    }
  }

  /** AudioContextの生成/再開のみ — user gestureの同期区間で呼ぶ。
   *  音源の取得・decodeは待たない（操縦を塞がない）。 */
  resumeFromGesture(): Promise<OutputState> {
    if (this.resumeFlight) return this.resumeFlight;
    this.resumeFlight = (async () => {
      try {
        if (!this.ctx) {
          this.ctx = new AudioContext();
          this.master = this.ctx.createGain();
          this.bgmBus = this.ctx.createGain();
          this.sfxBus = this.ctx.createGain();
          this.bgmBus.connect(this.master);
          this.sfxBus.connect(this.master);
          this.master.connect(this.ctx.destination);
          this.applyGains();
          // OS/タブ要因であとからsuspendされる経路を観測する
          this.ctx.onstatechange = () => {
            if (!this.ctx) return;
            this.outputState =
              this.ctx.state === "running" ? "running" : "suspended";
          };
        }
        if (this.ctx.state !== "running") {
          await this.ctx.resume();
        }
        this.outputState =
          this.ctx.state === "running" ? "running" : "denied";
        return this.outputState;
      } catch {
        this.outputState = this.ctx ? "denied" : "failed";
        return this.outputState;
      } finally {
        this.resumeFlight = null;
      }
    })();
    return this.resumeFlight;
  }

  /** 音源manifest取得+全asset decode — single-flight。操縦を待たせない。 */
  preloadAssets(): Promise<AssetsState> {
    if (this.assetsState === "ready") return Promise.resolve("ready");
    if (this.preloadFlight) return this.preloadFlight;
    this.preloadFlight = (async () => {
      this.assetsState = "loading";
      try {
        const res = await fetch("/audio/manifest.json");
        if (!res.ok) throw new Error(`manifest ${res.status}`);
        const m = await res.json();
        const manifest: Record<string, AudioAsset> = m.assets || {};
        const ids: AudioAssetId[] = [
          "bgm_suika",
          "sfx_swing",
          "sfx_success",
          "sfx_miss",
        ];
        // decodeは再生ctxに依存しない — ctx未作成ならOfflineAudioContextで
        // decodeする（AudioBufferはcontext非依存で再生可能）
        const ctx: BaseAudioContext =
          this.ctx ?? new OfflineAudioContext(1, 1, 44100);
        const decoded = new Map<AudioAssetId, AudioBuffer>();
        for (const id of ids) {
          const a = manifest[id];
          if (!a) {
            this.assetErrors[id] = "missing";
            throw new Error(`missing ${id}`);
          }
          const buf = await fetch(`/audio/${a.file}`);
          if (!buf.ok) {
            this.assetErrors[id] = `http ${buf.status}`;
            throw new Error(`fetch ${id} ${buf.status}`);
          }
          const raw = await buf.arrayBuffer();
          if (raw.byteLength !== a.bytes) {
            this.assetErrors[id] = "size_mismatch";
            throw new Error(`size ${id}`);
          }
          try {
            decoded.set(id, await ctx.decodeAudioData(raw));
          } catch {
            this.assetErrors[id] = "decode_failed";
            throw new Error(`decode ${id}`);
          }
        }
        // 全assetが揃った時点でcommit — 不足bufferを持ったままreadyにしない
        this.buffers = decoded;
        this.assetsState = "ready";
        return "ready";
      } catch {
        this.assetsState = "failed";
        return "failed";
      } finally {
        this.preloadFlight = null;
      }
    })();
    return this.preloadFlight;
  }

  /** 互換ラッパー（旧 arm）— 明示操作で resume + preload を開始する。
   *  decode完了を待たずに返す。 */
  arm(): Promise<OutputState> {
    void this.preloadAssets();
    return this.resumeFromGesture();
  }

  setMuted(m: boolean) {
    this.muted = m;
    this.applyGains();
    try {
      localStorage.setItem(LS_KEY, JSON.stringify({ muted: m }));
    } catch {
      /* ignore */
    }
  }

  /** 音声操縦のarming状態 — BGM/SFXをduckしてASRとの衝突を抑える */
  setVoiceArmed(armed: boolean) {
    if (this.voiceArmed === armed) return;
    this.voiceArmed = armed;
    this.applyGains();
  }

  /** 出力デバイスの実情報（R7 2.3 — setSinkId対応可否と現在sinkをUIへ出す） */
  outputInfo(): { sinkId: string | null; setSinkId: boolean; state: string } {
    const ctx = this.ctx as (AudioContext & { sinkId?: string }) | null;
    return {
      sinkId: ctx && typeof ctx.sinkId === "string" ? ctx.sinkId : null,
      setSinkId: !!ctx && "setSinkId" in ctx,
      state: ctx?.state ?? "none",
    };
  }

  /** テスト音（R7 2.3 — 実スピーカーへの到達を利用者が確認する短いbeep） */
  async testTone(): Promise<OutputState> {
    const st = await this.resumeFromGesture();
    if (st !== "running" || !this.ctx || !this.master) return st;
    const t = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    osc.frequency.value = 880;
    g.gain.setValueAtTime(0.0, t);
    g.gain.linearRampToValueAtTime(0.25, t + 0.02);
    g.gain.setValueAtTime(0.25, t + 0.25);
    g.gain.linearRampToValueAtTime(0.0, t + 0.4);
    osc.connect(g);
    g.connect(this.master);
    osc.start(t);
    osc.stop(t + 0.45);
    return st;
  }

  /** 実発話active — 追加duck（開始時に更に落とし、終端後に穏やかに戻す） */
  setSpeechActive(active: boolean) {
    if (this.speechActive === active) return;
    this.speechActive = active;
    this.applyGains();
  }

  private applyGains() {
    if (!this.ctx || !this.master || !this.bgmBus || !this.sfxBus) return;
    const t = this.ctx.currentTime;
    this.master.gain.setTargetAtTime(this.muted ? 0 : 1, t, 0.02);
    const duck = (this.voiceArmed ? BGM_DUCK_DB : 0) +
      (this.speechActive ? SPEECH_DUCK_DB : 0);
    const sfxDuck = (this.voiceArmed ? SFX_DUCK_DB : 0) +
      (this.speechActive ? SPEECH_DUCK_DB : 0);
    this.bgmBus.gain.setTargetAtTime(dbToGain(duck) * 0.5, t, 0.04);
    this.sfxBus.gain.setTargetAtTime(dbToGain(sfxDuck) * 0.9, t, 0.04);
  }

  private playSfx(id: AudioAssetId) {
    const buf = this.buffers.get(id);
    if (!this.ctx || !this.sfxBus || !buf) return;
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.sfxBus);
    this.sfxSources.add(src);
    src.onended = () => this.sfxSources.delete(src);
    src.start();
  }

  private cancelSfx() {
    for (const s of this.sfxSources) {
      try {
        s.onended = null;
        s.stop();
      } catch {
        /* 既に停止済み */
      }
    }
    this.sfxSources.clear();
  }

  private startBgm() {
    const buf = this.buffers.get("bgm_suika");
    if (!this.ctx || !this.bgmBus || !buf || this.bgmPlaying) return;
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.loop = true;
    // 個別fade用gain — bus gainとは独立にフェードする
    const g = this.ctx.createGain();
    g.gain.value = 1;
    src.connect(g);
    g.connect(this.bgmBus);
    src.start();
    this.bgmSrc = src;
    this.bgmFade = g;
    this.bgmPlaying = true;
  }

  private stopBgm(fadeS = 0.6) {
    if (!this.ctx || !this.bgmSrc || !this.bgmPlaying) return;
    const src = this.bgmSrc;
    const g = this.bgmFade;
    this.bgmSrc = null;
    this.bgmFade = null;
    this.bgmPlaying = false;
    try {
      if (g) {
        g.gain.setValueAtTime(g.gain.value, this.ctx.currentTime);
        g.gain.linearRampToValueAtTime(0, this.ctx.currentTime + fadeS);
      }
      src.stop(this.ctx.currentTime + fadeS + 0.05);
    } catch {
      /* 既に停止済み */
    }
  }

  /**
   * game/stateの実snapshotを消費する唯一の制御面。
   * 順序: 旧run無効化 → 新snapshot登録 → BGM現在状態同期 → 有効な新eventのSFX。
   */
  reconcile(snap: AudioSnapshot) {
    const runId = snap.runId ?? null;

    // 1) run失効/変更 — BGM停止・run所有SFXをcancel・cursor失効
    if (this.runId !== runId) {
      this.stopBgm(0.15);
      this.cancelSfx();
      this.runId = runId;
      this.hwSeq = runId
        ? snap.events.reduce((m, e) => Math.max(m, e.event_seq || 0), 0)
        : 0;
      // 新runのsnapshot分は鳴らさない（購読開始点をhigh-waterに固定）が、
      // round開始済みか終端かは現在状態として登録する — BGM判定の正本。
      this.sawRoundStarted = snap.events.some(
        (e) => e.kind === "round_started",
      );
      this.sawTerminal = snap.events.some((e) => TERMINAL_KINDS.has(e.kind));
      this.lastEventKind = snap.events.length
        ? snap.events[snap.events.length - 1].kind
        : null;
    }

    // 2) BGM現在状態の同期 — eventではなく「現在snapshot」の状態から決める。
    //    assetsが後からreadyになった・出力がunlockされた・通信が復帰した
    //    場合でも、同じrunが実プレイ中ならreconcileでBGMを始める/再開する。
    for (const e of snap.events) {
      if (e.kind === "round_started") this.sawRoundStarted = true;
      if (TERMINAL_KINDS.has(e.kind)) this.sawTerminal = true;
    }
    const roundActive =
      !!runId && this.sawRoundStarted && !this.sawTerminal && snap.running;
    const wantBgm =
      roundActive &&
      snap.fresh &&
      this.assetsState === "ready" &&
      this.outputState === "running" &&
      this.ctx?.state === "running";
    if (wantBgm && !this.bgmPlaying) this.startBgm();
    if (!wantBgm && this.bgmPlaying) this.stopBgm(roundActive ? 0.6 : 0.25);

    // 3) 有効な新eventのみSFX — cursor越えの重複・snapshot由来の過去分は鳴らさない
    if (!runId) return;
    for (const ev of snap.events) {
      if ((ev.event_seq || 0) <= this.hwSeq) continue;
      this.hwSeq = ev.event_seq;
      this.handleEvent(ev);
      this.lastEventKind = ev.kind;
    }
  }

  private handleEvent(ev: GameAudioEvent) {
    // SFXは音源と出力が生きている時だけ — audio未readyでもcursorは進める
    // （後から準備完了しても過去SFXを連続再生しない）
    const audible =
      this.assetsState === "ready" && this.outputState === "running";
    switch (ev.kind) {
      case "round_started":
        // BGMはreconcileの状態同期で処理 — event個別には何もしない
        break;
      case "strike_started":
        if (audible) this.playSfx("sfx_swing");
        break;
      case "strike_missed":
        if (audible) this.playSfx("sfx_miss"); // BGMは継続
        break;
      case "round_succeeded":
        this.stopBgm(0.8);
        if (audible) this.playSfx("sfx_success");
        break;
      case "round_failed":
        this.stopBgm(0.4);
        // 直前の空振りと同一原因の失敗音を二重に鳴らさない
        // （最後のstrike_missedがすぐ前のイベントなら今回はBGM停止のみ。
        //  独立した時間切れ等の失敗音までは抑制しない）
        if (audible && this.lastEventKind !== "strike_missed") {
          this.playSfx("sfx_miss");
        }
        break;
      case "round_aborted":
      case "control_fault":
        // operator中断・観測/通信断・転倒 — BGM停止のみ。安全異常を
        // 面白い失敗音で装飾しない
        this.stopBgm(0.25);
        break;
    }
  }

  /** 後方互換 — 旧呼出し側があればreconcileへ寄せるための診断用 */
  resetRound() {
    this.reconcile({ runId: null, events: [], running: false, fresh: true });
  }
}
