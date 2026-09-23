/**
 * LocalAsrController — AudioWorklet PCM16/16kHz集音 → 専用ASRサービス
 * （Qwen3-ASR、Thorローカル）→ テキスト確定 → 通常dispatch経路（R6 P1-D、
 * R7: epoch公開・queue鮮度・実デバイス情報・ASR往復時刻を報告）。
 *
 * 原則:
 * - 明示armのみでマイク起動。ページ表示だけでは開始しない。
 * - ASRサービスは転写のみ — 指令化は必ず既存dispatch（/api/game/command・
 *   /api/intents）へ通す。ASR経路が直接motor指令を発行しない。
 * - 停止語は認識テキスト確定時に解釈・送信キューへ先行して送る。
 * - in-flight ASR要求はsessionあたり1つ — 追加発話は最新1件のみ残す
 *   （bounded queue。3D転送や制御経路と無制限queueを共有しない）。
 *   滞留が QUEUE_STALE_MS を越えた発話は古い音声として捨てる —
 *   「STOP後の遅着ASRで動作が再開」しないための鮮度ゲート（R7 1.4/S09）。
 * - 集音断（track ended）・権利剥奪・ASR故障はhealth=failedでarmを解き、
 *   voice由来の継続動作（jog heartbeat）を止める。
 * - pagehide/owner変更/reset後の自動再armはしない。
 * - 録音は発話区間のみ・メモリ内。永続化しない。
 */

export type AsrHealth =
  | "off"
  | "starting"
  | "listening"
  | "speech" // 発話区間内（VAD active）
  | "transcribing" // ASR往復中
  | "failed";

export interface AsrResult {
  text: string;
  latency_ms?: number;
  engine?: string;
}

/** 集音系の実情報 — UIのデバイス表示用（R7 2.3）。 */
export interface AsrInputInfo {
  sampleRate: number; // AudioContextの実sampleRate（要求と一致しない場合はresample済みと表示）
  requestedSampleRate: number;
  resampled: boolean; // worklet内で16kHzへlinear resampleしている
  channelCount: number | null;
  deviceId: string | null;
  deviceLabel: string | null;
  ctxState: string; // running/suspended等
}

/** ASR往復の時刻分離 — 体感応答を「発話終了→転写→受理→実動作」で測る（R7 4.2）。 */
export interface AsrTiming {
  sentAtMs: number;      // ASRへ送った時刻（発話区間確定直後）
  doneAtMs: number;      // ASR応答受領
  serverLatencyMs: number | null; // ASR内部推論ms（応答body由来）
  queueWaitMs: number;   // in-flight占有で待った時間（0=即送信）
}

interface LocalAsrHandlers {
  onCommit: (text: string, timing?: AsrTiming) => void;
  onStop: () => void;
  onHealth: (h: AsrHealth, detail?: string) => void;
  onInterim: (text: string) => void; // VAD区間表示（「聞き取り中」）
  onSpeech?: (active: boolean) => void; // 実発話active — BGM ducking用
  onInputInfo?: (info: AsrInputInfo) => void; // 実デバイス情報（R7 2.3）
  onLevel?: (rms: number) => void; // 入力レベルmeter（R7 2.3）
}

import { isStopCommandText } from "./voice";

const ASR_TIMEOUT_MS = 8000;
const MIN_UTT_SAMPLES = 16000 * 0.25; // 250ms未満の発話は捨てる（ノイズ突発）
// in-flight占有で滞留した発話の鮮度上限 — これを越えて処理された
// 音声は「今の意図」とは言えないため捨てる（S09: 古い発話再送なし）。
const QUEUE_STALE_MS = 4000;
const TARGET_RATE = 16000;

export class LocalAsrController {
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private node: AudioWorkletNode | null = null;
  private srcNode: MediaStreamAudioSourceNode | null = null;
  private armed = false;
  private epoch = 0;
  private inFlight: AbortController | null = null;
  private queued: { buf: ArrayBuffer; at: number } | null = null; // 最新1件のみ保持
  private h: LocalAsrHandlers;
  private sessionId: string | null = null;

  constructor(h: LocalAsrHandlers) {
    this.h = h;
  }

  isArmed() {
    return this.armed;
  }

  /** 現epoch — disarm/故障で単調増加。UIはdispatch時に照合して
   *  遅着ASR由来の指令を失効できる（voice_epoch結合）。 */
  epochNow() {
    return this.epoch;
  }

  /** 明示arm — getUserMediaはここでのみ呼ぶ（gesture区間から）。
   *  deviceId指定で入力デバイスを固定できる（R7 2.3 デバイス選択）。 */
  async arm(sessionId: string, deviceId?: string): Promise<boolean> {
    if (this.armed) return true;
    this.sessionId = sessionId;
    this.armed = true;
    const myEpoch = ++this.epoch;
    this.h.onHealth("starting");
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
    } catch (e) {
      this.armed = false;
      this.h.onHealth("failed", `mic_${(e as DOMException)?.name ?? "denied"}`);
      return false;
    }
    // 許可待ち中にdisarmされた場合は取得済みstreamを必ずstopする（R7 2.3）
    if (myEpoch !== this.epoch) {
      try { this.stream.getTracks().forEach((t) => t.stop()); } catch { /* noop */ }
      this.stream = null;
      return false;
    }
    try {
      // 16kHzで開く — 実sampleRateが違う環境ではworklet内でlinear resampleする
      this.ctx = new AudioContext({ sampleRate: TARGET_RATE });
      if (this.ctx.state !== "running") {
        // 一部環境ではsuspendedで作成される — gesture区間内なのでresume試行
        try { await this.ctx.resume(); } catch { /* noop */ }
      }
      if (this.ctx.state !== "running") {
        throw new Error("audioctx_not_running");
      }
      await this.ctx.audioWorklet.addModule("/asr-worklet.js");
      const actualRate = this.ctx.sampleRate;
      const resampled = actualRate !== TARGET_RATE;
      this.srcNode = this.ctx.createMediaStreamSource(this.stream);
      this.node = new AudioWorkletNode(this.ctx, "kotoba-asr-capture");
      this.node.port.onmessage = (ev) => this.onWorklet(ev, myEpoch);
      // workletへ実sampleRateを通知 — 16k固定前提を暗黙にしない
      this.node.port.postMessage({
        type: "config",
        inputRate: actualRate,
        targetRate: TARGET_RATE,
      });
      this.srcNode.connect(this.node);
      // worklet出力は使わないが、process()を回すためdestinationへ結ぶ
      // （gain 0のまま — 音声出力経路へは一切出さない）
      const g = this.ctx.createGain();
      g.gain.value = 0;
      this.node.connect(g);
      g.connect(this.ctx.destination);
      // track消失（マイク抜き・権利剥奪）を監視 — 自動復旧しない
      const track = this.stream.getAudioTracks()[0];
      track.onended = () => {
        if (myEpoch !== this.epoch) return;
        this.armed = false;
        this.cleanup();
        this.h.onHealth("failed", "track_ended");
      };
      // 実デバイス情報をUIへ（R7 2.3 — 設定/許可/実rate/デバイスを見せる）
      const st = track.getSettings() as MediaTrackSettings;
      this.h.onInputInfo?.({
        sampleRate: actualRate,
        requestedSampleRate: TARGET_RATE,
        resampled,
        channelCount: typeof st.channelCount === "number" ? st.channelCount : null,
        deviceId: typeof st.deviceId === "string" ? st.deviceId : null,
        deviceLabel: track.label || null,
        ctxState: this.ctx.state,
      });
    } catch (e) {
      this.armed = false;
      this.cleanup();
      this.h.onHealth("failed", `worklet_${(e as Error)?.name ?? "error"}`);
      return false;
    }
    this.h.onHealth("listening");
    return true;
  }

  disarm() {
    this.armed = false;
    this.epoch += 1; // 旧epochの遅着を全て失効
    if (this.inFlight) { this.inFlight.abort(); this.inFlight = null; }
    this.queued = null;
    this.cleanup();
    this.h.onSpeech?.(false);
    this.h.onInterim("");
    this.h.onLevel?.(0);
    this.h.onHealth("off");
  }

  private cleanup() {
    try { this.node?.port.postMessage("reset"); } catch { /* ignore */ }
    try { this.node?.disconnect(); } catch { /* ignore */ }
    try { this.srcNode?.disconnect(); } catch { /* ignore */ }
    try { this.stream?.getTracks().forEach((t) => t.stop()); } catch { /* ignore */ }
    try { void this.ctx?.close(); } catch { /* ignore */ }
    this.node = null;
    this.srcNode = null;
    this.stream = null;
    this.ctx = null;
  }

  private onWorklet(ev: MessageEvent, epoch: number) {
    if (epoch !== this.epoch || !this.armed) return;
    const m = ev.data;
    if (m?.type === "vad") {
      this.h.onSpeech?.(!!m.active);
      this.h.onInterim(m.active ? "●" : "");
      if (m.active) this.h.onHealth("speech");
      else if (this.armed) this.h.onHealth("listening");
    } else if (m?.type === "level") {
      this.h.onLevel?.(typeof m.rms === "number" ? m.rms : 0);
    } else if (m?.type === "utt" && m.buf instanceof ArrayBuffer) {
      if (m.buf.byteLength < MIN_UTT_SAMPLES * 2) return; // 短すぎる突発は捨てる
      this.submitUtt(m.buf, epoch);
    }
  }

  /**
   * 発話区間のPCM16をASRへ送る。in-flightは1本 — 実行中なら最新1件を
   * キューへ置き、古い保留は捨てる（bounded queue）。滞留が
   * QUEUE_STALE_MSを越えた発話は鮮度切れとして捨て、古い音声が
   * 遅れて指令化しないようにする（R7 1.4/S05/S09）。
   */
  private submitUtt(buf: ArrayBuffer, epoch: number) {
    if (this.inFlight) {
      this.queued = { buf, at: Date.now() }; // latest-wins — 古い未処理発話を溜めない
      return;
    }
    void this.recognize(buf, epoch, 0);
  }

  private async recognize(buf: ArrayBuffer, epoch: number, queueWaitMs: number) {
    const ac = new AbortController();
    this.inFlight = ac;
    this.h.onHealth("transcribing");
    const sentAtMs = Date.now();
    const timer = setTimeout(() => ac.abort(), ASR_TIMEOUT_MS);
    try {
      const r = await fetch(
        `/api/asr/recognize?session_id=${encodeURIComponent(this.sessionId ?? "")}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/octet-stream" },
          body: buf,
          signal: ac.signal,
        },
      );
      if (epoch !== this.epoch || !this.armed) return;
      const j = (await r.json().catch(() => ({}))) as AsrResult & { detail?: string };
      const timing: AsrTiming = {
        sentAtMs,
        doneAtMs: Date.now(),
        serverLatencyMs: typeof j.latency_ms === "number" ? j.latency_ms : null,
        queueWaitMs,
      };
      if (!r.ok) {
        // ASR故障はarmを解く（voice由来の継続動作を止める）。
        // 403/503/timeoutを区別したdetailをUIへ渡す（R7 1.2/1.4）。
        this.armed = false;
        this.cleanup();
        const detail =
          typeof j.detail === "string" && j.detail ? `asr_${r.status}:${j.detail}` : `asr_${r.status}`;
        this.h.onHealth("failed", detail);
        return;
      }
      const text = (j.text || "").trim();
      if (!text) {
        if (this.armed) this.h.onHealth("listening");
        return;
      }
      // 停止語は最優先 — 確定テキストが停止語なら指令dispatchより先に送る
      if (isStopCommandText(text)) {
        this.h.onInterim("");
        this.h.onStop();
        if (this.armed) this.h.onHealth("listening");
        return;
      }
      this.h.onInterim("");
      this.h.onCommit(text, timing);
      if (this.armed) this.h.onHealth("listening");
    } catch {
      if (epoch !== this.epoch || !this.armed) return;
      // abort/timeout/network — 継続不能な故障としてarmを解く
      this.armed = false;
      this.cleanup();
      this.h.onHealth("failed", "asr_unreachable");
    } finally {
      clearTimeout(timer);
      if (this.inFlight === ac) this.inFlight = null;
      // 保留中の最新発話を処理（古いものは捨て済み）。
      // 鮮度切れ（QUEUE_STALE_MS超過）はここで棄却する — 混雑時に
      // 遅れて届いた古い音声を指令として再送しない（S09）。
      const next = this.queued;
      this.queued = null;
      if (next && this.armed && epoch === this.epoch) {
        const waited = Date.now() - next.at;
        if (waited <= QUEUE_STALE_MS) {
          void this.recognize(next.buf, epoch, waited);
        }
        // 鮮度切れは黙って捨てる（表示はlastTiming/healthで追跡可）
      }
    }
  }
}
