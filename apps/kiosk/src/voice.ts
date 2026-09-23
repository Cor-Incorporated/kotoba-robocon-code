/**
 * VoiceController — Chrome音声操縦（R5-05）。
 *
 * 原則:
 * - 明示armのみでマイク起動。ページ表示だけでは開始しない。
 * - 既定経路は on-device Web Speech。processLocally代入は対応の証明に
 *   しない — SpeechRecognition.available({processLocally:true}) の実結果を
 *   probeして表示し、非対応・pack不在・未確認を区別する。
 *   第三者ASRへの黙ったfallbackはしない。
 * - interimは字幕/候補状態のみ。確定は final または「辞書完全一致が
 *   安定＋無音境界」を満たした候補。interim文字列だけでactuateしない。
 * - 停止語はfinalを待たない — interimに明確な停止語が現れた時点で
 *   安定待ち・解釈・送信キューより先にstopNow相当へ送る。
 * - epoch = arm/再開/provider変更の世代。旧epochの遅着結果は捨てる。
 *   同一(utterance,epoch)内で指令は一度だけ。
 * - マイク/recognizer故障はheartbeat対象 — voice健康が損なわれたら
 *   armを解き、voice由来の操作継続を止める。
 */

export type VoiceLocality =
  | "on-device" // available({processLocally:true})=availableを確認
  | "pack-required" // downloadable — 明示installが必要
  | "unverified" // probe APIなし — クラウド経路の可能性を隠さない
  | "unavailable"; // 非対応

export type VoiceHealth =
  | "off"
  | "starting"
  | "listening"
  | "restarting"
  | "failed";

export interface VoiceProbe {
  hasSR: boolean;
  secureContext: boolean;
  hasAvailableApi: boolean;
  hasInstallApi: boolean;
  locality: VoiceLocality;
  detail: string;
}

// 停止語 — finalを待たず即応する明確な群。
// 「やめ」等の部分文字・「ストップしないで」の否定をSTOPへ誤配しないため、
// 停止語の直前後に否定語が無いことを確認する。
const STOP_WORDS = ["止まって", "止まれ", "とまって", "とまれ", "ストップ",
                    "停止", "危ない", "あぶない"];
const NEGATIONS = ["ないで", "なし", "やめない", "ずに"];

// 号令辞書 — interim安定commitの対象は「完全な辞書一致」のみ
// （部分一致「ま」→「前」等で過早発火しない）。能力profile由来の語に限定。
const COMMAND_LEXICON = new Set([
  "前", "前へ", "進んで", "後ろ", "後ろへ", "下がって", "後ろに下がって",
  "少し前", "少し右", "少し左", "右", "左",
  "右を向いて", "左を向いて", "後ろを向いて", "後ろを振り返って",
  "振り返って", "止まって", "止まれ", "ストップ",
  "割って", "打って", "棒を振って", "終わって",
]);

/** 停止語判定 — 否定・部分文字をSTOPへ誤配しない。
 *  VoiceControllerとLocalAsrControllerで共有（停止契約は経路不変）。 */
export function isStopCommandText(t: string): boolean {
  const s = t.replace(/\s+/g, "");
  for (const w of STOP_WORDS) {
    const i = s.indexOf(w);
    if (i < 0) continue;
    const before = s.slice(0, i);
    const after = s.slice(i + w.length);
    // 「ストップしないで」「止まらないで」はSTOPではない
    if (NEGATIONS.some((n) => before.endsWith(n) || after.startsWith(n))) {
      continue;
    }
    return true;
  }
  return false;
}

const INTERIM_STABLE_MS = 180; // 候補安定時間（probe値150〜250の帯）
const RESTART_MAX = 5; // 連続再起動の上限（無限restartで健康を偽装しない）
const RESTART_DELAY_MS = 250;

export function probeVoice(): Promise<VoiceProbe> {
  const SR =
    (window as any).SpeechRecognition ||
    (window as any).webkitSpeechRecognition;
  const base: VoiceProbe = {
    hasSR: !!SR,
    secureContext: window.isSecureContext === true,
    hasAvailableApi: !!(SR && typeof SR.available === "function"),
    hasInstallApi: !!(SR && typeof SR.install === "function"),
    locality: "unavailable",
    detail: "",
  };
  if (!SR) {
    base.detail = "SpeechRecognition未実装";
    return Promise.resolve(base);
  }
  if (!base.secureContext) {
    base.detail = "secure contextではありません";
    return Promise.resolve(base);
  }
  if (!base.hasAvailableApi) {
    // Chrome 139未満等 — ローカル処理の可否をAPIで確認できない
    base.locality = "unverified";
    base.detail = "on-device確認APIなし（Chrome 139+で検収）";
    return Promise.resolve(base);
  }
  return SR.available({ langs: ["ja-JP"], processLocally: true })
    .then((v: string) => {
      if (v === "available") {
        base.locality = "on-device";
        base.detail = "ja-JP on-device利用可";
      } else if (v === "downloadable" || v === "downloading") {
        base.locality = "pack-required";
        base.detail = `ja-JPローカルpack要install (${v})`;
      } else {
        base.locality = "unavailable";
        base.detail = `on-device非対応 (${v}) — クラウドASRへは黙って落としません`;
      }
      return base;
    })
    .catch(() => {
      base.locality = "unverified";
      base.detail = "available()呼出し失敗 — locality未確認";
      return base;
    });
}

interface VoiceHandlers {
  onCommit: (text: string) => void; // 確定指令 → 通常/ゲーム共通dispatch
  onStop: () => void; // 即時STOP — 解釈・キューを迂回
  onInterim: (text: string) => void; // 字幕表示（指令化しない）
  onHealth: (h: VoiceHealth, detail?: string) => void;
}

export class VoiceController {
  private recog: any = null;
  private epoch = 0;
  private restarts = 0;
  private restartTimer: ReturnType<typeof setTimeout> | null = null;
  private armed = false;
  private interimCandidate = "";
  private interimKey = "";
  private interimSince = 0;
  private interimTimer: ReturnType<typeof setTimeout> | null = null;
  private committedKeys = new Set<string>();
  private stoppedKeys = new Set<string>();
  private h: VoiceHandlers;

  constructor(h: VoiceHandlers) {
    this.h = h;
  }

  isArmed() {
    return this.armed;
  }

  arm(localityVerified: boolean) {
    const SR =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;
    if (!SR || this.armed) return;
    this.armed = true;
    this.restarts = 0;
    this.startSession(localityVerified);
  }

  disarm() {
    this.armed = false;
    this.epoch += 1; // 旧epochの遅着を全て失効
    if (this.restartTimer) clearTimeout(this.restartTimer);
    this.restartTimer = null;
    this.clearInterim();
    try {
      this.recog?.stop();
    } catch {
      /* ignore */
    }
    this.recog = null;
    this.h.onHealth("off");
  }

  private startSession(localityVerified: boolean) {
    const SR =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;
    const myEpoch = ++this.epoch;
    const r = new SR();
    r.lang = "ja-JP";
    r.continuous = true; // 号令モードは連続セッション — 端点制御ではない
    r.interimResults = true;
    r.maxAlternatives = 3;
    if (localityVerified) {
      try {
        r.processLocally = true; // probe済み時のみ設定
      } catch {
        /* non-fatal */
      }
    }
    this.h.onHealth("starting");
    r.onstart = () => {
      if (myEpoch !== this.epoch) return;
      this.restarts = 0;
      this.h.onHealth("listening");
    };
    r.onresult = (e: any) => {
      if (myEpoch !== this.epoch || !this.armed) return; // 旧epochは捨てる
      this.handleResults(e, myEpoch);
    };
    r.onerror = (e: any) => {
      if (myEpoch !== this.epoch) return;
      const kind = e?.error || "unknown";
      if (kind === "no-speech" || kind === "aborted") {
        // 無音タイムアウトは故障ではない — onendで有限再起動へ
        return;
      }
      // not-allowed/audio-capture/network/service-not-allowed等は故障
      this.armed = false;
      this.h.onHealth("failed", kind);
      this.h.onInterim("");
    };
    r.onend = () => {
      if (myEpoch !== this.epoch) return;
      this.clearInterim();
      this.h.onInterim("");
      if (!this.armed) {
        this.h.onHealth("off");
        return;
      }
      // continuous engineの終了 — 有限回・単一timerで再起動。
      // 完了しない再起動でhealthを偽装しない。
      if (this.restarts >= RESTART_MAX) {
        this.armed = false;
        this.h.onHealth("failed", "restart_exhausted");
        return;
      }
      this.restarts += 1;
      this.h.onHealth("restarting");
      this.restartTimer = setTimeout(() => {
        if (!this.armed) return;
        this.startSession(localityVerified);
      }, RESTART_DELAY_MS);
    };
    this.recog = r;
    try {
      r.start();
    } catch {
      this.armed = false;
      this.h.onHealth("failed", "start_refused");
    }
  }

  private clearInterim() {
    this.interimCandidate = "";
    this.interimKey = "";
    this.interimSince = 0;
    if (this.interimTimer) clearTimeout(this.interimTimer);
    this.interimTimer = null;
  }

  /** 停止語判定 — 否定・部分文字をSTOPへ誤配しない */
  private isStopCommand(t: string): boolean {
    return isStopCommandText(t);
  }

  private handleResults(e: any, epoch: number) {
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const res = e.results[i];
      const uttKey = `${epoch}:${i}`;
      // STOP最優先 — interim/final問わず明確な停止語で即送
      // （同一utterance内で重複排除、別発話の再停止は妨げない）
      const head = (res[0]?.transcript || "").trim();
      if (head && !this.stoppedKeys.has(uttKey) && this.isStopCommand(head)) {
        this.stoppedKeys.add(uttKey);
        this.committedKeys.add(uttKey); // 停止は確定 — 後着変化で再指令化しない
        this.clearInterim();
        this.h.onInterim("");
        this.h.onStop();
        continue;
      }
      if (res.isFinal) {
        const tx = head;
        // epoch+utterance(resultIndex)で一度だけ — interim安定commit
        // 済みのfinal・同一発話の重複finalは再送しない
        if (tx && !this.committedKeys.has(uttKey)) {
          this.committedKeys.add(uttKey);
          this.h.onCommit(tx);
        }
      } else {
        interim += head;
        this.trackInterim(head, epoch, uttKey);
      }
    }
    this.h.onInterim(interim);
  }

  /**
   * interimの早期commit — 「完全な辞書一致」が安定した場合のみ。
   * Web Speechから独立VADの無音境界は取れないため、安定時間のみで
   * 判定し、未完の構文（数値・接続詞を含む表現）は辞書外＝commitしない。
   * commit keyはutteranceのresultIndexと同一 — 後着finalを再送しない。
   */
  private trackInterim(text: string, epoch: number, uttKey: string) {
    const t = text.trim();
    if (!t || !COMMAND_LEXICON.has(t)) {
      if (this.interimKey === uttKey) this.clearInterim();
      return;
    }
    if (t === this.interimCandidate && this.interimKey === uttKey) {
      return; // 安定継続中 — timerは既に走っている
    }
    this.interimKey = uttKey;
    this.interimCandidate = t;
    this.interimSince = Date.now();
    if (this.interimTimer) clearTimeout(this.interimTimer);
    this.interimTimer = setTimeout(() => {
      if (epoch !== this.epoch || !this.armed) return;
      if (this.interimKey !== uttKey || this.interimCandidate !== t) return;
      if (Date.now() - this.interimSince < INTERIM_STABLE_MS - 20) return;
      // 安定した完全辞書一致 — 発話終端(final)を待たずcommit
      if (this.committedKeys.has(uttKey)) return;
      this.committedKeys.add(uttKey);
      this.clearInterim();
      this.h.onInterim("");
      this.h.onCommit(t);
    }, INTERIM_STABLE_MS);
  }
}
