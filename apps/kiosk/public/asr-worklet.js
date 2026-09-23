// kotoba ASR capture worklet — PCM16/16kHz化とVAD（発話区間検出）。
// メインスレッドへ {type:'pcm', buf:Int16Array} と {type:'vad', active:bool}
// をpostMessageする。認識・指令化はこのファイルでは行わない（転送のみ）。
//
// R7: 実AudioContext sampleRateが16kHzと一致しない環境（Firefox/実機HWで
// 要求rateが無視される場合がある）へ対応 — {type:'config', inputRate,
// targetRate} を受け、不一致時はlinear resampleして16kHz PCM16を吐く。
// {type:'level', rms} を~10Hzで出しUIの入力レベルmeterに供する。
//
// VADはエネルギーベースの保守的実装:
//   - RMS > ON閾値 が ON_HOLD_MS 継続 → 発話開始
//   - RMS < OFF閾値 が OFF_HOLD_MS 継続 → 発話終了（utterance確定）
//   - 最大発話 MAX_UTT_S を越えたら強制終端（無限bufferを防ぐ）
// 閾値はキャリブレーション不要の固定値（短い号令を取りこぼさないよう
// 立ち上がりを緩く、終端をやや遅くする）。
const ON_RMS = 0.012;      // 発話開始エネルギー（正規化16bit）
const OFF_RMS = 0.008;     // 発話継続の下限（ヒステリシス）
const ON_HOLD_MS = 80;     // 立ち上がり判定 — 「前」の短い母音を拾う
const OFF_HOLD_MS = 450;   // 終端判定 — 語尾の息切れで切らない程度
const MAX_UTT_S = 12;      // 1発話の上限（bounded queueの前提）
const PRE_ROLL_MS = 240;   // 立ち上がり前の音声を保持（頭切れ防止）
const LEVEL_EVERY_S = 0.1; // level meterの報告間隔

class KotobaAsrCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.active = false;          // 発話区間内か
    this.onHoldMs = 0;
    this.offHoldMs = 0;
    this.utt = [];                // 現発話のInt16チャンク（出力rate）
    this.uttLen = 0;
    this.pre = [];                // 直前リング（頭切れ防止・出力rate）
    this.preLen = 0;
    // resample状態（inputRate≠targetRateのとき使用）
    this.inRate = 0;              // 0=未設定 — 既定は「input==sampleRate」
    this.outRate = 0;
    this.rsTail = null;           // 前ブロック末尾のFloat32（補間の跨ぎ用）
    this.rsPos = 0;               // 次に生成する出力sampleに対応する入力位置
    this.lastLevel = 0;           // level報告の時刻管理
    this.port.onmessage = (e) => {
      if (e.data === "flush") this._flush();
      else if (e.data === "reset") {
        this.utt = []; this.uttLen = 0; this.active = false;
        this.onHoldMs = 0; this.offHoldMs = 0;
        this.rsTail = null; this.rsPos = 0;
      } else if (e.data && e.data.type === "config") {
        this.inRate = e.data.inputRate || 0;
        this.outRate = e.data.targetRate || 0;
        this.rsTail = null; this.rsPos = 0;
      }
    };
  }

  _flush() {
    if (this.uttLen === 0) return;
    const out = new Int16Array(this.uttLen);
    let o = 0;
    for (const c of this.utt) { out.set(c, o); o += c.length; }
    this.port.postMessage({ type: "utt", buf: out.buffer }, [out.buffer]);
    this.utt = [];
    this.uttLen = 0;
  }

  // float32入力（inputRate）→ int16出力（outRate）。不一致時はlinear補間。
  // ブロック境界の連続性のため前ブロック末尾1sampleと小数位置を保持する。
  _toPcm16(input) {
    const inRate = this.inRate || sampleRate;
    const outRate = this.outRate || inRate;
    if (inRate === outRate) {
      const pcm = new Int16Array(input.length);
      for (let i = 0; i < input.length; i++) {
        const s = Math.max(-1, Math.min(1, input[i]));
        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      return pcm;
    }
    const ratio = inRate / outRate;
    // 前ブロック末尾を先頭へ繋げた拡張入力で補間の連続性を保つ
    const tail = this.rsTail;
    const ext = new Float32Array((tail ? tail.length : 0) + input.length);
    if (tail) ext.set(tail, 0);
    ext.set(input, tail ? tail.length : 0);
    const cap = Math.floor((ext.length - 1 - this.rsPos) / ratio) + 1;
    const pcm = new Int16Array(Math.max(0, cap));
    let n = 0;
    let pos = this.rsPos;
    while (pos + 1 < ext.length) {
      const i = Math.floor(pos);
      const f = pos - i;
      const s = ext[i] * (1 - f) + ext[i + 1] * f;
      const c = Math.max(-1, Math.min(1, s));
      pcm[n++] = c < 0 ? c * 0x8000 : c * 0x7fff;
      pos += ratio;
    }
    // 次ブロックへ — 未消化の先頭位置と末尾sampleを保持
    this.rsPos = pos - Math.floor(pos);
    this.rsTail = ext.slice(Math.floor(pos) - 1 >= 0 ? Math.floor(pos) - 1 : 0);
    this.rsPos += Math.floor(pos) - (ext.length - this.rsTail.length);
    return pcm.subarray(0, n);
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch || ch.length === 0) return true;
    // float32 → int16（実sampleRate≠16kならresampleして16kHzを出す）
    const inRate = this.inRate || sampleRate;
    const n = ch.length;
    let sum = 0;
    for (let i = 0; i < n; i++) sum += ch[i] * ch[i];
    const rms = Math.sqrt(sum / n);
    const frameMs = (n / inRate) * 1000;
    const pcm = this._toPcm16(ch);
    const outRate = this.outRate || inRate;

    // 入力レベルを一定周期で報告（UI meter用 — VADとは独立）
    if (currentTime - this.lastLevel >= LEVEL_EVERY_S) {
      this.lastLevel = currentTime;
      this.port.postMessage({ type: "level", rms });
    }

    // 直前リングを常に更新（発話頭の欠落を防ぐ）
    this.pre.push(pcm);
    this.preLen += pcm.length;
    const preCap = Math.ceil((PRE_ROLL_MS / 1000) * outRate);
    while (this.preLen > preCap) { this.preLen -= this.pre[0].length; this.pre.shift(); }

    if (!this.active) {
      if (rms > ON_RMS) {
        this.onHoldMs += frameMs;
        if (this.onHoldMs >= ON_HOLD_MS) {
          this.active = true;
          this.offHoldMs = 0;
          // pre-rollを発話の先頭へ付ける
          for (const c of this.pre) { this.utt.push(c); this.uttLen += c.length; }
          this.port.postMessage({ type: "vad", active: true });
        }
      } else {
        this.onHoldMs = 0;
      }
      return true;
    }

    // 発話中 — bufferへ積む
    this.utt.push(pcm);
    this.uttLen += pcm.length;
    if (rms < OFF_RMS) {
      this.offHoldMs += frameMs;
      if (this.offHoldMs >= OFF_HOLD_MS) {
        this.active = false;
        this.onHoldMs = 0;
        this._flush();
        this.port.postMessage({ type: "vad", active: false });
      }
    } else {
      this.offHoldMs = 0;
    }
    if (this.uttLen / outRate >= MAX_UTT_S) {
      this.active = false;
      this.onHoldMs = 0;
      this._flush();
      this.port.postMessage({ type: "vad", active: false });
    }
    return true;
  }
}

registerProcessor("kotoba-asr-capture", KotobaAsrCapture);
