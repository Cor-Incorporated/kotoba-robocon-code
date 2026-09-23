import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

const htmlPath = process.env.RENDER_HTML ??
  new URL("../../apps/kiosk/public/render/index.html", import.meta.url);
const html = readFileSync(htmlPath, "utf8");
const begin = html.indexOf("<script>");
const end = html.indexOf("</script>", begin);
assert.ok(begin >= 0 && end > begin, "kotoba bootstrap script exists");

function startWorker() {
  const reports = [];
  const forwarded = [];
  const intervals = [];
  let injected = null;
  class NativeWorker { constructor(url) { this.url = url; } }
  class FakeXHR {
    open() {}
    send() { this.responseText = ""; }
  }
  const page = {
    window: { Worker: NativeWorker, devicePixelRatio: 1 },
    location: { search: "?rg=test-generation" },
    document: { visibilityState: "visible" },
    URLSearchParams,
    URL: class { static createObjectURL(blob) { injected = blob.parts[0]; return "blob:patched"; } },
    Blob: class { constructor(parts) { this.parts = parts; } },
    XMLHttpRequest: FakeXHR,
    console,
  };
  vm.runInNewContext(html.slice(begin + 8, end), page);
  new page.window.Worker("blob:viser-worker");
  assert.equal(typeof injected, "string");

  class Socket {
    constructor() { this.readyState = 0; this.listeners = {}; }
    addEventListener(event, cb) { (this.listeners[event] ??= []).push(cb); }
    emit(event, value = {}) { for (const cb of this.listeners[event] ?? []) cb(value); }
  }
  const self = {
    WebSocket: Socket,
    addEventListener() {},
    postMessage(data, transfer) { forwarded.push({ data, transfer }); },
  };
  const worker = {
    self,
    WebSocket: Socket,
    BroadcastChannel: class { postMessage(data) { reports.push(data.kotobaRender); } },
    setInterval(cb) { intervals.push(cb); },
    setTimeout() { return 1; },
    clearTimeout() {},
    Date,
    console,
  };
  vm.runInNewContext(injected, worker);
  const ws = new self.WebSocket("ws://local");
  ws.readyState = 1;
  ws.emit("open");
  ws.emit("message", { data: new ArrayBuffer(128) });
  return { reports, forwarded, intervals, self, ws };
}

test("PM01 node と replay 完了を両方復号した現socketだけscene成立", () => {
  const h = startWorker();
  const transfer = [new ArrayBuffer(8)];
  const node = { type: "BatchedGlbMessage", name: "/bodies/PM01/group0" };
  const nodeBatch = { type: "message_batch", messages: [node] };
  h.self.postMessage(nodeBatch, transfer);
  assert.equal(h.forwarded[0].data, nodeBatch);
  assert.equal(h.forwarded[0].transfer, transfer, "transferable buffer is unchanged");
  assert.equal(h.reports.some((m) => m.ev === "scene_node"), false);
  h.self.postMessage({ type: "message_batch", messages: [{ type: "ReplayDoneMessage" }] });
  assert.equal(h.reports.filter((m) => m.ev === "scene_node" && m.cid === 1).length, 1);
  h.intervals[0]();
  assert.equal(h.reports.at(-1).scene, true);
  assert.ok(h.reports.at(-1).bytes < 50_000_000, "scene must not depend on byte threshold");
});

test("新socketは旧scene証拠を継承せず、静止中もACKを継続", () => {
  const h = startWorker();
  h.self.postMessage({ type: "message_batch", messages: [
    { type: "BatchedGlbMessage", name: "/bodies/PM01/group0" },
    { type: "ReplayDoneMessage" },
  ] });
  h.intervals[0]();
  assert.equal(h.reports.at(-1).scene, true);
  const ws2 = new h.self.WebSocket("ws://local");
  ws2.readyState = 1;
  ws2.emit("open");
  ws2.emit("message", { data: new ArrayBuffer(24) });
  h.intervals[0]();
  assert.equal(h.reports.at(-1).scene, false);
  assert.equal(h.reports.at(-1).cid, 2);
  h.intervals[0]();
  assert.equal(h.reports.at(-1).ev, "ack", "idle socket keeps reporting liveness");
});
