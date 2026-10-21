// ============================================================================
// AskMe AI + DeepSeek -> OpenAI-compatible gateway (Node.js, zero dependencies)
// ----------------------------------------------------------------------------
// Exposes AskMe AI and DeepSeek web chat as OpenAI-compatible API for tools
// like 9Router, Cursor, Cline, OpenCode, Codex CLI, etc.
//
//   Run:   DEEPSEEK_TOKEN=your-deepseek-usertoken node askme-openai-gateway.mjs
//   Env:   PORT=8787
//          MODEL_ID=askme-ai            (AskMe backend, no login needed)
//          DS_MODEL_ID=deepseek         (DeepSeek backend, needs DEEPSEEK_TOKEN)
//          API_KEY=optional-shared-secret
//          ASKME_ORIGIN=https://www.ask-me-ai.app
//          DS_ORIGIN=https://chat.deepseek.com
//          DEEPSEEK_TOKEN=...           (see below)
//   Needs: Node.js 18+ (global fetch). Ships its DeepSeek PoW solver as
//          sha3_wasm_bg.wasm (DeepSeek's own DeepSeekHashV1 solver).
//
// DeepSeek login token: log in at https://chat.deepseek.com/ in your browser,
// open devtools -> Application -> Local Storage -> chat.deepseek.com, copy the
// `userToken` value. (It grants full account access: keep it secret, rotate by
// logging out if it ever leaks.) The gateway also needs it as DEEPSEEK_TOKEN.
//
// Then: Base URL http://localhost:8787/v1, models `askme-ai` / `deepseek`.
//
// Endpoints: GET /v1/models, POST /v1/chat/completions (stream + non-stream),
// GET /healthz, GET /. AskMe/DeepSeek knobs (temperature/tools/vision) don't
// exist, so those fields are accepted but ignored. Requests are stateless.
// ============================================================================

import http from "node:http";
import crypto from "node:crypto";
import fs from "node:fs";

// ------------------------------- config ------------------------------------
const PORT = Number(process.env.PORT || 8787);
const MODEL_ID = process.env.MODEL_ID || "askme-ai";
const DS_MODEL_ID = process.env.DS_MODEL_ID || "deepseek";
const DS_SEARCH_MODEL_ID = process.env.DS_SEARCH_MODEL_ID || "deepseek-search";
const API_KEY = process.env.API_KEY || "";
const ASKME_ORIGIN = (process.env.ASKME_ORIGIN || "https://www.ask-me-ai.app").replace(/\/+$/, "");
const DS_ORIGIN = (process.env.DS_ORIGIN || "https://chat.deepseek.com").replace(/\/+$/, "");
const DEEPSEEK_TOKEN = process.env.DEEPSEEK_TOKEN || "";
// Stable device id derived from the token: a real client keeps one device
// id for life, while random-per-restart is a bot tell.
function dsDeviceId() {
  try {
    const h = crypto.createHash("sha256").update("ds-device|" + DEEPSEEK_TOKEN).digest("hex");
    return h.slice(0, 8) + "-" + h.slice(8, 12) + "-4" + h.slice(13, 16) + "-a" + h.slice(17, 20) + "-" + h.slice(20, 32);
  } catch (err) {
    return "00000000-0000-4000-8000-000000000000";
  }
}
const DS_DEVICE_ID = dsDeviceId();
// How long to reuse one chat session (a human keeps a tab open instead of
// opening a fresh single-shot session per message). 0 = always fresh.
function dsEnvMs(name, def) {
  const v = process.env[name];
  if (v == null || v === "") return def;
  const n = Number(v);
  return Number.isFinite(n) && n >= 0 ? n : def;
}
const DS_SESSION_REUSE_MS = dsEnvMs("DS_SESSION_REUSE_MS", 20 * 60 * 1000);
// Max extra random pause before a turn (actual pause is 150ms + up to this).
// 0 = no pause. Small jitter so requests stop looking mechanical.
const DS_HUMAN_DELAY_MS = dsEnvMs("DS_HUMAN_DELAY_MS", 800);
let dsSessionCache = { id: null, at: 0 };
function dsSleep(ms) { return new Promise(function (res) { setTimeout(res, ms); }); }
async function dsHumanPause() {
  if (!(DS_HUMAN_DELAY_MS > 0)) return;
  await dsSleep(150 + Math.random() * DS_HUMAN_DELAY_MS);
}
const APP_VERSION = "unknown";
const MAX_PROMPT_CHARS = 12000;
const ASKME_TIMEOUT_MS = 120000;
const DS_TIMEOUT_MS = 180000;

// ---------------- DeepSeek PoW: bundled WASM solver --------------------------
// DeepSeekHashV1 uses a NON-STANDARD hash (plain SHA3-256 does NOT match), so
// the gateway ships DeepSeek's own solver module, sha3_wasm_bg.wasm
// (downloaded from https://fe-static.deepseek.com/chat/), and drives its
// wasm_solve() directly. Verified live: difficulty-144000 challenges solve in
// well under a second. ABI: malloc(size, align), wasm_solve(retptr, cptr,
// clen, pptr, plen, difficulty) -> [found:i32, answer:f64].
let DS_WASM = null;
try {
  const wasmBytes = fs.readFileSync(new URL("./sha3_wasm_bg.wasm", import.meta.url));
  DS_WASM = (await WebAssembly.instantiate(wasmBytes)).instance.exports;
} catch (err) {
  console.log("DeepSeek WASM solver unavailable (" + (err && err.message) + "): DeepSeek mode will not work.");
}

function dsWasmSolve(challengeHex, prefix, limit) {
  // Returns the PoW answer (number) or null when unsolvable.
  const malloc = DS_WASM.__wbindgen_export_0;
  const mem = function () { return new Uint8Array(DS_WASM.memory.buffer); };
  const view = function () { return new DataView(DS_WASM.memory.buffer); };
  const writeStr = function (s) {
    const b = new TextEncoder().encode(s);
    const ptr = malloc(b.length, 1) >>> 0;
    mem().set(b, ptr);
    return [ptr, b.length];
  };
  const cp = writeStr(challengeHex);
  const pp = writeStr(prefix);
  const rp = malloc(64, 1) >>> 0;
  DS_WASM.wasm_solve(rp, cp[0], cp[1], pp[0], pp[1], limit);
  if (view().getInt32(rp, true) === 0) return null;
  return view().getFloat64(rp + 8, true);
}

// --------------------------- OpenAI mapping --------------------------------
function estimateTokens(text) {
  return Math.ceil(String(text || "").length / 4);
}

function extractTextContent(content) {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    const parts = [];
    for (const p of content) {
      if (p && p.type === "text" && typeof p.text === "string") parts.push(p.text);
      else if (p && p.type === "image_url") parts.push("[image omitted: not supported by this bridge]");
    }
    return parts.join("\n");
  }
  return String(content == null ? "" : content);
}

function openaiError(status, message, type) {
  const err = new Error(message);
  err.status = status;
  err.openaiType = type || (status === 400 ? "invalid_request_error" : "server_error");
  return err;
}

function buildPromptText(messages, maxChars) {
  const limit = maxChars || MAX_PROMPT_CHARS;
  if (!Array.isArray(messages) || messages.length === 0) {
    throw openaiError(400, "messages array is required and must not be empty", "invalid_request_error");
  }
  let system = "";
  const turns = [];
  for (const m of messages) {
    const role = (m && m.role) || "user";
    const text = extractTextContent(m && m.content).trim();
    if (!text) continue;
    if (role === "system") system += (system ? "\n" : "") + text;
    else if (role === "assistant") turns.push({ who: "Assistant", text: text });
    else if (role === "tool" || role === "function") turns.push({ who: "Tool result", text: text });
    else turns.push({ who: "User", text: text });
  }
  if (turns.length === 0) throw openaiError(400, "no usable message content found", "invalid_request_error");
  const block = function () {
    return turns.map(function (t) { return t.who + ": " + t.text; }).join("\n\n");
  };
  while (turns.length > 1 && (system.length + block().length) > limit) turns.shift();
  let prompt = (system ? "System: " + system + "\n\n" : "") + block();
  if (prompt.length > limit) prompt = prompt.slice(prompt.length - limit);
  return prompt;
}

function chartToText(ev) {
  const lines = [];
  lines.push("[chart" + (ev.title ? ": " + ev.title : "") + "]");
  try {
    const opt = ev.option || {};
    const cats = (opt.xAxis && opt.xAxis.data) || [];
    const series = (opt.series && opt.series[0]) || {};
    const vals = series.data || [];
    if (cats.length && vals.length) {
      lines.push(cats.join(" | "));
      lines.push(vals.map(function (v) { return String(v); }).join(" | "));
    } else {
      lines.push(JSON.stringify(opt).slice(0, 1000));
    }
  } catch (err) {
    lines.push("(chart data unavailable)");
  }
  return lines.join("\n");
}

function newCompletionId() {
  return "chatcmpl-askme-" + crypto.randomBytes(12).toString("hex");
}

function openaiChunk(id, model, delta, finishReason) {
  return "data: " + JSON.stringify({
    id: id,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model: model,
    choices: [{ index: 0, delta: delta, finish_reason: finishReason || null }]
  });
}

// ============================ AskMe backend ================================
const BCG = [1, 2, 2, 4, 8, 32, 5, 160];
const AUW = [123, 47, 145, 68, 232, 60, 165, 105];
const AS0 = [210, 23, 142, 243, 86, 186, 13, 193];
const SUFFIXES = ["Zy3", "A7x", "Lp2", "Hj4", "K9m", "Qw8", "Xs1", "Nv7"];

let askmeServerTimeOffsetMs = 0;

function bWQ(a, g) {
  let q = (1095977805 ^ (g & 65535)) >>> 0;
  for (let r = 0; r < a.length; r++) {
    const factor = (((g >>> ((r * 3) % 32)) & 7) + 1);
    q = (((q << 5) >>> 0) + q * factor + a[r]) >>> 0;
  }
  return (q ^ 1161906481 ^ ((g >>> 16) & 65535)) >>> 0;
}

function bWPConcat(g) {
  const h = [];
  for (let qi = 0; qi < 8; qi++) {
    const p = (((g >>> (qi * 4)) & 15) + 1);
    const suf = SUFFIXES[qi];
    for (let k = 0; k < suf.length; k++) h.push((((suf.charCodeAt(k) ^ 90) * p) & 255));
  }
  const m = [];
  for (let q = 7; q >= 0; q--) m.push(Math.floor(g / Math.pow(256, q)) & 255);
  return h.concat(m);
}

function bUv(a, seed) {
  let s = seed;
  const q = new Array(a.length);
  for (let r = 0; r < a.length; r++) {
    const mod = r % 4;
    if (mod === 0) q[r] = (a[r] ^ ((r * 7 + 13) & 255) ^ (s & 255)) & 255;
    else if (mod === 1) q[r] = ((a[r] + ((r * 11 + 17) & 255) - (s & 63)) & 255);
    else if (mod === 2) q[r] = (a[r] ^ ((r * 13 + 23) & 255) ^ ((s >>> 2) & 15)) & 255;
    else q[r] = ((a[r] - ((r * 17 + 31) & 127) + ((s >>> 4) & 31)) & 255);
    s = (((Math.imul(s, 1103515245) + 12345) >>> 0) >>> 16) & 0x7FFF;
  }
  return q;
}

function askmeMakeRand16(h) {
  const rnd = crypto.randomBytes(16);
  const hour = Math.floor(h / 3600000) % 24;
  const minute = Math.floor(h / 60000) % 60;
  const t = ((hour * 60 + minute) ^ 43981) & 255;
  const e = Buffer.alloc(16);
  for (let i = 0; i < 16; i++) e[i] = (i % 4 === 0) ? ((rnd[i] ^ t) & 255) : rnd[i];
  return e;
}

function askmeGenToken(deviceId, nowMs) {
  const h = nowMs;
  const g = Math.floor(h / 300000);
  const e = askmeMakeRand16(h);
  const d = Buffer.from(deviceId, "utf8");
  const b = [];
  for (let s = 7; s >= 0; s--) b.push(Math.floor(g / Math.pow(256, s)) & 255);
  b.push(d.length & 255);
  for (const x of d) b.push(x);
  for (const x of e) b.push(x);
  const r = bWQ(b, g);
  for (let s = 3; s >= 0; s--) b.push((r >>> (s * 8)) & 255);

  const sConcat = bWPConcat(g);
  const l = crypto.createHash("sha256").update(Buffer.from(sConcat)).digest();
  const k = [];
  for (let i = 0; i < l.length; i++) {
    const sb = l[i], ri = i % 8, o = BCG[ri], n = AUW[ri];
    k.push((((((sb ^ o) >>> 0) + (o & n) - (sb | n)) & 255) ^ AS0[ri]) & 255);
  }
  const p64 = Buffer.alloc(64);
  Buffer.from(k).copy(p64);
  const nHmac = crypto.createHmac("sha256", p64).update(Buffer.from(b)).digest();

  const p0 = e[0];
  const mVal = ((g & 65535) ^ p0) & 0xFFFF;
  const lVal = (((g >>> 16) & 65535) ^ (p0 >>> 8)) & 0xFFFF;
  const pList = Array.from(nHmac);
  pList.push((mVal >>> 8) & 255, mVal & 255, (lVal >>> 8) & 255, lVal & 255);
  const kHash = crypto.createHash("sha256").update(Buffer.from(pList)).digest();

  const j = [];
  for (let s = 0; s < 32; s++) j.push((s % 2 === 0) ? nHmac[s] : kHash[s]);
  const f = b.concat(j);
  const seed = (Math.floor(g * 31337) & 0x7FFFFFFF);
  return Buffer.from(bUv(f, seed)).toString("base64").replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

const GATEWAY_DEVICE_ID = crypto.randomUUID();

function parseSseBlock(block) {
  const events = [];
  for (const line of block.split("\n")) {
    const t = line.trim();
    if (!t.startsWith("data:")) continue;
    const payload = t.slice(5).trim();
    if (!payload) continue;
    try {
      events.push(JSON.parse(payload));
    } catch (err) { /* ignore partial JSON */ }
  }
  return events;
}

async function askmeStream(prompt, onEvent, signal) {
  const body = {
    message: prompt,
    deviceId: GATEWAY_DEVICE_ID,
    utcOffsetMinutes: -new Date().getTimezoneOffset()
  };

  const doFetch = async function (token) {
    const ctl = new AbortController();
    const timer = setTimeout(function () { ctl.abort(); }, ASKME_TIMEOUT_MS);
    const onAbort = function () { ctl.abort(); };
    if (signal) {
      if (signal.aborted) ctl.abort();
      else signal.addEventListener("abort", onAbort, { once: true });
    }
    try {
      return await fetch(ASKME_ORIGIN + "/api/chat/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Cache-Control": "no-cache",
          "Accept": "text/event-stream",
          "X-App-Token": token,
          "X-App-Version": APP_VERSION,
          "X-AskMe-Streaming-Protocol": "1",
          "X-AskMe-Visualization-Protocol": "1",
          "X-AskMe-Music-Protocol": "1"
        },
        body: JSON.stringify(body),
        signal: ctl.signal
      });
    } finally {
      clearTimeout(timer);
      if (signal) signal.removeEventListener("abort", onAbort);
    }
  };

  let res = await doFetch(askmeGenToken(GATEWAY_DEVICE_ID, Date.now() + askmeServerTimeOffsetMs));
  if (res.status === 401) {
    let serverTime = null;
    try {
      const j = JSON.parse(await res.text());
      if (j && j.serverTime) serverTime = j.serverTime;
    } catch (err) { /* ignore */ }
    if (!serverTime) throw openaiError(502, "AskMe rejected the request (unauthorized)", "server_error");
    askmeServerTimeOffsetMs = serverTime - Date.now();
    res = await doFetch(askmeGenToken(GATEWAY_DEVICE_ID, serverTime));
    if (res.status === 401) throw openaiError(502, "AskMe rejected the request (unauthorized)", "server_error");
  }
  if (!res.ok || !res.body) {
    const txt = await res.text().catch(function () { return ""; });
    throw openaiError(502, "AskMe request failed (HTTP " + res.status + "): " + String(txt).slice(0, 200), "server_error");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const step = await reader.read();
    if (step.done) break;
    buf += decoder.decode(step.value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const evs = parseSseBlock(block);
      for (const ev of evs) onEvent(ev);
    }
  }
  buf += decoder.decode();
  if (buf.trim()) {
    for (const ev of parseSseBlock(buf)) onEvent(ev);
  }
}

// ============================ DeepSeek backend =============================
// Reverse-engineered from the chat.deepseek.com web client:
// - Bearer userToken auth, per-request PoW (DeepSeekHashV1 = SHA3-256 mining)
// - POST /api/v0/chat_session/create {}
// - POST /api/v0/chat/create_pow_challenge {target_path}
// - POST /api/v0/chat/completion {chat_session_id, parent_message_id, ...}
// - SSE stream: `event: <name>` / `data: <json>`; deltas are JSON-patch-ish ops.

const DS_TARGET_PATH = "/api/v0/chat/completion";
const DS_MODEL_TYPE = "deepseek-chat";

function dsHeaders(extra) {
  if (!DEEPSEEK_TOKEN) {
    throw openaiError(400, "DeepSeek backend needs DEEPSEEK_TOKEN (your chat.deepseek.com login `userToken`). Set it and restart the gateway.", "invalid_request_error");
  }
  // The web client sends these on every API call; /chat/completion answers
  // 40300 MISSING_HEADER without them (verified live).
  return Object.assign({
    "Content-Type": "application/json",
    "Accept": "*/*",
    "Origin": DS_ORIGIN,
    "Referer": DS_ORIGIN + "/",
    "Authorization": "Bearer " + DEEPSEEK_TOKEN,
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "x-device-id": DS_DEVICE_ID,
    "x-device-model": "Chrome",
    "x-client-bundle-id": "com.deepseek.chat",
    "x-client-platform": "web",
    "x-client-version": "1.0.0",
    "x-client-locale": "en_US",
    "x-client-timezone-offset": "0"
  }, extra || {});
}

async function dsPost(path, body, extraHeaders, timeoutMs) {
  const ctl = new AbortController();
  const timer = setTimeout(function () { ctl.abort(); }, timeoutMs || 30000);
  try {
    const res = await fetch(DS_ORIGIN + path, {
      method: "POST",
      headers: dsHeaders(extraHeaders),
      body: JSON.stringify(body == null ? {} : body),
      signal: ctl.signal
    });
    const text = await res.text();
    let json = null;
    try { json = JSON.parse(text); } catch (err) { /* keep null */ }
    return { status: res.status, json: json, text: text };
  } finally {
    clearTimeout(timer);
  }
}

function dsCheckBiz(json, what) {
  // DeepSeek wraps results: {code, msg, data:{biz_code, biz_msg, biz_data}}
  if (!json || typeof json.code !== "undefined") {
    const code = json && json.code;
    if (code === 40003 || code === 401) throw openaiError(502, "DeepSeek rejected the login token (INVALID_TOKEN). Paste a fresh `userToken` as DEEPSEEK_TOKEN.", "server_error");
    if (code !== 0 && code != null) throw openaiError(502, "DeepSeek " + what + " failed: code=" + code + " msg=" + ((json && json.msg) || "?"), "server_error");
  }
  const data = json && json.data;
  if (data && typeof data.biz_code !== "undefined" && data.biz_code !== 0) {
    throw openaiError(502, "DeepSeek " + what + " failed: biz_code=" + data.biz_code + " biz_msg=" + (data.biz_msg || "?"), "server_error");
  }
  return data && data.biz_data;
}

// Solve DeepSeekHashV1 via the bundled WASM: smallest i in [0, difficulty)
// with DeepSeekHash(salt + "_" + expireAt + "_" + i) === challenge.
// Returns {answer, ms} or null (the server occasionally hands out
// unsolvable challenges — caller refetches).
function dsSolvePow(challenge) {
  const prefix = challenge.salt + "_" + challenge.expireAt + "_";
  const target = challenge.challenge;
  const limit = Number(challenge.difficulty);
  if (!DS_WASM || !challenge.salt || !target || !(limit > 0) || limit > 1e9) return null;
  const t0 = Date.now();
  const ans = dsWasmSolve(target, prefix, limit);
  if (ans == null) return null;
  return { answer: ans, ms: Date.now() - t0 };
}

async function dsPowHeader() {
  // Returns { "X-DS-PoW-Response": base64({...}) }. Up to 5 fetch+solve
  // rounds: unsolvable challenges get discarded and refetched (verified live
  // that issuance is reliable: 46/46 solvable in testing, so 5 misses means
  // something else is wrong and completion will fail loudly below).
  for (let attempt = 1; attempt <= 5; attempt++) {
    try {
      const r = await dsPost("/api/v0/chat/create_pow_challenge", { target_path: DS_TARGET_PATH });
      const biz = dsCheckBiz(r.json, "create_pow_challenge");
      const c = (biz && biz.challenge) || biz || {};
      const challenge = {
        algorithm: c.algorithm || "DeepSeekHashV1",
        challenge: c.challenge,
        salt: c.salt,
        difficulty: c.difficulty,
        signature: c.signature,
        expireAt: c.expire_at != null ? c.expire_at : c.expireAt
      };
      if (!challenge.challenge || !challenge.salt) return {};
      const sol = dsSolvePow(challenge);
      if (sol == null) {
        console.log("DeepSeek PoW challenge had no solution, refetching (attempt " + attempt + "/5).");
        // Back off before refetching: hammering challenge fetches looks automated.
        await dsSleep(attempt * 600 + Math.random() * 500);
        continue;
      }
      const payload = {
        algorithm: challenge.algorithm,
        challenge: challenge.challenge,
        salt: challenge.salt,
        answer: sol.answer,
        signature: challenge.signature,
        target_path: DS_TARGET_PATH
      };
      return { "X-DS-PoW-Response": Buffer.from(JSON.stringify(payload), "utf8").toString("base64") };
    } catch (err) {
      console.log("DeepSeek PoW skipped/failed (" + (err && err.message) + "); trying completion anyway.");
      return {};
    }
  }
  console.log("DeepSeek PoW: 5 unsolvable challenges in a row; trying completion anyway.");
  return {};
}

function dsParseSseBlock(block) {
  // Returns [{event, data}] — data parsed as JSON when possible.
  const out = [];
  let event = "message";
  const datas = [];
  const flush = function () {
    if (datas.length) {
      const raw = datas.join("\n");
      let data = raw;
      try { data = JSON.parse(raw); } catch (err) { /* keep raw */ }
      out.push({ event: event, data: data });
    }
    event = "message";
    datas.length = 0;
  };
  for (const line of block.split("\n")) {
    const t = line.trim();
    if (!t) { flush(); continue; }
    if (t.startsWith("event:")) { event = t.slice(6).trim(); }
    else if (t.startsWith("data:")) { datas.push(t.slice(5).trim()); }
  }
  flush();
  return out;
}

// Live text from the real DeepSeek SSE shapes (captured live — the stream has
// NO "delta" events; text arrives as data blocks in these shapes):
// - full snapshot: {v:{response:{fragments:[{type:"RESPONSE",content:"..."}]}}}
// - content append: {o:"APPEND", p:".../content", v:"..."}
// - bare token:     {v:"..."}  (no o/p — the common per-chunk shape)
// Returns true when the block signals completion. `refs` (optional array)
// collects web-search result entries {cite_index,title,url} from
// {p:".../results", v:[...]} blocks for a Sources section.
function dsStreamText(data, sink, refs) {
  let finished = false;
  // `top` = this object is a whole data block (not nested inside a BATCH):
  // only a top-level bare {v:"..."} is a text token. BATCH children carry
  // metadata (token usage, quasi_status) whose bare strings must be ignored.
  const walk = function (d, top) {
    if (!d || typeof d !== "object") return;
    if (typeof d.p === "string" && /results/i.test(d.p) && Array.isArray(d.v)) {
      if (refs) {
        for (const r of d.v) {
          if (r && typeof r.url === "string" && r.url) {
            refs.push({ cite_index: r.cite_index, title: String(r.title || r.url), url: r.url });
          }
        }
      }
      return;
    }
    if (typeof d.o === "string") {
      if (d.o === "BATCH" && Array.isArray(d.v)) { for (const x of d.v) walk(x, false); return; }
      if (d.o === "SET" && d.p === "response/status" && d.v === "FINISHED") { finished = true; return; }
      if (d.o === "APPEND" && typeof d.v === "string" &&
          typeof d.p === "string" && /content/i.test(d.p)) { sink(d.v); return; }
      return;
    }
    if (top && typeof d.v === "string") { sink(d.v); return; }
    const resp = d.v && d.v.response;
    if (resp && Array.isArray(resp.fragments)) {
      for (const f of resp.fragments) {
        if (f && f.type === "RESPONSE" && typeof f.content === "string" && f.content) sink(f.content);
      }
    }
  };
  walk(data, true);
  return finished;
}

async function dsFetchHistoryText(sessionId) {
  try {
    const ctl = new AbortController();
    const timer = setTimeout(function () { ctl.abort(); }, 30000);
    let res;
    try {
      res = await fetch(DS_ORIGIN + "/api/v0/chat/history_messages?chat_session_id=" + encodeURIComponent(sessionId), {
        headers: dsHeaders({ Accept: "application/json" }),
        signal: ctl.signal
      });
    } finally {
      clearTimeout(timer);
    }
    const json = await res.json().catch(function () { return null; });
    const biz = dsCheckBiz(json, "history_messages");
    const msgs = (biz && (biz.messages || biz.history_messages || biz.chat_messages)) || [];
    // Walk from the end: last assistant message with text fragments wins.
    for (let i = msgs.length - 1; i >= 0; i--) {
      const m = msgs[i] || {};
      const role = String(m.role || m.message_role || "").toUpperCase();
      if (role && role !== "ASSISTANT" && role !== "AI") continue;
      const frags = m.fragments || m.fragment || [];
      const parts = [];
      for (const f of frags) {
        // Only real answer text: skip the user's echo, the search-status
        // fragment ("Found N web pages") and reasoning traces.
        const t = String((f && (f.fragmentType || f.fragment_type || f.type)) || "").toUpperCase();
        if (t === "THINK" || t === "THINKING" || t === "SEARCH" || t === "REQUEST") continue;
        const c = f && (f.content != null ? f.content : f.text);
        if (typeof c === "string" && c) parts.push(c);
      }
      if (parts.length) return parts.join("");
      if (typeof m.content === "string" && m.content) return m.content;
    }
    return "";
  } catch (err) {
    console.log("DeepSeek history fallback failed (" + (err && err.message) + ").");
    return "";
  }
}

// Reuse one chat session for a while, like a human keeping a tab open,
// instead of a fresh single-shot session per request (a bot tell).
async function dsSessionId() {
  if (dsSessionCache.id && (Date.now() - dsSessionCache.at) < DS_SESSION_REUSE_MS) return dsSessionCache.id;
  const s = await dsPost("/api/v0/chat_session/create", {});
  const sBiz = dsCheckBiz(s.json, "chat_session/create");
  const session = (sBiz && sBiz.chat_session) || sBiz || {};
  const sessionId = session.id || session.chat_session_id;
  if (!sessionId) throw openaiError(502, "DeepSeek session create gave no id.", "server_error");
  dsSessionCache = { id: sessionId, at: Date.now() };
  return sessionId;
}

// Full DeepSeek turn. onText(chunk) receives live text (may be empty if the
// server only sends full snapshots). Resolves to the authoritative text.
// When `search` is true the model browses the web; cited sources are
// appended as a Sources section and also streamed via onText.
async function deepseekTurn(prompt, onText, signal, search) {
  // Small random pause so request cadence stops looking mechanical.
  await dsHumanPause();
  const sessionId = await dsSessionId();

  // 2. PoW (best effort)
  const powHeaders = await dsPowHeader();

  // 3. completion stream
  const ctl = new AbortController();
  const timer = setTimeout(function () { ctl.abort(); }, DS_TIMEOUT_MS);
  const onAbort = function () { ctl.abort(); };
  if (signal) {
    if (signal.aborted) ctl.abort();
    else signal.addEventListener("abort", onAbort, { once: true });
  }
  let streamed = "";
  const refs = [];
  try {
    const res = await fetch(DS_ORIGIN + DS_TARGET_PATH, {
      method: "POST",
      headers: dsHeaders(Object.assign({ Accept: "text/event-stream" }, powHeaders)),
      body: JSON.stringify({
        chat_session_id: sessionId,
        parent_message_id: null,
        model_type: DS_MODEL_TYPE,
        prompt: prompt,
        ref_file_ids: [],
        thinking_enabled: false,
        search_enabled: !!search,
        action: null,
        preempt: false
      }),
      signal: ctl.signal
    });
    const ctype = (res.headers.get("content-type") || "").toLowerCase();
    if (!res.ok || !res.body || ctype.includes("application/json")) {
      const txt = await res.text().catch(function () { return ""; });
      if (/40300|MISSING_HEADER/.test(txt)) {
        throw openaiError(502, "DeepSeek completion rejected (40300 MISSING_HEADER): the gateway reached DeepSeek but DeepSeek says required headers/PoW are missing. Likely causes: (1) old gateway code without device headers — re-download the latest askme-openai-gateway.mjs / worker file and restart/redeploy; (2) Node gateway missing sha3_wasm_bg.wasm side-by-side (startup log says 'WASM solver unavailable' — keep both files in the same folder); (3) Cloudflare free plan — DeepSeek PoW needs ~0.5s CPU, use the local Node gateway or a paid plan (AskMe works on free); (4) PoW solve failed — see gateway console for 'DeepSeek PoW' lines. Raw: " + String(txt).slice(0, 200), "server_error");
      }
      throw openaiError(502, "DeepSeek completion failed (HTTP " + res.status + "): " + String(txt).slice(0, 300), "server_error");
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let finished = false;
    for (;;) {
      const step = await reader.read();
      if (step.done) break;
      buf += decoder.decode(step.value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        for (const ev of dsParseSseBlock(block)) {
          if (ev.event === "close") {
            finished = true;
          } else if (typeof ev.data !== "undefined") {
            if (dsStreamText(ev.data, function (chunk) {
              streamed += chunk;
              onText(chunk);
            }, refs)) finished = true;
          }
        }
      }
      if (finished) break;
    }
    try { await reader.cancel(); } catch (err) { /* ignore */ }
  } finally {
    clearTimeout(timer);
    if (signal) signal.removeEventListener("abort", onAbort);
  }

  // 4. authoritative text via history (reconcile with anything streamed)
  const full = await dsFetchHistoryText(sessionId);
  let text;
  if (full) {
    if (!streamed) text = full;
    else if (full.startsWith(streamed)) {
      const tail = full.slice(streamed.length);
      if (tail) onText(tail);
      text = full;
    } else if (streamed.startsWith(full)) text = streamed;
    else text = full; // conflict: trust history
  } else {
    text = streamed;
  }
  // 5. cited web sources (search mode): appended AFTER reconciliation so the
  // prefix comparison above sees the raw model text.
  const sources = dsSourcesSection(refs);
  if (sources) {
    onText(sources);
    text += sources;
  }
  return text;
}

// Dedupe collected {cite_index,title,url} refs into a "\n\nSources:..." tail.
function dsSourcesSection(refs) {
  const seen = new Set();
  const items = [];
  for (const r of refs || []) {
    const n = Number(r && r.cite_index);
    const key = (Number.isFinite(n) ? n : 0) + "|" + (r && r.url);
    if (!r || !r.url || seen.has(key)) continue;
    seen.add(key);
    items.push({ n: Number.isFinite(n) ? n : 1e9, title: String((r && r.title) || r.url), url: r.url });
    if (items.length >= 40) break;
  }
  if (!items.length) return "";
  items.sort(function (a, b) { return a.n - b.n; });
  const top = items.slice(0, 10);
  return "\n\nSources:\n" + top.map(function (r) {
    return "[" + (r.n > 1e8 ? "?" : r.n) + "] " + r.title + " — " + r.url;
  }).join("\n");
}

// ------------------------------ HTTP layer ---------------------------------
function cors(res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  res.setHeader("Access-Control-Allow-Private-Network", "true");
  res.setHeader("Access-Control-Max-Age", "86400");
}

function sendJson(res, status, obj) {
  const data = JSON.stringify(obj);
  res.writeHead(status, { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(data) });
  res.end(data);
}

function checkAuth(req) {
  if (!API_KEY) return;
  const hdr = req.headers["authorization"] || "";
  if (hdr !== "Bearer " + API_KEY) {
    throw openaiError(401, "invalid API key", "invalid_request_error");
  }
}

function readBody(req) {
  return new Promise(function (resolve, reject) {
    const chunks = [];
    req.on("data", function (c) { chunks.push(c); });
    req.on("end", function () { resolve(Buffer.concat(chunks).toString("utf8")); });
    req.on("error", reject);
  });
}

function isDeepSeekModel(model) {
  return model === DS_MODEL_ID || /^deepseek/i.test(model || "");
}

async function handleChatCompletions(req, res) {
  let payload;
  try {
    payload = JSON.parse(await readBody(req));
  } catch (err) {
    throw openaiError(400, "invalid JSON body", "invalid_request_error");
  }
  const model = payload.model || MODEL_ID;
  const stream = !!payload.stream;
  const prompt = buildPromptText(payload.messages, MAX_PROMPT_CHARS);
  const promptTokens = estimateTokens(prompt);
  const id = newCompletionId();
  const useDS = isDeepSeekModel(model);
  const useSearch = model === DS_SEARCH_MODEL_ID;

  if (!stream) {
    let text = "";
    if (useDS) {
      text = await deepseekTurn(prompt, function () {}, req.aborted ? AbortSignal.abort() : undefined, useSearch);
    } else {
      await askmeStream(prompt, function (ev) {
        if (ev && ev.type === "text" && typeof ev.content === "string") text += ev.content;
        else if (ev && ev.type === "chart") text += (text ? "\n" : "") + chartToText(ev);
      }, req.aborted ? AbortSignal.abort() : undefined);
    }
    const ct = estimateTokens(text);
    sendJson(res, 200, {
      id: id,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: model,
      choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: "stop" }],
      usage: { prompt_tokens: promptTokens, completion_tokens: ct, total_tokens: promptTokens + ct }
    });
    return;
  }

  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive"
  });
  const write = function (line) {
    if (!res.writableEnded) res.write(line + "\n\n");
  };
  write(openaiChunk(id, model, { role: "assistant" }, null));
  if (typeof res.flushHeaders === "function") res.flushHeaders();

  let aborted = false;
  req.on("close", function () { aborted = true; });
  const signal = AbortSignal.timeout ? AbortSignal.timeout(DS_TIMEOUT_MS) : undefined;

  try {
    const emit = function (chunk) {
      if (!aborted && !res.writableEnded && chunk) write(openaiChunk(id, model, { content: chunk }, null));
    };
    if (useDS) {
      await deepseekTurn(prompt, emit, signal, useSearch);
    } else {
      await askmeStream(prompt, function (ev) {
        if (aborted || res.writableEnded) return;
        if (ev && ev.type === "text" && typeof ev.content === "string" && ev.content) {
          emit(ev.content);
        } else if (ev && ev.type === "chart") {
          emit("\n" + chartToText(ev));
        }
      }, signal);
    }
    if (!aborted && !res.writableEnded) {
      write(openaiChunk(id, model, {}, "stop"));
      res.end("data: [DONE]\n\n");
    }
  } catch (err) {
    if (!res.headersSent) throw err;
    try { res.end(); } catch (e2) { /* ignore */ }
  }
}

function modelList() {
  const list = [{ id: MODEL_ID, object: "model", created: Math.floor(Date.now() / 1000), owned_by: "askme-bridge" }];
  if (DS_MODEL_ID && DS_MODEL_ID !== MODEL_ID) {
    list.push({ id: DS_MODEL_ID, object: "model", created: Math.floor(Date.now() / 1000), owned_by: "deepseek-bridge" });
  }
  if (DS_SEARCH_MODEL_ID && DS_SEARCH_MODEL_ID !== MODEL_ID && DS_SEARCH_MODEL_ID !== DS_MODEL_ID) {
    list.push({ id: DS_SEARCH_MODEL_ID, object: "model", created: Math.floor(Date.now() / 1000), owned_by: "deepseek-bridge" });
  }
  return { object: "list", data: list };
}

const server = http.createServer(function (req, res) {
  cors(res);
  (async function () {
    try {
      const url = new URL(req.url || "/", "http://localhost");
      const path = url.pathname.replace(/\/+$/, "") || "/";
      if (req.method === "OPTIONS") {
        res.writeHead(204);
        res.end();
        return;
      }
      if (req.method === "GET" && (path === "/" || path === "/healthz")) {
        sendJson(res, 200, { ok: true, name: "askme-deepseek-gateway", models: [MODEL_ID, DS_MODEL_ID], endpoints: ["GET /v1/models", "POST /v1/chat/completions"] });
        return;
      }
      if (req.method === "GET" && (path === "/v1/models" || path === "/models")) {
        if (API_KEY) checkAuth(req);
        sendJson(res, 200, modelList());
        return;
      }
      if (req.method === "POST" && (path === "/v1/chat/completions" || path === "/chat/completions")) {
        checkAuth(req);
        await handleChatCompletions(req, res);
        return;
      }
      sendJson(res, 404, { error: { message: "not found: " + path, type: "invalid_request_error", code: 404 } });
    } catch (err) {
      const status = (err && err.status) || 500;
      if (req.method === "POST" && res.headersSent) {
        try { res.end(); } catch (e2) { /* ignore */ }
        return;
      }
      sendJson(res, status, { error: { message: String((err && err.message) || "unknown error"), type: String((err && err.openaiType) || "server_error"), code: status } });
    }
  })();
});

server.listen(PORT, function () {
  console.log("askme-deepseek-gateway on http://localhost:" + PORT + "/v1  (models: " + MODEL_ID + ", " + DS_MODEL_ID + ")");
  if (!API_KEY) console.log("(no API_KEY set: accepting any bearer token)");
  if (!DEEPSEEK_TOKEN) console.log("(no DEEPSEEK_TOKEN set: `" + DS_MODEL_ID + "` model will return an error until you set it)");
});
