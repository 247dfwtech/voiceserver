/**
 * VapiClone Voice Server -- Standalone WebSocket server for Twilio Media Streams.
 * Runs on a GPU VPS with Ollama (LLM), Whisper (STT), and Piper (TTS).
 *
 * Features:
 * - Concurrent call session management
 * - HTTP IPC endpoint for vapiclone (Railway) to register calls
 * - Reports call events back to vapiclone's API via HTTP
 * - Health check with Ollama connectivity status
 * - Available Ollama models listing
 * - Graceful shutdown on SIGTERM/SIGINT
 * - Memory monitoring and periodic health logging
 */

import "dotenv/config"; // Load .env file before anything else
import { WebSocketServer, WebSocket } from "ws";
import { createServer, type IncomingMessage, type ServerResponse } from "http";
import { CallSession, type CallSessionConfig, type CostBreakdown } from "./voice-pipeline/call-session";
import { runPostCallAnalysis } from "./voice-pipeline/analysis-runner";
import { transferCallWithDial } from "./voice-pipeline/call-transfer";
import { getOllamaActiveRequests, getOllamaMaxParallel, incrementOllama, decrementOllama } from "./ollama-concurrency";
import { createLLMProvider } from "./providers";
import type { LLMToolDefinition, LLMToolCall, LLMMessage } from "./providers/llm/interface";
import { modelManager } from "./model-manager";
import { warmupKokoro, checkKokoroHealth, KokoroTTS } from "./providers/tts/kokoro";
import { pcm16kToMulaw } from "./voice-pipeline/audio-utils";
import { createHash } from "crypto";
import * as fs from "fs/promises";
import * as path from "path";
import { checkQwen3Health } from "./providers/tts/qwen3";
import { voiceCloneManager } from "./providers/tts/kokoclone";
import { chatterboxVoiceManager } from "./providers/tts/chatterbox";
import { metricsBuffer, startMetricsCollection, stopMetricsCollection, setSessionCountProvider, collectSnapshot } from "./gpu-monitor";
import { exec } from "child_process";

// ---- Configuration ----

const WS_PORT = parseInt(process.env.WS_PORT || "8765", 10);
const IPC_PORT = parseInt(process.env.IPC_PORT || "8766", 10);
const MAX_SESSIONS = parseInt(process.env.MAX_CONCURRENT_CALLS || "20", 10);
const SERVER_ID = `voice-${process.pid}-${Date.now().toString(36)}`;
const HEALTH_LOG_INTERVAL = 60_000; // Log health every 60s
const STALE_SESSION_TIMEOUT = 3600_000; // 1 hour max call

const VAPICLONE_API_URL = process.env.VAPICLONE_API_URL || "";
const VAPICLONE_API_KEY = process.env.VAPICLONE_API_KEY || "";
const IPC_SECRET = process.env.IPC_SECRET || "";
const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434/v1";

// ---- State ----

interface SessionEntry {
  session: CallSession;
  ws: WebSocket;
  config: CallSessionConfig;
  startedAt: number;
  twilioCallSid?: string;
  transferInitiated?: boolean;
}

const sessions = new Map<string, SessionEntry>();
const pendingConfigs = new Map<string, { config: CallSessionConfig; createdAt: number }>();
const startedAt = Date.now();
let totalCallsHandled = 0;
let isShuttingDown = false;

const MAX_PENDING_CONFIGS = 200;
const MAX_IPC_BODY_BYTES = 1_000_000; // 1MB
const PENDING_CONFIG_TTL_MS = 60_000; // 60s TTL for pending configs

/** Helper: log and return IPC error response */
function ipcError(res: ServerResponse, err: unknown, context: string, status = 400): void {
  const msg = err instanceof Error ? err.message : "Bad request";
  console.error(`[ipc] ${context}:`, err instanceof Error ? err.message : err);
  if (!res.headersSent) {
    res.writeHead(status, { "Content-Type": "application/json" });
  }
  res.end(JSON.stringify({ error: msg }));
}

// Sweep stale pending configs every 30s
setInterval(() => {
  const now = Date.now();
  for (const [callId, entry] of pendingConfigs) {
    if (now - entry.createdAt > PENDING_CONFIG_TTL_MS) {
      pendingConfigs.delete(callId);
      console.warn(`[voice-server] Swept stale pending config for call ${callId} (registered ${Math.round((now - entry.createdAt) / 1000)}s ago, WebSocket never connected)`);
    }
  }
}, 30_000);

// Singleton KokoroTTS instance for /tts/test preview calls.
// Now stateless (HTTP client), so this is just a convenience to avoid re-constructing.
let kokoroTTSSingleton: import("./providers/tts/kokoro").KokoroTTS | null = null;
async function getOrCreateKokoroSingleton(voiceId: string): Promise<import("./providers/tts/kokoro").KokoroTTS> {
  const { KokoroTTS } = await import("./providers/tts/kokoro");
  if (!kokoroTTSSingleton || (kokoroTTSSingleton as any).voiceId !== voiceId) {
    kokoroTTSSingleton = new KokoroTTS({ provider: "kokoro", voiceId, speed: 1.0 });
  }
  return kokoroTTSSingleton;
}

// ---- WebSocket Server ----

const wss = new WebSocketServer({
  port: WS_PORT,
  host: "0.0.0.0",
  maxPayload: 65536, // 64KB -- Twilio chunks are ~200 bytes
  verifyClient: () => !isShuttingDown,
});

console.log(`[voice-server] -----------------------------------------------`);
console.log(`[voice-server] Server ID: ${SERVER_ID}`);
console.log(`[voice-server] WebSocket: ws://0.0.0.0:${WS_PORT} (maxPayload: 64KB)`);
console.log(`[voice-server] IPC HTTP:  http://0.0.0.0:${IPC_PORT}`);
console.log(`[voice-server] Max sessions: ${MAX_SESSIONS}`);
console.log(`[voice-server] Ollama URL: ${OLLAMA_URL}`);
console.log(`[voice-server] VapiClone API: ${VAPICLONE_API_URL || "(not configured)"}`);
console.log(`[voice-server] -----------------------------------------------`);

// Initialize model manager (loads config, creates data dir, auto-pulls default LLM)
modelManager.initialize().then(() => {
  console.log(`[voice-server] Model manager ready (default LLM: ${process.env.DEFAULT_LLM || "llama3.2:3b-4k"})`);
}).catch((err) => {
  console.error("[voice-server] Model manager initialization failed:", err);
});

// Pre-warm Kokoro TTS so first call doesn't wait 6+ seconds for model load
warmupKokoro();

// Pre-warm LLM so first call doesn't wait for model load into GPU VRAM
(async () => {
  try {
    const ollamaUrl = (process.env.OLLAMA_URL || "http://localhost:11434/v1").replace(/\/v1\/?$/, "");
    const defaultLLM = process.env.DEFAULT_LLM || "llama3.2:3b-4k";
    console.log(`[voice-server] Pre-warming LLM: ${defaultLLM}...`);
    const res = await fetch(`${ollamaUrl}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: defaultLLM, prompt: "hi", stream: false }),
      signal: AbortSignal.timeout(120_000),
    });
    if (res.ok) {
      console.log(`[voice-server] LLM ${defaultLLM} warmed up and loaded into GPU VRAM`);
    } else {
      console.warn(`[voice-server] LLM warmup returned HTTP ${res.status}`);
    }
  } catch (err) {
    console.warn(`[voice-server] LLM warmup failed (non-fatal):`, err instanceof Error ? err.message : err);
  }
})();

// Start GPU/CPU/memory metrics collection (30s interval, 3-day ring buffer)
setSessionCountProvider(() => sessions.size);
startMetricsCollection();

wss.on("connection", (ws: WebSocket) => {
  if (isShuttingDown) {
    ws.close(1001, "Server shutting down");
    return;
  }

  let callId: string | null = null;
  let streamSid: string | null = null;

  ws.on("message", async (data: Buffer) => {
    try {
      const msg = JSON.parse(data.toString());

      switch (msg.event) {
        case "connected":
          console.log("[voice-server] Twilio WebSocket connected");
          break;

        case "start": {
          streamSid = msg.streamSid || msg.start?.streamSid || null;
          callId = msg.start?.customParameters?.callId || null;

          if (!callId) {
            console.error("[voice-server] No callId in stream parameters");
            ws.close();
            return;
          }

          // Check session limit
          if (sessions.size >= MAX_SESSIONS) {
            console.error(
              `[voice-server] Session limit reached (${MAX_SESSIONS}), rejecting call ${callId}`
            );
            ws.close(1013, "Max concurrent calls reached");
            return;
          }

          if (!streamSid) {
            console.warn(`[voice-server] WARNING: No streamSid in start event — outbound audio will not work!`);
          }

          console.log(
            `[voice-server] Stream started: callId=${callId}, streamSid=${streamSid} ` +
              `(${sessions.size + 1}/${MAX_SESSIONS} sessions)`
          );

          const pendingEntry = pendingConfigs.get(callId);
          if (!pendingEntry) {
            console.error(`[voice-server] No pending config for callId=${callId}`);
            ws.close();
            return;
          }
          pendingConfigs.delete(callId);
          const config = pendingEntry.config;

          const session = new CallSession(config);

          // Handle audio output back to Twilio
          session.on("audio", (base64Audio: string) => {
            if (ws.readyState === WebSocket.OPEN && streamSid) {
              ws.send(
                JSON.stringify({
                  event: "media",
                  streamSid,
                  media: { payload: base64Audio },
                })
              );
            }
          });

          // Handle clear audio (interrupt)
          session.on("clear_audio", () => {
            if (ws.readyState === WebSocket.OPEN && streamSid) {
              ws.send(JSON.stringify({ event: "clear", streamSid }));
            }
          });

          // Handle call transfer (provider-aware: Twilio or SignalWire)
          session.on("transfer", async (transferData: { destination?: string }) => {
            // Guard: prevent double transfer if already initiated
            const existingEntry = callId ? sessions.get(callId) : undefined;
            if (existingEntry?.transferInitiated) {
              console.log(`[voice-server] [${callId}] Transfer already initiated, skipping duplicate`);
              return;
            }
            const callProvider = config.provider || "twilio";
            console.log(`[voice-server] [${callId}] Transfer requested (${callProvider}), destination=${transferData.destination || "NONE"}`);
            if (transferData.destination) {
              try {
                const providerCallSid = msg.start?.callSid;
                if (providerCallSid) {
                  console.log(`[voice-server] [${callId}] Executing ${callProvider} transfer to ${transferData.destination}`);
                  const result = await transferCallWithDial({
                    callSid: providerCallSid,
                    destination: transferData.destination,
                    publicUrl: process.env.PUBLIC_URL || "",
                    callerNumber: config.customerNumber,
                    provider: callProvider as "twilio" | "signalwire",
                  });
                  if (!result.success) {
                    console.error(`[voice-server] [${callId}] Transfer failed:`, result.error);
                  } else {
                    console.log(`[voice-server] [${callId}] Transfer initiated successfully`);
                    // Mark transfer so stream-stop uses "call-forwarded" instead of "customer-ended-call"
                    const transferEntry = callId ? sessions.get(callId) : undefined;
                    if (transferEntry) transferEntry.transferInitiated = true;
                  }
                } else {
                  console.error(`[voice-server] [${callId}] Transfer failed: no callSid from ${callProvider}`);
                }
              } catch (err) {
                console.error(`[voice-server] [${callId}] Transfer error:`, err);
              }
            } else {
              console.error(`[voice-server] [${callId}] Transfer skipped: no destination`);
            }
          });

          // Handle voicemail detected
          session.on("voicemail_detected", () => {
            console.log(`[voice-server] [${callId}] Voicemail detected`);
          });

          // Handle call ended
          session.on("ended", async (endData) => {
            console.log(
              `[voice-server] Call ended: ${callId} reason=${endData.endedReason} ` +
                `duration=${endData.duration}s (${sessions.size - 1} sessions remaining)`
            );

            // Remove from tracking
            sessions.delete(callId!);
            totalCallsHandled++;

            // Run post-call processing
            try {
              await notifyCallEnded(callId!, endData, config);
            } catch (err) {
              console.error(`[voice-server] Post-call error:`, err);
            }

            // Close the Twilio WebSocket
            if (ws.readyState === WebSocket.OPEN) {
              ws.close();
            }
          });

          session.on("transcript", () => {
            // Future: broadcast to dashboard via WebSocket
          });

          session.on("user_speech", (text: string) => {
            console.log(`[voice-server] [${callId}] User: ${text}`);
          });

          session.on("tool_call", (toolData: { toolCall: { function: { name: string } } }) => {
            console.log(
              `[voice-server] [${callId}] Tool call: ${toolData.toolCall.function.name}`
            );
          });

          const entry: SessionEntry = {
            session,
            ws,
            config,
            startedAt: Date.now(),
            twilioCallSid: msg.start?.callSid,
          };
          sessions.set(callId, entry);

          // Start the session
          await session.start();
          break;
        }

        case "media": {
          if (!callId) break;
          const entry = sessions.get(callId);
          if (entry) {
            entry.session.handleAudio(msg.media.payload);
          }
          break;
        }

        case "stop": {
          if (callId) {
            const entry = sessions.get(callId);
            if (entry) {
              const reason = entry.transferInitiated ? "call-forwarded" : "customer-ended-call";
              console.log(`[voice-server] Stream stopped: callId=${callId} reason=${reason} (transferInitiated=${!!entry.transferInitiated})`);
              entry.session.endCall(reason);
            } else {
              console.log(`[voice-server] Stream stopped: callId=${callId} (no session)`);
            }
          }
          break;
        }
      }
    } catch (err) {
      console.error("[voice-server] Message parse error:", err);
    }
  });

  ws.on("close", () => {
    if (callId) {
      const entry = sessions.get(callId);
      if (entry && entry.session.getState() !== "ended") {
        entry.session.endCall("websocket-closed");
      }
    }
  });

  ws.on("error", (err) => {
    console.error(`[voice-server] WebSocket error (${callId || "unknown"}):`, err.message);
  });
});

// ---- Post-Call Processing ----

async function notifyCallEnded(
  callId: string,
  endData: {
    endedReason: string;
    transcript: string;
    duration: number;
    cost: CostBreakdown;
    overflow?: { used: boolean; count?: number; provider?: string | null; model?: string | null };
  },
  config: CallSessionConfig
): Promise<void> {
  // Run post-call analysis only on answered calls with real conversation
  let analysis: { summary?: string; successEvaluation?: string; analysisCost?: number; analysisProvider?: string; analysisModel?: string } = {};
  const analysisConfig = config.analysisConfig;
  const SKIP_ANALYSIS_REASONS = new Set(["voicemail", "stt-failure", "no-answer", "busy", "failed", "machine-detected"]);
  const hasRealConversation = endData.transcript && endData.transcript.split("\n").filter((l: string) => l.trim()).length >= 2;

  if (analysisConfig && !SKIP_ANALYSIS_REASONS.has(endData.endedReason) && hasRealConversation) {
    try {
      analysis = await runPostCallAnalysis(endData.transcript, analysisConfig);
      // Add analysis cost to the cost breakdown
      if (analysis.analysisCost && analysis.analysisCost > 0) {
        endData.cost.analysis = analysis.analysisCost;
        endData.cost.total = Math.round((endData.cost.total + analysis.analysisCost) * 10000) / 10000;
      }
    } catch (err) {
      console.error(`[voice-server] Analysis failed for ${callId}:`, err);
    }
  } else if (analysisConfig) {
    console.log(`[voice-server] Skipping analysis for ${callId}: reason=${endData.endedReason}, transcriptLines=${endData.transcript?.split("\n").filter((l: string) => l.trim()).length || 0}`);
  }

  // Send end-of-call report to VapiClone
  if (!VAPICLONE_API_URL) {
    console.warn(`[voice-server] VAPICLONE_API_URL not configured, skipping call event notification`);
    return;
  }

  try {
    const res = await fetch(`${VAPICLONE_API_URL}/api/webhooks/call-events`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${VAPICLONE_API_KEY}`,
        ...(IPC_SECRET ? { "x-ipc-secret": IPC_SECRET } : {}),
      },
      body: JSON.stringify({
        type: "end-of-call-report",
        callId,
        data: {
          endedReason: endData.endedReason,
          transcript: endData.transcript,
          duration: endData.duration,
          cost: endData.cost,
          sttProvider: config.transcriber?.provider || "unknown",
          sttModel: config.transcriber?.model || "unknown",
          llmProvider: config.model?.provider || "ollama",
          llmModel: config.model?.model || "unknown",
          analysis,
          overflow: endData.overflow,
          customer: {
            number: config.customerNumber,
            name: config.customerName,
            externalId: (config.metadata as Record<string, unknown>)?.externalId,
          },
          assistantId: config.assistantId,
          serverUrl: config.serverUrl,
        },
      }),
      signal: AbortSignal.timeout(15000),
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    }

    console.log(`[voice-server] Call event sent to VapiClone for ${callId}`);
  } catch (err) {
    console.error(`[voice-server] Failed to send call event to VapiClone:`, err);
  }
}

// ---- IPC HTTP Server ----

/** Mask phone number for logs: +12145551234 -> +1214***1234 */
function maskPhone(phone: string): string {
  if (!phone || phone.length < 7) return "***";
  return phone.slice(0, 4) + "***" + phone.slice(-4);
}

/** Read IPC request body with size limit */
function readBody(req: IncomingMessage, maxBytes: number): Promise<string> {
  return new Promise((resolve, reject) => {
    let body = "";
    let size = 0;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > maxBytes) {
        req.destroy();
        reject(new Error("Body too large"));
        return;
      }
      body += chunk;
    });
    req.on("end", () => resolve(body));
    req.on("error", reject);
  });
}

/** Check if Ollama is reachable and return model info */
async function checkOllama(): Promise<{ ok: boolean; models?: string[]; error?: string }> {
  try {
    const ollamaBaseUrl = OLLAMA_URL.replace(/\/v1\/?$/, "");
    const res = await fetch(`${ollamaBaseUrl}/api/tags`, {
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) {
      return { ok: false, error: `HTTP ${res.status}` };
    }
    const data = await res.json() as { models?: { name: string }[] };
    const models = (data.models || []).map((m: { name: string }) => m.name);
    return { ok: true, models };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
}

function handleIPC(req: IncomingMessage, res: ServerResponse): void {
  const url = new URL(req.url || "/", `http://localhost:${IPC_PORT}`);

  // GET /health -- public health check (enhanced with GPU data)
  if (req.method === "GET" && url.pathname === "/health") {
    const mem = process.memoryUsage();
    const latest = metricsBuffer.getLatest();

    // Check Ollama and TTS service health concurrently
    Promise.all([
      checkOllama().catch(() => ({ ok: false, error: "check failed" })),
      checkKokoroHealth().catch(() => false),
      checkQwen3Health().catch(() => false),
    ]).then(([ollamaStatus, kokoroOk, qwen3Ok]) => {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          ok: !isShuttingDown,
          sessions: sessions.size,
          maxSessions: MAX_SESSIONS,
          totalCallsHandled,
          uptime: Math.round((Date.now() - startedAt) / 1000),
          memory: {
            rss: Math.round(mem.rss / 1024 / 1024),
            heapUsed: Math.round(mem.heapUsed / 1024 / 1024),
          },
          gpu: latest?.gpu || null,
          cpu: latest?.cpu || null,
          systemMemory: latest?.mem || null,
          processes: latest?.processes || [],
          disk: latest?.disk || null,
          network: latest?.network || null,
          ollama: ollamaStatus,
          tts: {
            kokoro: { ok: kokoroOk, url: process.env.KOKORO_API_URL || "http://localhost:8880" },
            qwen3: { ok: qwen3Ok, url: process.env.QWEN3_API_URL || "http://localhost:8881" },
          },
        })
      );
    });
    return;
  }

  // GET /health/gpu -- latest GPU snapshot only
  if (req.method === "GET" && url.pathname === "/health/gpu") {
    collectSnapshot().then((snap) => {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(snap));
    }).catch((err) => {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
    });
    return;
  }

  // GET /metrics/history -- historical metrics for charts
  if (req.method === "GET" && url.pathname === "/metrics/history") {
    const rangeParam = url.searchParams.get("range") || "1h";
    const maxPoints = parseInt(url.searchParams.get("maxPoints") || "300", 10);

    const rangeMs: Record<string, number> = {
      "1h": 60 * 60 * 1000,
      "6h": 6 * 60 * 60 * 1000,
      "24h": 24 * 60 * 60 * 1000,
      "3d": 3 * 24 * 60 * 60 * 1000,
    };
    const ms = rangeMs[rangeParam] || rangeMs["1h"];
    const since = Date.now() - ms;
    const snapshots = metricsBuffer.getRange(since, maxPoints);

    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ snapshots, count: snapshots.length, rangeMs: ms }));
    return;
  }

  // GET /metrics -- Prometheus-compatible metrics (public)
  if (req.method === "GET" && url.pathname === "/metrics") {
    const mem = process.memoryUsage();
    const lines = [
      `# HELP voice_active_sessions Current active call sessions`,
      `# TYPE voice_active_sessions gauge`,
      `voice_active_sessions ${sessions.size}`,
      `# HELP voice_max_sessions Maximum concurrent sessions`,
      `# TYPE voice_max_sessions gauge`,
      `voice_max_sessions ${MAX_SESSIONS}`,
      `# HELP voice_total_calls_handled Total calls handled since startup`,
      `# TYPE voice_total_calls_handled counter`,
      `voice_total_calls_handled ${totalCallsHandled}`,
      `# HELP voice_pending_configs Pending call configs waiting for Twilio`,
      `# TYPE voice_pending_configs gauge`,
      `voice_pending_configs ${pendingConfigs.size}`,
      `# HELP voice_memory_rss_bytes Process RSS memory in bytes`,
      `# TYPE voice_memory_rss_bytes gauge`,
      `voice_memory_rss_bytes ${mem.rss}`,
      `# HELP voice_memory_heap_used_bytes Heap used in bytes`,
      `# TYPE voice_memory_heap_used_bytes gauge`,
      `voice_memory_heap_used_bytes ${mem.heapUsed}`,
      `# HELP voice_uptime_seconds Server uptime in seconds`,
      `# TYPE voice_uptime_seconds gauge`,
      `voice_uptime_seconds ${Math.round((Date.now() - startedAt) / 1000)}`,
    ];
    res.writeHead(200, { "Content-Type": "text/plain" });
    res.end(lines.join("\n") + "\n");
    return;
  }

  // ---- All remaining endpoints require IPC secret ----
  if (IPC_SECRET) {
    const authHeader = req.headers["x-ipc-secret"] || "";
    if (authHeader !== IPC_SECRET) {
      res.writeHead(401, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Unauthorized" }));
      return;
    }
  }

  // GET /models -- list available Ollama models (enhanced with model-manager)
  if (req.method === "GET" && url.pathname === "/models") {
    modelManager
      .getFullStatus()
      .then((status) => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(status));
      })
      .catch((err) => {
        res.writeHead(502, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({
            error: "Failed to get model status",
            details: err instanceof Error ? err.message : String(err),
          })
        );
      });
    return;
  }

  // ---- Model Management Endpoints ----

  // GET /models/status -- Full model status
  if (req.method === "GET" && url.pathname === "/models/status") {
    modelManager
      .getFullStatus()
      .then((status) => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(status));
      })
      .catch((err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
      });
    return;
  }

  // GET /models/llm -- List installed LLM models
  if (req.method === "GET" && url.pathname === "/models/llm") {
    modelManager
      .listLLMModels()
      .then((models) => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ models }));
      })
      .catch((err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
      });
    return;
  }

  // POST /models/llm/pull -- Pull/install an Ollama model
  if (req.method === "POST" && url.pathname === "/models/llm/pull") {
    readBody(req, MAX_IPC_BODY_BYTES)
      .then(async (body) => {
        const { name } = JSON.parse(body);
        if (!name) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Missing 'name' in request body" }));
          return;
        }

        const result = await modelManager.pullLLMModel(name, (status, completed, total) => {
          // Progress is logged server-side; could be streamed via SSE in future
          console.log(`[model-manager] Pull ${name}: ${status} ${completed || ""}/${total || ""}`);
        });

        if (result.success) {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true, name }));
        } else {
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: result.error }));
        }
      })
      .catch((err) => ipcError(res, err, `${req.method} ${url.pathname}`));
    return;
  }

  // POST /models/llm/activate -- Set active LLM
  if (req.method === "POST" && url.pathname === "/models/llm/activate") {
    readBody(req, MAX_IPC_BODY_BYTES)
      .then(async (body) => {
        const { name } = JSON.parse(body);
        if (!name) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Missing 'name' in request body" }));
          return;
        }

        const result = await modelManager.activateLLM(name);
        if (result.success) {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true, activeLLM: name }));
        } else {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: result.error }));
        }
      })
      .catch((err) => ipcError(res, err, `${req.method} ${url.pathname}`));
    return;
  }

  // DELETE /models/llm/:name -- Remove an installed LLM model
  if (req.method === "DELETE" && url.pathname.startsWith("/models/llm/")) {
    const modelName = decodeURIComponent(url.pathname.slice("/models/llm/".length));
    if (!modelName) {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Missing model name in URL" }));
      return;
    }

    modelManager
      .deleteLLMModel(modelName)
      .then((result) => {
        if (result.success) {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true, deleted: modelName }));
        } else {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: result.error }));
        }
      })
      .catch((err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
      });
    return;
  }

  // GET /models/stt -- List installed STT models
  if (req.method === "GET" && url.pathname === "/models/stt") {
    modelManager
      .listSTTModels()
      .then((models) => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ models, catalog: modelManager.getSTTCatalog() }));
      })
      .catch((err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
      });
    return;
  }

  // POST /models/stt/install -- Install a Whisper model
  if (req.method === "POST" && url.pathname === "/models/stt/install") {
    readBody(req, MAX_IPC_BODY_BYTES)
      .then(async (body) => {
        const { name } = JSON.parse(body);
        if (!name) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Missing 'name' in request body" }));
          return;
        }

        const result = await modelManager.installSTTModel(name);
        if (result.success) {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true, name }));
        } else {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: result.error }));
        }
      })
      .catch((err) => ipcError(res, err, `${req.method} ${url.pathname}`));
    return;
  }

  // POST /models/stt/activate -- Set active STT model
  if (req.method === "POST" && url.pathname === "/models/stt/activate") {
    readBody(req, MAX_IPC_BODY_BYTES)
      .then(async (body) => {
        const { name } = JSON.parse(body);
        if (!name) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Missing 'name' in request body" }));
          return;
        }

        const result = await modelManager.activateSTT(name);
        if (result.success) {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true, activeSTT: name }));
        } else {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: result.error }));
        }
      })
      .catch((err) => ipcError(res, err, `${req.method} ${url.pathname}`));
    return;
  }

  // DELETE /models/stt/:name -- Remove an installed STT model
  if (req.method === "DELETE" && url.pathname.startsWith("/models/stt/")) {
    const modelName = decodeURIComponent(url.pathname.slice("/models/stt/".length));
    if (!modelName) {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Missing model name in URL" }));
      return;
    }

    modelManager
      .deleteSTTModel(modelName)
      .then((result) => {
        if (result.success) {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true, deleted: modelName }));
        } else {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: result.error }));
        }
      })
      .catch((err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
      });
    return;
  }

  // GET /models/tts -- List installed TTS voices
  if (req.method === "GET" && url.pathname === "/models/tts") {
    modelManager
      .listTTSVoices()
      .then((voices) => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ voices, catalog: modelManager.getTTSCatalog() }));
      })
      .catch((err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
      });
    return;
  }

  // POST /models/tts/install -- Install a Piper voice
  if (req.method === "POST" && url.pathname === "/models/tts/install") {
    readBody(req, MAX_IPC_BODY_BYTES)
      .then(async (body) => {
        const { name } = JSON.parse(body);
        if (!name) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Missing 'name' in request body" }));
          return;
        }

        const result = await modelManager.installTTSVoice(name);
        if (result.success) {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true, name }));
        } else {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: result.error }));
        }
      })
      .catch((err) => ipcError(res, err, `${req.method} ${url.pathname}`));
    return;
  }

  // POST /models/tts/activate -- Set active TTS voice
  if (req.method === "POST" && url.pathname === "/models/tts/activate") {
    readBody(req, MAX_IPC_BODY_BYTES)
      .then(async (body) => {
        const { name } = JSON.parse(body);
        if (!name) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Missing 'name' in request body" }));
          return;
        }

        const result = await modelManager.activateTTS(name);
        if (result.success) {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true, activeTTS: name }));
        } else {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: result.error }));
        }
      })
      .catch((err) => ipcError(res, err, `${req.method} ${url.pathname}`));
    return;
  }

  // DELETE /models/tts/:name -- Remove an installed TTS voice
  if (req.method === "DELETE" && url.pathname.startsWith("/models/tts/")) {
    const voiceName = decodeURIComponent(url.pathname.slice("/models/tts/".length));
    if (!voiceName) {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Missing voice name in URL" }));
      return;
    }

    modelManager
      .deleteTTSVoice(voiceName)
      .then((result) => {
        if (result.success) {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true, deleted: voiceName }));
        } else {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: result.error }));
        }
      })
      .catch((err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
      });
    return;
  }

  // GET /models/search -- Search HuggingFace for compatible models
  if (req.method === "GET" && url.pathname === "/models/search") {
    const query = url.searchParams.get("q") || "";
    const type = (url.searchParams.get("type") || "llm") as "llm" | "stt" | "tts";

    if (!query && type !== "stt") {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Missing 'q' query parameter" }));
      return;
    }

    if (!["llm", "stt", "tts"].includes(type)) {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Invalid 'type' parameter. Must be llm, stt, or tts." }));
      return;
    }

    modelManager
      .searchHuggingFace(query, type)
      .then((results) => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ results, query, type }));
      })
      .catch((err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
      });
    return;
  }

  // ---- Settings / API Key Management ----

  // Allowed settings keys that can be read/written via IPC
  const ALLOWED_SETTINGS: Record<string, { envKey: string; sensitive: boolean }> = {
    deepgram_api_key: { envKey: "DEEPGRAM_API_KEY", sensitive: true },
    openai_api_key: { envKey: "OPENAI_API_KEY", sensitive: true },
    elevenlabs_api_key: { envKey: "ELEVENLABS_API_KEY", sensitive: true },
    deepseek_api_key: { envKey: "DEEPSEEK_API_KEY", sensitive: true },
    openrouter_api_key: { envKey: "OPENROUTER_API_KEY", sensitive: true },
    cerebras_api_key: { envKey: "CEREBRAS_API_KEY", sensitive: true },
    groq_api_key: { envKey: "GROQ_API_KEY", sensitive: true },
    deepinfra_api_key: { envKey: "DEEPINFRA_API_KEY", sensitive: true },
    groq_tts_api_key: { envKey: "GROQ_TTS_API_KEY", sensitive: true },
    unrealspeech_api_key: { envKey: "UNREALSPEECH_API_KEY", sensitive: true },
    groq_stt_api_key: { envKey: "GROQ_STT_API_KEY", sensitive: true },
    twilio_account_sid: { envKey: "TWILIO_ACCOUNT_SID", sensitive: true },
    twilio_auth_token: { envKey: "TWILIO_AUTH_TOKEN", sensitive: true },
    default_llm: { envKey: "DEFAULT_LLM", sensitive: false },
    whisper_model: { envKey: "WHISPER_MODEL", sensitive: false },
    kokoro_voice: { envKey: "KOKORO_VOICE", sensitive: false },
    ollama_flash_attention: { envKey: "OLLAMA_FLASH_ATTENTION", sensitive: false },
    ollama_num_parallel: { envKey: "OLLAMA_NUM_PARALLEL", sensitive: false },
    sw_project_id: { envKey: "SW_PROJECT_ID", sensitive: true },
    sw_auth_token: { envKey: "SW_AUTH_TOKEN", sensitive: true },
    sw_space_url: { envKey: "SW_SPACE_URL", sensitive: false },
    overflow_llm_provider: { envKey: "OVERFLOW_LLM_PROVIDER", sensitive: false },
    overflow_llm_model: { envKey: "OVERFLOW_LLM_MODEL", sensitive: false },
  };

  // GET /settings -- List current settings (sensitive values masked)
  if (req.method === "GET" && url.pathname === "/settings") {
    const settings: Record<string, string | null> = {};
    for (const [key, def] of Object.entries(ALLOWED_SETTINGS)) {
      const val = process.env[def.envKey] || null;
      if (def.sensitive && val) {
        settings[key] = val.slice(0, 4) + "..." + val.slice(-4);
      } else {
        settings[key] = val;
      }
    }
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ settings }));
    return;
  }

  // PUT /settings -- Update one or more settings (persists to .env and process.env)
  if (req.method === "PUT" && url.pathname === "/settings") {
    readBody(req, MAX_IPC_BODY_BYTES)
      .then(async (body) => {
        const updates = JSON.parse(body) as Record<string, string>;
        const applied: string[] = [];

        for (const [key, value] of Object.entries(updates)) {
          const def = ALLOWED_SETTINGS[key];
          if (!def) {
            console.warn(`[settings] Ignoring unknown setting: ${key}`);
            continue;
          }
          // Update in-memory env
          process.env[def.envKey] = value;
          applied.push(key);
          console.log(`[settings] Updated ${def.envKey} in process.env`);
        }

        // Persist to .env file
        if (applied.length > 0) {
          try {
            const fs = await import("fs");
            const path = await import("path");
            const envPath = process.env.ENV_FILE_PATH || path.join(process.cwd(), ".env");
            let envContent = "";
            try {
              envContent = fs.readFileSync(envPath, "utf-8");
            } catch {
              envContent = "";
            }

            for (const key of applied) {
              const def = ALLOWED_SETTINGS[key]!;
              const val = process.env[def.envKey] || "";
              const regex = new RegExp(`^${def.envKey}=.*$`, "m");
              if (regex.test(envContent)) {
                envContent = envContent.replace(regex, `${def.envKey}=${val}`);
              } else {
                envContent += `\n${def.envKey}=${val}`;
              }
            }

            fs.writeFileSync(envPath, envContent);
            console.log(`[settings] Persisted ${applied.length} setting(s) to ${envPath}`);
          } catch (err) {
            console.warn(`[settings] Failed to persist to .env file:`, err instanceof Error ? err.message : err);
            // Not fatal — in-memory env is already updated
          }

          // Restart Ollama PM2 process if ollama-specific settings changed
          const ollamaKeys = ["ollama_flash_attention", "ollama_num_parallel"];
          const ollamaChanged = applied.some((k) => ollamaKeys.includes(k));
          if (ollamaChanged) {
            try {
              const { exec: execCb } = await import("child_process");
              execCb("pm2 restart ollama --update-env", (err, stdout, stderr) => {
                if (err) {
                  console.warn(`[settings] Failed to restart Ollama PM2:`, err.message);
                } else {
                  console.log(`[settings] Restarted Ollama PM2 process to apply new env vars`);
                }
              });
            } catch (err) {
              console.warn(`[settings] pm2 restart ollama failed:`, err);
            }
          }
        }

        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: true, applied }));
      })
      .catch((err) => ipcError(res, err, `PUT ${url.pathname}`));
    return;
  }

  // POST /register-call -- Register pending call config
  if (req.method === "POST" && url.pathname === "/register-call") {
    if (isShuttingDown) {
      res.writeHead(503, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Server shutting down" }));
      return;
    }

    readBody(req, MAX_IPC_BODY_BYTES)
      .then((body) => {
        const { callId, config } = JSON.parse(body);

        if (sessions.size >= MAX_SESSIONS) {
          res.writeHead(429, { "Content-Type": "application/json" });
          res.end(
            JSON.stringify({
              error: "Max concurrent calls reached",
              sessions: sessions.size,
              maxSessions: MAX_SESSIONS,
            })
          );
          return;
        }

        if (pendingConfigs.size >= MAX_PENDING_CONFIGS) {
          res.writeHead(429, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Too many pending configs" }));
          return;
        }

        registerCallConfig(callId, config);
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: true, sessions: sessions.size }));
      })
      .catch((err) => ipcError(res, err, `POST ${url.pathname}`));
    return;
  }

  // GET /sessions -- List active sessions (PII masked)
  if (req.method === "GET" && url.pathname === "/sessions") {
    let overflowActiveCount = 0;
    const sessionList = Array.from(sessions.entries()).map(([id, entry]) => {
      const isOverflow = entry.session.isUsingOverflow();
      if (isOverflow) overflowActiveCount++;
      return {
        callId: id,
        assistantId: entry.config.assistantId,
        customerNumber: maskPhone(entry.config.customerNumber),
        state: entry.session.getState(),
        duration: Math.round((Date.now() - entry.startedAt) / 1000),
        isOverflow,
      };
    });

    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({
      sessions: sessionList,
      count: sessionList.length,
      ollama: { active: getOllamaActiveRequests(), max: getOllamaMaxParallel() },
      overflow: {
        active: overflowActiveCount,
        provider: process.env.OVERFLOW_LLM_PROVIDER || null,
        model: process.env.OVERFLOW_LLM_MODEL || null,
      },
    }));
    return;
  }

  // POST /end-call/:id -- Force-end a specific call
  if (req.method === "POST" && url.pathname.startsWith("/end-call/")) {
    const targetCallId = url.pathname.split("/end-call/")[1];
    const entry = sessions.get(targetCallId);
    if (entry) {
      entry.session.endCall("admin-force-ended");
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: true, callId: targetCallId }));
    } else {
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Session not found" }));
    }
    return;
  }

  // POST /amd-result/:callId -- Forward Twilio AMD result to active session
  if (req.method === "POST" && url.pathname.startsWith("/amd-result/")) {
    const targetCallId = url.pathname.split("/amd-result/")[1];
    let body = "";
    req.on("data", (chunk: Buffer) => { body += chunk.toString(); });
    req.on("end", () => {
      try {
        const { answeredBy } = JSON.parse(body);
        console.log(`[voice-server] AMD result for ${targetCallId}: ${answeredBy}`);
        const entry = sessions.get(targetCallId);
        if (entry && entry.session.voicemailDetector) {
          // Use forceAmdResult to properly handle machine_start (waits for beep)
          // vs machine_end_beep (speaks immediately)
          entry.session.voicemailDetector.forceAmdResult(answeredBy);
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true }));
        } else {
          res.writeHead(404, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Session not found or no voicemail detector" }));
        }
      } catch (err) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Invalid JSON" }));
      }
    });
    return;
  }

  // POST /services/:name/:action -- Start/stop GPU services via PM2
  // Allowed services: kokoro-fastapi, qwen3-tts, ollama
  if (req.method === "POST" && url.pathname.startsWith("/services/")) {
    const parts = url.pathname.split("/"); // ["", "services", name, action]
    const serviceName = parts[2];
    const action = parts[3]; // "start" or "stop"
    const ALLOWED_SERVICES = ["kokoro-fastapi", "qwen3-tts", "ollama"];

    if (!ALLOWED_SERVICES.includes(serviceName)) {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: `Unknown service: ${serviceName}` }));
      return;
    }
    if (action !== "start" && action !== "stop") {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: `Invalid action: ${action}. Use start or stop.` }));
      return;
    }

    const pm2Cmd = action === "start" ? "restart" : "stop";
    // exec imported at top level
    exec(`export PATH=/opt/nvm/versions/node/v24.12.0/bin:$PATH && pm2 ${pm2Cmd} ${serviceName}`, (err, stdout, stderr) => {
      if (err) {
        console.error(`[voice-server] PM2 ${pm2Cmd} ${serviceName} failed:`, stderr);
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: stderr || err.message }));
      } else {
        console.log(`[voice-server] PM2 ${pm2Cmd} ${serviceName}: OK`);
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: true, service: serviceName, action }));
      }
    });
    return;
  }

  // GET /services -- List service statuses via PM2
  if (req.method === "GET" && url.pathname === "/services") {
    // exec imported at top level
    exec("export PATH=/opt/nvm/versions/node/v24.12.0/bin:$PATH && pm2 jlist", (err, stdout) => {
      if (err) {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Failed to get PM2 status" }));
        return;
      }
      try {
        const pm2List = JSON.parse(stdout);
        const services = pm2List
          .filter((p: any) => ["kokoro-fastapi", "qwen3-tts", "ollama"].includes(p.name))
          .map((p: any) => ({
            name: p.name,
            status: p.pm2_env?.status || "unknown",
            pid: p.pid,
            uptime: p.pm2_env?.pm_uptime ? Date.now() - p.pm2_env.pm_uptime : 0,
            restarts: p.pm2_env?.restart_time || 0,
            memory: Math.round((p.monit?.memory || 0) / 1024 / 1024),
            cpu: p.monit?.cpu || 0,
          }));
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ services }));
      } catch {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Failed to parse PM2 output" }));
      }
    });
    return;
  }

  // ---- Voice Cloning (KokoClone) Endpoints ----

  // GET /voice-clone/status -- Check if KokoClone is installed
  if (req.method === "GET" && url.pathname === "/voice-clone/status") {
    voiceCloneManager
      .isInstalled()
      .then(async (installed) => {
        const voices = await voiceCloneManager.listClonedVoices();
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ installed, voices, count: voices.length }));
      })
      .catch((err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
      });
    return;
  }

  // POST /voice-clone/install -- Install KokoClone dependencies
  if (req.method === "POST" && url.pathname === "/voice-clone/install") {
    voiceCloneManager
      .install()
      .then((result) => {
        if (result.success) {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true, message: "KokoClone installed successfully" }));
        } else {
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: result.error }));
        }
      })
      .catch((err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
      });
    return;
  }

  // POST /voice-clone/uninstall -- Uninstall KokoClone dependencies
  if (req.method === "POST" && url.pathname === "/voice-clone/uninstall") {
    voiceCloneManager
      .uninstall()
      .then((result) => {
        if (result.success) {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true, message: "KokoClone uninstalled" }));
        } else {
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: result.error }));
        }
      })
      .catch((err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
      });
    return;
  }

  // GET /voice-clone/voices -- List all cloned voices
  if (req.method === "GET" && url.pathname === "/voice-clone/voices") {
    voiceCloneManager
      .listClonedVoices()
      .then((voices) => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ voices }));
      })
      .catch((err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
      });
    return;
  }

  // POST /voice-clone/create -- Upload reference audio and create a cloned voice
  // Expects multipart-like JSON: { name: string, audio: base64, filename: string }
  if (req.method === "POST" && url.pathname === "/voice-clone/create") {
    readBody(req, 10_000_000) // 10MB max for audio upload
      .then(async (body) => {
        const { name, audio, filename } = JSON.parse(body);

        if (!name || !audio) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Missing 'name' and 'audio' (base64) in request body" }));
          return;
        }

        // Decode base64 audio
        const audioBuffer = Buffer.from(audio, "base64");

        if (audioBuffer.length < 1000) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Audio file too small. Need 3-10 seconds of clear speech." }));
          return;
        }

        if (audioBuffer.length > 5_000_000) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Audio file too large (max 5MB). Use a 3-10 second clip." }));
          return;
        }

        const result = await voiceCloneManager.createClonedVoice(
          name,
          audioBuffer,
          filename || "reference.wav"
        );

        if (result.success) {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true, voice: result.voice }));
        } else {
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: result.error }));
        }
      })
      .catch((err) => ipcError(res, err, `${req.method} ${url.pathname}`));
    return;
  }

  // DELETE /voice-clone/voices/:id -- Delete a cloned voice
  if (req.method === "DELETE" && url.pathname.startsWith("/voice-clone/voices/")) {
    const voiceId = decodeURIComponent(url.pathname.slice("/voice-clone/voices/".length));
    if (!voiceId) {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Missing voice ID in URL" }));
      return;
    }

    voiceCloneManager
      .deleteClonedVoice(voiceId)
      .then((result) => {
        if (result.success) {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true, deleted: voiceId }));
        } else {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: result.error }));
        }
      })
      .catch((err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
      });
    return;
  }

  // POST /voice-clone/test -- Test a cloned voice with sample text
  if (req.method === "POST" && url.pathname === "/voice-clone/test") {
    readBody(req, MAX_IPC_BODY_BYTES)
      .then(async (body) => {
        const { voiceId, text } = JSON.parse(body);

        if (!voiceId || !text) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Missing 'voiceId' and 'text' in request body" }));
          return;
        }

        try {
          const { KokoCloneTTS: KCT } = await import("./providers/tts/kokoclone");
          const tts = new KCT({ provider: "kokoclone", voiceId, speed: 1.0 });
          const audio = await tts.synthesize(text);

          // Return base64-encoded WAV-like PCM
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({
            ok: true,
            audio: audio.toString("base64"),
            sampleRate: 16000,
            format: "pcm_s16le",
            durationMs: Math.round((audio.length / 2) / 16000 * 1000),
          }));
        } catch (err) {
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
        }
      })
      .catch((err) => ipcError(res, err, `${req.method} ${url.pathname}`));
    return;
  }

  // ---- Chatterbox Turbo Voice Cloning Endpoints ----

  // GET /chatterbox/status -- Check if Chatterbox Turbo is installed
  if (req.method === "GET" && url.pathname === "/chatterbox/status") {
    chatterboxVoiceManager
      .isInstalled()
      .then(async (installed) => {
        const voices = await chatterboxVoiceManager.listVoices();
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ installed, voices, count: voices.length }));
      })
      .catch((err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
      });
    return;
  }

  // GET /chatterbox/voices -- List all Chatterbox voices
  if (req.method === "GET" && url.pathname === "/chatterbox/voices") {
    chatterboxVoiceManager
      .listVoices()
      .then((voices) => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ voices }));
      })
      .catch((err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
      });
    return;
  }

  // POST /chatterbox/create -- Upload reference audio and create a Chatterbox voice
  if (req.method === "POST" && url.pathname === "/chatterbox/create") {
    readBody(req, 10_000_000) // 10MB max for audio upload
      .then(async (body) => {
        const { name, audio, filename } = JSON.parse(body);

        if (!name || !audio) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Missing 'name' and 'audio' (base64) in request body" }));
          return;
        }

        const audioBuffer = Buffer.from(audio, "base64");

        if (audioBuffer.length < 1000) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Audio file too small. Need ~10 seconds of clear speech." }));
          return;
        }

        if (audioBuffer.length > 5_000_000) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Audio file too large (max 5MB). Use a ~10 second clip." }));
          return;
        }

        const result = await chatterboxVoiceManager.createVoice(
          name,
          audioBuffer,
          filename || "reference.wav"
        );

        if (result.success) {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true, voice: result.voice }));
        } else {
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: result.error }));
        }
      })
      .catch((err) => ipcError(res, err, `${req.method} ${url.pathname}`));
    return;
  }

  // DELETE /chatterbox/voices/:id -- Delete a Chatterbox voice
  if (req.method === "DELETE" && url.pathname.startsWith("/chatterbox/voices/")) {
    const voiceId = decodeURIComponent(url.pathname.slice("/chatterbox/voices/".length));
    if (!voiceId) {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Missing voice ID in URL" }));
      return;
    }

    chatterboxVoiceManager
      .deleteVoice(voiceId)
      .then((result) => {
        if (result.success) {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true, deleted: voiceId }));
        } else {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: result.error }));
        }
      })
      .catch((err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
      });
    return;
  }

  // POST /chatterbox/test -- Test a Chatterbox voice with sample text
  if (req.method === "POST" && url.pathname === "/chatterbox/test") {
    readBody(req, MAX_IPC_BODY_BYTES)
      .then(async (body) => {
        const { voiceId, text } = JSON.parse(body);

        if (!voiceId || !text) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Missing 'voiceId' and 'text' in request body" }));
          return;
        }

        try {
          const { ChatterboxTurboTTS: CBT } = await import("./providers/tts/chatterbox");
          const tts = new CBT({ provider: "chatterbox", voiceId });
          const audio = await tts.synthesize(text);

          const wavBuffer = pcmToWav(audio, 16000, 1, 16);
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({
            ok: true,
            audio: wavBuffer.toString("base64"),
            mimeType: "audio/wav",
            sampleRate: 16000,
            durationMs: Math.round((audio.length / 2) / 16000 * 1000),
          }));
        } catch (err) {
          console.error("[voice-server] Chatterbox test error:", err);
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
        }
      })
      .catch((err) => ipcError(res, err, `${req.method} ${url.pathname}`));
    return;
  }

  // ---- Qwen3-TTS Voice Cloning Endpoints ----

  // GET /qwen3/status -- Check Qwen3-TTS container health + list voices
  if (req.method === "GET" && url.pathname === "/qwen3/status") {
    import("./providers/tts/qwen3").then(async ({ checkQwen3Health, qwen3VoiceManager }) => {
      const healthy = await checkQwen3Health();
      const voices = await qwen3VoiceManager.listVoices();
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ installed: healthy, healthy, voices, count: voices.length }));
    }).catch((err) => {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
    });
    return;
  }

  // GET /qwen3/voices -- List all Qwen3 cloned voices
  if (req.method === "GET" && url.pathname === "/qwen3/voices") {
    import("./providers/tts/qwen3").then(async ({ qwen3VoiceManager }) => {
      const voices = await qwen3VoiceManager.listVoices();
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ voices }));
    }).catch((err) => {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
    });
    return;
  }

  // POST /qwen3/create -- Upload reference audio and create a Qwen3 cloned voice
  if (req.method === "POST" && url.pathname === "/qwen3/create") {
    readBody(req, 10_000_000)
      .then(async (body) => {
        const { name, audio, filename, language, transcript } = JSON.parse(body);

        if (!name || !audio) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Missing 'name' and 'audio' (base64) in request body" }));
          return;
        }

        const audioBuffer = Buffer.from(audio, "base64");

        if (audioBuffer.length < 1000) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Audio file too small. Need 5-20 seconds of clear speech." }));
          return;
        }

        if (audioBuffer.length > 10_000_000) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Audio file too large (max 10MB). Use a 5-20 second clip." }));
          return;
        }

        const { qwen3VoiceManager } = await import("./providers/tts/qwen3");
        const voice = await qwen3VoiceManager.createVoice(
          name,
          audioBuffer,
          language || "en",
          transcript
        );

        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: true, voice }));
      })
      .catch((err) => ipcError(res, err, `${req.method} ${url.pathname}`));
    return;
  }

  // DELETE /qwen3/voices/:id -- Delete a Qwen3 voice
  if (req.method === "DELETE" && url.pathname.startsWith("/qwen3/voices/")) {
    const voiceId = decodeURIComponent(url.pathname.slice("/qwen3/voices/".length));
    if (!voiceId) {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Missing voice ID in URL" }));
      return;
    }

    import("./providers/tts/qwen3").then(async ({ qwen3VoiceManager }) => {
      const deleted = await qwen3VoiceManager.deleteVoice(voiceId);
      if (deleted) {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: true, deleted: voiceId }));
      } else {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: `Failed to delete voice ${voiceId}` }));
      }
    }).catch((err) => {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
    });
    return;
  }

  // POST /qwen3/test -- Test synthesis with a Qwen3 voice
  if (req.method === "POST" && url.pathname === "/qwen3/test") {
    readBody(req, MAX_IPC_BODY_BYTES)
      .then(async (body) => {
        const { voiceId, text } = JSON.parse(body);

        if (!voiceId || !text) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Missing 'voiceId' and 'text' in request body" }));
          return;
        }

        try {
          const { Qwen3TTS } = await import("./providers/tts/qwen3");
          const tts = new Qwen3TTS({ provider: "qwen3", voiceId });
          const audio = await tts.synthesize(text);

          const wavBuffer = pcmToWav(audio, 16000, 1, 16);
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({
            ok: true,
            audio: wavBuffer.toString("base64"),
            mimeType: "audio/wav",
            sampleRate: 16000,
            durationMs: Math.round((audio.length / 2) / 16000 * 1000),
          }));
        } catch (err) {
          console.error("[voice-server] Qwen3 test error:", err);
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
        }
      })
      .catch((err) => ipcError(res, err, `${req.method} ${url.pathname}`));
    return;
  }

  // ---- Fish Speech S2 voice management endpoints ----

  // GET /fish-speech/status -- Check Fish Speech health + list voices
  if (req.method === "GET" && url.pathname === "/fish-speech/status") {
    import("./providers/tts/fish-speech").then(async ({ checkFishSpeechHealth, fishSpeechVoiceManager }) => {
      const healthy = await checkFishSpeechHealth();
      const voices = await fishSpeechVoiceManager.listVoices();
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ installed: healthy, healthy, voices, count: voices.length }));
    }).catch((err) => {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
    });
    return;
  }

  // GET /fish-speech/voices -- List all Fish Speech cloned voices
  if (req.method === "GET" && url.pathname === "/fish-speech/voices") {
    import("./providers/tts/fish-speech").then(async ({ fishSpeechVoiceManager }) => {
      const voices = await fishSpeechVoiceManager.listVoices();
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ voices }));
    }).catch((err) => {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
    });
    return;
  }

  // POST /fish-speech/create -- Upload reference audio and create a Fish Speech cloned voice
  if (req.method === "POST" && url.pathname === "/fish-speech/create") {
    readBody(req, 10_000_000)
      .then(async (body) => {
        const { name, audio, filename, transcript } = JSON.parse(body);

        if (!name || !audio) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Missing 'name' and 'audio' (base64) in request body" }));
          return;
        }

        const audioBuffer = Buffer.from(audio, "base64");

        if (audioBuffer.length < 1000) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Audio file too small. Need 15-30 seconds of clear speech for best results." }));
          return;
        }

        if (audioBuffer.length > 10_000_000) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Audio file too large (max 10MB). Use a 15-30 second clip." }));
          return;
        }

        const { fishSpeechVoiceManager } = await import("./providers/tts/fish-speech");
        const voice = await fishSpeechVoiceManager.createVoice(
          name,
          audioBuffer,
          transcript
        );

        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: true, voice }));
      })
      .catch((err) => ipcError(res, err, `${req.method} ${url.pathname}`));
    return;
  }

  // DELETE /fish-speech/voices/:id -- Delete a Fish Speech voice
  if (req.method === "DELETE" && url.pathname.startsWith("/fish-speech/voices/")) {
    const voiceId = decodeURIComponent(url.pathname.slice("/fish-speech/voices/".length));
    if (!voiceId) {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Missing voice ID in URL" }));
      return;
    }

    import("./providers/tts/fish-speech").then(async ({ fishSpeechVoiceManager }) => {
      const deleted = await fishSpeechVoiceManager.deleteVoice(voiceId);
      if (deleted) {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: true, deleted: voiceId }));
      } else {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: `Failed to delete voice ${voiceId}` }));
      }
    }).catch((err) => {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
    });
    return;
  }

  // POST /fish-speech/test -- Test synthesis with a Fish Speech voice
  if (req.method === "POST" && url.pathname === "/fish-speech/test") {
    readBody(req, MAX_IPC_BODY_BYTES)
      .then(async (body) => {
        const { voiceId, text } = JSON.parse(body);

        if (!text) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Missing 'text' in request body" }));
          return;
        }

        try {
          const { FishSpeechTTS } = await import("./providers/tts/fish-speech");
          const tts = new FishSpeechTTS({ provider: "fish-speech", voiceId: voiceId || "" });
          const audio = await tts.synthesize(text);

          const wavBuffer = pcmToWav(audio, 16000, 1, 16);
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({
            ok: true,
            audio: wavBuffer.toString("base64"),
            mimeType: "audio/wav",
            sampleRate: 16000,
            durationMs: Math.round((audio.length / 2) / 16000 * 1000),
          }));
        } catch (err) {
          console.error("[voice-server] Fish Speech test error:", err);
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
        }
      })
      .catch((err) => ipcError(res, err, `${req.method} ${url.pathname}`));
    return;
  }

  // GET /logs -- Return recent log lines for real-time debugging from vapiclone UI
  if (req.method === "GET" && url.pathname === "/logs") {
    import("fs").then(async (fs) => {
      const lines = Math.min(parseInt(url.searchParams.get("lines") || "100", 10), 2000);
      const type = url.searchParams.get("type") || "all"; // "out", "error", or "all"
      try {
        const readLastLines = (filePath: string, n: number): string[] => {
          try {
            const content = fs.readFileSync(filePath, "utf8");
            const all = content.split("\n").filter(Boolean);
            return all.slice(-n);
          } catch { return []; }
        };
        const outLines = (type === "all" || type === "out")
          ? readLastLines("/var/log/voiceserver/out.log", lines)
          : [];
        const errLines = (type === "all" || type === "error")
          ? readLastLines("/var/log/voiceserver/error.log", lines)
          : [];
        // Merge and sort by timestamp prefix
        const merged = [...outLines, ...errLines].sort();
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: true, lines: merged.slice(-lines), total: merged.length }));
      } catch (err) {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: String(err) }));
      }
    });
    return;
  }

  // POST /tts/test -- Synthesize a voice sample for preview in the UI
  // Returns base64-encoded WAV audio
  if (req.method === "POST" && url.pathname === "/tts/test") {
    readBody(req, MAX_IPC_BODY_BYTES)
      .then(async (body) => {
        const { text, voiceId, provider } = JSON.parse(body);
        const sampleText = text || "Hello! I'm your AI assistant. How can I help you today?";
        const sampleVoice = voiceId || "af_heart";
        const sampleProvider = provider || "kokoro";

        try {
          let audioBuffer: Buffer;
          let sampleRate: number;

          if (sampleProvider === "kokoro" || !sampleProvider || sampleProvider === "piper") {
            const tts = await getOrCreateKokoroSingleton(sampleVoice);
            audioBuffer = await tts.synthesize(sampleText, undefined, sampleVoice);
            sampleRate = 16000;

            if (audioBuffer.length === 0) {
              kokoroTTSSingleton = null;
              res.writeHead(500, { "Content-Type": "application/json" });
              res.end(JSON.stringify({ error: "TTS synthesis returned empty audio. Voice pack may still be downloading — try again in 10s." }));
              return;
            }
          } else if (sampleProvider === "kokoclone" || sampleProvider === "clone") {
            const { KokoCloneTTS } = await import("./providers/tts/kokoclone");
            const tts = new KokoCloneTTS({ provider: "kokoclone", voiceId: sampleVoice, speed: 1.0 });
            audioBuffer = await tts.synthesize(sampleText);
            sampleRate = 16000;

            if (audioBuffer.length === 0) {
              res.writeHead(500, { "Content-Type": "application/json" });
              res.end(JSON.stringify({ error: "KokoClone TTS returned empty audio. Check that the cloned voice exists and dependencies are installed." }));
              return;
            }
          } else if (sampleProvider === "qwen3" || sampleProvider === "qwen3-tts") {
            const { Qwen3TTS } = await import("./providers/tts/qwen3");
            const tts = new Qwen3TTS({ provider: "qwen3", voiceId: sampleVoice });
            audioBuffer = await tts.synthesize(sampleText);
            sampleRate = 16000;

            if (audioBuffer.length === 0) {
              res.writeHead(500, { "Content-Type": "application/json" });
              res.end(JSON.stringify({ error: "Qwen3-TTS returned empty audio. Check that the service is running and the voice exists." }));
              return;
            }
          } else if (sampleProvider === "chatterbox" || sampleProvider === "chatterbox-turbo") {
            const { ChatterboxTurboTTS } = await import("./providers/tts/chatterbox");
            const tts = new ChatterboxTurboTTS({ provider: "chatterbox", voiceId: sampleVoice });
            audioBuffer = await tts.synthesize(sampleText);
            sampleRate = 16000;

            if (audioBuffer.length === 0) {
              res.writeHead(500, { "Content-Type": "application/json" });
              res.end(JSON.stringify({ error: "Chatterbox TTS returned empty audio." }));
              return;
            }
          } else if (sampleProvider === "unrealspeech") {
            const { UnrealSpeechTTS } = await import("./providers/tts/unrealspeech");
            const tts = new UnrealSpeechTTS({ provider: "unrealspeech", voiceId: sampleVoice });
            audioBuffer = await tts.synthesize(sampleText);
            sampleRate = 16000;
          } else {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: `Provider '${sampleProvider}' not supported for preview` }));
            return;
          }

          // Convert raw PCM s16le to WAV by prepending WAV header
          const wavBuffer = pcmToWav(audioBuffer, sampleRate, 1, 16);

          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({
            ok: true,
            audio: wavBuffer.toString("base64"),
            mimeType: "audio/wav",
            sampleRate,
            durationMs: Math.round((audioBuffer.length / 2) / sampleRate * 1000),
          }));
        } catch (err) {
          // Reset singleton on error so next request gets a fresh Python process
          kokoroTTSSingleton = null;
          console.error("[voice-server] TTS test error:", err);
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
        }
      })
      .catch((err) => ipcError(res, err, `${req.method} ${url.pathname}`));
    return;
  }

  // ---- POST /test-chat — LLM chat using same providers/tools as live calls ----
  if (req.method === "POST" && url.pathname === "/test-chat") {
    readBody(req, MAX_IPC_BODY_BYTES)
      .then(async (raw) => {
        const body = JSON.parse(raw);
        const { systemPrompt, messages, tools: bodyTools, toolMode, triggerPhrases, model } = body;

        if (!systemPrompt || !messages || !model) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Missing required fields: systemPrompt, messages, model" }));
          return;
        }

        // --- Overflow LLM routing (same as call-session.ts lines 872-904) ---
        const isOllama = model.provider === "ollama" || !model.provider;
        let usingOverflow = false;
        let decrementOnComplete = false;
        let llmProvider;

        try {
          if (isOllama) {
            const active = getOllamaActiveRequests();
            const max = getOllamaMaxParallel();
            const overflowProvider = process.env.OVERFLOW_LLM_PROVIDER;
            const overflowModel = process.env.OVERFLOW_LLM_MODEL;

            if (active >= max && overflowProvider && overflowModel) {
              console.log(`[test-chat] Overflow triggered: ollama ${active}/${max} → ${overflowProvider}/${overflowModel}`);
              llmProvider = createLLMProvider({
                provider: overflowProvider,
                model: overflowModel,
                temperature: model.temperature,
                maxTokens: model.maxTokens,
              });
              usingOverflow = true;
            } else {
              incrementOllama();
              decrementOnComplete = true;
              llmProvider = createLLMProvider(model);
            }
          } else {
            llmProvider = createLLMProvider(model);
          }
        } catch (err) {
          ipcError(res, err, "POST /test-chat create provider", 500);
          return;
        }

        // --- Build tool definitions (same as call-session.ts lines 906-954) ---
        const useTriggerPhrases = toolMode === "trigger-phrases";
        const safeTools = bodyTools || [];

        const llmTools: LLMToolDefinition[] = useTriggerPhrases ? [] : safeTools
          .filter((t: any) => t.functionDefinition)
          .map((t: any) => ({
            type: "function" as const,
            function: {
              name: t.functionDefinition.name,
              description: t.functionDefinition.description,
              parameters: t.functionDefinition.parameters,
            },
          }));

        if (!useTriggerPhrases) {
          for (const t of safeTools) {
            if (t.type === "endCall" && !t.functionDefinition) {
              llmTools.push({ type: "function", function: { name: "end_call", description: "End the phone call", parameters: { type: "object", properties: {} } } });
            }
            if (t.type === "transferCall" && !t.functionDefinition) {
              llmTools.push({ type: "function", function: { name: t.name, description: t.description || "Transfer the call", parameters: { type: "object", properties: {} } } });
            }
            if (t.type === "dtmf" && !t.functionDefinition) {
              llmTools.push({ type: "function", function: { name: t.name, description: t.description || "Send DTMF tones", parameters: { type: "object", properties: {} } } });
            }
          }
        }

        // --- Build LLM messages ---
        const llmMessages: LLMMessage[] = [
          { role: "system", content: systemPrompt },
          ...messages.map((m: any) => ({ role: m.role as "user" | "assistant", content: m.content })),
        ];

        // --- Run streamCompletion as a Promise ---
        const releaseSlot = () => {
          if (decrementOnComplete) {
            decrementOnComplete = false;
            decrementOllama();
          }
        };

        const TIMEOUT_MS = 30_000;
        let timedOut = false;

        try {
          const result = await new Promise<{ fullText: string; toolCalls: LLMToolCall[]; ignoredToolCall: boolean }>((resolve, reject) => {
            let fullText = "";
            const collectedToolCalls: LLMToolCall[] = [];
            let ignoredToolCall = false;

            const timeout = setTimeout(() => {
              timedOut = true;
              cancelFn?.();
              releaseSlot();
              reject(new Error("LLM request timed out after 30s"));
            }, TIMEOUT_MS);

            let cancelFn: (() => void) | null = null;
            const { cancel } = llmProvider.streamCompletion({
              messages: llmMessages,
              tools: llmTools.length > 0 ? llmTools : undefined,
              onToken: (token: string) => { fullText += token; },
              onToolCall: (toolCall: LLMToolCall) => {
                if (useTriggerPhrases) {
                  ignoredToolCall = true;
                  return;
                }
                collectedToolCalls.push(toolCall);
              },
              onDone: () => {
                clearTimeout(timeout);
                if (!timedOut) resolve({ fullText, toolCalls: collectedToolCalls, ignoredToolCall });
              },
            });
            cancelFn = cancel;
          });

          releaseSlot();

          let reply = result.fullText;

          // --- Handle tool-call-only response in trigger-phrases mode (retry) ---
          if (useTriggerPhrases && result.ignoredToolCall && !reply.trim()) {
            console.log("[test-chat] LLM produced only tool call, no text — retrying with clean messages");
            const cleanMessages: LLMMessage[] = llmMessages.filter(
              (m: any) => m.role !== "tool" && !m.tool_calls && !m.tool_call_id
            );
            cleanMessages.push({ role: "system", content: "Respond naturally to the customer. Do not attempt to call any functions or tools. Just speak." });

            const retryResult = await new Promise<{ fullText: string }>((resolve, reject) => {
              let fullText = "";
              const timeout = setTimeout(() => { reject(new Error("Retry timed out")); }, TIMEOUT_MS);
              llmProvider.streamCompletion({
                messages: cleanMessages,
                onToken: (token: string) => { fullText += token; },
                onToolCall: () => {},
                onDone: () => { clearTimeout(timeout); resolve({ fullText }); },
              });
            });
            reply = retryResult.fullText;
          }

          // --- Strip <think> blocks ---
          reply = reply.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
          reply = reply.replace(/<think>[\s\S]*/g, "").trim();
          if (!reply) {
            reply = "I'm sorry, could you repeat that? I want to make sure I understand correctly.";
          }

          // --- Trigger phrase scanning ---
          const responseToolCalls: { name: string; action: string; arguments?: string }[] = [];

          if (useTriggerPhrases && triggerPhrases && triggerPhrases.length > 0) {
            const lower = reply.toLowerCase();
            for (const tp of triggerPhrases) {
              if (lower.includes(tp.phrase.toLowerCase())) {
                console.log(`[test-chat] Trigger phrase matched: "${tp.phrase}" → ${tp.toolName}`);
                const tool = safeTools.find((t: any) => t.name === tp.toolName || t.type === tp.toolName);
                if (tool) {
                  responseToolCalls.push({ name: tool.name || tp.toolName, action: tool.type === "endCall" ? "endCall" : tool.type === "transferCall" ? "transfer" : tool.type === "dtmf" ? "dtmf" : "function" });
                } else if (tp.toolName === "end_call" || tp.toolName === "endCall") {
                  responseToolCalls.push({ name: "end_call", action: "endCall" });
                } else if (tp.toolName === "transfer" || tp.toolName === "transferCall") {
                  responseToolCalls.push({ name: "transferCall", action: "transfer" });
                }
                break; // Only first match
              }
            }
          }

          // --- In tools mode, map LLM tool calls ---
          if (!useTriggerPhrases && result.toolCalls.length > 0) {
            for (const tc of result.toolCalls) {
              const matchingTool = safeTools.find((t: any) =>
                (t.functionDefinition && t.functionDefinition.name === tc.function.name) || t.name === tc.function.name
              );
              const action = matchingTool?.type === "endCall" ? "endCall"
                : matchingTool?.type === "transferCall" ? "transfer"
                : matchingTool?.type === "dtmf" ? "dtmf"
                : tc.function.name === "end_call" ? "endCall"
                : "function";
              responseToolCalls.push({ name: tc.function.name, action, arguments: tc.function.arguments });
            }
          }

          const responseBody: any = { reply };
          if (responseToolCalls.length > 0) responseBody.toolCalls = responseToolCalls;
          if (usingOverflow) responseBody.overflow = true;

          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify(responseBody));
        } catch (err) {
          releaseSlot();
          ipcError(res, err, "POST /test-chat LLM", 500);
        }
      })
      .catch((err) => ipcError(res, err, "POST /test-chat"));
    return;
  }

  res.writeHead(404);
  res.end("Not found");
}

/** Convert raw PCM (s16le) buffer to WAV format */
function pcmToWav(pcm: Buffer, sampleRate: number, numChannels: number, bitDepth: number): Buffer {
  const dataSize = pcm.length;
  const header = Buffer.alloc(44);
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + dataSize, 4);
  header.write("WAVE", 8);
  header.write("fmt ", 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20); // PCM
  header.writeUInt16LE(numChannels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE((sampleRate * numChannels * bitDepth) / 8, 28);
  header.writeUInt16LE((numChannels * bitDepth) / 8, 32);
  header.writeUInt16LE(bitDepth, 34);
  header.write("data", 36);
  header.writeUInt32LE(dataSize, 40);
  return Buffer.concat([header, pcm]);
}

const ipcServer = createServer(handleIPC);
// Bind to 0.0.0.0 so it's accessible from Railway and other external services
ipcServer.listen(IPC_PORT, "0.0.0.0", () => {
  console.log(`[voice-server] IPC endpoints: /health, /models, /models/status, /models/{llm,stt,tts}, /models/search, /tts/test, /test-chat, /sessions, /metrics, /settings, /register-call, /end-call/:id, /voice-clone/*, /chatterbox/*`);
});

// ---- Call Config Registration ----

const AUDIO_CACHE_DIR = path.join(process.env.DATA_DIR || "/data", "audio-first-messages");

// Ensure audio cache directory exists on startup
fs.mkdir(AUDIO_CACHE_DIR, { recursive: true }).catch(() => {});

function registerCallConfig(callId: string, config: CallSessionConfig): void {
  pendingConfigs.set(callId, { config, createdAt: Date.now() });

  // Audio First Message: check for cached mulaw file or synthesize and cache
  if (
    config.audioFirstMessage &&
    config.firstMessage &&
    config.firstMessageMode !== "assistant-waits-for-user" &&
    config.voice?.provider === "kokoro"
  ) {
    let firstMsg = config.firstMessage;
    if (config.variableValues) {
      firstMsg = firstMsg.replace(/\{\{(\w+)\}\}/g, (_, key) => config.variableValues?.[key] || "");
    }

    const agentName = config.variableValues?.agentName || "default";
    const textHash = createHash("md5").update(firstMsg).digest("hex").slice(0, 8);
    const cacheFile = path.join(AUDIO_CACHE_DIR, `${config.assistantId}_${agentName}_${textHash}.ulaw`);

    fs.readFile(cacheFile).then((mulawBuffer) => {
      // Cache hit — instant playback
      const entry = pendingConfigs.get(callId);
      if (entry) {
        entry.config.preSynthesizedFirstMessageMulaw = mulawBuffer;
        if (!entry.config.metadata) entry.config.metadata = {};
        (entry.config.metadata as Record<string, unknown>).firstMessageSource = "audio-cache";
        console.log(`[voice-server] Audio cache HIT for ${callId} (${cacheFile}, ${mulawBuffer.length} bytes)`);
      }
    }).catch(() => {
      // Cache miss — synthesize, convert to mulaw, save, and use
      const tts = new KokoroTTS({ provider: "kokoro", voiceId: config.voice!.voiceId });
      const startMs = Date.now();
      tts.synthesize(firstMsg).then(async (pcm16k) => {
        const mulaw = pcm16kToMulaw(pcm16k);
        // Save to cache for future calls
        await fs.writeFile(cacheFile, mulaw).catch((e) =>
          console.warn(`[voice-server] Failed to cache audio file: ${e.message}`)
        );
        const entry = pendingConfigs.get(callId);
        if (entry) {
          entry.config.preSynthesizedFirstMessageMulaw = mulaw;
          if (!entry.config.metadata) entry.config.metadata = {};
          (entry.config.metadata as Record<string, unknown>).firstMessageSource = "live-tts";
          console.log(`[voice-server] Audio cache MISS — synthesized and cached for ${callId} (${mulaw.length} bytes, ${Date.now() - startMs}ms, saved to ${cacheFile})`);
        }
      }).catch((err) => {
        console.warn(`[voice-server] Audio first message synth failed for ${callId}: ${err.message} — will synthesize live`);
      });
    });
    return;
  }

  // Standard pre-synthesize first message with Kokoro while waiting for Twilio WebSocket
  if (
    config.firstMessage &&
    config.firstMessageMode !== "assistant-waits-for-user" &&
    config.voice?.provider === "kokoro"
  ) {
    let firstMsg = config.firstMessage;
    if (config.variableValues) {
      firstMsg = firstMsg.replace(/\{\{(\w+)\}\}/g, (_, key) => config.variableValues?.[key] || "");
    }

    const tts = new KokoroTTS({ provider: "kokoro", voiceId: config.voice.voiceId });
    const startMs = Date.now();
    tts.synthesize(firstMsg).then((pcm16k) => {
      const entry = pendingConfigs.get(callId);
      if (entry) {
        entry.config.preSynthesizedFirstMessage = pcm16k;
        console.log(`[voice-server] Pre-synthesized first message for ${callId} (${pcm16k.length} bytes, ${Date.now() - startMs}ms)`);
      }
    }).catch((err) => {
      console.warn(`[voice-server] Pre-synth failed for ${callId}: ${err.message} — will synthesize live`);
    });
  }
}

// ---- Periodic Health & Stale Session Cleanup ----

const healthInterval = setInterval(() => {
  const mem = process.memoryUsage();
  const uptimeSec = Math.round((Date.now() - startedAt) / 1000);
  const uptimeMin = Math.round(uptimeSec / 60);

  console.log(
    `[voice-server] Health: sessions=${sessions.size}/${MAX_SESSIONS} ` +
      `total=${totalCallsHandled} pending=${pendingConfigs.size} ` +
      `mem=${Math.round(mem.rss / 1024 / 1024)}MB ` +
      `uptime=${uptimeMin}m`
  );

  // Check for stale sessions (calls exceeding max duration)
  const now = Date.now();
  for (const [callId, entry] of sessions) {
    const durationMs = now - entry.startedAt;
    if (durationMs > STALE_SESSION_TIMEOUT) {
      console.warn(
        `[voice-server] Stale session detected: ${callId} (${Math.round(durationMs / 60000)}min)`
      );
      entry.session.endCall("max-duration-safety-timeout");
    }
  }
}, HEALTH_LOG_INTERVAL);

// ---- Graceful Shutdown ----

async function gracefulShutdown(signal: string): Promise<void> {
  if (isShuttingDown) return;
  isShuttingDown = true;

  console.log(`\n[voice-server] ${signal} received -- starting graceful shutdown...`);
  console.log(`[voice-server] Active sessions: ${sessions.size}`);

  // Stop metrics collection
  stopMetricsCollection();

  // Stop accepting new connections
  wss.close();
  ipcServer.close();
  clearInterval(healthInterval);

  // End all active sessions
  const endPromises: Promise<void>[] = [];
  for (const [callId, entry] of sessions) {
    console.log(`[voice-server] Ending session: ${callId}`);
    entry.session.endCall("server-shutdown");
    endPromises.push(
      new Promise<void>((resolve) => {
        const timeout = setTimeout(resolve, 5000);
        entry.session.once("ended", () => {
          clearTimeout(timeout);
          resolve();
        });
      })
    );
  }

  // Wait for all sessions to end (max 10 seconds)
  if (endPromises.length > 0) {
    console.log(`[voice-server] Waiting for ${endPromises.length} sessions to drain...`);
    await Promise.race([
      Promise.all(endPromises),
      new Promise((resolve) => setTimeout(resolve, 10000)),
    ]);
  }

  console.log(
    `[voice-server] Shutdown complete. Handled ${totalCallsHandled} total calls.`
  );
  process.exit(0);
}

process.on("SIGTERM", () => gracefulShutdown("SIGTERM"));
process.on("SIGINT", () => gracefulShutdown("SIGINT"));

// ---- Uncaught Error Handlers ----

process.on("uncaughtException", (err) => {
  console.error("[voice-server] UNCAUGHT EXCEPTION:", err);
  // Don't exit -- keep serving existing calls
});

process.on("unhandledRejection", (reason) => {
  console.error("[voice-server] UNHANDLED REJECTION:", reason);
  // Don't exit -- keep serving existing calls
});
