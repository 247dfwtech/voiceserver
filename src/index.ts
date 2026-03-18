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
import { modelManager } from "./model-manager";
import { voiceCloneManager } from "./providers/tts/kokoclone";
import { chatterboxVoiceManager } from "./providers/tts/chatterbox";
import { metricsBuffer, startMetricsCollection, stopMetricsCollection, setSessionCountProvider, collectSnapshot } from "./gpu-monitor";

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
}

const sessions = new Map<string, SessionEntry>();
const pendingConfigs = new Map<string, CallSessionConfig>();
const startedAt = Date.now();
let totalCallsHandled = 0;
let isShuttingDown = false;

const MAX_PENDING_CONFIGS = 200;
const MAX_IPC_BODY_BYTES = 1_000_000; // 1MB

// Singleton KokoroTTS instance — keeps the Python process alive between /tts/test calls
// so the model doesn't need to reload on every preview request (cold start is 60-90s)
let kokoroTTSSingleton: import("./providers/tts/kokoro").KokoroTTS | null = null;
async function getOrCreateKokoroSingleton(voiceId: string): Promise<import("./providers/tts/kokoro").KokoroTTS> {
  const { KokoroTTS } = await import("./providers/tts/kokoro");
  if (!kokoroTTSSingleton) {
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
  console.log("[voice-server] Model manager ready (default LLM: qwen3.5:9b)");
}).catch((err) => {
  console.error("[voice-server] Model manager initialization failed:", err);
});

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
          streamSid = msg.streamSid;
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

          console.log(
            `[voice-server] Stream started: callId=${callId}, streamSid=${streamSid} ` +
              `(${sessions.size + 1}/${MAX_SESSIONS} sessions)`
          );

          const config = pendingConfigs.get(callId);
          if (!config) {
            console.error(`[voice-server] No pending config for callId=${callId}`);
            ws.close();
            return;
          }
          pendingConfigs.delete(callId);

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

          // Handle call transfer
          session.on("transfer", async (transferData: { destination?: string }) => {
            if (transferData.destination) {
              try {
                const twilioCallSid = msg.start?.callSid;
                if (twilioCallSid) {
                  const result = await transferCallWithDial({
                    callSid: twilioCallSid,
                    destination: transferData.destination,
                    publicUrl: process.env.PUBLIC_URL || "",
                    callerNumber: config.customerNumber,
                  });
                  if (!result.success) {
                    console.error(`[voice-server] Transfer failed:`, result.error);
                  }
                }
              } catch (err) {
                console.error(`[voice-server] Transfer error:`, err);
              }
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
          console.log(`[voice-server] Stream stopped: callId=${callId}`);
          if (callId) {
            const entry = sessions.get(callId);
            if (entry) {
              entry.session.endCall("twilio-stream-stopped");
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
  },
  config: CallSessionConfig
): Promise<void> {
  // Run post-call analysis if configured
  let analysis: { summary?: string; successEvaluation?: string } = {};
  const analysisConfig = config.analysisConfig;
  if (analysisConfig) {
    try {
      analysis = await runPostCallAnalysis(endData.transcript, analysisConfig);
    } catch (err) {
      console.error(`[voice-server] Analysis failed for ${callId}:`, err);
    }
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
          analysis,
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

    // Check Ollama connectivity asynchronously
    checkOllama().then((ollamaStatus) => {
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
          ollama: ollamaStatus,
        })
      );
    }).catch(() => {
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
          ollama: { ok: false, error: "check failed" },
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
      .catch((err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : "Bad request" }));
      });
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
      .catch((err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : "Bad request" }));
      });
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
      .catch((err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : "Bad request" }));
      });
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
      .catch((err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : "Bad request" }));
      });
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
      .catch((err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : "Bad request" }));
      });
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
      .catch((err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : "Bad request" }));
      });
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
      .catch(() => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Bad request" }));
      });
    return;
  }

  // GET /sessions -- List active sessions (PII masked)
  if (req.method === "GET" && url.pathname === "/sessions") {
    const sessionList = Array.from(sessions.entries()).map(([id, entry]) => ({
      callId: id,
      assistantId: entry.config.assistantId,
      customerNumber: maskPhone(entry.config.customerNumber),
      state: entry.session.getState(),
      duration: Math.round((Date.now() - entry.startedAt) / 1000),
    }));

    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ sessions: sessionList, count: sessionList.length }));
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
      .catch((err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : "Bad request" }));
      });
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
      .catch((err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : "Bad request" }));
      });
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
      .catch((err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : "Bad request" }));
      });
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
      .catch((err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : "Bad request" }));
      });
    return;
  }

  // GET /logs -- Return recent log lines for real-time debugging from vapiclone UI
  if (req.method === "GET" && url.pathname === "/logs") {
    import("fs").then(async (fs) => {
      const lines = parseInt(url.searchParams.get("lines") || "100", 10);
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
            sampleRate = 16000; // synthesize returns 16kHz after resampling

            // If synthesis returned empty audio (e.g. voice pack 404 from HuggingFace),
            // reset the singleton so the next request starts a fresh Python process.
            if (audioBuffer.length === 0) {
              kokoroTTSSingleton = null;
              res.writeHead(500, { "Content-Type": "application/json" });
              res.end(JSON.stringify({ error: "TTS synthesis returned empty audio. Voice pack may still be downloading — try again in 10s." }));
              return;
            }
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
      .catch((err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : "Bad request" }));
      });
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
  console.log(`[voice-server] IPC endpoints: /health, /models, /models/status, /models/{llm,stt,tts}, /models/search, /tts/test, /sessions, /metrics, /register-call, /end-call/:id, /voice-clone/*, /chatterbox/*`);
});

// ---- Call Config Registration ----

function registerCallConfig(callId: string, config: CallSessionConfig): void {
  pendingConfigs.set(callId, config);

  // Auto-expire after 60 seconds if Twilio never connects
  setTimeout(() => {
    if (pendingConfigs.has(callId)) {
      pendingConfigs.delete(callId);
      console.warn(`[voice-server] Pending config expired: ${callId}`);
    }
  }, 60000);
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
