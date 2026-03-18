import { EventEmitter } from "events";
import WebSocket from "ws";
import type { STTProvider, STTConfig } from "./interface";

/**
 * Deepgram STT provider -- uses Deepgram's Flux turn-based streaming API (v2).
 *
 * Flux is purpose-built for voice agents with native end-of-turn detection
 * using both acoustic and semantic cues. No manual VAD/silence detection needed.
 *
 * Protocol: WebSocket to wss://api.deepgram.com/v2/listen
 * Audio in: raw PCM 16-bit 16kHz mono (linear16) sent as binary frames
 * Events out: StartOfTurn, Update (interim), EagerEndOfTurn, EndOfTurn, TurnResumed
 *
 * @see https://developers.deepgram.com/reference/speech-to-text/listen-flux
 */

/** Available Deepgram models for the turn-based (v2) and standard (v1) APIs */
export const DEEPGRAM_MODELS = [
  // Turn-based (v2 / Flux) -- recommended for voice agents
  { id: "flux-general-en", name: "Flux General (English)", api: "v2", description: "Conversational STT with native end-of-turn detection (recommended for voice agents)" },
  // Standard streaming (v1 / Nova)
  { id: "nova-3-general", name: "Nova-3 General", api: "v1", description: "Latest general-purpose model, highest accuracy" },
  { id: "nova-3-medical", name: "Nova-3 Medical", api: "v1", description: "Medical terminology optimized" },
  { id: "nova-2-general", name: "Nova-2 General", api: "v1", description: "Previous generation, stable" },
];

const DEFAULT_MODEL = "flux-general-en";
const DEEPGRAM_WS_BASE = "wss://api.deepgram.com";
const RECONNECT_DELAY_MS = 2000;
const MAX_RECONNECTS = 5;
const KEEPALIVE_INTERVAL_MS = 5_000; // v1 only — Deepgram v1 closes after 10s of no audio/keepalive

export class DeepgramSTT extends EventEmitter implements STTProvider {
  private config: STTConfig;
  private apiKey: string;
  private model: string;
  private ws: WebSocket | null = null;
  private closed = false;
  private reconnectCount = 0;
  private keepaliveTimer: ReturnType<typeof setInterval> | null = null;
  private currentTurnTranscript = "";
  private inTurn = false;

  constructor(config: STTConfig, apiKey: string) {
    super();
    this.config = { ...config, acceptsMulaw: true }; // Deepgram accepts raw mulaw — skip PCM conversion
    this.apiKey = apiKey;
    this.model = config.model || DEFAULT_MODEL;
  }

  async start(): Promise<void> {
    await this.connect();
  }

  private audioBytesSent = 0;

  send(audio: Buffer): void {
    if (this.closed || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    this.audioBytesSent += audio.length;
    // Log first audio send and periodically
    if (this.audioBytesSent === audio.length || this.audioBytesSent % 32000 < audio.length) {
      console.log(`[stt/deepgram] Audio sent: ${this.audioBytesSent} bytes total, chunk=${audio.length}B, ws=${this.ws.readyState}`);
    }
    this.ws.send(audio);
  }

  async finish(): Promise<void> {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "Finalize" }));
    }
  }

  close(): void {
    this.closed = true;
    this.stopKeepalive();
    if (this.ws) {
      try {
        if (this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({ type: "CloseStream" }));
        }
        this.ws.close();
      } catch {}
      this.ws = null;
    }
    this.removeAllListeners();
  }

  /** Update keyterms mid-stream (Flux Configure message) */
  updateKeyterms(keyterms: string[]): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: "Configure",
        keyterms,
      }));
    }
  }

  private async connect(): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      if (this.closed) { reject(new Error("DeepgramSTT is closed")); return; }

      const modelInfo = DEEPGRAM_MODELS.find((m) => m.id === this.model);
      const apiVersion = modelInfo?.api === "v1" ? "v1" : "v2";

      // Build WebSocket URL with query parameters
      // Use mulaw 8kHz directly from Twilio — no PCM conversion needed, better quality
      const params = new URLSearchParams();
      params.set("model", this.model);
      params.set("encoding", "mulaw");
      params.set("sample_rate", "8000");

      if (apiVersion === "v2") {
        // Flux turn-based settings
        params.set("eot_threshold", "0.7");
        params.set("eot_timeout_ms", "5000");
      }

      // Add keywords as keyterms for Flux, or as keywords for Nova
      if (this.config.keywords && this.config.keywords.length > 0) {
        for (const kw of this.config.keywords) {
          params.append("keyterm", kw);
        }
      }

      if (this.config.language && this.config.language !== "en") {
        params.set("language", this.config.language);
      }

      const wsUrl = `${DEEPGRAM_WS_BASE}/${apiVersion}/listen?${params.toString()}`;

      console.log(`[stt/deepgram] Connecting to ${apiVersion} API (model=${this.model})...`);

      this.ws = new WebSocket(wsUrl, {
        headers: {
          Authorization: `Token ${this.apiKey}`,
        },
      });

      let connected = false;

      this.ws.on("open", () => {
        connected = true;
        this.reconnectCount = 0;
        console.log(`[stt/deepgram] Connected (model=${this.model}, api=${apiVersion})`);
        this.startKeepalive();
        resolve();
      });

      this.ws.on("message", (data: WebSocket.Data) => {
        try {
          const msg = JSON.parse(data.toString());
          if (apiVersion === "v2") {
            this.handleV2Message(msg);
          } else {
            this.handleV1Message(msg);
          }
        } catch (err) {
          console.error("[stt/deepgram] Failed to parse message:", err);
        }
      });

      this.ws.on("error", (err: Error) => {
        console.error("[stt/deepgram] WebSocket error:", err.message);
        if (!connected) {
          reject(err);
        } else {
          this.emit("error", err);
        }
      });

      this.ws.on("close", (code: number, reason: Buffer) => {
        this.stopKeepalive();
        const reasonStr = reason.toString() || "unknown";
        console.warn(`[stt/deepgram] WebSocket closed: code=${code} reason=${reasonStr}`);

        if (!this.closed && this.reconnectCount < MAX_RECONNECTS) {
          this.reconnectCount++;
          console.log(`[stt/deepgram] Reconnecting (${this.reconnectCount}/${MAX_RECONNECTS})...`);
          setTimeout(() => {
            this.connect().catch((err) => {
              this.emit("error", err);
            });
          }, RECONNECT_DELAY_MS);
        } else if (!this.closed) {
          this.emit("error", new Error(`Deepgram WebSocket closed after ${MAX_RECONNECTS} reconnect attempts`));
        }
      });
    });
  }

  /**
   * Handle Flux v2 turn-based messages.
   *
   * Events: StartOfTurn, Update, EagerEndOfTurn, TurnResumed, EndOfTurn
   */
  private handleV2Message(msg: Record<string, unknown>): void {
    // Connected confirmation
    if (msg.type === "ListenV2Connected") {
      console.log(`[stt/deepgram] Session established: request_id=${msg.request_id}`);
      return;
    }

    // Configuration success/failure
    if (msg.type === "ListenV2ConfigureSuccess") {
      console.log(`[stt/deepgram] Configure accepted`);
      return;
    }
    if (msg.type === "ListenV2ConfigureFailure") {
      console.error(`[stt/deepgram] Configure rejected:`, msg);
      return;
    }

    // Fatal error
    if (msg.type === "ListenV2FatalError") {
      const errMsg = `Deepgram fatal error: ${msg.description} (code=${msg.code})`;
      console.error(`[stt/deepgram] ${errMsg}`);
      this.emit("error", new Error(errMsg));
      return;
    }

    // Turn events (ListenV2TurnInfo)
    const event = msg.event as string | undefined;
    const transcript = (msg.transcript as string) || "";

    switch (event) {
      case "StartOfTurn":
        this.inTurn = true;
        this.currentTurnTranscript = "";
        this.emit("speech_started");
        break;

      case "Update":
        // Interim transcript — emit as non-final
        if (transcript) {
          this.currentTurnTranscript = transcript;
          this.emit("transcript", {
            text: transcript,
            isFinal: false,
            confidence: undefined,
          });
        }
        break;

      case "EagerEndOfTurn":
        // Early end-of-turn signal — emit current transcript as final
        // but don't trigger utterance_end yet (TurnResumed might follow)
        if (transcript) {
          this.currentTurnTranscript = transcript;
        }
        break;

      case "TurnResumed":
        // Speaker continued after EagerEndOfTurn — keep listening
        // No action needed, Updates will continue
        break;

      case "EndOfTurn": {
        // Definitive end of turn — emit final transcript and utterance_end
        const finalText = transcript || this.currentTurnTranscript;
        if (finalText.trim()) {
          this.emit("transcript", {
            text: finalText.trim(),
            isFinal: true,
            confidence: (msg.end_of_turn_confidence as number) ?? undefined,
          });
          this.emit("utterance_end");
        }
        this.inTurn = false;
        this.currentTurnTranscript = "";
        break;
      }
    }
  }

  /**
   * Handle Nova v1 standard streaming messages (fallback for non-Flux models).
   */
  private handleV1Message(msg: Record<string, unknown>): void {
    // Standard Deepgram v1 response
    const channel = (msg.channel as { alternatives?: Array<{ transcript?: string; confidence?: number }> }) || {};
    const alt = channel.alternatives?.[0];
    if (!alt || !alt.transcript) return;

    const isFinal = (msg.is_final as boolean) ?? false;
    const speechFinal = (msg.speech_final as boolean) ?? false;

    if (alt.transcript.trim()) {
      // Emit speech_started on first transcript if not already in turn
      if (!this.inTurn) {
        this.inTurn = true;
        this.emit("speech_started");
      }

      this.emit("transcript", {
        text: alt.transcript.trim(),
        isFinal,
        confidence: alt.confidence,
      });

      if (isFinal) {
        this.currentTurnTranscript += (this.currentTurnTranscript ? " " : "") + alt.transcript.trim();
      }

      // speech_final = true means Deepgram detected end of utterance
      if (speechFinal) {
        this.emit("utterance_end");
        this.inTurn = false;
        this.currentTurnTranscript = "";
      }
    }
  }

  private startKeepalive(): void {
    this.stopKeepalive();
    // Use WebSocket ping frames instead of text KeepAlive messages.
    // Deepgram docs warn that sending KeepAlive as text data frames
    // interleaved with binary audio "will cause the audio processing to choke."
    // Twilio sends continuous audio so text KeepAlive is redundant during calls,
    // but ping frames keep the TCP connection alive without disrupting audio processing.
    this.keepaliveTimer = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.ping();
      }
    }, KEEPALIVE_INTERVAL_MS);
  }

  private stopKeepalive(): void {
    if (this.keepaliveTimer) {
      clearInterval(this.keepaliveTimer);
      this.keepaliveTimer = null;
    }
  }
}
