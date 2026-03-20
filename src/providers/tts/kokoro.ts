import type { TTSProvider, TTSConfig } from "./interface";

/**
 * Kokoro-82M TTS provider -- self-hosted, free, Apache 2.0, #1 TTS Arena.
 *
 * Calls the Kokoro-FastAPI HTTP service (OpenAI-compatible /v1/audio/speech).
 * Each synthesis is an independent HTTP request -- true concurrent synthesis
 * with no shared queue or subprocess bottleneck.
 *
 * Key advantages:
 *   - Near-human naturalness (#1 in TTS Spaces Arena)
 *   - Sub-0.3s latency on GPU, 210x real-time on RTX 4090
 *   - True concurrency: 50-100+ simultaneous streams on RTX 4090
 *   - Only ~1.8GB VRAM (82M params + FastAPI overhead)
 *   - 67 voices, Apache 2.0 license
 */

// Available Kokoro voices (American English subset -- most useful for sales calls)
export const KOKORO_VOICES = [
  // American Female
  { id: "af_heart", name: "Heart", gender: "female", accent: "american", description: "Warm, natural (DEFAULT)" },
  { id: "af_alloy", name: "Alloy", gender: "female", accent: "american", description: "Clear, professional" },
  { id: "af_aoede", name: "Aoede", gender: "female", accent: "american", description: "Melodic, engaging" },
  { id: "af_bella", name: "Bella", gender: "female", accent: "american", description: "Friendly, approachable" },
  { id: "af_jessica", name: "Jessica", gender: "female", accent: "american", description: "Confident, articulate" },
  { id: "af_kore", name: "Kore", gender: "female", accent: "american", description: "Youthful, energetic" },
  { id: "af_nicole", name: "Nicole", gender: "female", accent: "american", description: "Smooth, calm" },
  { id: "af_nova", name: "Nova", gender: "female", accent: "american", description: "Bright, modern" },
  { id: "af_river", name: "River", gender: "female", accent: "american", description: "Flowing, pleasant" },
  { id: "af_sarah", name: "Sarah", gender: "female", accent: "american", description: "Conversational" },
  { id: "af_sky", name: "Sky", gender: "female", accent: "american", description: "Light, airy" },
  // American Male
  { id: "am_adam", name: "Adam", gender: "male", accent: "american", description: "Deep, authoritative" },
  { id: "am_echo", name: "Echo", gender: "male", accent: "american", description: "Rich, resonant" },
  { id: "am_eric", name: "Eric", gender: "male", accent: "american", description: "Professional, clear" },
  { id: "am_fenrir", name: "Fenrir", gender: "male", accent: "american", description: "Strong, commanding" },
  { id: "am_liam", name: "Liam", gender: "male", accent: "american", description: "Warm, trustworthy" },
  { id: "am_michael", name: "Michael", gender: "male", accent: "american", description: "Versatile, natural" },
  { id: "am_onyx", name: "Onyx", gender: "male", accent: "american", description: "Smooth, deep" },
  // British Female
  { id: "bf_emma", name: "Emma", gender: "female", accent: "british", description: "British, refined" },
  { id: "bf_isabella", name: "Isabella", gender: "female", accent: "british", description: "British, elegant" },
  // British Male
  { id: "bm_george", name: "George", gender: "male", accent: "british", description: "British, distinguished" },
  { id: "bm_lewis", name: "Lewis", gender: "male", accent: "british", description: "British, casual" },
];

const DEFAULT_VOICE = "af_heart";
const KOKORO_SAMPLE_RATE = 24000; // Kokoro-FastAPI outputs at 24kHz
const PIPELINE_SAMPLE_RATE = 16000; // Voice pipeline expects 16kHz
const KOKORO_API_URL = process.env.KOKORO_API_URL || "http://localhost:8880";

// ---- Resampler (kept from original -- proven, no dependencies) ----

function _resampleTo16k(input: Buffer, sourceRate: number): Buffer {
  if (sourceRate === PIPELINE_SAMPLE_RATE) return input;
  const ratio = sourceRate / PIPELINE_SAMPLE_RATE;
  const inputSamples = input.length / 2;
  const outputSamples = Math.floor(inputSamples / ratio);
  const output = Buffer.alloc(outputSamples * 2);
  for (let i = 0; i < outputSamples; i++) {
    const srcPos = i * ratio;
    const srcIndex = Math.floor(srcPos);
    const frac = srcPos - srcIndex;
    const s0 = input.readInt16LE(Math.min(srcIndex, inputSamples - 1) * 2);
    const s1 = input.readInt16LE(Math.min(srcIndex + 1, inputSamples - 1) * 2);
    output.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(s0 + frac * (s1 - s0)))), i * 2);
  }
  return output;
}

// ---- Health check ----

export async function checkKokoroHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${KOKORO_API_URL}/health`, {
      signal: AbortSignal.timeout(5000),
    });
    return res.ok;
  } catch {
    return false;
  }
}

/** Pre-warm: poll Kokoro-FastAPI health until it's ready (model may still be loading) */
export async function warmupKokoro(): Promise<void> {
  console.log(`[tts/kokoro] Waiting for Kokoro-FastAPI at ${KOKORO_API_URL}...`);
  for (let i = 0; i < 30; i++) {
    if (await checkKokoroHealth()) {
      console.log("[tts/kokoro] Kokoro-FastAPI is healthy and ready");
      return;
    }
    await new Promise((r) => setTimeout(r, 2000));
  }
  console.warn("[tts/kokoro] Kokoro-FastAPI health check failed after 60s -- TTS may not work");
}

// ---- TTS Provider ----

export class KokoroTTS implements TTSProvider {
  private voiceId: string;
  private speed: number;

  constructor(config: TTSConfig) {
    this.voiceId = config.voiceId || process.env.KOKORO_VOICE || DEFAULT_VOICE;
    this.speed = config.speed || 1.0;

    // Validate voice ID (warn but don't block -- FastAPI has more voices than our list)
    const validVoice = KOKORO_VOICES.find((v) => v.id === this.voiceId);
    if (!validVoice) {
      console.warn(`[tts/kokoro] Voice "${this.voiceId}" not in preset list, using anyway (FastAPI may support it)`);
    }
  }

  /**
   * Synthesize text to PCM audio via Kokoro-FastAPI HTTP endpoint.
   * Each call is independent -- no shared queue, true concurrency.
   */
  async synthesize(
    text: string,
    onChunk?: (chunk: Buffer) => void,
    voiceOverride?: string
  ): Promise<Buffer> {
    if (!text.trim()) return Buffer.alloc(0);

    const voice = voiceOverride || this.voiceId;
    const response = await fetch(`${KOKORO_API_URL}/v1/audio/speech`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "kokoro",
        input: text,
        voice,
        speed: this.speed,
        response_format: "pcm",
      }),
      signal: AbortSignal.timeout(90_000),
    });

    if (!response.ok) {
      const errBody = await response.text().catch(() => "");
      throw new Error(`Kokoro-FastAPI error ${response.status}: ${errBody.substring(0, 200)}`);
    }

    const arrayBuffer = await response.arrayBuffer();
    const pcm24k = Buffer.from(arrayBuffer);
    if (pcm24k.length === 0) return Buffer.alloc(0);

    const pcm16k = _resampleTo16k(pcm24k, KOKORO_SAMPLE_RATE);
    console.log(`[tts/kokoro] Synthesized voice=${voice} chars=${text.length} pcm16k=${pcm16k.length}bytes duration=${(pcm16k.length / 2 / PIPELINE_SAMPLE_RATE).toFixed(2)}s`);
    onChunk?.(pcm16k);
    return pcm16k;
  }

  /**
   * Streaming synthesis via chunked HTTP response.
   * Falls back to full-buffer-then-chunk if streaming isn't available.
   */
  synthesizeStream(
    text: string,
    onChunk: (chunk: Buffer) => void,
    onDone: () => void,
    onError?: (err: Error) => void
  ): { cancel: () => void } {
    const controller = new AbortController();
    let cancelled = false;

    (async () => {
      try {
        const response = await fetch(`${KOKORO_API_URL}/v1/audio/speech`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: "kokoro",
            input: text,
            voice: this.voiceId,
            speed: this.speed,
            response_format: "pcm",
          }),
          signal: controller.signal,
        });

        if (!response.ok || !response.body) {
          throw new Error(`Kokoro-FastAPI error ${response.status}`);
        }

        // Stream the response body, resample, and emit in 100ms chunks
        const reader = (response.body as any).getReader() as ReadableStreamDefaultReader<Uint8Array>;
        let residual = Buffer.alloc(0);
        const CHUNK_SIZE_16K = 3200; // 100ms at 16kHz 16-bit mono
        // Minimum input bytes needed to produce one 16kHz output chunk
        const MIN_INPUT_BYTES = Math.ceil(CHUNK_SIZE_16K * (KOKORO_SAMPLE_RATE / PIPELINE_SAMPLE_RATE)) * 2;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          if (cancelled) { reader.cancel(); return; }

          residual = Buffer.concat([residual, Buffer.from(value)]);

          // Emit complete chunks as they accumulate
          while (residual.length >= MIN_INPUT_BYTES) {
            // Align to sample boundary (2 bytes per sample)
            const take = MIN_INPUT_BYTES - (MIN_INPUT_BYTES % 2);
            const inputChunk = residual.subarray(0, take);
            residual = residual.subarray(take);
            if (!cancelled) {
              onChunk(_resampleTo16k(inputChunk, KOKORO_SAMPLE_RATE));
            }
          }
        }

        // Flush remaining audio
        if (residual.length >= 2 && !cancelled) {
          // Align to sample boundary
          const aligned = residual.subarray(0, residual.length - (residual.length % 2));
          if (aligned.length > 0) {
            onChunk(_resampleTo16k(aligned, KOKORO_SAMPLE_RATE));
          }
        }

        if (!cancelled) onDone();
      } catch (err) {
        if (cancelled || controller.signal.aborted) return;
        const error = err instanceof Error ? err : new Error(String(err));
        console.error("[tts/kokoro] Stream error:", error.message);
        if (onError) {
          onError(error);
        } else {
          onDone(); // Fallback: signal completion so session doesn't hang
        }
      }
    })();

    return {
      cancel: () => {
        cancelled = true;
        controller.abort();
      },
    };
  }

  /** No-op: HTTP client is stateless */
  destroy(): void {}
}
