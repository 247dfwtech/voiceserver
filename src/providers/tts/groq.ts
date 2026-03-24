import type { TTSProvider, TTSConfig } from "./interface";

/**
 * Groq TTS provider — PlayAI voices via Groq LPU inference.
 *
 * Uses the OpenAI-compatible /v1/audio/speech endpoint.
 * Returns PCM 24kHz, resampled to 16kHz for the voice pipeline.
 */

const GROQ_TTS_API_URL = "https://api.groq.com/openai/v1/audio/speech";
const PIPELINE_SAMPLE_RATE = 16000;
// Groq returns WAV with a 44-byte header; PCM sample rate is 24kHz
const GROQ_SAMPLE_RATE = 24000;

export class GroqTTS implements TTSProvider {
  private voiceId: string;
  private speed: number;
  private model: string;

  constructor(config: TTSConfig) {
    this.voiceId = config.voiceId || "Fritz-PlayAI";
    this.speed = config.speed || 1.0;
    this.model = config.model || "playai-tts";
  }

  async synthesize(
    text: string,
    onChunk?: (chunk: Buffer) => void
  ): Promise<Buffer> {
    if (!text.trim()) return Buffer.alloc(0);

    const apiKey = process.env.GROQ_TTS_API_KEY;
    if (!apiKey) {
      throw new Error("GROQ_TTS_API_KEY not set");
    }

    const response = await fetch(GROQ_TTS_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: this.model,
        input: text,
        voice: this.voiceId,
        response_format: "pcm",
        speed: this.speed,
      }),
      signal: AbortSignal.timeout(15000),
    });

    if (!response.ok) {
      const errBody = await response.text().catch(() => "");
      throw new Error(
        `Groq TTS error ${response.status}: ${errBody.substring(0, 200)}`
      );
    }

    const arrayBuffer = await response.arrayBuffer();
    const pcmRaw = Buffer.from(arrayBuffer);
    if (pcmRaw.length === 0) return Buffer.alloc(0);

    const pcm16k = _resampleTo16k(pcmRaw, GROQ_SAMPLE_RATE);
    console.log(
      `[tts/groq] Synthesized voice=${this.voiceId} chars=${text.length} pcm16k=${pcm16k.length}bytes duration=${(pcm16k.length / 2 / PIPELINE_SAMPLE_RATE).toFixed(2)}s`
    );
    onChunk?.(pcm16k);
    return pcm16k;
  }

  synthesizeStream(
    text: string,
    onChunk: (chunk: Buffer) => void,
    onDone: () => void,
    onError?: (err: Error) => void
  ): { cancel: () => void } {
    const controller = new AbortController();

    (async () => {
      try {
        if (!text.trim()) {
          onDone();
          return;
        }

        const apiKey = process.env.GROQ_TTS_API_KEY;
        if (!apiKey) {
          throw new Error("GROQ_TTS_API_KEY not set");
        }

        const response = await fetch(GROQ_TTS_API_URL, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${apiKey}`,
          },
          body: JSON.stringify({
            model: this.model,
            input: text,
            voice: this.voiceId,
            response_format: "pcm",
            speed: this.speed,
          }),
          signal: controller.signal,
        });

        if (!response.ok || !response.body) {
          const errBody = await response.text().catch(() => "");
          throw new Error(
            `Groq TTS error ${response.status}: ${errBody.substring(0, 200)}`
          );
        }

        // Read the full response and deliver as resampled PCM chunks
        const arrayBuffer = await response.arrayBuffer();
        const pcmRaw = Buffer.from(arrayBuffer);

        if (pcmRaw.length === 0) {
          onDone();
          return;
        }

        const pcm16k = _resampleTo16k(pcmRaw, GROQ_SAMPLE_RATE);
        console.log(
          `[tts/groq] Synthesized voice=${this.voiceId} chars=${text.length} pcm16k=${pcm16k.length}bytes duration=${(pcm16k.length / 2 / PIPELINE_SAMPLE_RATE).toFixed(2)}s`
        );

        // Deliver in ~100ms chunks (3200 bytes at 16kHz 16-bit mono)
        const CHUNK_SIZE = 3200;
        for (let offset = 0; offset < pcm16k.length; offset += CHUNK_SIZE) {
          if (controller.signal.aborted) return;
          onChunk(pcm16k.subarray(offset, Math.min(offset + CHUNK_SIZE, pcm16k.length)));
        }

        onDone();
      } catch (err: any) {
        if (!controller.signal.aborted) {
          console.error(`[tts/groq] Error: ${err.message}`);
          onError?.(err);
        }
      }
    })();

    return {
      cancel: () => controller.abort(),
    };
  }
}

/** Resample PCM16 from sourceSampleRate to 16kHz via linear interpolation */
function _resampleTo16k(pcmBuf: Buffer, sourceSampleRate: number): Buffer {
  if (sourceSampleRate === PIPELINE_SAMPLE_RATE) return pcmBuf;

  const ratio = sourceSampleRate / PIPELINE_SAMPLE_RATE;
  const srcSamples = pcmBuf.length / 2;
  const dstSamples = Math.floor(srcSamples / ratio);
  const out = Buffer.alloc(dstSamples * 2);

  for (let i = 0; i < dstSamples; i++) {
    const srcPos = i * ratio;
    const idx = Math.floor(srcPos);
    const frac = srcPos - idx;

    const s0 = pcmBuf.readInt16LE(Math.min(idx, srcSamples - 1) * 2);
    const s1 = pcmBuf.readInt16LE(Math.min(idx + 1, srcSamples - 1) * 2);
    const sample = Math.round(s0 + frac * (s1 - s0));

    out.writeInt16LE(Math.max(-32768, Math.min(32767, sample)), i * 2);
  }

  return out;
}
