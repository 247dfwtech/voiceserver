import type { TTSProvider, TTSConfig } from "./interface";

/**
 * ElevenLabs TTS provider — cloud streaming via REST API.
 *
 * Uses /v1/text-to-speech/{voice_id}/stream endpoint with pcm_16000 output.
 * Returns PCM 16kHz 16-bit mono — matches the pipeline directly, no resampling needed.
 */

const ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1";
const PIPELINE_SAMPLE_RATE = 16000;

export class ElevenLabsTTS implements TTSProvider {
  private voiceId: string;
  private model: string;
  private stability: number;
  private similarityBoost: number;

  constructor(config: TTSConfig) {
    this.voiceId = config.voiceId || "21m00Tcm4TlvDq8ikWAM"; // Rachel (default)
    this.model = config.model || "eleven_turbo_v2_5";
    this.stability = config.stability ?? 0.5;
    this.similarityBoost = 0.75;
  }

  async synthesize(
    text: string,
    onChunk?: (chunk: Buffer) => void
  ): Promise<Buffer> {
    if (!text.trim()) return Buffer.alloc(0);

    const apiKey = process.env.ELEVENLABS_API_KEY;
    if (!apiKey) {
      throw new Error("ELEVENLABS_API_KEY not set");
    }

    const response = await fetch(
      `${ELEVENLABS_API_URL}/text-to-speech/${this.voiceId}/stream?output_format=pcm_16000`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "xi-api-key": apiKey,
        },
        body: JSON.stringify({
          text,
          model_id: this.model,
          voice_settings: {
            stability: this.stability,
            similarity_boost: this.similarityBoost,
          },
        }),
        signal: AbortSignal.timeout(15000),
      }
    );

    if (!response.ok) {
      const errBody = await response.text().catch(() => "");
      throw new Error(
        `ElevenLabs TTS error ${response.status}: ${errBody.substring(0, 200)}`
      );
    }

    const arrayBuffer = await response.arrayBuffer();
    const pcm16k = Buffer.from(arrayBuffer);
    if (pcm16k.length === 0) return Buffer.alloc(0);

    console.log(
      `[tts/elevenlabs] Synthesized voice=${this.voiceId} model=${this.model} chars=${text.length} pcm16k=${pcm16k.length}bytes duration=${(pcm16k.length / 2 / PIPELINE_SAMPLE_RATE).toFixed(2)}s`
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
    let cancelled = false;

    (async () => {
      try {
        if (!text.trim()) {
          onDone();
          return;
        }

        const apiKey = process.env.ELEVENLABS_API_KEY;
        if (!apiKey) {
          throw new Error("ELEVENLABS_API_KEY not set");
        }

        const response = await fetch(
          `${ELEVENLABS_API_URL}/text-to-speech/${this.voiceId}/stream?output_format=pcm_16000`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "xi-api-key": apiKey,
            },
            body: JSON.stringify({
              text,
              model_id: this.model,
              voice_settings: {
                stability: this.stability,
                similarity_boost: this.similarityBoost,
              },
            }),
            signal: controller.signal,
          }
        );

        if (!response.ok || !response.body) {
          const errBody = await response.text().catch(() => "");
          throw new Error(
            `ElevenLabs TTS error ${response.status}: ${errBody.substring(0, 200)}`
          );
        }

        // Stream the response — already pcm_16000, deliver in ~100ms chunks
        const reader = (response.body as any).getReader() as ReadableStreamDefaultReader<Uint8Array>;
        let residual = Buffer.alloc(0);
        const CHUNK_SIZE = 3200; // 100ms at 16kHz 16-bit mono

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          if (cancelled) {
            reader.cancel();
            return;
          }

          residual = Buffer.concat([residual, Buffer.from(value)]);

          while (residual.length >= CHUNK_SIZE) {
            onChunk(residual.subarray(0, CHUNK_SIZE));
            residual = residual.subarray(CHUNK_SIZE);
          }
        }

        // Flush remaining audio
        if (residual.length >= 2 && !cancelled) {
          const aligned = residual.subarray(0, residual.length - (residual.length % 2));
          if (aligned.length > 0) {
            onChunk(aligned);
          }
        }

        if (!cancelled) {
          console.log(
            `[tts/elevenlabs] Stream complete voice=${this.voiceId} model=${this.model} chars=${text.length}`
          );
          onDone();
        }
      } catch (err: any) {
        if (!cancelled && !controller.signal.aborted) {
          console.error(`[tts/elevenlabs] Error: ${err.message}`);
          onError?.(err);
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
}
