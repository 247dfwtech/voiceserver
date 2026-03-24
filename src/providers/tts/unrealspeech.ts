import type { TTSProvider, TTSConfig } from "./interface";

/**
 * UnrealSpeech V8 TTS provider — cloud-based, 30+ voices, multilingual.
 *
 * Uses the /stream endpoint for low-latency synthesis (~300ms).
 * Returns PCM mulaw at 8kHz for telephony, resampled to 16kHz PCM.
 */

const UNREALSPEECH_API_URL = "https://api.v8.unrealspeech.com";

export class UnrealSpeechTTS implements TTSProvider {
  private voiceId: string;
  private speed: number;

  constructor(config: TTSConfig) {
    this.voiceId = config.voiceId || "Sierra";
    this.speed = config.speed ?? 0.0;
  }

  async synthesize(
    text: string,
    onChunk?: (chunk: Buffer) => void
  ): Promise<Buffer> {
    const apiKey = process.env.UNREALSPEECH_API_KEY;
    if (!apiKey) {
      throw new Error("UNREALSPEECH_API_KEY not set");
    }

    const response = await fetch(`${UNREALSPEECH_API_URL}/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        Text: text,
        VoiceId: this.voiceId,
        Bitrate: "192k",
        Speed: this.speed,
        Pitch: 1.0,
        Codec: "pcm_s16le",
        Temperature: 0.25,
      }),
      signal: AbortSignal.timeout(15000),
    });

    if (!response.ok) {
      const errBody = await response.text();
      throw new Error(
        `UnrealSpeech error ${response.status}: ${errBody.substring(0, 200)}`
      );
    }

    const arrayBuffer = await response.arrayBuffer();
    const pcm22k = Buffer.from(arrayBuffer);

    // Resample from 22050Hz to 16000Hz (linear interpolation)
    const pcm16k = _resampleTo16k(pcm22k, 22050);

    if (onChunk) {
      onChunk(pcm16k);
    }

    return pcm16k;
  }

  synthesizeStream(
    text: string,
    onChunk: (chunk: Buffer) => void,
    onDone: () => void,
    onError?: (err: Error) => void
  ): { cancel: () => void } {
    const controller = new AbortController();

    this.synthesize(text, onChunk)
      .then(() => onDone())
      .catch((err) => {
        if (!controller.signal.aborted) {
          onError?.(err);
        }
      });

    return {
      cancel: () => controller.abort(),
    };
  }
}

/** Resample PCM16 from sourceSampleRate to 16kHz via linear interpolation */
function _resampleTo16k(
  pcmBuf: Buffer,
  sourceSampleRate: number
): Buffer {
  if (sourceSampleRate === 16000) return pcmBuf;

  const ratio = sourceSampleRate / 16000;
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

    out.writeInt16LE(
      Math.max(-32768, Math.min(32767, sample)),
      i * 2
    );
  }

  return out;
}
