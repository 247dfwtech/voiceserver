import * as fs from "fs";
import * as path from "path";
import type { TTSProvider, TTSConfig } from "./interface";

/**
 * Qwen3-TTS provider -- voice cloning with 3-second reference audio.
 *
 * Calls the Qwen3-TTS FastAPI HTTP service (OpenAI-compatible /v1/audio/speech).
 * Supports both preset voices and cloned voices via voice library profiles.
 *
 * Key features:
 *   - 3-second voice cloning from reference audio
 *   - 10 languages supported
 *   - 0.6B model: ~1.5-2GB VRAM, 20-30 concurrent streams on RTX 4090
 *   - Apache 2.0 license
 */

const QWEN3_API_URL = process.env.QWEN3_API_URL || "http://localhost:8881";
const QWEN3_VOICES_DIR = process.env.QWEN3_VOICES_DIR || "/data/qwen3-voices";
const QWEN3_SAMPLE_RATE = 24000;
const PIPELINE_SAMPLE_RATE = 16000;

// Preset voices (Base model doesn't support presets — these are for reference only).
// Voice cloning is the primary use case for Qwen3-TTS; preset voices should use Kokoro.
export const QWEN3_PRESET_VOICES: { id: string; name: string; gender: string; language: string; description: string }[] = [];

export const QWEN3_LANGUAGES = [
  { code: "en", name: "English" },
  { code: "zh", name: "Chinese" },
  { code: "ja", name: "Japanese" },
  { code: "ko", name: "Korean" },
  { code: "fr", name: "French" },
  { code: "de", name: "German" },
  { code: "es", name: "Spanish" },
  { code: "it", name: "Italian" },
  { code: "pt", name: "Portuguese" },
  { code: "nl", name: "Dutch" },
];

// ---- Language code mapping (dashboard stores ISO codes, API needs full names) ----

const LANG_CODE_TO_NAME: Record<string, string> = {
  en: "english", zh: "chinese", ja: "japanese", ko: "korean",
  fr: "french", de: "german", es: "spanish", it: "italian",
  pt: "portuguese", nl: "dutch", ru: "russian",
};

function resolveLanguage(lang: string): string {
  if (!lang) return "auto";
  const lower = lang.toLowerCase();
  return LANG_CODE_TO_NAME[lower] || lower;
}

// ---- Resampler ----

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

export async function checkQwen3Health(): Promise<boolean> {
  try {
    const res = await fetch(`${QWEN3_API_URL}/health`, {
      signal: AbortSignal.timeout(5000),
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function warmupQwen3(): Promise<void> {
  console.log(`[tts/qwen3] Waiting for Qwen3-TTS at ${QWEN3_API_URL}...`);
  for (let i = 0; i < 45; i++) {
    if (await checkQwen3Health()) {
      console.log("[tts/qwen3] Qwen3-TTS is healthy and ready");
      return;
    }
    await new Promise((r) => setTimeout(r, 2000));
  }
  console.warn("[tts/qwen3] Qwen3-TTS health check failed after 90s -- voice cloning may not work");
}

// ---- Voice Manager ----

export interface Qwen3Voice {
  id: string;
  name: string;
  language: string;
  refText?: string;
  createdAt: string;
  type: "cloned";
  provider: "qwen3";
}

export class Qwen3VoiceManager {
  private voicesDir: string;
  private profilesDir: string;

  constructor() {
    this.voicesDir = QWEN3_VOICES_DIR;
    this.profilesDir = path.join(this.voicesDir, "profiles");
  }

  async initialize(): Promise<void> {
    try {
      await fs.promises.mkdir(this.profilesDir, { recursive: true });
    } catch (err) {
      console.warn(`[tts/qwen3] Could not create voices directory: ${err instanceof Error ? err.message : err}`);
    }
  }

  async createVoice(
    name: string,
    audioBuffer: Buffer,
    language: string,
    transcript?: string
  ): Promise<Qwen3Voice> {
    await this.initialize();

    // Generate a safe directory name from the voice name
    const safeName = name.toLowerCase().replace(/[^a-z0-9_-]/g, "_").substring(0, 50);
    const id = `qwen3_${safeName}_${Date.now().toString(36)}`;
    const profileDir = path.join(this.profilesDir, id);

    await fs.promises.mkdir(profileDir, { recursive: true });

    // Convert uploaded audio to proper WAV (browser may send M4A, WebM, OGG, etc.)
    const wavBuffer = await this.convertToWav(audioBuffer, profileDir);
    await fs.promises.writeFile(path.join(profileDir, "reference.wav"), wavBuffer);

    // Write metadata
    const meta = {
      name,
      language,
      ref_text: transcript || "",
      created_at: new Date().toISOString(),
    };
    await fs.promises.writeFile(
      path.join(profileDir, "meta.json"),
      JSON.stringify(meta, null, 2)
    );

    const voice: Qwen3Voice = {
      id,
      name,
      language,
      refText: transcript,
      createdAt: meta.created_at,
      type: "cloned",
      provider: "qwen3",
    };

    console.log(`[tts/qwen3] Created voice profile: ${id} (${name}, ${language})`);
    return voice;
  }

  async listVoices(): Promise<Qwen3Voice[]> {
    try {
      await this.initialize();
      const entries = await fs.promises.readdir(this.profilesDir, { withFileTypes: true });
      const voices: Qwen3Voice[] = [];

      for (const entry of entries) {
        if (!entry.isDirectory()) continue;
        try {
          const metaPath = path.join(this.profilesDir, entry.name, "meta.json");
          const metaRaw = await fs.promises.readFile(metaPath, "utf-8");
          const meta = JSON.parse(metaRaw);
          voices.push({
            id: entry.name,
            name: meta.name || entry.name,
            language: meta.language || "en",
            refText: meta.ref_text,
            createdAt: meta.created_at || "",
            type: "cloned",
            provider: "qwen3",
          });
        } catch {
          // Skip invalid profiles
        }
      }

      return voices;
    } catch {
      return [];
    }
  }

  async deleteVoice(id: string): Promise<boolean> {
    try {
      const profileDir = path.join(this.profilesDir, id);
      await fs.promises.rm(profileDir, { recursive: true, force: true });
      console.log(`[tts/qwen3] Deleted voice profile: ${id}`);
      return true;
    } catch (err) {
      console.error(`[tts/qwen3] Failed to delete voice ${id}:`, err instanceof Error ? err.message : err);
      return false;
    }
  }

  getReferencePath(voiceId: string): string {
    return path.join(this.profilesDir, voiceId, "reference.wav");
  }

  /** Get metadata for a cloned voice (ref_text, language) */
  async getVoiceMeta(voiceId: string): Promise<{ ref_text?: string; language?: string } | null> {
    try {
      const metaPath = path.join(this.profilesDir, voiceId, "meta.json");
      const raw = await fs.promises.readFile(metaPath, "utf-8");
      return JSON.parse(raw);
    } catch {
      return null;
    }
  }

  /** Check if a voiceId is a cloned voice (vs preset) */
  isClonedVoice(voiceId: string): boolean {
    return voiceId.startsWith("qwen3_");
  }

  /** Convert any audio format to 24kHz mono WAV using ffmpeg */
  private async convertToWav(input: Buffer, workDir: string): Promise<Buffer> {
    const { execFile } = await import("child_process");
    const { promisify } = await import("util");
    const execFileAsync = promisify(execFile);

    const inputPath = path.join(workDir, "input_raw");
    const outputPath = path.join(workDir, "reference.wav");

    await fs.promises.writeFile(inputPath, input);

    try {
      await execFileAsync("ffmpeg", [
        "-y", "-i", inputPath,
        "-ar", "24000", "-ac", "1", "-sample_fmt", "s16",
        "-f", "wav", outputPath,
      ], { timeout: 30000 });

      const wavData = await fs.promises.readFile(outputPath);
      // Clean up temp input file
      await fs.promises.unlink(inputPath).catch(() => {});
      return wavData;
    } catch (err) {
      await fs.promises.unlink(inputPath).catch(() => {});
      console.warn("[tts/qwen3] ffmpeg conversion failed, saving raw audio:", err instanceof Error ? err.message : err);
      return input;
    }
  }
}

export const qwen3VoiceManager = new Qwen3VoiceManager();

// ---- TTS Provider ----

export class Qwen3TTS implements TTSProvider {
  private voiceId: string;
  private speed: number;

  constructor(config: TTSConfig) {
    this.voiceId = config.voiceId || "Vivian";
    this.speed = config.speed || 1.0;
  }

  async synthesize(
    text: string,
    onChunk?: (chunk: Buffer) => void,
    voiceOverride?: string
  ): Promise<Buffer> {
    if (!text.trim()) return Buffer.alloc(0);

    const voiceId = voiceOverride || this.voiceId;
    const isCloned = qwen3VoiceManager.isClonedVoice(voiceId);

    let response: Response;

    if (isCloned) {
      // Cloned voice: use /v1/audio/voice-clone with reference audio + text
      const refPath = qwen3VoiceManager.getReferencePath(voiceId);
      const meta = await qwen3VoiceManager.getVoiceMeta(voiceId);

      const refAudio = await import("fs").then(f => f.promises.readFile(refPath));
      const refAudioB64 = refAudio.toString("base64");

      response = await fetch(`${QWEN3_API_URL}/v1/audio/voice-clone`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input: text,
          ref_audio: refAudioB64,
          ref_text: meta?.ref_text || "",
          language: resolveLanguage(meta?.language || "en"),
          response_format: "pcm",
          speed: this.speed,
        }),
        signal: AbortSignal.timeout(90_000),
      });
    } else {
      // Preset voice: use standard /v1/audio/speech
      response = await fetch(`${QWEN3_API_URL}/v1/audio/speech`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "qwen3-tts",
          input: text,
          voice: voiceId,
          speed: this.speed,
          response_format: "pcm",
        }),
        signal: AbortSignal.timeout(90_000),
      });
    }

    if (!response.ok) {
      const errBody = await response.text().catch(() => "");
      throw new Error(`Qwen3-TTS error ${response.status}: ${errBody.substring(0, 200)}`);
    }

    const arrayBuffer = await response.arrayBuffer();
    const pcmRaw = Buffer.from(arrayBuffer);
    if (pcmRaw.length === 0) return Buffer.alloc(0);

    const pcm16k = _resampleTo16k(pcmRaw, QWEN3_SAMPLE_RATE);
    console.log(`[tts/qwen3] Synthesized voice=${voiceId} cloned=${isCloned} chars=${text.length} pcm16k=${pcm16k.length}bytes`);
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
        const voiceId = this.voiceId;
        const isCloned = qwen3VoiceManager.isClonedVoice(voiceId);

        let response: Response;

        if (isCloned) {
          const refPath = qwen3VoiceManager.getReferencePath(voiceId);
          const meta = await qwen3VoiceManager.getVoiceMeta(voiceId);
          const refAudio = await import("fs").then(f => f.promises.readFile(refPath));

          response = await fetch(`${QWEN3_API_URL}/v1/audio/voice-clone`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              input: text,
              ref_audio: refAudio.toString("base64"),
              ref_text: meta?.ref_text || "",
              language: resolveLanguage(meta?.language || "en"),
              response_format: "pcm",
              speed: this.speed,
            }),
            signal: controller.signal,
          });
        } else {
          response = await fetch(`${QWEN3_API_URL}/v1/audio/speech`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              model: "qwen3-tts",
              input: text,
              voice: voiceId,
              speed: this.speed,
              response_format: "pcm",
            }),
            signal: controller.signal,
          });
        }

        if (!response.ok || !response.body) {
          const errBody = await response.text().catch(() => "");
          throw new Error(`Qwen3-TTS stream error ${response.status}: ${errBody.substring(0, 200)}`);
        }

        const reader = (response.body as any).getReader() as ReadableStreamDefaultReader<Uint8Array>;
        let residual = Buffer.alloc(0);
        const CHUNK_SIZE_16K = 3200;
        const MIN_INPUT_BYTES = Math.ceil(CHUNK_SIZE_16K * (QWEN3_SAMPLE_RATE / PIPELINE_SAMPLE_RATE)) * 2;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          if (cancelled) { reader.cancel(); return; }

          residual = Buffer.concat([residual, Buffer.from(value)]);

          while (residual.length >= MIN_INPUT_BYTES) {
            const take = MIN_INPUT_BYTES - (MIN_INPUT_BYTES % 2);
            const inputChunk = residual.subarray(0, take);
            residual = residual.subarray(take);
            if (!cancelled) {
              onChunk(_resampleTo16k(inputChunk, QWEN3_SAMPLE_RATE));
            }
          }
        }

        if (residual.length >= 2 && !cancelled) {
          const aligned = residual.subarray(0, residual.length - (residual.length % 2));
          if (aligned.length > 0) {
            onChunk(_resampleTo16k(aligned, QWEN3_SAMPLE_RATE));
          }
        }

        if (!cancelled) onDone();
      } catch (err) {
        if (cancelled || controller.signal.aborted) return;
        const error = err instanceof Error ? err : new Error(String(err));
        console.error("[tts/qwen3] Stream error:", error.message);
        if (onError) {
          onError(error);
        } else {
          onDone();
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

  destroy(): void {}
}
