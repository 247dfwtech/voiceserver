import * as fs from "fs";
import * as path from "path";
import type { TTSProvider, TTSConfig } from "./interface";

/**
 * Fish Speech S2 Pro TTS provider -- voice cloning with 10-30s reference audio.
 *
 * Calls the Fish Speech HTTP API server (self-hosted on GPU).
 * Supports both persistent cloned voices and zero-shot cloning.
 *
 * Key features:
 *   - Best English quality (0.99% WER, 81.88% EmergentTTS win rate)
 *   - Zero-shot voice cloning from 10-30s reference audio
 *   - SGLang continuous batching for 20+ concurrent requests
 *   - ~150-300ms TTFA on RTX 4090
 *   - Apache 2.0 license (self-hosted)
 */

const FISH_SPEECH_API_URL = process.env.FISH_SPEECH_API_URL || "http://localhost:8882";
const FISH_SPEECH_VOICES_DIR = process.env.FISH_SPEECH_VOICES_DIR || "/data/fish-speech-voices";
const FISH_SPEECH_SAMPLE_RATE = 44100;
const PIPELINE_SAMPLE_RATE = 16000;

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

export async function checkFishSpeechHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${FISH_SPEECH_API_URL}/v1/health`, {
      signal: AbortSignal.timeout(5000),
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function warmupFishSpeech(): Promise<void> {
  console.log(`[tts/fish-speech] Waiting for Fish Speech at ${FISH_SPEECH_API_URL}...`);
  for (let i = 0; i < 45; i++) {
    if (await checkFishSpeechHealth()) {
      console.log("[tts/fish-speech] Fish Speech is healthy and ready");
      return;
    }
    await new Promise((r) => setTimeout(r, 2000));
  }
  console.warn("[tts/fish-speech] Fish Speech health check failed after 90s -- voice cloning may not work");
}

// ---- Voice Manager ----

export interface FishSpeechVoice {
  id: string;
  name: string;
  refText?: string;
  createdAt: string;
  type: "cloned";
  provider: "fish-speech";
}

export class FishSpeechVoiceManager {
  private voicesDir: string;
  private profilesDir: string;

  constructor() {
    this.voicesDir = FISH_SPEECH_VOICES_DIR;
    this.profilesDir = path.join(this.voicesDir, "profiles");
  }

  async initialize(): Promise<void> {
    try {
      await fs.promises.mkdir(this.profilesDir, { recursive: true });
    } catch (err) {
      console.warn(`[tts/fish-speech] Could not create voices directory: ${err instanceof Error ? err.message : err}`);
    }
  }

  async createVoice(
    name: string,
    audioBuffer: Buffer,
    transcript?: string
  ): Promise<FishSpeechVoice> {
    await this.initialize();

    // Generate a safe directory name
    const safeName = name.toLowerCase().replace(/[^a-z0-9_-]/g, "_").substring(0, 50);
    const id = `fish_${safeName}_${Date.now().toString(36)}`;
    const profileDir = path.join(this.profilesDir, id);

    await fs.promises.mkdir(profileDir, { recursive: true });

    // Convert uploaded audio to WAV via ffmpeg (24kHz mono for optimal cloning)
    const wavBuffer = await this.convertToWav(audioBuffer, profileDir);
    await fs.promises.writeFile(path.join(profileDir, "reference.wav"), wavBuffer);

    // Write metadata
    const meta = {
      name,
      ref_text: transcript || "",
      created_at: new Date().toISOString(),
    };
    await fs.promises.writeFile(
      path.join(profileDir, "meta.json"),
      JSON.stringify(meta, null, 2)
    );

    // Also register with Fish Speech server via /v1/references/add
    try {
      const refAudioB64 = wavBuffer.toString("base64");
      const resp = await fetch(`${FISH_SPEECH_API_URL}/v1/references/add`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id,
          audio: refAudioB64,
          text: transcript || "",
        }),
        signal: AbortSignal.timeout(30_000),
      });
      if (!resp.ok) {
        const errBody = await resp.text().catch(() => "");
        console.warn(`[tts/fish-speech] Failed to register voice with server: ${resp.status} ${errBody.substring(0, 200)}`);
      } else {
        console.log(`[tts/fish-speech] Voice registered with server: ${id}`);
      }
    } catch (err) {
      console.warn(`[tts/fish-speech] Could not register voice with server (server may be offline): ${err instanceof Error ? err.message : err}`);
    }

    const voice: FishSpeechVoice = {
      id,
      name,
      refText: transcript,
      createdAt: meta.created_at,
      type: "cloned",
      provider: "fish-speech",
    };

    console.log(`[tts/fish-speech] Created voice profile: ${id} (${name})`);
    return voice;
  }

  async listVoices(): Promise<FishSpeechVoice[]> {
    try {
      await this.initialize();
      const entries = await fs.promises.readdir(this.profilesDir, { withFileTypes: true });
      const voices: FishSpeechVoice[] = [];

      for (const entry of entries) {
        if (!entry.isDirectory()) continue;
        try {
          const metaPath = path.join(this.profilesDir, entry.name, "meta.json");
          const metaRaw = await fs.promises.readFile(metaPath, "utf-8");
          const meta = JSON.parse(metaRaw);
          voices.push({
            id: entry.name,
            name: meta.name || entry.name,
            refText: meta.ref_text,
            createdAt: meta.created_at || "",
            type: "cloned",
            provider: "fish-speech",
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

      // Also remove from Fish Speech server
      try {
        await fetch(`${FISH_SPEECH_API_URL}/v1/references/delete?reference_id=${encodeURIComponent(id)}`, {
          method: "DELETE",
          signal: AbortSignal.timeout(10_000),
        });
      } catch {
        // Server may be offline
      }

      console.log(`[tts/fish-speech] Deleted voice profile: ${id}`);
      return true;
    } catch (err) {
      console.error(`[tts/fish-speech] Failed to delete voice ${id}:`, err instanceof Error ? err.message : err);
      return false;
    }
  }

  getReferencePath(voiceId: string): string {
    return path.join(this.profilesDir, voiceId, "reference.wav");
  }

  async getVoiceMeta(voiceId: string): Promise<{ ref_text?: string } | null> {
    try {
      const metaPath = path.join(this.profilesDir, voiceId, "meta.json");
      const raw = await fs.promises.readFile(metaPath, "utf-8");
      return JSON.parse(raw);
    } catch {
      return null;
    }
  }

  isClonedVoice(voiceId: string): boolean {
    return voiceId.startsWith("fish_");
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
        "-ar", "44100", "-ac", "1", "-sample_fmt", "s16",
        "-f", "wav", outputPath,
      ], { timeout: 30000 });

      const wavData = await fs.promises.readFile(outputPath);
      await fs.promises.unlink(inputPath).catch(() => {});
      return wavData;
    } catch (err) {
      await fs.promises.unlink(inputPath).catch(() => {});
      console.warn("[tts/fish-speech] ffmpeg conversion failed, saving raw audio:", err instanceof Error ? err.message : err);
      return input;
    }
  }
}

export const fishSpeechVoiceManager = new FishSpeechVoiceManager();

// ---- TTS Provider ----

export class FishSpeechTTS implements TTSProvider {
  private voiceId: string;
  private speed: number;

  constructor(config: TTSConfig) {
    this.voiceId = config.voiceId || "";
    this.speed = config.speed || 1.0;
  }

  async synthesize(
    text: string,
    onChunk?: (chunk: Buffer) => void,
    voiceOverride?: string
  ): Promise<Buffer> {
    if (!text.trim()) return Buffer.alloc(0);

    const voiceId = voiceOverride || this.voiceId;
    const isCloned = fishSpeechVoiceManager.isClonedVoice(voiceId);

    let body: Record<string, unknown>;

    if (isCloned) {
      // Use persistent reference_id (registered via /v1/references/add)
      body = {
        text,
        reference_id: voiceId,
        format: "pcm",
        sample_rate: FISH_SPEECH_SAMPLE_RATE,
        streaming: false,
        normalize: true,
        temperature: 0.7,
        top_p: 0.8,
        repetition_penalty: 1.1,
      };
    } else {
      // No voice specified — use default voice (or could add inline reference)
      body = {
        text,
        format: "pcm",
        sample_rate: FISH_SPEECH_SAMPLE_RATE,
        streaming: false,
        normalize: true,
        temperature: 0.7,
        top_p: 0.8,
        repetition_penalty: 1.1,
      };
    }

    const response = await fetch(`${FISH_SPEECH_API_URL}/v1/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(90_000),
    });

    if (!response.ok) {
      const errBody = await response.text().catch(() => "");
      throw new Error(`Fish Speech error ${response.status}: ${errBody.substring(0, 200)}`);
    }

    const arrayBuffer = await response.arrayBuffer();
    const pcmRaw = Buffer.from(arrayBuffer);
    if (pcmRaw.length === 0) return Buffer.alloc(0);

    const pcm16k = _resampleTo16k(pcmRaw, FISH_SPEECH_SAMPLE_RATE);
    console.log(`[tts/fish-speech] Synthesized voice=${voiceId} cloned=${isCloned} chars=${text.length} pcm16k=${pcm16k.length}bytes`);
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
        const isCloned = fishSpeechVoiceManager.isClonedVoice(voiceId);

        let body: Record<string, unknown>;

        if (isCloned) {
          body = {
            text,
            reference_id: voiceId,
            format: "pcm",
            sample_rate: FISH_SPEECH_SAMPLE_RATE,
            streaming: true,
            normalize: true,
            temperature: 0.7,
            top_p: 0.8,
            repetition_penalty: 1.1,
          };
        } else {
          body = {
            text,
            format: "pcm",
            sample_rate: FISH_SPEECH_SAMPLE_RATE,
            streaming: true,
            normalize: true,
            temperature: 0.7,
            top_p: 0.8,
            repetition_penalty: 1.1,
          };
        }

        const response = await fetch(`${FISH_SPEECH_API_URL}/v1/tts`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: controller.signal,
        });

        if (!response.ok || !response.body) {
          const errBody = await response.text().catch(() => "");
          throw new Error(`Fish Speech stream error ${response.status}: ${errBody.substring(0, 200)}`);
        }

        const reader = (response.body as any).getReader() as ReadableStreamDefaultReader<Uint8Array>;
        let residual = Buffer.alloc(0);
        const CHUNK_SIZE_16K = 3200; // 100ms at 16kHz 16-bit mono
        const MIN_INPUT_BYTES = Math.ceil(CHUNK_SIZE_16K * (FISH_SPEECH_SAMPLE_RATE / PIPELINE_SAMPLE_RATE)) * 2;

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
              onChunk(_resampleTo16k(inputChunk, FISH_SPEECH_SAMPLE_RATE));
            }
          }
        }

        if (residual.length >= 2 && !cancelled) {
          const aligned = residual.subarray(0, residual.length - (residual.length % 2));
          if (aligned.length > 0) {
            onChunk(_resampleTo16k(aligned, FISH_SPEECH_SAMPLE_RATE));
          }
        }

        if (!cancelled) onDone();
      } catch (err) {
        if (cancelled || controller.signal.aborted) return;
        const error = err instanceof Error ? err : new Error(String(err));
        console.error("[tts/fish-speech] Stream error:", error.message);
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
