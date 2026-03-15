import { spawn } from "child_process";
import * as fs from "fs";
import * as path from "path";
import type { TTSProvider, TTSConfig } from "./interface";

/**
 * KokoClone -- Voice cloning for Kokoro-82M TTS.
 *
 * Uses the KokoClone project (Apache 2.0) to clone any voice from a
 * 3-10 second audio sample, then synthesizes speech using that cloned voice.
 *
 * Architecture:
 *   1. Upload a reference audio sample (3-10s WAV/MP3)
 *   2. KokoClone extracts voice characteristics via Kanade voice conversion
 *   3. Kokoro-ONNX synthesizes base audio from text
 *   4. Kanade converts the synthesized audio to match the reference voice
 *
 * Dependencies: kokoro-onnx, kanade-tokenizer, torchaudio
 * All free, all Apache 2.0 or MIT licensed.
 */

const CLONED_VOICES_DIR = process.env.CLONED_VOICES_DIR || "/data/cloned-voices";
const KOKOCLONE_SAMPLE_RATE = 24000;

export interface ClonedVoice {
  id: string;
  name: string;
  referenceFile: string;
  createdAt: string;
}

/**
 * Manages cloned voice profiles -- upload reference audio, list, delete.
 * Used by the model-manager and IPC endpoints.
 */
export class VoiceCloneManager {
  private voicesDir: string;

  constructor() {
    this.voicesDir = CLONED_VOICES_DIR;
  }

  async initialize(): Promise<void> {
    await fs.promises.mkdir(this.voicesDir, { recursive: true });
    await fs.promises.mkdir(path.join(this.voicesDir, "references"), { recursive: true });
  }

  /** Check if KokoClone dependencies are installed */
  async isInstalled(): Promise<boolean> {
    try {
      const { stdout } = await new Promise<{ stdout: string; stderr: string }>((resolve, reject) => {
        const proc = spawn("python3", ["-c", "import kokoro_onnx; from kanade import KanadeModel; print('ok')"], {
          stdio: ["pipe", "pipe", "pipe"],
        });
        let stdout = "";
        let stderr = "";
        proc.stdout.on("data", (d: Buffer) => (stdout += d.toString()));
        proc.stderr.on("data", (d: Buffer) => (stderr += d.toString()));
        proc.on("close", (code) => (code === 0 ? resolve({ stdout, stderr }) : reject(new Error(stderr))));
        proc.on("error", reject);
      });
      return stdout.includes("ok");
    } catch {
      return false;
    }
  }

  /** Install KokoClone dependencies */
  async install(): Promise<{ success: boolean; error?: string }> {
    try {
      console.log("[kokoclone] Installing KokoClone dependencies...");

      const cmds = [
        "pip3 install --no-cache-dir kokoro-onnx torchaudio soundfile",
        "pip3 install --no-cache-dir git+https://github.com/frothywater/kanade-tokenizer",
      ];

      for (const cmd of cmds) {
        await new Promise<void>((resolve, reject) => {
          const proc = spawn("bash", ["-c", cmd], { stdio: ["pipe", "pipe", "pipe"] });
          let stderr = "";
          proc.stderr.on("data", (d: Buffer) => (stderr += d.toString()));
          proc.on("close", (code) => (code === 0 ? resolve() : reject(new Error(`Install failed: ${stderr}`))));
          proc.on("error", reject);
        });
      }

      await this.initialize();
      console.log("[kokoclone] KokoClone dependencies installed successfully");
      return { success: true };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error("[kokoclone] Install failed:", msg);
      return { success: false, error: msg };
    }
  }

  /** Uninstall KokoClone dependencies */
  async uninstall(): Promise<{ success: boolean; error?: string }> {
    try {
      await new Promise<void>((resolve, reject) => {
        const proc = spawn("bash", ["-c", "pip3 uninstall -y kokoro-onnx kanade-tokenizer"], {
          stdio: ["pipe", "pipe", "pipe"],
        });
        proc.on("close", () => resolve());
        proc.on("error", reject);
      });
      return { success: true };
    } catch (err) {
      return { success: false, error: err instanceof Error ? err.message : String(err) };
    }
  }

  /** Save a reference audio file and register the cloned voice */
  async createClonedVoice(
    name: string,
    audioBuffer: Buffer,
    filename: string
  ): Promise<{ success: boolean; voice?: ClonedVoice; error?: string }> {
    try {
      await this.initialize();

      // Generate a unique ID
      const id = `clone_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;

      // Save reference audio
      const ext = path.extname(filename) || ".wav";
      const refFilename = `${id}${ext}`;
      const refPath = path.join(this.voicesDir, "references", refFilename);
      await fs.promises.writeFile(refPath, audioBuffer);

      // Save voice metadata
      const voice: ClonedVoice = {
        id,
        name,
        referenceFile: refFilename,
        createdAt: new Date().toISOString(),
      };

      const manifestPath = path.join(this.voicesDir, "manifest.json");
      let manifest: ClonedVoice[] = [];
      try {
        const raw = await fs.promises.readFile(manifestPath, "utf-8");
        manifest = JSON.parse(raw);
      } catch {
        // No manifest yet
      }
      manifest.push(voice);
      await fs.promises.writeFile(manifestPath, JSON.stringify(manifest, null, 2));

      console.log(`[kokoclone] Created cloned voice: ${name} (${id})`);
      return { success: true, voice };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return { success: false, error: msg };
    }
  }

  /** List all cloned voices */
  async listClonedVoices(): Promise<ClonedVoice[]> {
    try {
      const manifestPath = path.join(this.voicesDir, "manifest.json");
      const raw = await fs.promises.readFile(manifestPath, "utf-8");
      return JSON.parse(raw) as ClonedVoice[];
    } catch {
      return [];
    }
  }

  /** Delete a cloned voice */
  async deleteClonedVoice(id: string): Promise<{ success: boolean; error?: string }> {
    try {
      const manifestPath = path.join(this.voicesDir, "manifest.json");
      let manifest: ClonedVoice[] = [];
      try {
        const raw = await fs.promises.readFile(manifestPath, "utf-8");
        manifest = JSON.parse(raw);
      } catch {
        return { success: false, error: "No cloned voices found" };
      }

      const voice = manifest.find((v) => v.id === id);
      if (!voice) {
        return { success: false, error: `Cloned voice "${id}" not found` };
      }

      // Delete reference file
      try {
        await fs.promises.unlink(path.join(this.voicesDir, "references", voice.referenceFile));
      } catch {
        // File may already be gone
      }

      // Update manifest
      manifest = manifest.filter((v) => v.id !== id);
      await fs.promises.writeFile(manifestPath, JSON.stringify(manifest, null, 2));

      console.log(`[kokoclone] Deleted cloned voice: ${voice.name} (${id})`);
      return { success: true };
    } catch (err) {
      return { success: false, error: err instanceof Error ? err.message : String(err) };
    }
  }

  /** Get the reference audio file path for a cloned voice */
  async getReferencePath(voiceId: string): Promise<string | null> {
    const voices = await this.listClonedVoices();
    const voice = voices.find((v) => v.id === voiceId);
    if (!voice) return null;
    return path.join(this.voicesDir, "references", voice.referenceFile);
  }
}

/**
 * KokoClone TTS provider -- synthesizes speech using a cloned voice.
 *
 * Uses Kokoro-ONNX for base synthesis, then Kanade for voice conversion
 * to match the reference speaker.
 */
export class KokoCloneTTS implements TTSProvider {
  private config: TTSConfig;
  private voiceId: string;
  private speed: number;
  private voiceCloneManager: VoiceCloneManager;

  constructor(config: TTSConfig) {
    this.config = config;
    this.voiceId = config.voiceId || "";
    this.speed = config.speed || 1.0;
    this.voiceCloneManager = new VoiceCloneManager();
  }

  async synthesize(
    text: string,
    onChunk?: (chunk: Buffer) => void
  ): Promise<Buffer> {
    const refPath = await this.voiceCloneManager.getReferencePath(this.voiceId);
    if (!refPath) {
      throw new Error(`Cloned voice "${this.voiceId}" not found. Upload a reference audio first.`);
    }

    return new Promise((resolve, reject) => {
      const pythonScript = `
import sys, json, struct
import numpy as np

try:
    from kokoro_onnx import Kokoro
    from kanade import KanadeModel, vocode
    import torchaudio
    import torch
except ImportError as e:
    print(f"KOKOCLONE_ERROR Missing dependency: {e}", file=sys.stderr, flush=True)
    sys.stdout.buffer.write(struct.pack("<I", 0))
    sys.stdout.buffer.flush()
    sys.exit(0)

try:
    # Load models
    kokoro = Kokoro.from_pretrained()
    kanade = KanadeModel.from_pretrained()
    vocoder = kanade.vocoder

    # Load reference audio
    ref_audio, ref_sr = torchaudio.load("${refPath.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}")
    if ref_sr != 24000:
        ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, 24000)
    ref_audio = ref_audio.mean(dim=0)  # mono

    # Synthesize base audio with Kokoro
    text = ${JSON.stringify(text)}
    speed = ${this.speed}
    samples, sr = kokoro.create(text, voice="af_heart", speed=speed, lang="en-us")

    # Convert to torch tensor
    source_audio = torch.from_numpy(samples).float()

    # Voice conversion with Kanade
    with torch.no_grad():
        converted_mel = kanade.voice_conversion(source_audio.unsqueeze(0), ref_audio.unsqueeze(0))
        converted_audio = vocode(vocoder, converted_mel)

    # Convert to int16 PCM
    audio_np = converted_audio.squeeze().cpu().numpy()
    audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
    pcm_bytes = audio_int16.tobytes()

    sys.stdout.buffer.write(struct.pack("<I", len(pcm_bytes)))
    sys.stdout.buffer.write(pcm_bytes)
    sys.stdout.buffer.flush()

    print(f"KOKOCLONE_SYNTH chars={len(text)} samples={len(audio_int16)} duration={len(audio_int16)/24000:.2f}s", file=sys.stderr, flush=True)

except Exception as e:
    print(f"KOKOCLONE_ERROR {e}", file=sys.stderr, flush=True)
    sys.stdout.buffer.write(struct.pack("<I", 0))
    sys.stdout.buffer.flush()
`;

      const proc = spawn("python3", ["-u", "-c", pythonScript], {
        stdio: ["pipe", "pipe", "pipe"],
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
      });

      let responseBuf = Buffer.alloc(0);
      let expectedLength: number | null = null;

      proc.stdout.on("data", (data: Buffer) => {
        responseBuf = Buffer.concat([responseBuf, data]);

        if (expectedLength === null && responseBuf.length >= 4) {
          expectedLength = responseBuf.readUInt32LE(0);
          responseBuf = responseBuf.subarray(4);
        }

        if (expectedLength !== null && responseBuf.length >= expectedLength) {
          const pcmData = responseBuf.subarray(0, expectedLength);
          if (pcmData.length === 0) {
            resolve(Buffer.alloc(0));
            return;
          }
          const resampled = this.resampleTo16k(pcmData, KOKOCLONE_SAMPLE_RATE);
          onChunk?.(resampled);
          resolve(resampled);
        }
      });

      proc.stderr.on("data", (data: Buffer) => {
        const msg = data.toString().trim();
        if (msg.includes("KOKOCLONE_ERROR")) {
          console.error(`[tts/kokoclone] ${msg}`);
        } else if (msg.includes("KOKOCLONE_SYNTH")) {
          console.log(`[tts/kokoclone] ${msg}`);
        }
      });

      proc.on("error", (err) => {
        reject(new Error(`KokoClone Python error: ${err.message}`));
      });

      proc.on("close", (code) => {
        if (expectedLength === null || (expectedLength !== null && responseBuf.length < expectedLength)) {
          reject(new Error(`KokoClone process exited with code ${code} before producing audio`));
        }
      });

      // Timeout
      setTimeout(() => {
        proc.kill("SIGTERM");
        reject(new Error("KokoClone TTS timed out after 60s"));
      }, 60_000);
    });
  }

  synthesizeStream(
    text: string,
    onChunk: (chunk: Buffer) => void,
    onDone: () => void
  ): { cancel: () => void } {
    let cancelled = false;

    this.synthesize(text)
      .then((audio) => {
        if (cancelled) return;
        if (audio.length > 0) {
          const chunkSize = 3200;
          for (let i = 0; i < audio.length; i += chunkSize) {
            if (cancelled) return;
            onChunk(audio.subarray(i, Math.min(i + chunkSize, audio.length)));
          }
        }
        if (!cancelled) onDone();
      })
      .catch((err) => {
        if (!cancelled) {
          console.error("[tts/kokoclone] Stream error:", err.message);
          onDone();
        }
      });

    return {
      cancel: () => {
        cancelled = true;
      },
    };
  }

  private resampleTo16k(input: Buffer, sourceRate: number): Buffer {
    if (sourceRate === 16000) return input;
    const ratio = sourceRate / 16000;
    const inputSamples = input.length / 2;
    const outputSamples = Math.floor(inputSamples / ratio);
    const output = Buffer.alloc(outputSamples * 2);
    for (let i = 0; i < outputSamples; i++) {
      const srcPos = i * ratio;
      const srcIndex = Math.floor(srcPos);
      const frac = srcPos - srcIndex;
      const s0 = input.readInt16LE(Math.min(srcIndex, inputSamples - 1) * 2);
      const s1 = input.readInt16LE(Math.min(srcIndex + 1, inputSamples - 1) * 2);
      const sample = Math.round(s0 + frac * (s1 - s0));
      output.writeInt16LE(Math.max(-32768, Math.min(32767, sample)), i * 2);
    }
    return output;
  }
}

// Singleton manager
export const voiceCloneManager = new VoiceCloneManager();
