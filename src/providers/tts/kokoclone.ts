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
const KOKOCLONE_SAMPLE_RATE = 16000; // vocos vocoder outputs 16kHz PCM

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
        const proc = spawn("python3", ["-c", "import kokoro_onnx; from kanade_tokenizer import KanadeModel; print('ok')"], {
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
 *
 * Performance: Uses a persistent Python daemon per voice ID so models are
 * loaded once (~10s cold start) and reused across all synthesis calls (~1s each).
 */

// Daemon Python script — loads models once, processes requests in a loop via stdin/stdout
const KOKOCLONE_DAEMON_SCRIPT = `
import sys, struct, json, os
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"

try:
    from kokoro_onnx import Kokoro
    from kanade_tokenizer import KanadeModel, load_vocoder, vocode
    import torchaudio
    import torch
except ImportError as e:
    print(f"KOKOCLONE_ERROR Missing dependency: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

try:
    # Load models once (HuggingFace weights cached locally after first run)
    kokoro = Kokoro("/models/kokoro/kokoro-v1.0.onnx", "/models/kokoro/voices-v1.0.bin")
    kanade = KanadeModel.from_pretrained(repo_id="frothywater/kanade-tokenizer")
    vocoder = load_vocoder()  # vocos vocoder, outputs 16kHz PCM
    ref_cache = {}  # cache reference audio by path
    print("KOKOCLONE_READY", file=sys.stderr, flush=True)
except Exception as e:
    import traceback
    print(f"KOKOCLONE_ERROR model load failed: {e}", file=sys.stderr, flush=True)
    print(traceback.format_exc(), file=sys.stderr, flush=True)
    sys.exit(1)

# Request loop: read JSON from stdin, write PCM length+data to stdout
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        text = req["text"]
        ref_path = req["refPath"]
        speed = float(req.get("speed", 1.0))
        start = __import__("time").time()

        # Cache reference audio (avoid reloading each synthesis)
        if ref_path not in ref_cache:
            ref_audio, ref_sr = torchaudio.load(ref_path)
            if ref_sr != 16000:
                ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, 16000)
            ref_cache[ref_path] = ref_audio.mean(dim=0)
        ref_audio = ref_cache[ref_path]

        # Synthesize base audio with Kokoro (24kHz output)
        samples, sr = kokoro.create(text, voice="af_heart", speed=speed, lang="en-us")

        # Resample to 16kHz for Kanade
        source_audio = torch.from_numpy(samples).float()
        if sr != 16000:
            source_audio = torchaudio.functional.resample(source_audio, sr, 16000)

        # Voice conversion: source content + reference speaker style
        with torch.no_grad():
            mel_spec = kanade.voice_conversion(source_audio, ref_audio)
            converted_audio = vocode(vocoder, mel_spec.unsqueeze(0))

        # Convert to int16 PCM (vocos outputs 16kHz — no resampling needed in Node.js)
        audio_np = converted_audio.squeeze().cpu().numpy()
        audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
        pcm_bytes = audio_int16.tobytes()

        sys.stdout.buffer.write(struct.pack("<I", len(pcm_bytes)))
        sys.stdout.buffer.write(pcm_bytes)
        sys.stdout.buffer.flush()

        elapsed = __import__("time").time() - start
        print(f"KOKOCLONE_SYNTH chars={len(text)} samples={len(audio_int16)} dur={len(audio_int16)/16000:.2f}s elapsed={elapsed:.2f}s", file=sys.stderr, flush=True)

    except Exception as e:
        import traceback
        print(f"KOKOCLONE_ERROR {e}", file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        # Write zero-length response so Node.js doesn't hang
        sys.stdout.buffer.write(struct.pack("<I", 0))
        sys.stdout.buffer.flush()
`;

interface DaemonState {
  proc: ReturnType<typeof spawn>;
  ready: boolean;
  readyPromise: Promise<void>;
  pending: Array<{ resolve: (buf: Buffer) => void; reject: (err: Error) => void; responseBuf: Buffer; expectedLength: number | null }>;
}

// Global daemon map: voiceId → daemon state (models stay loaded across all calls)
const _kokocloneDaemons = new Map<string, DaemonState>();

function getOrStartDaemon(voiceId: string): DaemonState {
  if (_kokocloneDaemons.has(voiceId)) {
    return _kokocloneDaemons.get(voiceId)!;
  }

  const proc = spawn("python3", ["-u", "-c", KOKOCLONE_DAEMON_SCRIPT], {
    stdio: ["pipe", "pipe", "pipe"],
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });

  let readyResolve: () => void;
  let readyReject: (err: Error) => void;
  const readyPromise = new Promise<void>((res, rej) => {
    readyResolve = res;
    readyReject = rej;
  });

  const state: DaemonState = { proc, ready: false, readyPromise, pending: [] };
  _kokocloneDaemons.set(voiceId, state);

  let stdoutBuf = Buffer.alloc(0);

  proc.stdout.on("data", (data: Buffer) => {
    stdoutBuf = Buffer.concat([stdoutBuf, data]);

    // Process all complete responses in the buffer
    while (state.pending.length > 0) {
      const current = state.pending[0];

      if (current.expectedLength === null) {
        if (stdoutBuf.length < 4) break;
        current.expectedLength = stdoutBuf.readUInt32LE(0);
        stdoutBuf = stdoutBuf.subarray(4);
      }

      if (stdoutBuf.length < current.expectedLength) break;

      // Got a complete response
      const pcmData = stdoutBuf.subarray(0, current.expectedLength);
      stdoutBuf = stdoutBuf.subarray(current.expectedLength);
      state.pending.shift();
      current.resolve(Buffer.from(pcmData));
    }
  });

  proc.stderr.on("data", (data: Buffer) => {
    const msg = data.toString();
    for (const line of msg.split("\n")) {
      const l = line.trim();
      if (!l) continue;
      if (l.includes("KOKOCLONE_READY")) {
        state.ready = true;
        readyResolve!();
        console.log(`[tts/kokoclone] Daemon ready for voice ${voiceId}`);
      } else if (l.includes("KOKOCLONE_ERROR")) {
        console.error(`[tts/kokoclone] ${l}`);
        if (!state.ready) readyReject!(new Error(l));
      } else if (l.includes("KOKOCLONE_SYNTH")) {
        console.log(`[tts/kokoclone] ${l}`);
      }
    }
  });

  proc.on("error", (err) => {
    console.error(`[tts/kokoclone] Daemon process error: ${err.message}`);
    _kokocloneDaemons.delete(voiceId);
    if (!state.ready) readyReject!(err);
    for (const p of state.pending) p.reject(err);
    state.pending = [];
  });

  proc.on("close", (code) => {
    console.log(`[tts/kokoclone] Daemon for voice ${voiceId} exited (code ${code})`);
    _kokocloneDaemons.delete(voiceId);
    if (!state.ready) readyReject!(new Error(`Daemon exited before ready (code ${code})`));
    for (const p of state.pending) p.reject(new Error(`Daemon exited unexpectedly (code ${code})`));
    state.pending = [];
  });

  return state;
}

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

    const daemon = getOrStartDaemon(this.voiceId);

    // Wait for daemon to be ready (models loaded) — up to 90s for first cold start
    if (!daemon.ready) {
      const timeout = new Promise<never>((_, rej) =>
        setTimeout(() => rej(new Error("KokoClone daemon startup timed out (90s)")), 90_000)
      );
      await Promise.race([daemon.readyPromise, timeout]);
    }

    // Enqueue synthesis request
    return new Promise<Buffer>((resolve, reject) => {
      const entry = { resolve: (pcm: Buffer) => {
        if (pcm.length === 0) { resolve(Buffer.alloc(0)); return; }
        const resampled = this.resampleTo16k(pcm, KOKOCLONE_SAMPLE_RATE);
        onChunk?.(resampled);
        resolve(resampled);
      }, reject, responseBuf: Buffer.alloc(0), expectedLength: null as number | null };
      daemon.pending.push(entry);

      // Send request to daemon
      const req = JSON.stringify({ text, refPath, speed: this.speed }) + "\n";
      daemon.proc.stdin!.write(req);

      // Per-request timeout
      const timer = setTimeout(() => {
        const idx = daemon.pending.indexOf(entry);
        if (idx !== -1) daemon.pending.splice(idx, 1);
        reject(new Error("KokoClone synthesis timed out after 60s"));
      }, 60_000);

      // Clear timer on completion
      const origResolve = entry.resolve;
      entry.resolve = (pcm: Buffer) => { clearTimeout(timer); origResolve(pcm); };
      const origReject = entry.reject;
      entry.reject = (err: Error) => { clearTimeout(timer); origReject(err); };
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
