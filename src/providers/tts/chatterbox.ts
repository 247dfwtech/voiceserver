import { spawn, ChildProcess } from "child_process";
import * as fs from "fs";
import * as path from "path";
import type { TTSProvider, TTSConfig } from "./interface";

/**
 * Chatterbox Turbo TTS provider -- voice cloning via Resemble AI's Chatterbox.
 *
 * Uses a persistent Python subprocess running inside the "chatterbox" conda env.
 * The model loads once into GPU memory (~10-30s cold start), then accepts
 * text + reference audio path on stdin, returning raw PCM audio on stdout.
 *
 * Requires:
 *   - conda env "chatterbox" with python=3.11
 *   - Chatterbox repo installed (pip install -e .)
 *   - CUDA GPU
 *
 * Voice management is handled by ChatterboxVoiceManager (similar to VoiceCloneManager
 * from kokoclone.ts) -- stores reference audio clips and manifest.json.
 */

const CHATTERBOX_VOICES_DIR = process.env.CHATTERBOX_VOICES_DIR || "/data/chatterbox-voices";
const CONDA_PATH = process.env.CONDA_PATH || "/root/miniconda3/bin/conda";

// ---- Voice Management ----

export interface ChatterboxVoice {
  id: string;
  name: string;
  referenceFile: string;
  createdAt: string;
}

/**
 * Manages Chatterbox voice profiles -- upload reference audio, list, delete.
 */
export class ChatterboxVoiceManager {
  private voicesDir: string;

  constructor() {
    this.voicesDir = CHATTERBOX_VOICES_DIR;
  }

  async initialize(): Promise<void> {
    await fs.promises.mkdir(this.voicesDir, { recursive: true });
    await fs.promises.mkdir(path.join(this.voicesDir, "references"), { recursive: true });
  }

  /** Check if Chatterbox Turbo is available in the conda env */
  async isInstalled(): Promise<boolean> {
    try {
      const { stdout } = await new Promise<{ stdout: string; stderr: string }>((resolve, reject) => {
        const proc = spawn(CONDA_PATH, [
          "run", "-n", "chatterbox", "python3", "-c", "from chatterbox.tts_turbo import ChatterboxTurboTTS; print('ok')",
        ], { stdio: ["pipe", "pipe", "pipe"] });
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

  /** Save a reference audio file and register the voice */
  async createVoice(
    name: string,
    audioBuffer: Buffer,
    filename: string
  ): Promise<{ success: boolean; voice?: ChatterboxVoice; error?: string }> {
    try {
      await this.initialize();

      const id = `cb_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
      const ext = path.extname(filename) || ".wav";
      const refFilename = `${id}${ext}`;
      const refPath = path.join(this.voicesDir, "references", refFilename);
      await fs.promises.writeFile(refPath, audioBuffer);

      const voice: ChatterboxVoice = {
        id,
        name,
        referenceFile: refFilename,
        createdAt: new Date().toISOString(),
      };

      const manifestPath = path.join(this.voicesDir, "manifest.json");
      let manifest: ChatterboxVoice[] = [];
      try {
        const raw = await fs.promises.readFile(manifestPath, "utf-8");
        manifest = JSON.parse(raw);
      } catch {
        // No manifest yet
      }
      manifest.push(voice);
      await fs.promises.writeFile(manifestPath, JSON.stringify(manifest, null, 2));

      console.log(`[chatterbox] Created voice: ${name} (${id})`);
      return { success: true, voice };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return { success: false, error: msg };
    }
  }

  /** List all Chatterbox voices */
  async listVoices(): Promise<ChatterboxVoice[]> {
    try {
      const manifestPath = path.join(this.voicesDir, "manifest.json");
      const raw = await fs.promises.readFile(manifestPath, "utf-8");
      return JSON.parse(raw) as ChatterboxVoice[];
    } catch {
      return [];
    }
  }

  /** Delete a voice */
  async deleteVoice(id: string): Promise<{ success: boolean; error?: string }> {
    try {
      const manifestPath = path.join(this.voicesDir, "manifest.json");
      let manifest: ChatterboxVoice[] = [];
      try {
        const raw = await fs.promises.readFile(manifestPath, "utf-8");
        manifest = JSON.parse(raw);
      } catch {
        return { success: false, error: "No voices found" };
      }

      const voice = manifest.find((v) => v.id === id);
      if (!voice) {
        return { success: false, error: `Voice "${id}" not found` };
      }

      try {
        await fs.promises.unlink(path.join(this.voicesDir, "references", voice.referenceFile));
      } catch {
        // File may already be gone
      }

      manifest = manifest.filter((v) => v.id !== id);
      await fs.promises.writeFile(manifestPath, JSON.stringify(manifest, null, 2));

      console.log(`[chatterbox] Deleted voice: ${voice.name} (${id})`);
      return { success: true };
    } catch (err) {
      return { success: false, error: err instanceof Error ? err.message : String(err) };
    }
  }

  /** Get the reference audio file path for a voice */
  async getReferencePath(voiceId: string): Promise<string | null> {
    const voices = await this.listVoices();
    const voice = voices.find((v) => v.id === voiceId);
    if (!voice) return null;
    return path.join(this.voicesDir, "references", voice.referenceFile);
  }
}

// Singleton manager
export const chatterboxVoiceManager = new ChatterboxVoiceManager();

// ---- Persistent Python Subprocess (Kokoro-style singleton) ----

let _proc: ChildProcess | null = null;
let _ready = false;
let _readyPromise: Promise<void> | null = null;
let _sharedQueue: Promise<Buffer> = Promise.resolve(Buffer.alloc(0));
let _sampleRate = 24000; // Will be updated from CHATTERBOX_READY message

// Binary protocol state
let _responseBuf = Buffer.alloc(0);
let _expectedLength: number | null = null;
let _pendingResolve: ((buf: Buffer) => void) | null = null;
let _pendingReject: ((err: Error) => void) | null = null;

// Idle timeout to free GPU memory when not in use
let _idleTimer: ReturnType<typeof setTimeout> | null = null;
const IDLE_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes

function _resetIdleTimer() {
  if (_idleTimer) clearTimeout(_idleTimer);
  _idleTimer = setTimeout(() => {
    if (_proc && !_pendingResolve) {
      console.log("[tts/chatterbox] Idle timeout reached, shutting down persistent process to free GPU memory");
      _proc.kill("SIGTERM");
      _proc = null;
      _ready = false;
      _readyPromise = null;
    }
  }, IDLE_TIMEOUT_MS);
}

function _buildChatterboxScript(): string {
  return `
import sys, json, struct, os

# Redirect text output to stderr before importing (torch prints to stdout)
_binary_stdout = sys.stdout.buffer
sys.stdout = sys.stderr

# Set HF_TOKEN to bypass auth check (model is already cached locally)
if not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = "skip"

import torch
import torchaudio
import numpy as np
from chatterbox.tts_turbo import ChatterboxTurboTTS

model = ChatterboxTurboTTS.from_pretrained(device="cuda")
print(f"CHATTERBOX_READY sr={model.sr}", file=sys.stderr, flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        cmd = json.loads(line)
        text = cmd.get("text", "")
        ref_path = cmd.get("ref_path", "")

        if not text or not ref_path:
            _binary_stdout.write(struct.pack("<I", 0))
            _binary_stdout.flush()
            continue

        wav = model.generate(text, audio_prompt_path=ref_path)
        audio_np = wav.squeeze().cpu().numpy()
        audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
        pcm_bytes = audio_int16.tobytes()

        _binary_stdout.write(struct.pack("<I", len(pcm_bytes)))
        _binary_stdout.write(pcm_bytes)
        _binary_stdout.flush()

        print(f"CHATTERBOX_SYNTH chars={len(text)} sr={model.sr} samples={len(audio_int16)} duration={len(audio_int16)/model.sr:.2f}s", file=sys.stderr, flush=True)

    except Exception as e:
        print(f"CHATTERBOX_ERROR {e}", file=sys.stderr, flush=True)
        _binary_stdout.write(struct.pack("<I", 0))
        _binary_stdout.flush()
`.trim();
}

function _ensureChatterboxReady(): Promise<void> {
  if (_ready && _proc && !_proc.killed) return Promise.resolve();
  if (_readyPromise) return _readyPromise;

  _readyPromise = new Promise<void>((resolve, reject) => {
    console.log("[tts/chatterbox] Starting persistent Chatterbox Turbo Python process...");

    _proc = spawn(CONDA_PATH, [
      "run", "-n", "chatterbox", "python3", "-u", "-c", _buildChatterboxScript(),
    ], {
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });

    let stderrBuf = "";
    let resolved = false;

    // Handle binary stdout — length-prefixed PCM
    _proc.stdout!.on("data", (data: Buffer) => {
      _responseBuf = Buffer.concat([_responseBuf, data]);

      while (true) {
        if (_expectedLength === null && _responseBuf.length >= 4) {
          _expectedLength = _responseBuf.readUInt32LE(0);
          _responseBuf = _responseBuf.subarray(4);
        }
        if (_expectedLength !== null && _responseBuf.length >= _expectedLength) {
          const pcm = _responseBuf.subarray(0, _expectedLength);
          _responseBuf = _responseBuf.subarray(_expectedLength);
          _expectedLength = null;

          const cb = _pendingResolve;
          _pendingResolve = null;
          _pendingReject = null;
          cb?.(Buffer.from(pcm));
        } else {
          break;
        }
      }
    });

    _proc.stderr!.on("data", (data: Buffer) => {
      const text = data.toString();
      stderrBuf += text;
      for (const line of text.split("\n").filter((l) => l.trim())) {
        if (line.includes("CHATTERBOX_READY")) {
          // Parse sample rate: "CHATTERBOX_READY sr=24000"
          const srMatch = line.match(/sr=(\d+)/);
          if (srMatch) _sampleRate = parseInt(srMatch[1], 10);
          if (!resolved) {
            resolved = true;
            _ready = true;
            console.log(`[tts/chatterbox] Chatterbox Turbo loaded (sr=${_sampleRate}, shared process)`);
            resolve();
          }
        } else if (line.includes("CHATTERBOX_SYNTH")) {
          console.log(`[tts/chatterbox] ${line.trim()}`);
        } else if (line.includes("CHATTERBOX_ERROR")) {
          console.error(`[tts/chatterbox] ${line.trim()}`);
        }
      }
    });

    _proc.on("error", (err) => {
      console.error("[tts/chatterbox] Python process error:", err.message);
      _ready = false; _readyPromise = null; _proc = null;
      _pendingReject?.(err);
      _pendingResolve = null; _pendingReject = null;
      if (!resolved) reject(err);
    });

    _proc.on("exit", (code) => {
      console.warn(`[tts/chatterbox] Python process exited (code ${code})`);
      _ready = false; _proc = null; _readyPromise = null;
      if (_pendingReject) {
        _pendingReject(new Error(`Chatterbox process exited (code ${code})`));
        _pendingResolve = null; _pendingReject = null;
      }
      if (!resolved) reject(new Error(`Chatterbox process exited (code ${code}): ${stderrBuf.slice(-300)}`));
    });

    // Model load timeout (120s -- GPU model loading can be slow)
    setTimeout(() => {
      if (!resolved) {
        resolved = true;
        _proc?.kill("SIGTERM");
        _ready = false; _readyPromise = null;
        reject(new Error(`Chatterbox model load timed out after 120s. stderr: ${stderrBuf.slice(-300)}`));
      }
    }, 120_000);
  });

  return _readyPromise;
}

async function _doSynthesize(
  text: string,
  refPath: string,
  onChunk?: (chunk: Buffer) => void
): Promise<Buffer> {
  await _ensureChatterboxReady();

  return new Promise((resolve, reject) => {
    if (!_proc || !_proc.stdin || !_proc.stdout) {
      reject(new Error("Chatterbox Python process not available"));
      return;
    }
    if (_pendingResolve) {
      reject(new Error("Chatterbox is already processing a request (queue bug)"));
      return;
    }

    _resetIdleTimer();

    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      _pendingResolve = null; _pendingReject = null;
      reject(new Error(`Chatterbox TTS timed out after 90s for: "${text.substring(0, 50)}"`));
    }, 90_000);

    _pendingResolve = (pcmData) => {
      if (timedOut) return;
      clearTimeout(timer);
      if (pcmData.length === 0) { resolve(Buffer.alloc(0)); return; }
      const resampled = _resampleTo16k(pcmData, _sampleRate);
      onChunk?.(resampled);
      resolve(resampled);
    };
    _pendingReject = (err) => {
      if (timedOut) return;
      clearTimeout(timer);
      reject(err);
    };

    _proc.stdin.write(JSON.stringify({ text, ref_path: refPath }) + "\n");
  });
}

function _resampleTo16k(input: Buffer, sourceRate: number): Buffer {
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
    output.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(s0 + frac * (s1 - s0)))), i * 2);
  }
  return output;
}

// ---- TTS Provider ----

export class ChatterboxTurboTTS implements TTSProvider {
  private config: TTSConfig;
  private voiceId: string;
  private voiceManager: ChatterboxVoiceManager;

  constructor(config: TTSConfig) {
    this.config = config;
    this.voiceId = config.voiceId || "";
    this.voiceManager = new ChatterboxVoiceManager();
  }

  async synthesize(
    text: string,
    onChunk?: (chunk: Buffer) => void
  ): Promise<Buffer> {
    const refPath = await this.voiceManager.getReferencePath(this.voiceId);
    if (!refPath) {
      throw new Error(`Chatterbox voice "${this.voiceId}" not found. Upload a reference audio first.`);
    }

    // Serialize through shared queue (one request at a time)
    const prev = _sharedQueue;
    const result = prev.catch(() => {}).then(() =>
      _doSynthesize(text, refPath, onChunk)
    );
    _sharedQueue = result.catch(() => Buffer.alloc(0));
    return result;
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
          const chunkSize = 3200; // 100ms at 16kHz 16-bit
          for (let i = 0; i < audio.length; i += chunkSize) {
            if (cancelled) return;
            onChunk(audio.subarray(i, Math.min(i + chunkSize, audio.length)));
          }
        }
        if (!cancelled) onDone();
      })
      .catch((err) => {
        if (!cancelled) {
          console.error("[tts/chatterbox] Stream error:", err.message);
          onDone();
        }
      });

    return { cancel: () => { cancelled = true; } };
  }

  destroy(): void {}
}
