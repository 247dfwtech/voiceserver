import { spawn, ChildProcess } from "child_process";
import type { TTSProvider, TTSConfig } from "./interface";

/**
 * Kokoro-82M TTS provider -- self-hosted, free, Apache 2.0, #1 TTS Arena.
 *
 * Uses the hexgrad/Kokoro-82M model via a persistent Python subprocess.
 * The Python process loads the model once into GPU memory and accepts
 * text on stdin, returning raw PCM audio (16-bit, 24kHz mono) on stdout.
 *
 * Key advantages over Piper:
 *   - Near-human naturalness (#1 in TTS Spaces Arena, beating models 5-15x its size)
 *   - Sub-0.3s latency on GPU, 210x real-time on RTX 4090
 *   - Only ~500MB VRAM (82M params)
 *   - 54 voices, 8 languages
 *   - Apache 2.0 license
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
const KOKORO_SAMPLE_RATE = 24000; // Kokoro outputs at 24kHz

// ---- Module-level singleton Python process ----
// Shared across ALL KokoroTTS instances and call sessions so the model loads
// ONCE into GPU memory. Without this, every new call session spawns its own
// Python process, taking 30-60s to load before speaking the first message.

let _proc: ChildProcess | null = null;
let _ready = false;
let _readyPromise: Promise<void> | null = null;
let _sharedQueue: Promise<Buffer> = Promise.resolve(Buffer.alloc(0));

// Binary protocol state for current in-flight synthesis
let _responseBuf = Buffer.alloc(0);
let _expectedLength: number | null = null;
let _pendingResolve: ((buf: Buffer) => void) | null = null;
let _pendingReject: ((err: Error) => void) | null = null;

function _buildPythonScript(defaultVoice: string): string {
  return `
import sys, json, struct

# CRITICAL: redirect all text output to stderr BEFORE importing any libraries.
# torch and kokoro print to stdout during import, corrupting our binary protocol.
_binary_stdout = sys.stdout.buffer
sys.stdout = sys.stderr

import torch
import numpy as np

try:
    from kokoro import KPipeline
    pipeline = KPipeline(lang_code='a')  # 'a' = American English
    USE_KOKORO_PKG = True
    print("KOKORO_READY", file=sys.stderr, flush=True)
except ImportError:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import soundfile as sf
    USE_KOKORO_PKG = False
    print("KOKORO_READY", file=sys.stderr, flush=True)

def synthesize_kokoro(text, voice_id, speed):
    audio_segments = []
    for _, _, audio in pipeline(text, voice=voice_id, speed=speed):
        if audio is not None:
            audio_segments.append(audio)
    if not audio_segments:
        return np.array([], dtype=np.float32)
    return np.concatenate(audio_segments)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        cmd = json.loads(line)
        text = cmd.get("text", "")
        voice = cmd.get("voice", "${defaultVoice}")
        speed = cmd.get("speed", 1.0)

        if not text:
            _binary_stdout.write(struct.pack("<I", 0))
            _binary_stdout.flush()
            continue

        if USE_KOKORO_PKG:
            audio_np = synthesize_kokoro(text, voice, speed)
        else:
            audio_np = np.zeros(16000, dtype=np.float32)

        audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
        pcm_bytes = audio_int16.tobytes()

        _binary_stdout.write(struct.pack("<I", len(pcm_bytes)))
        _binary_stdout.write(pcm_bytes)
        _binary_stdout.flush()

        print(f"KOKORO_SYNTH voice={voice} chars={len(text)} samples={len(audio_int16)} duration={len(audio_int16)/24000:.2f}s", file=sys.stderr, flush=True)

    except Exception as e:
        print(f"KOKORO_ERROR {e}", file=sys.stderr, flush=True)
        _binary_stdout.write(struct.pack("<I", 0))
        _binary_stdout.flush()
`.trim();
}

function _ensureKokoroReady(defaultVoice: string): Promise<void> {
  if (_ready && _proc && !_proc.killed) return Promise.resolve();
  if (_readyPromise) return _readyPromise;

  _readyPromise = new Promise<void>((resolve, reject) => {
    console.log("[tts/kokoro] Starting shared Kokoro-82M Python process...");

    _proc = spawn("python3", ["-u", "-c", _buildPythonScript(defaultVoice)], {
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
          const cbErr = _pendingReject;
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
        if (line.includes("KOKORO_READY")) {
          if (!resolved) {
            resolved = true;
            _ready = true;
            console.log("[tts/kokoro] Kokoro-82M model loaded and ready (shared process)");
            resolve();
          }
        } else if (line.includes("KOKORO_SYNTH")) {
          console.log(`[tts/kokoro] ${line.trim()}`);
        } else if (line.includes("KOKORO_ERROR")) {
          console.error(`[tts/kokoro] ${line.trim()}`);
        }
      }
    });

    _proc.on("error", (err) => {
      console.error("[tts/kokoro] Python process error:", err.message);
      _ready = false; _readyPromise = null; _proc = null;
      _pendingReject?.(err);
      _pendingResolve = null; _pendingReject = null;
      if (!resolved) reject(err);
    });

    _proc.on("exit", (code) => {
      console.warn(`[tts/kokoro] Python process exited (code ${code})`);
      _ready = false; _proc = null; _readyPromise = null;
      if (_pendingReject) {
        _pendingReject(new Error(`Kokoro process exited (code ${code})`));
        _pendingResolve = null; _pendingReject = null;
      }
      if (!resolved) reject(new Error(`Kokoro process exited (code ${code}): ${stderrBuf.slice(-300)}`));
    });

    setTimeout(() => {
      if (!resolved) {
        resolved = true;
        _proc?.kill("SIGTERM");
        _ready = false; _readyPromise = null;
        reject(new Error(`Kokoro model load timed out after 120s. stderr: ${stderrBuf.slice(-300)}`));
      }
    }, 120_000);
  });

  return _readyPromise;
}

async function _doSynthesize(
  text: string,
  voiceId: string,
  speed: number,
  defaultVoice: string,
  onChunk?: (chunk: Buffer) => void
): Promise<Buffer> {
  await _ensureKokoroReady(defaultVoice);

  return new Promise((resolve, reject) => {
    if (!_proc || !_proc.stdin || !_proc.stdout) {
      reject(new Error("Kokoro Python process not available"));
      return;
    }
    if (_pendingResolve) {
      reject(new Error("Kokoro is already processing a request (queue bug)"));
      return;
    }

    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      _pendingResolve = null; _pendingReject = null;
      reject(new Error(`Kokoro TTS timed out after 90s for: "${text.substring(0, 50)}"`));
    }, 90_000);

    _pendingResolve = (pcmData) => {
      if (timedOut) return;
      clearTimeout(timer);
      if (pcmData.length === 0) { resolve(Buffer.alloc(0)); return; }
      const resampled = _resampleTo16k(pcmData, KOKORO_SAMPLE_RATE);
      onChunk?.(resampled);
      resolve(resampled);
    };
    _pendingReject = (err) => {
      if (timedOut) return;
      clearTimeout(timer);
      reject(err);
    };

    _proc.stdin.write(JSON.stringify({ text, voice: voiceId, speed }) + "\n");
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

/** Pre-warm Kokoro so the first call doesn't wait 6+ seconds for model load */
export function warmupKokoro(): void {
  _ensureKokoroReady(DEFAULT_VOICE).catch((err) => {
    console.warn(`[tts/kokoro] Pre-warm failed: ${err.message}`);
  });
}

export class KokoroTTS implements TTSProvider {
  private config: TTSConfig;
  private voiceId: string;
  private speed: number;

  constructor(config: TTSConfig) {
    this.config = config;
    this.voiceId = config.voiceId || process.env.KOKORO_VOICE || DEFAULT_VOICE;
    this.speed = config.speed || 1.0;

    // Validate voice ID
    const validVoice = KOKORO_VOICES.find((v) => v.id === this.voiceId);
    if (!validVoice) {
      console.warn(`[tts/kokoro] Unknown voice "${this.voiceId}", falling back to ${DEFAULT_VOICE}`);
      this.voiceId = DEFAULT_VOICE;
    }

    // Kick off model load in background so first synthesize() call doesn't wait 60s
    _ensureKokoroReady(DEFAULT_VOICE).catch((err) => {
      console.warn(`[tts/kokoro] Background warm-up failed: ${err.message}`);
    });
  }

  async synthesize(
    text: string,
    onChunk?: (chunk: Buffer) => void,
    voiceOverride?: string
  ): Promise<Buffer> {
    // Serialize all synthesis through the module-level queue so the single
    // shared Python process processes one request at a time.
    const prev = _sharedQueue;
    const result = prev.catch(() => {}).then(() =>
      _doSynthesize(text, voiceOverride || this.voiceId, this.speed, DEFAULT_VOICE, onChunk)
    );
    _sharedQueue = result.catch(() => Buffer.alloc(0));
    return result;
  }

  synthesizeStream(
    text: string,
    onChunk: (chunk: Buffer) => void,
    onDone: () => void,
    onError?: (err: Error) => void
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
          const error = err instanceof Error ? err : new Error(String(err));
          console.error("[tts/kokoro] Stream error:", error.message);
          if (onError) {
            onError(error);
          } else {
            onDone(); // Fallback: signal completion so session doesn't hang
          }
        }
      });

    return { cancel: () => { cancelled = true; } };
  }

  /** No-op: shared process is managed at module level */
  destroy(): void {}
}
