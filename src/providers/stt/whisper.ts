import { EventEmitter } from "events";
import { spawn, ChildProcess } from "child_process";
import { tmpdir } from "os";
import { join } from "path";
import { writeFile, unlink } from "fs/promises";
import type { STTProvider, STTConfig } from "./interface";

/**
 * Whisper STT provider -- uses faster-whisper Python package via persistent subprocess.
 *
 * Strategy: Buffer audio until an utterance boundary (silence detection),
 * then write to a temp WAV file and send the path to the persistent Python process.
 *
 * Uses the same module-level singleton pattern as Kokoro TTS and Granite STT:
 * one Python process loads the model once into GPU memory and handles all
 * transcription requests via stdin/stdout protocol.
 *
 * Latency: ~0.3-1s per utterance on GPU, ~1-3s on CPU.
 */

// Module-level singleton so all call sessions share one loaded model process
let whisperProcess: ChildProcess | null = null;
let whisperReady = false;
let whisperReadyPromise: Promise<void> | null = null;
let pendingResolve: ((text: string) => void) | null = null;
let pendingReject: ((err: Error) => void) | null = null;
let stdoutBuffer = "";

function buildWhisperScript(modelSize: string, language: string): string {
  return `
import sys, json

# Redirect all stdout noise to stderr BEFORE importing heavy libs
_orig_stdout = sys.stdout.buffer
sys.stdout = sys.stderr

from faster_whisper import WhisperModel

model_size = "${modelSize}"
language = "${language}"

print(f"WHISPER_LOADING model={model_size}", flush=True)

# Use GPU if available (auto detects CUDA), fallback to CPU with int8
try:
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    print(f"WHISPER_READY device=cuda compute=float16", flush=True)
except Exception:
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print(f"WHISPER_READY device=cpu compute=int8", flush=True)

# Main loop: read JSON commands from stdin
# Format: {"path": "/tmp/audio.wav", "keywords": ["solar", "Freedom Forever"]}
# Or plain file path for backward compatibility
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    try:
        # Parse JSON or treat as plain file path
        initial_prompt = None
        if line.startswith("{"):
            cmd = json.loads(line)
            audio_path = cmd["path"]
            keywords = cmd.get("keywords", [])
            if keywords:
                initial_prompt = ", ".join(keywords)
        else:
            audio_path = line

        segments, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
            initial_prompt=initial_prompt,
        )
        transcription = " ".join(seg.text.strip() for seg in segments).strip()
        _orig_stdout.write((transcription + "\\n").encode())
        _orig_stdout.flush()
        print(f"WHISPER_OK lang={info.language} prob={info.language_probability:.2f} chars={len(transcription)}", flush=True)

    except Exception as e:
        _orig_stdout.write(f"ERROR: {e}\\n".encode())
        _orig_stdout.flush()
        print(f"WHISPER_ERR {e}", flush=True)
`.trim();
}

function ensureWhisperProcess(modelSize: string, language: string): Promise<void> {
  if (whisperReady && whisperProcess && !whisperProcess.killed) return Promise.resolve();
  if (whisperReadyPromise) return whisperReadyPromise;

  whisperReadyPromise = new Promise<void>((resolve, reject) => {
    console.log(`[stt/whisper] Starting Whisper persistent Python process (model=${modelSize})...`);

    const script = buildWhisperScript(modelSize, language);
    whisperProcess = spawn("python3", ["-u", "-c", script], {
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });

    let stderrBuf = "";
    let resolved = false;

    // Parse stdout lines — each line is a transcription response or ERROR:
    whisperProcess.stdout!.on("data", (data: Buffer) => {
      stdoutBuffer += data.toString();
      const newline = stdoutBuffer.indexOf("\n");
      if (newline !== -1) {
        const line = stdoutBuffer.slice(0, newline).trim();
        stdoutBuffer = stdoutBuffer.slice(newline + 1);

        if (pendingResolve) {
          const cb = pendingResolve;
          const cbErr = pendingReject;
          pendingResolve = null;
          pendingReject = null;

          if (line.startsWith("ERROR:")) {
            cbErr?.(new Error(line.slice(6).trim()));
          } else {
            cb(line);
          }
        }
      }
    });

    whisperProcess.stderr!.on("data", (data: Buffer) => {
      const text = data.toString();
      stderrBuf += text;

      for (const line of text.split("\n").filter((l) => l.trim())) {
        if (line.includes("WHISPER_READY")) {
          if (!resolved) {
            resolved = true;
            whisperReady = true;
            console.log(`[stt/whisper] ${line.trim()}`);
            resolve();
          }
        } else if (line.includes("WHISPER_OK") || line.includes("WHISPER_LOADING")) {
          console.log(`[stt/whisper] ${line.trim()}`);
        } else if (line.includes("WHISPER_ERR")) {
          console.error(`[stt/whisper] ${line.trim()}`);
        }
      }
    });

    whisperProcess.on("error", (err) => {
      console.error("[stt/whisper] Python process error:", err.message);
      whisperReady = false;
      whisperReadyPromise = null;
      whisperProcess = null;
      pendingReject?.(err);
      pendingResolve = null;
      pendingReject = null;
      if (!resolved) reject(err);
    });

    whisperProcess.on("exit", (code) => {
      console.warn(`[stt/whisper] Python process exited with code ${code}`);
      whisperReady = false;
      whisperProcess = null;
      whisperReadyPromise = null;
      if (pendingReject) {
        pendingReject(new Error(`Whisper process exited (code ${code})`));
        pendingResolve = null;
        pendingReject = null;
      }
      if (!resolved) reject(new Error(`Whisper process exited (code ${code}): ${stderrBuf.slice(-500)}`));
    });

    // Allow up to 120s for first cold-start model download + load
    setTimeout(() => {
      if (!resolved) {
        resolved = true;
        whisperProcess?.kill("SIGTERM");
        whisperReady = false;
        whisperReadyPromise = null;
        reject(new Error(`Whisper model load timed out after 120s. stderr: ${stderrBuf.slice(-300)}`));
      }
    }, 120_000);
  });

  return whisperReadyPromise;
}

async function transcribeFile(audioPath: string, modelSize: string, language: string, keywords?: string[]): Promise<string> {
  await ensureWhisperProcess(modelSize, language);

  return new Promise<string>((resolve, reject) => {
    if (!whisperProcess || !whisperProcess.stdin) {
      reject(new Error("Whisper process not available"));
      return;
    }

    if (pendingResolve) {
      reject(new Error("Whisper is already processing a request"));
      return;
    }

    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      pendingResolve = null;
      pendingReject = null;
      reject(new Error("Whisper transcription timed out after 15s"));
    }, 15_000);

    pendingResolve = (text) => {
      if (!timedOut) {
        clearTimeout(timer);
        resolve(text);
      }
    };
    pendingReject = (err) => {
      if (!timedOut) {
        clearTimeout(timer);
        reject(err);
      }
    };

    // Send JSON with path + keywords for initial_prompt biasing
    const cmd = JSON.stringify({ path: audioPath, keywords: keywords || [] });
    whisperProcess.stdin.write(cmd + "\n");
  });
}

export class WhisperSTT extends EventEmitter implements STTProvider {
  private config: STTConfig;
  private audioBuffer: Buffer[] = [];
  private isProcessing = false;
  private silenceFrames = 0;
  private speechDetected = false;
  private closed = false;

  private modelSize: string;
  private language: string;
  private keywords: string[];
  private speechThreshold = 500;
  private silenceThresholdFrames = 50; // 50 * 20ms = 1000ms of silence (phone conversations need longer pauses)

  constructor(config: STTConfig) {
    super();
    this.config = config;
    this.modelSize = config.model || process.env.WHISPER_MODEL || "small.en";
    this.language = config.language || "en";
    this.keywords = config.keywords || [];
  }

  async start(): Promise<void> {
    // Kick off model loading in the background so first utterance isn't slow
    ensureWhisperProcess(this.modelSize, this.language).catch((err) => {
      console.warn(`[stt/whisper] Background model load failed: ${err.message}`);
    });
  }

  send(audio: Buffer): void {
    if (this.closed) return;

    this.audioBuffer.push(audio);

    // Simple VAD: detect speech/silence transitions
    const rms = this.calculateRMS(audio);

    if (rms > this.speechThreshold) {
      if (!this.speechDetected) {
        this.speechDetected = true;
        this.emit("speech_started");
      }
      this.silenceFrames = 0;
    } else if (this.speechDetected) {
      this.silenceFrames++;

      // End of utterance detected
      if (this.silenceFrames >= this.silenceThresholdFrames) {
        this.processBufferedAudio();
        this.silenceFrames = 0;
        this.speechDetected = false;
      }
    }
  }

  async finish(): Promise<void> {
    if (this.audioBuffer.length > 0 && this.speechDetected) {
      await this.processBufferedAudio();
    }
  }

  close(): void {
    this.closed = true;
    this.audioBuffer = [];
    this.removeAllListeners();
  }

  private async processBufferedAudio(): Promise<void> {
    if (this.isProcessing || this.audioBuffer.length === 0) return;
    this.isProcessing = true;

    const pcmData = Buffer.concat(this.audioBuffer);
    this.audioBuffer = [];

    try {
      const transcript = await this.transcribeWithWhisper(pcmData);
      if (transcript.trim()) {
        this.emit("transcript", {
          text: transcript.trim(),
          isFinal: true,
          confidence: 0.9,
        });
        this.emit("utterance_end");
      }
    } catch (err) {
      this.emit("error", err instanceof Error ? err : new Error(String(err)));
    } finally {
      this.isProcessing = false;
    }
  }

  private async transcribeWithWhisper(pcmData: Buffer): Promise<string> {
    const tmpPath = join(tmpdir(), `whisper-${Date.now()}.wav`);
    try {
      const wavBuffer = this.pcmToWav(pcmData, 16000, 16, 1);
      await writeFile(tmpPath, wavBuffer);
      return await transcribeFile(tmpPath, this.modelSize, this.language, this.keywords);
    } finally {
      try { await unlink(tmpPath); } catch {}
    }
  }

  private pcmToWav(pcm: Buffer, sampleRate: number, bitsPerSample: number, channels: number): Buffer {
    const byteRate = sampleRate * channels * (bitsPerSample / 8);
    const blockAlign = channels * (bitsPerSample / 8);
    const dataSize = pcm.length;
    const headerSize = 44;

    const header = Buffer.alloc(headerSize);
    header.write("RIFF", 0);
    header.writeUInt32LE(dataSize + headerSize - 8, 4);
    header.write("WAVE", 8);
    header.write("fmt ", 12);
    header.writeUInt32LE(16, 16);
    header.writeUInt16LE(1, 20);
    header.writeUInt16LE(channels, 22);
    header.writeUInt32LE(sampleRate, 24);
    header.writeUInt32LE(byteRate, 28);
    header.writeUInt16LE(blockAlign, 32);
    header.writeUInt16LE(bitsPerSample, 34);
    header.write("data", 36);
    header.writeUInt32LE(dataSize, 40);

    return Buffer.concat([header, pcm]);
  }

  private calculateRMS(buffer: Buffer): number {
    let sum = 0;
    const samples = buffer.length / 2;
    for (let i = 0; i < buffer.length; i += 2) {
      const sample = buffer.readInt16LE(i);
      sum += sample * sample;
    }
    return Math.sqrt(sum / samples);
  }
}
