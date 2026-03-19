import { EventEmitter } from "events";
import { spawn, ChildProcess } from "child_process";
import { tmpdir } from "os";
import { join } from "path";
import { writeFile, unlink } from "fs/promises";
import type { STTProvider, STTConfig } from "./interface";

/**
 * NVIDIA Parakeet TDT 0.6B v2 STT provider.
 *
 * Uses a PERSISTENT Python subprocess that loads the NeMo model once into GPU memory.
 * Parakeet TDT achieves 6.32% WER on LibriSpeech — best self-hosted accuracy for English.
 *
 * Key advantages:
 * - 6.32% WER (better than Whisper large-v3 at 2.7x speed)
 * - Token-and-Duration Transducer (TDT) architecture — streaming-ready
 * - Native 16kHz input (matches our PCM pipeline)
 * - ~0.2-0.5s per utterance on GPU
 * - Apache 2.0 license, free self-hosted
 *
 * Model: nvidia/parakeet-tdt-0.6b-v2 (via NVIDIA NeMo / HuggingFace)
 *
 * Protocol (same as Granite/Whisper):
 *   stdin:  <audio_file_path>\n
 *   stdout: <transcription_text>\n  (or "ERROR: <message>\n")
 */

// Module-level singleton so all call sessions share one loaded model process
let parakeetProcess: ChildProcess | null = null;
let parakeetReady = false;
let parakeetReadyPromise: Promise<void> | null = null;
let stdoutBuffer = "";

// FIFO queue for serialized transcription requests
interface QueueEntry {
  resolve: (text: string) => void;
  reject: (err: Error) => void;
  timer: ReturnType<typeof setTimeout>;
  audioPath: string;
}
const transcriptionQueue: QueueEntry[] = [];
let activeEntry: QueueEntry | null = null;

const DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v2";

function buildParakeetScript(modelId: string): string {
  return `
import sys, json

# Redirect all stdout noise to stderr BEFORE importing heavy libs
_orig_stdout = sys.stdout.buffer
sys.stdout = sys.stderr

import torch
import nemo.collections.asr as nemo_asr

model_id = "${modelId}"
use_cuda = torch.cuda.is_available()
device_str = "cuda" if use_cuda else "cpu"

print(f"PARAKEET_LOADING model={model_id} device={device_str}", flush=True)

# Load the pretrained Parakeet TDT model from NVIDIA NeMo / HuggingFace
model = nemo_asr.models.ASRModel.from_pretrained(model_id)
model = model.to(device_str)
model.eval()

print("PARAKEET_READY", flush=True)

# Main loop: read audio file paths from stdin, write transcription to stdout
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    try:
        # Parakeet TDT transcribes WAV files directly
        # Returns list of transcriptions (one per file)
        transcriptions = model.transcribe([line])

        # Handle both list-of-strings and Hypothesis objects
        if transcriptions and len(transcriptions) > 0:
            result = transcriptions[0]
            if hasattr(result, 'text'):
                transcription = result.text.strip()
            elif isinstance(result, str):
                transcription = result.strip()
            else:
                transcription = str(result).strip()
        else:
            transcription = ""

        _orig_stdout.write((transcription + "\\n").encode())
        _orig_stdout.flush()
        print(f"PARAKEET_OK chars={len(transcription)}", flush=True)

    except Exception as e:
        _orig_stdout.write(f"ERROR: {e}\\n".encode())
        _orig_stdout.flush()
        print(f"PARAKEET_ERR {e}", flush=True)
`.trim();
}

function ensureParakeetProcess(modelId: string): Promise<void> {
  if (parakeetReady && parakeetProcess && !parakeetProcess.killed) return Promise.resolve();
  if (parakeetReadyPromise) return parakeetReadyPromise;

  parakeetReadyPromise = new Promise<void>((resolve, reject) => {
    console.log(`[stt/parakeet] Starting Parakeet TDT persistent Python process (model=${modelId})...`);

    const script = buildParakeetScript(modelId);
    parakeetProcess = spawn("python3", ["-u", "-c", script], {
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });

    let stderrBuf = "";
    let resolved = false;

    // Parse stdout lines — each line is a transcription response or ERROR:
    parakeetProcess.stdout!.on("data", (data: Buffer) => {
      stdoutBuffer += data.toString();
      let newline: number;
      while ((newline = stdoutBuffer.indexOf("\n")) !== -1) {
        const line = stdoutBuffer.slice(0, newline).trim();
        stdoutBuffer = stdoutBuffer.slice(newline + 1);

        if (activeEntry) {
          const entry = activeEntry;
          activeEntry = null;
          clearTimeout(entry.timer);

          if (line.startsWith("ERROR:")) {
            entry.reject(new Error(line.slice(6).trim()));
          } else {
            entry.resolve(line);
          }

          // Process next queued request
          processNextInQueue();
        }
      }
    });

    parakeetProcess.stderr!.on("data", (data: Buffer) => {
      const text = data.toString();
      stderrBuf += text;

      for (const line of text.split("\n").filter((l) => l.trim())) {
        if (line.includes("PARAKEET_READY")) {
          if (!resolved) {
            resolved = true;
            parakeetReady = true;
            console.log("[stt/parakeet] Parakeet TDT model loaded and ready");
            resolve();
          }
        } else if (line.includes("PARAKEET_OK") || line.includes("PARAKEET_LOADING")) {
          console.log(`[stt/parakeet] ${line.trim()}`);
        } else if (line.includes("PARAKEET_ERR")) {
          console.error(`[stt/parakeet] ${line.trim()}`);
        }
      }
    });

    parakeetProcess.on("error", (err) => {
      console.error("[stt/parakeet] Python process error:", err.message);
      parakeetReady = false;
      parakeetReadyPromise = null;
      parakeetProcess = null;
      drainQueueWithError(err);
      if (!resolved) reject(err);
    });

    parakeetProcess.on("exit", (code) => {
      console.warn(`[stt/parakeet] Python process exited with code ${code}`);
      parakeetReady = false;
      parakeetProcess = null;
      parakeetReadyPromise = null;
      drainQueueWithError(new Error(`Parakeet process exited (code ${code})`));
      if (!resolved) reject(new Error(`Parakeet process exited (code ${code}): ${stderrBuf.slice(-500)}`));
    });

    // Allow up to 180s for first cold-start model download + load
    setTimeout(() => {
      if (!resolved) {
        resolved = true;
        parakeetProcess?.kill("SIGTERM");
        parakeetReady = false;
        parakeetReadyPromise = null;
        reject(new Error(`Parakeet model load timed out after 180s. stderr: ${stderrBuf.slice(-300)}`));
      }
    }, 180_000);
  });

  return parakeetReadyPromise;
}

/** Send the next queued request's command to the Python subprocess stdin */
function processNextInQueue(): void {
  if (activeEntry || transcriptionQueue.length === 0) return;
  if (!parakeetProcess || !parakeetProcess.stdin) return;

  activeEntry = transcriptionQueue.shift()!;
  parakeetProcess.stdin.write(activeEntry.audioPath + "\n");
}

/** Reject all queued + active entries (used on process crash/exit) */
function drainQueueWithError(err: Error): void {
  if (activeEntry) {
    clearTimeout(activeEntry.timer);
    activeEntry.reject(err);
    activeEntry = null;
  }
  while (transcriptionQueue.length > 0) {
    const entry = transcriptionQueue.shift()!;
    clearTimeout(entry.timer);
    entry.reject(err);
  }
}

async function transcribeFile(audioPath: string, modelId: string): Promise<string> {
  await ensureParakeetProcess(modelId);

  return new Promise<string>((resolve, reject) => {
    if (!parakeetProcess || !parakeetProcess.stdin) {
      reject(new Error("Parakeet process not available"));
      return;
    }

    const entry: QueueEntry = {
      resolve,
      reject,
      audioPath,
      timer: setTimeout(() => {
        const idx = transcriptionQueue.indexOf(entry);
        if (idx !== -1) {
          transcriptionQueue.splice(idx, 1);
        } else if (activeEntry === entry) {
          activeEntry = null;
          processNextInQueue();
        }
        reject(new Error("Parakeet transcription timed out after 15s"));
      }, 15_000),
    };

    transcriptionQueue.push(entry);

    if (!activeEntry) {
      processNextInQueue();
    }
  });
}

export class ParakeetSTT extends EventEmitter implements STTProvider {
  private config: STTConfig;
  private audioBuffer: Buffer[] = [];
  private isProcessing = false;
  private silenceFrames = 0;
  private speechDetected = false;
  private closed = false;

  private modelId: string;
  private speechThreshold: number;
  private silenceThresholdFrames: number;

  constructor(config: STTConfig) {
    super();
    this.config = config;
    // Accept model from config if it looks like a HuggingFace ID, otherwise use default
    this.modelId = config.model && config.model.includes("/") ? config.model : DEFAULT_MODEL;

    // Configurable VAD thresholds via provider-agnostic config
    const silenceMs = config.endOfTurnTimeoutMs ?? 1000;
    this.silenceThresholdFrames = Math.round(silenceMs / 20);

    this.speechThreshold = 500;
    if (config.confidenceThreshold && config.confidenceThreshold > 0.4) {
      this.speechThreshold = 500 + (config.confidenceThreshold - 0.4) * 1000;
    }
  }

  async start(): Promise<void> {
    // Kick off model loading in the background so first utterance isn't slow
    ensureParakeetProcess(this.modelId).catch((err) => {
      console.warn(`[stt/parakeet] Background model load failed: ${err.message}`);
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
      const transcript = await this.transcribeWithParakeet(pcmData);
      if (transcript.trim()) {
        this.emit("transcript", {
          text: transcript.trim(),
          isFinal: true,
          confidence: 0.95,
        });
        this.emit("utterance_end");
      }
    } catch (err) {
      this.emit("error", err instanceof Error ? err : new Error(String(err)));
    } finally {
      this.isProcessing = false;
    }
  }

  private async transcribeWithParakeet(pcmData: Buffer): Promise<string> {
    const tmpPath = join(tmpdir(), `parakeet-${Date.now()}.wav`);
    try {
      const wavBuffer = this.pcmToWav(pcmData, 16000, 16, 1);
      await writeFile(tmpPath, wavBuffer);
      return await transcribeFile(tmpPath, this.modelId);
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
