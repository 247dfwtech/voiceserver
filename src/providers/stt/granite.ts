import { EventEmitter } from "events";
import { spawn, ChildProcess } from "child_process";
import { tmpdir } from "os";
import { join } from "path";
import { writeFile, unlink } from "fs/promises";
import type { STTProvider, STTConfig } from "./interface";

/**
 * IBM Granite 4.0 1B Speech STT provider.
 *
 * Uses a PERSISTENT Python subprocess that loads the model once into GPU memory
 * and processes transcription requests via stdin/stdout without reloading.
 *
 * Key advantages over per-request subprocess:
 * - Model loads once (~30s cold start, then cached)
 * - Each transcription takes 0.5-2s instead of 15-30s
 * - No more 15s timeout failures
 *
 * Protocol:
 *   stdin:  <audio_file_path>\n
 *   stdout: <transcription_text>\n  (or "ERROR: <message>\n")
 */

// Module-level singleton so all call sessions share one loaded model process
let graniteProcess: ChildProcess | null = null;
let graniteReady = false;
let graniteReadyPromise: Promise<void> | null = null;
let pendingResolve: ((text: string) => void) | null = null;
let pendingReject: ((err: Error) => void) | null = null;
let stdoutBuffer = "";

const DEFAULT_MODEL = "ibm-granite/granite-4.0-1b-speech";

function getModelId(config: STTConfig): string {
  // Only use config.model if it looks like a valid HuggingFace ID (contains '/').
  // Reject Deepgram/Vapi model names like "flux-general-en", "nova-2-general".
  const configModel = config.model && config.model.includes("/") ? config.model : null;
  return configModel || DEFAULT_MODEL;
}

function buildPythonScript(modelId: string): string {
  return `
import sys

# Redirect all stdout noise to stderr before importing heavy libs
_orig_stdout = sys.stdout.buffer
sys.stdout = sys.stderr

import torch
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

model_id = "${modelId}"
use_cuda = torch.cuda.is_available()
device_str = "cuda" if use_cuda else "cpu"
torch_dtype = torch.float16 if use_cuda else torch.float32

print(f"GRANITE_LOADING model={model_id} device={device_str}", flush=True)

# Load processor and model directly.
# IMPORTANT: GraniteSpeechFeatureExtractor.__call__() does NOT accept a
# 'sampling_rate' keyword argument — hf_pipeline() always passes it, so we
# must bypass pipeline and call the model ourselves.
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch_dtype,
)
model = model.to(device_str)
model.eval()

print("GRANITE_READY", flush=True)

# Main loop: read audio file paths from stdin, write transcription to stdout
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    try:
        # Load WAV file (already 16kHz mono from our pcmToWav encoder)
        audio_array, sr = sf.read(line, dtype="float32")
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)  # stereo → mono

        # Call processor WITHOUT sampling_rate kwarg — Granite's feature
        # extractor has a fixed expected rate and rejects that argument.
        inputs = processor(audio_array, return_tensors="pt")

        # Move tensors to device; cast float tensors to model dtype
        inputs = {
            k: (v.to(device_str, dtype=torch_dtype) if v.dtype.is_floating_point else v.to(device_str))
            for k, v in inputs.items()
        }

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=440)

        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        _orig_stdout.write((transcription + "\\n").encode())
        _orig_stdout.flush()
        print(f"GRANITE_OK chars={len(transcription)}", flush=True)

    except Exception as e:
        _orig_stdout.write(f"ERROR: {e}\\n".encode())
        _orig_stdout.flush()
        print(f"GRANITE_ERR {e}", flush=True)
`.trim();
}

function ensureGraniteProcess(modelId: string): Promise<void> {
  if (graniteReady && graniteProcess && !graniteProcess.killed) return Promise.resolve();
  if (graniteReadyPromise) return graniteReadyPromise;

  graniteReadyPromise = new Promise<void>((resolve, reject) => {
    console.log("[stt/granite] Starting Granite persistent Python process...");

    const script = buildPythonScript(modelId);
    graniteProcess = spawn("python3", ["-u", "-c", script], {
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });

    let stderrBuf = "";
    let resolved = false;

    // Parse stdout lines — each line is a transcription response or ERROR:
    graniteProcess.stdout!.on("data", (data: Buffer) => {
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

    graniteProcess.stderr!.on("data", (data: Buffer) => {
      const text = data.toString();
      stderrBuf += text;

      for (const line of text.split("\n").filter((l) => l.trim())) {
        if (line.includes("GRANITE_READY")) {
          if (!resolved) {
            resolved = true;
            graniteReady = true;
            console.log("[stt/granite] Granite model loaded and ready");
            resolve();
          }
        } else if (line.includes("GRANITE_OK") || line.includes("GRANITE_LOADING")) {
          console.log(`[stt/granite] ${line.trim()}`);
        } else if (line.includes("GRANITE_ERR")) {
          console.error(`[stt/granite] ${line.trim()}`);
        }
      }
    });

    graniteProcess.on("error", (err) => {
      console.error("[stt/granite] Python process error:", err.message);
      graniteReady = false;
      graniteReadyPromise = null;
      graniteProcess = null;
      pendingReject?.(err);
      pendingResolve = null;
      pendingReject = null;
      if (!resolved) reject(err);
    });

    graniteProcess.on("exit", (code) => {
      console.warn(`[stt/granite] Python process exited with code ${code}`);
      graniteReady = false;
      graniteProcess = null;
      graniteReadyPromise = null;
      if (pendingReject) {
        pendingReject(new Error(`Granite process exited (code ${code})`));
        pendingResolve = null;
        pendingReject = null;
      }
      if (!resolved) reject(new Error(`Granite process exited (code ${code}): ${stderrBuf.slice(-500)}`));
    });

    // Allow up to 180s for first cold-start model download + load
    setTimeout(() => {
      if (!resolved) {
        resolved = true;
        graniteProcess?.kill("SIGTERM");
        graniteReady = false;
        graniteReadyPromise = null;
        reject(new Error(`Granite model load timed out after 180s. stderr: ${stderrBuf.slice(-300)}`));
      }
    }, 180_000);
  });

  return graniteReadyPromise;
}

async function transcribeFile(audioPath: string, modelId: string): Promise<string> {
  await ensureGraniteProcess(modelId);

  return new Promise<string>((resolve, reject) => {
    if (!graniteProcess || !graniteProcess.stdin) {
      reject(new Error("Granite process not available"));
      return;
    }

    if (pendingResolve) {
      reject(new Error("Granite is already processing a request"));
      return;
    }

    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      pendingResolve = null;
      pendingReject = null;
      reject(new Error("Granite transcription timed out after 30s"));
    }, 30_000);

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

    graniteProcess.stdin.write(audioPath + "\n");
  });
}

export class GraniteSTT extends EventEmitter implements STTProvider {
  private config: STTConfig;
  private audioBuffer: Buffer[] = [];
  private isProcessing = false;
  private silenceFrames = 0;
  private speechDetected = false;
  private closed = false;

  private modelId: string;
  private speechThreshold = 500;
  private silenceThresholdFrames = 25; // 25 * 20ms = 500ms of silence

  constructor(config: STTConfig) {
    super();
    this.config = config;
    this.modelId = getModelId(config);
  }

  async start(): Promise<void> {
    // Kick off model loading in the background so first utterance isn't slow
    ensureGraniteProcess(this.modelId).catch((err) => {
      console.warn(`[stt/granite] Background model load failed: ${err.message}`);
    });
  }

  send(audio: Buffer): void {
    if (this.closed) return;

    this.audioBuffer.push(audio);

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
      const transcript = await this.transcribeWithGranite(pcmData);
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

  private async transcribeWithGranite(pcmData: Buffer): Promise<string> {
    const tmpPath = join(tmpdir(), `granite-${Date.now()}.wav`);
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
