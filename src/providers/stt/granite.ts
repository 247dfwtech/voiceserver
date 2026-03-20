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
let stdoutBuffer = "";

// FIFO queue for serialized transcription requests (Granite processes one at a time)
interface QueueEntry {
  resolve: (text: string) => void;
  reject: (err: Error) => void;
  timer: ReturnType<typeof setTimeout>;
  audioPath: string;
}
const transcriptionQueue: QueueEntry[] = [];
let activeEntry: QueueEntry | null = null;

const DEFAULT_MODEL = "ibm-granite/granite-4.0-1b-speech";

function getModelId(config: STTConfig): string {
  // Only use config.model if it looks like a valid HuggingFace ID (contains '/').
  // Reject Deepgram/Vapi model names like "flux-general-en", "nova-2-general".
  const configModel = config.model && config.model.includes("/") ? config.model : null;
  return configModel || DEFAULT_MODEL;
}

function sanitizePythonString(s: string): string {
  return s.replace(/[^a-zA-Z0-9.\-_\/]/g, "");
}

function buildPythonScript(modelId: string): string {
  const safeModelId = sanitizePythonString(modelId);
  return `
import sys

# Redirect all stdout noise to stderr before importing heavy libs
_orig_stdout = sys.stdout.buffer
sys.stdout = sys.stderr

import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

model_id = "${safeModelId}"
use_cuda = torch.cuda.is_available()
device_str = "cuda" if use_cuda else "cpu"
# Granite docs specify bfloat16, not float16
torch_dtype = torch.bfloat16 if use_cuda else torch.float32

print(f"GRANITE_LOADING model={model_id} device={device_str}", flush=True)

# Granite 4.0 Speech is a MULTIMODAL CHAT model, not a standard ASR pipeline.
# The processor expects BOTH a text prompt (with <|audio|> token) AND audio.
# Passing audio alone routes to the text tokenizer → "Invalid text" error.
# Passing audio via hf_pipeline adds sampling_rate kwarg → also errors.
# Correct API (from HuggingFace model card):
#   model_inputs = processor(prompt_text, audio_tensor, device=device, return_tensors="pt")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
tokenizer = processor.tokenizer
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch_dtype,
)
model = model.to(device_str)
model.eval()

# Build the transcription prompt with chat template (done once, reused per request)
user_prompt = "<|audio|>can you transcribe the speech into a written format?"
chat = [{"role": "user", "content": user_prompt}]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

print("GRANITE_READY", flush=True)

# Main loop: read audio file paths from stdin, write transcription to stdout
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    try:
        # Load WAV as tensor (our pcmToWav writes 16kHz mono)
        wav, sr = torchaudio.load(line, normalize=True)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)  # stereo → mono

        # Process: pass BOTH text prompt AND audio tensor (multimodal API)
        model_inputs = processor(prompt, wav, device=device_str, return_tensors="pt").to(device_str)

        with torch.no_grad():
            model_outputs = model.generate(**model_inputs, max_new_tokens=200, do_sample=False, num_beams=1)

        # Extract only generated tokens (skip the input tokens)
        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
        transcription = tokenizer.batch_decode(new_tokens, add_special_tokens=False, skip_special_tokens=True)[0].strip()

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
      drainQueueWithError(err);
      if (!resolved) reject(err);
    });

    graniteProcess.on("exit", (code) => {
      console.warn(`[stt/granite] Python process exited with code ${code}`);
      graniteReady = false;
      graniteProcess = null;
      graniteReadyPromise = null;
      drainQueueWithError(new Error(`Granite process exited (code ${code})`));
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

/** Send the next queued request to the Python subprocess stdin */
function processNextInQueue(): void {
  if (activeEntry || transcriptionQueue.length === 0) return;
  if (!graniteProcess || !graniteProcess.stdin) return;

  activeEntry = transcriptionQueue.shift()!;
  graniteProcess.stdin.write(activeEntry.audioPath + "\n");
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
  await ensureGraniteProcess(modelId);

  return new Promise<string>((resolve, reject) => {
    if (!graniteProcess || !graniteProcess.stdin) {
      reject(new Error("Granite process not available"));
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
        reject(new Error("Granite transcription timed out after 15s"));
      }, 15_000),
    };

    transcriptionQueue.push(entry);

    if (!activeEntry) {
      processNextInQueue();
    }
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
  private speechThreshold = 200; // Lower for Twilio phone audio (mulaw→PCM16k has lower RMS)
  private silenceThresholdFrames = 50; // 50 * 20ms = 1000ms of silence

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

    const rms = this.calculateRMS(audio);

    if (rms > this.speechThreshold) {
      if (!this.speechDetected) {
        this.speechDetected = true;
        // Only start buffering when speech detected — discard pre-speech silence
        this.audioBuffer = [];
        this.emit("speech_started");
      }
      this.silenceFrames = 0;
      this.audioBuffer.push(audio);
    } else if (this.speechDetected) {
      // Buffer trailing silence (helps model with word boundaries)
      this.audioBuffer.push(audio);
      this.silenceFrames++;

      if (this.silenceFrames >= this.silenceThresholdFrames) {
        this.processBufferedAudio();
        this.silenceFrames = 0;
        this.speechDetected = false;
      }
    }
    // Pre-speech silence is discarded — not buffered
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
