import { EventEmitter } from "events";
import { spawn, type ChildProcess } from "child_process";
import { tmpdir } from "os";
import { join } from "path";
import { writeFile, unlink } from "fs/promises";
import type { STTProvider, STTConfig } from "./interface";

/**
 * IBM Granite 4.0 1B Speech STT provider.
 *
 * Uses the HuggingFace transformers pipeline for transcription.
 * Key advantages over Whisper:
 * - #1 on OpenASR leaderboard (better accuracy)
 * - Keyword biasing for names/acronyms (critical for sales calls)
 * - Smaller model (1B params, ~2GB VRAM vs Whisper large-v3's 3GB)
 * - Apache 2.0 license
 *
 * Strategy: Same batch approach as Whisper — buffer audio until
 * utterance boundary (silence detection), then transcribe.
 */
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
  private pythonScript: string;

  constructor(config: STTConfig) {
    super();
    this.config = config;
    // Only use config.model if it looks like a valid HuggingFace ID (contains '/').
    // Reject Deepgram/Vapi model names like "flux-general-en", "nova-2-general".
    const configModel = config.model && config.model.includes("/") ? config.model : null;
    this.modelId = configModel || "ibm-granite/granite-4.0-1b-speech";

    // Build keyword bias argument for the Python script
    const keywords = config.keywords?.length
      ? JSON.stringify(config.keywords)
      : "[]";

    // Python script that loads and runs the Granite model
    // The model is cached after first load, subsequent calls are fast
    this.pythonScript = `
import sys, json, torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

model_id = "${this.modelId}"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load model and processor (cached by transformers after first download)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    trust_remote_code=True,
).to(device)

# Read audio file path from stdin
audio_path = sys.stdin.readline().strip()
keywords = ${keywords}

import soundfile as sf
audio_data, sample_rate = sf.read(audio_path)

# Prepare inputs
inputs = processor(
    audio_data,
    sampling_rate=sample_rate,
    return_tensors="pt",
).to(device)

# Build generation kwargs
gen_kwargs = {"max_new_tokens": 440}

# Add keyword biasing if keywords provided
if keywords:
    gen_kwargs["decoder_input_ids"] = None
    # Granite supports keyword biasing through prompt
    keyword_str = ", ".join(keywords)
    gen_kwargs["prompt_ids"] = processor.get_prompt_ids(f"Keywords: {keyword_str}")

# Generate transcription
with torch.no_grad():
    predicted_ids = model.generate(**inputs, **gen_kwargs)

# Decode
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(transcription.strip())
`.trim();
  }

  async start(): Promise<void> {
    // Verify Python + transformers are available
    try {
      const proc = spawn("python3", ["-c", "import transformers; print('ok')"], { stdio: "pipe" });
      const result = await new Promise<string>((resolve, reject) => {
        let stdout = "";
        proc.stdout?.on("data", (d) => (stdout += d.toString()));
        proc.on("close", (code) => {
          if (code === 0 && stdout.includes("ok")) resolve(stdout);
          else reject(new Error("transformers not available"));
        });
        proc.on("error", reject);
      });
    } catch (err) {
      console.warn(`[stt/granite] ${err instanceof Error ? err.message : err}. Install: pip install transformers torch soundfile`);
    }
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
      const transcript = await this.transcribeWithGranite(pcmData);
      if (transcript.trim()) {
        this.emit("transcript", {
          text: transcript.trim(),
          isFinal: true,
          confidence: 0.95, // Granite is #1 on OpenASR
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

      const result = await new Promise<string>((resolve, reject) => {
        const proc: ChildProcess = spawn("python3", ["-c", this.pythonScript], {
          stdio: ["pipe", "pipe", "pipe"],
        });

        let stdout = "";
        let stderr = "";

        proc.stdout?.on("data", (data) => (stdout += data.toString()));
        proc.stderr?.on("data", (data) => (stderr += data.toString()));

        // Send the audio file path to stdin
        proc.stdin?.write(tmpPath + "\n");
        proc.stdin?.end();

        proc.on("close", (code) => {
          if (code === 0) {
            resolve(stdout.trim());
          } else {
            reject(new Error(`Granite STT failed (exit ${code}): ${stderr.slice(-500)}`));
          }
        });

        proc.on("error", reject);

        // Timeout after 15 seconds (first run downloads model)
        setTimeout(() => {
          proc.kill("SIGTERM");
          reject(new Error("Granite transcription timed out"));
        }, 15000);
      });

      return result;
    } finally {
      try { await unlink(tmpPath); } catch {}
    }
  }

  /** Convert raw PCM to WAV format */
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
