import { EventEmitter } from "events";
import { spawn, type ChildProcess } from "child_process";
import { tmpdir } from "os";
import { join } from "path";
import { writeFile, unlink } from "fs/promises";
import type { STTProvider, STTConfig } from "./interface";

/**
 * Whisper STT provider -- uses faster-whisper CLI for transcription.
 *
 * Strategy: Buffer audio until an utterance boundary (silence detection),
 * then write to a temp WAV file and run faster-whisper on it.
 *
 * This is a batch approach (not true streaming like Deepgram), but it's free.
 * Latency is ~0.5-2s per utterance depending on GPU/CPU.
 */
export class WhisperSTT extends EventEmitter implements STTProvider {
  private config: STTConfig;
  private audioBuffer: Buffer[] = [];
  private isProcessing = false;
  private silenceFrames = 0;
  private speechDetected = false;
  private closed = false;

  // Whisper process config
  private whisperModel: string;
  private whisperBinary: string;
  private speechThreshold = 500;
  private silenceThresholdFrames = 25; // 25 * 20ms = 500ms of silence

  constructor(config: STTConfig) {
    super();
    this.config = config;
    this.whisperModel = config.model || process.env.WHISPER_MODEL || "base.en";
    this.whisperBinary = process.env.WHISPER_BINARY || "faster-whisper";
  }

  async start(): Promise<void> {
    // Verify faster-whisper is available
    try {
      const proc = spawn(this.whisperBinary, ["--help"], { stdio: "pipe" });
      await new Promise<void>((resolve, reject) => {
        proc.on("close", (code) => {
          if (code === 0) resolve();
          else reject(new Error(`faster-whisper not found (exit code ${code}). Run scripts/install-whisper.sh`));
        });
        proc.on("error", () => reject(new Error("faster-whisper binary not found. Run scripts/install-whisper.sh")));
      });
    } catch (err) {
      console.warn(`[stt/whisper] ${err instanceof Error ? err.message : err}. Whisper STT may not work.`);
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

    // Collect buffered audio
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
    // Write PCM to temp WAV file
    const tmpPath = join(tmpdir(), `whisper-${Date.now()}.wav`);

    try {
      const wavBuffer = this.pcmToWav(pcmData, 16000, 16, 1);
      await writeFile(tmpPath, wavBuffer);

      // Run faster-whisper
      const result = await new Promise<string>((resolve, reject) => {
        const args = [
          tmpPath,
          "--model", this.whisperModel,
          "--language", this.config.language || "en",
          "--output_format", "txt",
          "--output_dir", tmpdir(),
        ];

        const proc: ChildProcess = spawn(this.whisperBinary, args, {
          stdio: ["pipe", "pipe", "pipe"],
        });

        let stdout = "";
        let stderr = "";

        proc.stdout?.on("data", (data) => (stdout += data.toString()));
        proc.stderr?.on("data", (data) => (stderr += data.toString()));

        proc.on("close", (code) => {
          if (code === 0) {
            resolve(stdout.trim() || stderr.trim());
          } else {
            reject(new Error(`Whisper failed (exit ${code}): ${stderr}`));
          }
        });

        proc.on("error", reject);

        // Timeout after 10 seconds
        setTimeout(() => {
          proc.kill("SIGTERM");
          reject(new Error("Whisper transcription timed out"));
        }, 10000);
      });

      return result;
    } finally {
      // Clean up temp file
      try {
        await unlink(tmpPath);
      } catch {
        // ignore
      }
      try {
        await unlink(tmpPath.replace(".wav", ".txt"));
      } catch {
        // ignore
      }
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
    header.writeUInt32LE(16, 16); // fmt chunk size
    header.writeUInt16LE(1, 20); // PCM format
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
