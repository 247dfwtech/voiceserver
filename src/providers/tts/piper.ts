import { spawn } from "child_process";
import type { TTSProvider, TTSConfig } from "./interface";

/**
 * Piper TTS provider -- self-hosted, free, fast neural TTS.
 *
 * Piper runs as a CLI tool: echo "text" | piper --model voice.onnx --output_raw
 * Output is raw PCM (16-bit, 16kHz or 22kHz depending on model).
 *
 * We resample to 16kHz to match our pipeline expectation.
 */
export class PiperTTS implements TTSProvider {
  private config: TTSConfig;
  private piperBinary: string;
  private modelPath: string;

  constructor(config: TTSConfig) {
    this.config = config;
    this.piperBinary = process.env.PIPER_BINARY || "piper";
    // Voice ID maps to model file path, e.g., "en_US-lessac-medium"
    const voiceId = config.voiceId || process.env.PIPER_VOICE || "en_US-lessac-medium";
    this.modelPath = process.env.PIPER_MODELS_DIR
      ? `${process.env.PIPER_MODELS_DIR}/${voiceId}.onnx`
      : voiceId;
  }

  async synthesize(
    text: string,
    onChunk?: (chunk: Buffer) => void
  ): Promise<Buffer> {
    return new Promise((resolve, reject) => {
      const args = [
        "--model", this.modelPath,
        "--output_raw",
        "--length_scale", String(1.0 / (this.config.speed || 1.0)),
      ];

      const proc = spawn(this.piperBinary, args, {
        stdio: ["pipe", "pipe", "pipe"],
      });

      const chunks: Buffer[] = [];
      let stderr = "";

      proc.stdout.on("data", (data: Buffer) => {
        // Piper outputs raw PCM at the model's sample rate (typically 22050)
        // Resample to 16kHz for our pipeline
        const resampled = this.resampleTo16k(data, 22050);
        chunks.push(resampled);
        onChunk?.(resampled);
      });

      proc.stderr.on("data", (data: Buffer) => {
        stderr += data.toString();
      });

      proc.on("close", (code) => {
        if (code === 0) {
          resolve(Buffer.concat(chunks));
        } else {
          reject(new Error(`Piper TTS failed (exit ${code}): ${stderr}`));
        }
      });

      proc.on("error", (err) => {
        reject(new Error(`Piper binary not found: ${err.message}. Run scripts/install-piper.sh`));
      });

      // Write text to stdin and close
      proc.stdin.write(text);
      proc.stdin.end();

      // Timeout after 15 seconds
      setTimeout(() => {
        proc.kill("SIGTERM");
        reject(new Error("Piper TTS timed out"));
      }, 15000);
    });
  }

  synthesizeStream(
    text: string,
    onChunk: (chunk: Buffer) => void,
    onDone: () => void,
    onError?: (err: Error) => void
  ): { cancel: () => void } {
    let cancelled = false;

    const args = [
      "--model", this.modelPath,
      "--output_raw",
      "--length_scale", String(1.0 / (this.config.speed || 1.0)),
    ];

    const proc = spawn(this.piperBinary, args, {
      stdio: ["pipe", "pipe", "pipe"],
    });

    proc.stdout.on("data", (data: Buffer) => {
      if (cancelled) return;
      const resampled = this.resampleTo16k(data, 22050);
      onChunk(resampled);
    });

    proc.on("close", () => {
      if (!cancelled) onDone();
    });

    proc.on("error", (err) => {
      if (!cancelled) {
        console.error("[tts/piper] Error:", err.message);
        if (onError) {
          onError(err);
        } else {
          onDone();
        }
      }
    });

    // Write text and close stdin
    proc.stdin.write(text);
    proc.stdin.end();

    return {
      cancel: () => {
        cancelled = true;
        proc.kill("SIGTERM");
      },
    };
  }

  /**
   * Simple linear resampling from sourceRate to 16000Hz.
   * Input and output are 16-bit signed PCM.
   */
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
