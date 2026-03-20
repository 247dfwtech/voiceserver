import { EventEmitter } from "events";
import { spawn, ChildProcess } from "child_process";
import type { STTProvider, STTConfig } from "./interface";

/**
 * Sherpa-ONNX Streaming STT provider.
 *
 * Uses sherpa-onnx's Python API directly via a persistent subprocess.
 * Each call session gets its own recognition stream within the shared process.
 * Audio is streamed in real-time as PCM 16kHz float32 — no batching, no WAV files.
 *
 * Key advantages:
 * - True streaming: transcripts arrive in real-time as user speaks
 * - Native concurrency: sherpa-onnx handles multiple streams internally
 * - CPU-only: doesn't compete with LLM/TTS for GPU
 * - ~20MB model, fast startup
 * - More accurate than Vosk
 *
 * Protocol (stdin/stdout JSON lines):
 *   stdin:  {"cmd":"audio","data":"<base64 PCM float32>"} or {"cmd":"finalize"} or {"cmd":"reset"}
 *   stdout: {"text":"...","final":true/false}
 */

// Module-level singleton
let sherpaProcess: ChildProcess | null = null;
let sherpaReady = false;
let sherpaReadyPromise: Promise<void> | null = null;

const MODEL_DIR = process.env.SHERPA_MODEL_DIR || "/models/sherpa-onnx/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17";

function sanitizePath(s: string): string {
  return s.replace(/[^a-zA-Z0-9.\-_\/]/g, "");
}

function buildSherpaScript(modelDir: string): string {
  const safeDir = sanitizePath(modelDir);
  return `
import sys, json, base64, struct, array

# Redirect prints to stderr
_orig_stdout = sys.stdout.buffer
sys.stdout = sys.stderr

import sherpa_onnx
import numpy as np

model_dir = "${safeDir}"

print(f"SHERPA_LOADING model={model_dir}", flush=True)

recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens=f"{model_dir}/tokens.txt",
    encoder=f"{model_dir}/encoder-epoch-99-avg-1.onnx",
    decoder=f"{model_dir}/decoder-epoch-99-avg-1.onnx",
    joiner=f"{model_dir}/joiner-epoch-99-avg-1.onnx",
    num_threads=4,
    sample_rate=16000,
    feature_dim=80,
    enable_endpoint_detection=True,
    rule1_min_trailing_silence=0.8,
    rule2_min_trailing_silence=0.4,
    rule3_min_utterance_length=8.0,
)

stream = recognizer.create_stream()
last_text = ""

print("SHERPA_READY", flush=True)

# Main loop: read JSON commands from stdin (readline for unbuffered reads)
while True:
    line = sys.stdin.readline()
    if not line:
        break  # EOF
    line = line.strip()
    if not line:
        continue

    try:
        cmd = json.loads(line)

        if cmd.get("cmd") == "audio":
            # Decode base64 PCM int16 data and convert to float32
            pcm_bytes = base64.b64decode(cmd["data"])
            # PCM is 16-bit signed integers, 16kHz mono
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Debug: log first few audio chunks and RMS
            global _audio_count
            try:
                _audio_count
            except NameError:
                _audio_count = 0
            _audio_count += 1
            if _audio_count <= 5 or _audio_count % 500 == 0:
                rms = float(np.sqrt(np.mean(samples**2)))
                print(f"SHERPA_AUDIO #{_audio_count}: {len(samples)} samples, rms={rms:.6f}", flush=True)

            stream.accept_waveform(16000, samples.tolist())

            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            text = recognizer.get_result(stream).strip()

            if text and text != last_text:
                is_endpoint = recognizer.is_endpoint(stream)
                result = {"text": text, "final": is_endpoint}
                _orig_stdout.write((json.dumps(result) + "\\n").encode())
                _orig_stdout.flush()

                if is_endpoint:
                    last_text = ""
                    recognizer.reset(stream)
                else:
                    last_text = text

        elif cmd.get("cmd") == "finalize":
            # Force finalize current recognition
            text = recognizer.get_result(stream).strip()
            if text:
                result = {"text": text, "final": True}
                _orig_stdout.write((json.dumps(result) + "\\n").encode())
                _orig_stdout.flush()
            last_text = ""
            recognizer.reset(stream)

            # Ack the finalize
            _orig_stdout.write((json.dumps({"finalized": True}) + "\\n").encode())
            _orig_stdout.flush()

        elif cmd.get("cmd") == "reset":
            last_text = ""
            recognizer.reset(stream)

    except Exception as e:
        _orig_stdout.write((json.dumps({"error": str(e)}) + "\\n").encode())
        _orig_stdout.flush()
        print(f"SHERPA_ERR {e}", flush=True)
`.trim();
}

function ensureSherpaProcess(): Promise<void> {
  if (sherpaReady && sherpaProcess && !sherpaProcess.killed) return Promise.resolve();
  if (sherpaReadyPromise) return sherpaReadyPromise;

  sherpaReadyPromise = new Promise<void>((resolve, reject) => {
    console.log(`[stt/sherpa] Starting Sherpa-ONNX persistent Python process...`);

    const script = buildSherpaScript(MODEL_DIR);
    sherpaProcess = spawn("python3", ["-u", "-c", script], {
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });

    let stderrBuf = "";
    let resolved = false;

    sherpaProcess.stderr!.on("data", (data: Buffer) => {
      const text = data.toString();
      stderrBuf += text;

      for (const line of text.split("\n").filter((l) => l.trim())) {
        if (line.includes("SHERPA_READY")) {
          if (!resolved) {
            resolved = true;
            sherpaReady = true;
            console.log("[stt/sherpa] Sherpa-ONNX model loaded and ready (CPU)");
            resolve();
          }
        } else if (line.includes("SHERPA_LOADING")) {
          console.log(`[stt/sherpa] ${line.trim()}`);
        } else if (line.includes("SHERPA_ERR")) {
          console.error(`[stt/sherpa] ${line.trim()}`);
        }
      }
    });

    sherpaProcess.on("error", (err) => {
      console.error("[stt/sherpa] Python process error:", err.message);
      sherpaReady = false;
      sherpaReadyPromise = null;
      sherpaProcess = null;
      if (!resolved) reject(err);
    });

    sherpaProcess.on("exit", (code) => {
      console.warn(`[stt/sherpa] Python process exited with code ${code}`);
      sherpaReady = false;
      sherpaProcess = null;
      sherpaReadyPromise = null;
      if (!resolved) reject(new Error(`Sherpa process exited (code ${code}): ${stderrBuf.slice(-500)}`));
    });

    setTimeout(() => {
      if (!resolved) {
        resolved = true;
        sherpaProcess?.kill("SIGTERM");
        sherpaReady = false;
        sherpaReadyPromise = null;
        reject(new Error(`Sherpa model load timed out after 30s`));
      }
    }, 30_000);
  });

  return sherpaReadyPromise;
}

export class SherpaSTT extends EventEmitter implements STTProvider {
  private config: STTConfig;
  private closed = false;
  private stdoutHandler: ((data: Buffer) => void) | null = null;
  private stdoutBuffer = "";
  private speechDetected = false;

  constructor(config: STTConfig) {
    super();
    this.config = config;
  }

  async start(): Promise<void> {
    await ensureSherpaProcess();

    // Reset the stream for this new session
    if (sherpaProcess?.stdin) {
      sherpaProcess.stdin.write(JSON.stringify({ cmd: "reset" }) + "\n");
    }

    // Attach stdout parser
    this.stdoutHandler = (data: Buffer) => {
      this.stdoutBuffer += data.toString();
      let newline: number;
      while ((newline = this.stdoutBuffer.indexOf("\n")) !== -1) {
        const line = this.stdoutBuffer.slice(0, newline).trim();
        this.stdoutBuffer = this.stdoutBuffer.slice(newline + 1);

        if (!line) continue;

        try {
          const parsed = JSON.parse(line);

          if (parsed.error) {
            this.emit("error", new Error(parsed.error));
            continue;
          }

          if (parsed.finalized) continue; // Ack from finalize command

          const text = parsed.text || "";
          const isFinal = parsed.final === true;

          if (text) {
            if (!this.speechDetected) {
              this.speechDetected = true;
              this.emit("speech_started");
            }

            this.emit("transcript", { text, isFinal, confidence: 0.9 });

            if (isFinal) {
              this.speechDetected = false;
              this.emit("utterance_end");
            }
          }
        } catch {
          // Non-JSON, skip
        }
      }
    };

    if (sherpaProcess?.stdout) {
      sherpaProcess.stdout.on("data", this.stdoutHandler);
    }
  }

  private sendCount = 0;

  send(audio: Buffer): void {
    if (this.closed || !sherpaProcess || !sherpaProcess.stdin) return;

    this.sendCount++;
    if (this.sendCount <= 3 || this.sendCount % 250 === 0) {
      console.log(`[stt/sherpa] send() #${this.sendCount}: ${audio.length} bytes, closed=${this.closed}, stdinWritable=${sherpaProcess.stdin?.writable}`);
    }

    // Send PCM int16 audio as base64 JSON command
    const b64 = audio.toString("base64");
    sherpaProcess.stdin.write(JSON.stringify({ cmd: "audio", data: b64 }) + "\n");
  }

  async finish(): Promise<void> {
    if (!sherpaProcess || !sherpaProcess.stdin) return;
    sherpaProcess.stdin.write(JSON.stringify({ cmd: "finalize" }) + "\n");
    // Wait briefly for final result
    await new Promise((resolve) => setTimeout(resolve, 200));
    if (this.stdoutHandler) {
      this.stdoutHandler(Buffer.alloc(0));
    }
  }

  close(): void {
    this.closed = true;
    if (this.stdoutHandler && sherpaProcess?.stdout) {
      sherpaProcess.stdout.removeListener("data", this.stdoutHandler);
    }
    this.stdoutHandler = null;
    this.stdoutBuffer = "";
    this.removeAllListeners();
    // Reset stream for next session (don't kill the process)
    if (sherpaProcess?.stdin) {
      sherpaProcess.stdin.write(JSON.stringify({ cmd: "reset" }) + "\n");
    }
  }
}
