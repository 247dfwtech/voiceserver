import { EventEmitter } from "events";
import { spawn, ChildProcess } from "child_process";
import type { STTProvider, STTConfig } from "./interface";

/**
 * Vosk STT provider — CPU-only, lightweight fallback.
 *
 * Key advantages:
 * - Runs on CPU only (no GPU needed) — perfect fallback when GPU STT fails
 * - Native 8kHz support (ideal for Twilio phone audio)
 * - ~50MB model size for English
 * - Offline, free, Apache 2.0
 * - Real-time streaming via persistent Python subprocess
 *
 * Uses vosk-api Python package with a persistent subprocess.
 * Audio is streamed in real-time (no file-based batching).
 *
 * Protocol:
 *   stdin:  raw PCM 16-bit 16kHz mono audio chunks (prefixed with 4-byte length)
 *   stdout: JSON lines {"text": "...", "partial": "..."} per Vosk protocol
 */

// Module-level singleton
let voskProcess: ChildProcess | null = null;
let voskReady = false;
let voskReadyPromise: Promise<void> | null = null;
let stdoutBuffer = "";

const DEFAULT_MODEL = "vosk-model-small-en-us-0.15";
const VOSK_MODELS_DIR = process.env.VOSK_MODELS_DIR || "/models/vosk";

function buildVoskScript(modelPath: string): string {
  return `
import sys, json, struct, os

# Redirect all stdout noise to stderr BEFORE importing
_orig_stdout = sys.stdout.buffer
sys.stdout = sys.stderr

from vosk import Model, KaldiRecognizer, SetLogLevel

# Suppress Vosk debug logging
SetLogLevel(-1)

model_path = "${modelPath}"

# Check if model exists, download if not
if not os.path.isdir(model_path):
    print(f"VOSK_LOADING downloading model to {model_path}", flush=True)
    from vosk import MODEL_LIST_URL
    import urllib.request, zipfile, io
    # Use vosk's built-in model download
    import vosk
    vosk_model_name = os.path.basename(model_path)
    # Download from alphacephei
    url = f"https://alphacephei.com/vosk/models/{vosk_model_name}.zip"
    print(f"VOSK_LOADING url={url}", flush=True)
    try:
        with urllib.request.urlopen(url) as resp:
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(os.path.dirname(model_path))
        print(f"VOSK_LOADING download complete", flush=True)
    except Exception as e:
        print(f"VOSK_ERR download failed: {e}", flush=True)
        sys.exit(1)

print(f"VOSK_LOADING model={model_path}", flush=True)
model = Model(model_path)

# 16kHz recognizer matching our PCM pipeline
rec = KaldiRecognizer(model, 16000)
rec.SetWords(True)

print("VOSK_READY", flush=True)

# Main loop: read length-prefixed PCM chunks from stdin
# Protocol: 4-byte little-endian length, then that many bytes of PCM audio
# Special: length=0 means finalize current utterance
while True:
    try:
        len_bytes = sys.stdin.buffer.read(4)
        if not len_bytes or len(len_bytes) < 4:
            break

        chunk_len = struct.unpack('<I', len_bytes)[0]

        if chunk_len == 0:
            # Finalize — get final result
            result = json.loads(rec.FinalResult())
            text = result.get("text", "").strip()
            _orig_stdout.write((json.dumps({"text": text, "final": True}) + "\\n").encode())
            _orig_stdout.flush()
            # Reset recognizer for next utterance
            rec = KaldiRecognizer(model, 16000)
            rec.SetWords(True)
            continue

        audio = sys.stdin.buffer.read(chunk_len)
        if len(audio) < chunk_len:
            break

        if rec.AcceptWaveform(audio):
            result = json.loads(rec.Result())
            text = result.get("text", "").strip()
            if text:
                _orig_stdout.write((json.dumps({"text": text, "final": True}) + "\\n").encode())
                _orig_stdout.flush()
        else:
            partial = json.loads(rec.PartialResult())
            partial_text = partial.get("partial", "").strip()
            if partial_text:
                _orig_stdout.write((json.dumps({"text": partial_text, "final": False}) + "\\n").encode())
                _orig_stdout.flush()

    except Exception as e:
        _orig_stdout.write((json.dumps({"error": str(e)}) + "\\n").encode())
        _orig_stdout.flush()
        print(f"VOSK_ERR {e}", flush=True)
`.trim();
}

function ensureVoskProcess(modelName: string): Promise<void> {
  if (voskReady && voskProcess && !voskProcess.killed) return Promise.resolve();
  if (voskReadyPromise) return voskReadyPromise;

  const modelPath = `${VOSK_MODELS_DIR}/${modelName}`;

  voskReadyPromise = new Promise<void>((resolve, reject) => {
    console.log(`[stt/vosk] Starting Vosk persistent Python process (model=${modelName})...`);

    const script = buildVoskScript(modelPath);
    voskProcess = spawn("python3", ["-u", "-c", script], {
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });

    let stderrBuf = "";
    let resolved = false;

    voskProcess.stdout!.on("data", (data: Buffer) => {
      stdoutBuffer += data.toString();
      // Handled by VoskSTT instance via onStdoutLine
    });

    voskProcess.stderr!.on("data", (data: Buffer) => {
      const text = data.toString();
      stderrBuf += text;

      for (const line of text.split("\n").filter((l) => l.trim())) {
        if (line.includes("VOSK_READY")) {
          if (!resolved) {
            resolved = true;
            voskReady = true;
            console.log("[stt/vosk] Vosk model loaded and ready (CPU)");
            resolve();
          }
        } else if (line.includes("VOSK_OK") || line.includes("VOSK_LOADING")) {
          console.log(`[stt/vosk] ${line.trim()}`);
        } else if (line.includes("VOSK_ERR")) {
          console.error(`[stt/vosk] ${line.trim()}`);
        }
      }
    });

    voskProcess.on("error", (err) => {
      console.error("[stt/vosk] Python process error:", err.message);
      voskReady = false;
      voskReadyPromise = null;
      voskProcess = null;
      if (!resolved) reject(err);
    });

    voskProcess.on("exit", (code) => {
      console.warn(`[stt/vosk] Python process exited with code ${code}`);
      voskReady = false;
      voskProcess = null;
      voskReadyPromise = null;
      if (!resolved) reject(new Error(`Vosk process exited (code ${code}): ${stderrBuf.slice(-500)}`));
    });

    // Allow up to 120s for model download + load
    setTimeout(() => {
      if (!resolved) {
        resolved = true;
        voskProcess?.kill("SIGTERM");
        voskReady = false;
        voskReadyPromise = null;
        reject(new Error(`Vosk model load timed out after 120s. stderr: ${stderrBuf.slice(-300)}`));
      }
    }, 120_000);
  });

  return voskReadyPromise;
}

export class VoskSTT extends EventEmitter implements STTProvider {
  private config: STTConfig;
  private closed = false;
  private modelName: string;
  private speechDetected = false;
  private silenceFrames = 0;
  private speechThreshold: number;
  private silenceThresholdFrames: number;
  private lastPartialText = "";
  private stdoutLineHandler: ((data: Buffer) => void) | null = null;

  constructor(config: STTConfig) {
    super();
    this.config = config;
    this.modelName = config.model || DEFAULT_MODEL;

    // VAD thresholds
    const silenceMs = config.endOfTurnTimeoutMs ?? 1200;
    this.silenceThresholdFrames = Math.round(silenceMs / 20);
    this.speechThreshold = 500;
    if (config.confidenceThreshold && config.confidenceThreshold > 0.4) {
      this.speechThreshold = 500 + (config.confidenceThreshold - 0.4) * 1000;
    }
  }

  async start(): Promise<void> {
    await ensureVoskProcess(this.modelName);

    // Attach stdout line parser for this session
    this.stdoutLineHandler = () => {
      let newline: number;
      while ((newline = stdoutBuffer.indexOf("\n")) !== -1) {
        const line = stdoutBuffer.slice(0, newline).trim();
        stdoutBuffer = stdoutBuffer.slice(newline + 1);

        if (!line) continue;

        try {
          const parsed = JSON.parse(line);
          if (parsed.error) {
            this.emit("error", new Error(parsed.error));
            continue;
          }

          const text = parsed.text || "";
          const isFinal = parsed.final === true;

          if (text) {
            this.emit("transcript", { text, isFinal, confidence: 0.85 });
            if (isFinal) {
              this.lastPartialText = "";
              this.emit("utterance_end");
            } else {
              this.lastPartialText = text;
            }
          }
        } catch {
          // Non-JSON line, skip
        }
      }
    };

    if (voskProcess?.stdout) {
      voskProcess.stdout.on("data", this.stdoutLineHandler);
    }
  }

  send(audio: Buffer): void {
    if (this.closed || !voskProcess || !voskProcess.stdin) return;

    // VAD for speech_started event
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
        // Send finalize signal (length=0) to get final result
        const finalizeLen = Buffer.alloc(4);
        finalizeLen.writeUInt32LE(0, 0);
        voskProcess.stdin.write(finalizeLen);
        this.speechDetected = false;
        this.silenceFrames = 0;
      }
    }

    // Stream audio to Vosk subprocess: 4-byte length prefix + PCM data
    const lenBuf = Buffer.alloc(4);
    lenBuf.writeUInt32LE(audio.length, 0);
    voskProcess.stdin.write(lenBuf);
    voskProcess.stdin.write(audio);

    // Process any pending stdout lines
    if (this.stdoutLineHandler) {
      this.stdoutLineHandler(Buffer.alloc(0));
    }
  }

  async finish(): Promise<void> {
    if (!voskProcess || !voskProcess.stdin) return;
    // Send finalize signal
    const finalizeLen = Buffer.alloc(4);
    finalizeLen.writeUInt32LE(0, 0);
    voskProcess.stdin.write(finalizeLen);

    // Wait a bit for final result
    await new Promise((resolve) => setTimeout(resolve, 200));
    if (this.stdoutLineHandler) {
      this.stdoutLineHandler(Buffer.alloc(0));
    }
  }

  close(): void {
    this.closed = true;
    if (this.stdoutLineHandler && voskProcess?.stdout) {
      voskProcess.stdout.removeListener("data", this.stdoutLineHandler);
    }
    this.stdoutLineHandler = null;
    this.removeAllListeners();
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
