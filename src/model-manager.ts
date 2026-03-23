/**
 * Model Manager -- Manages LLM (Ollama), STT (Sherpa-ONNX/Vosk/Granite/Deepgram), and TTS (Kokoro/Piper) models.
 *
 * V2: Optimized for 10+ concurrent calls.
 * Default STT: Sherpa-ONNX (CPU streaming, native concurrency up to 500 connections)
 * Fallback STT: Vosk (CPU-only, lightweight)
 * GPU STT: IBM Granite 4.0 1B Speech (self-hosted, keyword biasing)
 * Cloud STT: Deepgram Flux (paid, native end-of-turn detection)
 * Default LLM: llama3.2:3b (ultra-fast, 2GB VRAM, 0.7s first-speech)
 *
 * Default TTS: Kokoro-82M (#1 TTS Arena, near-human quality, Apache 2.0)
 * Fallback TTS: Piper (ultra-lightweight CPU option)
 *
 * Tracks installed models, active selections, and provides install/remove/activate operations.
 * Persists state to DATA_DIR/model-config.json.
 */

import * as fs from "fs";
import * as path from "path";
import { exec } from "child_process";

// ---- Configuration ----

const DATA_DIR = process.env.DATA_DIR || "/data";
const VOSK_MODELS_DIR = process.env.VOSK_MODELS_DIR || "/models/vosk";
const GRANITE_MODELS_DIR = process.env.GRANITE_MODELS_DIR || "/models/granite";
const PIPER_MODELS_DIR = process.env.PIPER_MODELS_DIR || "/models/piper";
const KOKORO_MODELS_DIR = process.env.KOKORO_MODELS_DIR || "/models/kokoro";
const OLLAMA_URL = (process.env.OLLAMA_URL || "http://localhost:11434/v1").replace(/\/v1\/?$/, "");
const CONFIG_PATH = path.join(DATA_DIR, "model-config.json");

// ---- Types ----

export interface InstalledModel {
  name: string;
  displayName?: string;
  size: string;
  provider: string;
  installedAt: string;
}

export type STTProviderName = "sherpa" | "vosk" | "granite" | "deepgram";

export interface ModelConfig {
  activeLLM: { provider: "ollama"; model: string } | null;
  activeSTT: { provider: STTProviderName; model: string } | null;
  activeTTS: { provider: "kokoro" | "piper" | "chatterbox" | "qwen3" | "kokoclone"; voice: string } | null;
  installedModels: {
    llm: Array<{ name: string; size: string; provider: "ollama"; installedAt: string }>;
    stt: Array<{ name: string; size: string; provider: STTProviderName; installedAt: string }>;
    tts: Array<{ name: string; size: string; provider: "kokoro" | "piper" | "chatterbox" | "qwen3" | "kokoclone"; installedAt: string }>;
  };
}

export interface STTModelInfo {
  name: string;
  size: string;
  description: string;
  provider: STTProviderName;
}

export interface HuggingFaceSearchResult {
  id: string;
  name: string;
  downloads: number;
  likes: number;
  description: string;
}

// ---- Sherpa-ONNX model catalog (CPU STT, streaming, best concurrency) ----

const SHERPA_MODEL_CATALOG: STTModelInfo[] = [
  {
    name: "sherpa-onnx-streaming-zipformer-en-20M",
    size: "~20MB",
    description: "Zipformer 20M English, CPU streaming, native concurrency (DEFAULT)",
    provider: "sherpa",
  },
];

// ---- Vosk model catalog (CPU STT) ----

const VOSK_MODEL_CATALOG: STTModelInfo[] = [
  {
    name: "vosk-model-small-en-us-0.15",
    size: "~40MB",
    description: "Small English model, CPU-only, fast (FALLBACK)",
    provider: "vosk",
  },
  {
    name: "vosk-model-en-us-0.22",
    size: "~1.8GB",
    description: "Large English model, CPU-only, higher accuracy",
    provider: "vosk",
  },
];

// ---- Hardcoded Granite STT model catalog ----

const GRANITE_MODEL_CATALOG: STTModelInfo[] = [
  {
    name: "ibm-granite/granite-4.0-1b-speech",
    size: "~2GB",
    description: "Granite 4.0 1B Speech, keyword biasing, Apache 2.0",
    provider: "granite",
  },
];

// ---- Kokoro-82M voice catalog (default TTS) ----

export interface TTSVoiceInfo {
  name: string;
  description: string;
  provider: "kokoro" | "piper" | "chatterbox";
  gender?: string;
  accent?: string;
}

const KOKORO_VOICE_CATALOG: TTSVoiceInfo[] = [
  // American Female
  { name: "af_heart", description: "Heart — Warm, natural (DEFAULT)", provider: "kokoro", gender: "female", accent: "american" },
  { name: "af_alloy", description: "Alloy — Clear, professional", provider: "kokoro", gender: "female", accent: "american" },
  { name: "af_aoede", description: "Aoede — Melodic, engaging", provider: "kokoro", gender: "female", accent: "american" },
  { name: "af_bella", description: "Bella — Friendly, approachable", provider: "kokoro", gender: "female", accent: "american" },
  { name: "af_jessica", description: "Jessica — Confident, articulate", provider: "kokoro", gender: "female", accent: "american" },
  { name: "af_kore", description: "Kore — Youthful, energetic", provider: "kokoro", gender: "female", accent: "american" },
  { name: "af_nicole", description: "Nicole — Smooth, calm", provider: "kokoro", gender: "female", accent: "american" },
  { name: "af_nova", description: "Nova — Bright, modern", provider: "kokoro", gender: "female", accent: "american" },
  { name: "af_river", description: "River — Flowing, pleasant", provider: "kokoro", gender: "female", accent: "american" },
  { name: "af_sarah", description: "Sarah — Conversational", provider: "kokoro", gender: "female", accent: "american" },
  { name: "af_sky", description: "Sky — Light, airy", provider: "kokoro", gender: "female", accent: "american" },
  // American Male
  { name: "am_adam", description: "Adam — Deep, authoritative", provider: "kokoro", gender: "male", accent: "american" },
  { name: "am_echo", description: "Echo — Rich, resonant", provider: "kokoro", gender: "male", accent: "american" },
  { name: "am_eric", description: "Eric — Professional, clear", provider: "kokoro", gender: "male", accent: "american" },
  { name: "am_fenrir", description: "Fenrir — Strong, commanding", provider: "kokoro", gender: "male", accent: "american" },
  { name: "am_liam", description: "Liam — Warm, trustworthy", provider: "kokoro", gender: "male", accent: "american" },
  { name: "am_michael", description: "Michael — Versatile, natural", provider: "kokoro", gender: "male", accent: "american" },
  { name: "am_onyx", description: "Onyx — Smooth, deep", provider: "kokoro", gender: "male", accent: "american" },
  // British Female
  { name: "bf_emma", description: "Emma — British, refined", provider: "kokoro", gender: "female", accent: "british" },
  { name: "bf_isabella", description: "Isabella — British, elegant", provider: "kokoro", gender: "female", accent: "british" },
  // British Male
  { name: "bm_george", description: "George — British, distinguished", provider: "kokoro", gender: "male", accent: "british" },
  { name: "bm_lewis", description: "Lewis — British, casual", provider: "kokoro", gender: "male", accent: "british" },
];

// ---- Common Piper voices catalog (fallback TTS) ----

const PIPER_VOICE_CATALOG: TTSVoiceInfo[] = [
  { name: "en_US-lessac-medium", description: "Female, natural (medium)", provider: "piper" },
  { name: "en_US-lessac-high", description: "Female, high quality", provider: "piper" },
  { name: "en_US-amy-medium", description: "Female (medium)", provider: "piper" },
  { name: "en_US-ryan-medium", description: "Male (medium)", provider: "piper" },
  { name: "en_US-arctic-medium", description: "Multiple speakers (medium)", provider: "piper" },
  { name: "en_US-libritts_r-medium", description: "Multiple speakers (medium)", provider: "piper" },
];

// ---- Helper: format bytes to human-readable ----

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)}KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(0)}MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)}GB`;
}

// ---- Helper: run shell command ----

function runCommand(cmd: string): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    exec(cmd, { timeout: 600_000 }, (error, stdout, stderr) => {
      if (error) {
        reject(error);
      } else {
        resolve({ stdout, stderr });
      }
    });
  });
}

// ---- ModelManager Class ----

export class ModelManager {
  private config: ModelConfig;
  private configLoaded = false;

  constructor() {
    this.config = this.defaultConfig();
  }

  // Default LLM to auto-pull on first boot if no LLM is configured
  private static readonly DEFAULT_LLM = "llama3.2:3b";

  private defaultConfig(): ModelConfig {
    return {
      activeLLM: { provider: "ollama", model: ModelManager.DEFAULT_LLM },
      activeSTT: { provider: "vosk", model: "vosk-model-en-us-0.22" },
      activeTTS: { provider: "kokoro", voice: "af_heart" },
      installedModels: {
        llm: [],
        stt: [],
        tts: [],
      },
    };
  }

  // ---- Config Persistence ----

  async initialize(): Promise<void> {
    // Ensure DATA_DIR exists
    try {
      await fs.promises.mkdir(DATA_DIR, { recursive: true });
    } catch {
      // Directory may already exist
    }

    await this.loadConfig();
    console.log("[model-manager] Initialized. Config loaded from", CONFIG_PATH);

    // Auto-pull default LLM on first boot if no LLM models are installed
    this.autoPullDefaultLLM().catch((err) => {
      console.error("[model-manager] Auto-pull default LLM failed:", err);
    });
  }

  /**
   * Checks if any LLM model is installed via Ollama. If not, automatically
   * pulls the default model (qwen3.5:9b) so the voiceserver is ready for
   * calls immediately after first boot.
   */
  private async autoPullDefaultLLM(): Promise<void> {
    const defaultModel = ModelManager.DEFAULT_LLM;

    try {
      // Check if Ollama is reachable
      const ollamaBaseUrl = OLLAMA_URL.replace(/\/v1\/?$/, "");
      const tagsRes = await fetch(`${ollamaBaseUrl}/api/tags`, {
        signal: AbortSignal.timeout(5000),
      });

      if (!tagsRes.ok) {
        console.warn("[model-manager] Ollama not reachable, skipping auto-pull");
        return;
      }

      const data = (await tagsRes.json()) as {
        models?: Array<{ name: string }>;
      };
      const installedModels = (data.models || []).map((m) => m.name);

      // Check if default model (or any variant of it) is already installed
      const hasDefault = installedModels.some(
        (m) => m === defaultModel || m.startsWith(defaultModel.split(":")[0])
      );

      if (hasDefault) {
        console.log(`[model-manager] Default LLM "${defaultModel}" already installed`);
        // Ensure it's set as active
        if (!this.config.activeLLM) {
          const exactMatch = installedModels.find((m) => m === defaultModel) || installedModels[0];
          this.config.activeLLM = { provider: "ollama", model: exactMatch };
          await this.saveConfig();
        }
        return;
      }

      if (installedModels.length > 0) {
        console.log(
          `[model-manager] Found ${installedModels.length} LLM model(s) already installed, skipping auto-pull`
        );
        // Set first installed model as active if none set
        if (!this.config.activeLLM) {
          this.config.activeLLM = { provider: "ollama", model: installedModels[0] };
          await this.saveConfig();
        }
        return;
      }

      // No models installed -- pull the default
      console.log(`[model-manager] No LLM models installed. Auto-pulling "${defaultModel}"...`);
      console.log(`[model-manager] This may take a few minutes on first boot.`);

      const result = await this.pullLLMModel(defaultModel, (status, completed, total) => {
        if (status === "downloading" && completed && total) {
          const pct = Math.round((completed / total) * 100);
          if (pct % 10 === 0) {
            console.log(`[model-manager] Auto-pull ${defaultModel}: ${pct}%`);
          }
        }
      });

      if (result.success) {
        console.log(`[model-manager] Auto-pull complete. Activating "${defaultModel}"...`);
        await this.activateLLM(defaultModel);
        console.log(`[model-manager] Default LLM "${defaultModel}" is ready for calls`);
      } else {
        console.error(`[model-manager] Auto-pull failed: ${result.error}`);
        console.error(`[model-manager] You can manually pull a model via POST /models/llm/pull`);
      }
    } catch (err) {
      console.warn(
        `[model-manager] Auto-pull skipped (Ollama may not be running yet):`,
        err instanceof Error ? err.message : String(err)
      );
    }
  }

  private async loadConfig(): Promise<void> {
    try {
      const raw = await fs.promises.readFile(CONFIG_PATH, "utf-8");
      const parsed = JSON.parse(raw) as Partial<ModelConfig>;
      this.config = {
        activeLLM: parsed.activeLLM ?? null,
        activeSTT: parsed.activeSTT ?? null,
        activeTTS: parsed.activeTTS ?? null,
        installedModels: {
          llm: parsed.installedModels?.llm ?? [],
          stt: parsed.installedModels?.stt ?? [],
          tts: parsed.installedModels?.tts ?? [],
        },
      };
      this.configLoaded = true;
    } catch {
      // File doesn't exist or is corrupt -- use defaults
      this.config = this.defaultConfig();
      this.configLoaded = true;
    }
  }

  private async saveConfig(): Promise<void> {
    try {
      await fs.promises.mkdir(DATA_DIR, { recursive: true });
      await fs.promises.writeFile(CONFIG_PATH, JSON.stringify(this.config, null, 2), "utf-8");
    } catch (err) {
      console.error("[model-manager] Failed to save config:", err);
    }
  }

  // ---- Status ----

  getConfig(): ModelConfig {
    return { ...this.config };
  }

  getActiveLLM(): string | null {
    return this.config.activeLLM?.model ?? null;
  }

  getActiveSTT(): string | null {
    return this.config.activeSTT?.model ?? null;
  }

  getActiveSTTProvider(): string | null {
    return this.config.activeSTT?.provider ?? null;
  }

  getActiveTTS(): string | null {
    return this.config.activeTTS?.voice ?? null;
  }

  getActiveTTSProvider(): string | null {
    return this.config.activeTTS?.provider ?? null;
  }

  async getFullStatus(): Promise<{
    activeLLM: ModelConfig["activeLLM"];
    activeSTT: ModelConfig["activeSTT"];
    activeTTS: ModelConfig["activeTTS"];
    installed: {
      llm: Array<InstalledModel & { active: boolean }>;
      stt: Array<InstalledModel & { active: boolean }>;
      tts: Array<InstalledModel & { active: boolean }>;
    };
    available: {
      stt: STTModelInfo[];
      tts: TTSVoiceInfo[];
    };
  }> {
    // Sync with Ollama and STT filesystem to get the real lists
    await this.syncLLMModels();
    const installedSTT = await this.listSTTModels();

    const activeLLMName = this.config.activeLLM?.model ?? null;
    const activeTTSName = this.config.activeTTS?.voice ?? null;

    return {
      activeLLM: this.config.activeLLM,
      activeSTT: this.config.activeSTT,
      activeTTS: this.config.activeTTS,
      installed: {
        llm: this.config.installedModels.llm.map((m) => ({
          ...m,
          active: m.name === activeLLMName,
        })),
        stt: installedSTT,
        tts: this.config.installedModels.tts.map((m) => ({
          ...m,
          active: m.name === activeTTSName,
        })),
      },
      available: {
        stt: [
          ...SHERPA_MODEL_CATALOG,
          ...VOSK_MODEL_CATALOG,
          ...GRANITE_MODEL_CATALOG,
        ],
        tts: [
          ...KOKORO_VOICE_CATALOG,
          ...PIPER_VOICE_CATALOG,
        ],
      },
    };
  }

  // ---- LLM Management (Ollama) ----

  /** Sync installed LLM list with Ollama's actual models */
  private async syncLLMModels(): Promise<void> {
    try {
      const res = await fetch(`${OLLAMA_URL}/api/tags`, {
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) return;

      const data = (await res.json()) as {
        models?: Array<{ name: string; size: number; modified_at?: string }>;
      };
      const ollamaModels = data.models || [];

      this.config.installedModels.llm = ollamaModels.map((m) => ({
        name: m.name,
        size: formatBytes(m.size),
        provider: "ollama" as const,
        installedAt: m.modified_at || new Date().toISOString(),
      }));

      // If active LLM is no longer installed, clear it
      if (this.config.activeLLM) {
        const stillInstalled = ollamaModels.some(
          (m) => m.name === this.config.activeLLM!.model
        );
        if (!stillInstalled) {
          this.config.activeLLM = null;
        }
      }

      await this.saveConfig();
    } catch (err) {
      console.error("[model-manager] Failed to sync LLM models with Ollama:", err);
    }
  }

  async listLLMModels(): Promise<Array<InstalledModel & { active: boolean }>> {
    await this.syncLLMModels();
    const activeName = this.config.activeLLM?.model ?? null;
    return this.config.installedModels.llm.map((m) => ({
      ...m,
      active: m.name === activeName,
    }));
  }

  async pullLLMModel(
    name: string,
    onProgress?: (status: string, completed?: number, total?: number) => void
  ): Promise<{ success: boolean; error?: string }> {
    try {
      console.log(`[model-manager] Pulling LLM model: ${name}`);

      const res = await fetch(`${OLLAMA_URL}/api/pull`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, stream: true }),
      });

      if (!res.ok) {
        const errText = await res.text();
        return { success: false, error: `Ollama returned HTTP ${res.status}: ${errText}` };
      }

      // Process the streaming response
      const reader = res.body?.getReader();
      if (!reader) {
        return { success: false, error: "No response body from Ollama" };
      }

      const decoder = new TextDecoder();
      let lastStatus = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value, { stream: true });
        // Ollama streams newline-delimited JSON
        const lines = text.split("\n").filter((l) => l.trim());

        for (const line of lines) {
          try {
            const parsed = JSON.parse(line) as {
              status?: string;
              completed?: number;
              total?: number;
              error?: string;
            };

            if (parsed.error) {
              return { success: false, error: parsed.error };
            }

            if (parsed.status) {
              lastStatus = parsed.status;
              if (onProgress) {
                onProgress(parsed.status, parsed.completed, parsed.total);
              }
            }
          } catch {
            // Skip malformed lines
          }
        }
      }

      console.log(`[model-manager] Pull complete for ${name}: ${lastStatus}`);

      // Sync models list
      await this.syncLLMModels();

      return { success: true };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`[model-manager] Pull failed for ${name}:`, msg);
      return { success: false, error: msg };
    }
  }

  async deleteLLMModel(name: string): Promise<{ success: boolean; error?: string }> {
    try {
      console.log(`[model-manager] Deleting LLM model: ${name}`);

      const res = await fetch(`${OLLAMA_URL}/api/delete`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });

      if (!res.ok) {
        const errText = await res.text();
        return { success: false, error: `Ollama returned HTTP ${res.status}: ${errText}` };
      }

      // Clear active if it was the deleted model
      if (this.config.activeLLM?.model === name) {
        this.config.activeLLM = null;
      }

      await this.syncLLMModels();
      return { success: true };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return { success: false, error: msg };
    }
  }

  async activateLLM(name: string): Promise<{ success: boolean; error?: string }> {
    // Sync and verify model is installed
    await this.syncLLMModels();

    const installed = this.config.installedModels.llm.find((m) => m.name === name);
    if (!installed) {
      return { success: false, error: `Model "${name}" is not installed. Pull it first.` };
    }

    // Set as active (only ONE LLM at a time)
    this.config.activeLLM = { provider: "ollama", model: name };
    await this.saveConfig();

    // Warm up the model by doing a dummy completion
    try {
      console.log(`[model-manager] Warming up LLM: ${name}`);
      const warmupRes = await fetch(`${OLLAMA_URL}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: name, prompt: "hello", stream: false }),
        signal: AbortSignal.timeout(120_000), // 2 min timeout for model loading
      });

      if (!warmupRes.ok) {
        console.warn(`[model-manager] Warmup returned HTTP ${warmupRes.status} but model is set as active`);
      } else {
        console.log(`[model-manager] LLM ${name} warmed up and loaded into GPU memory`);
      }
    } catch (err) {
      console.warn(`[model-manager] Warmup failed for ${name} (model still set as active):`, err);
    }

    return { success: true };
  }

  // ---- STT Management (Sherpa-ONNX + Vosk + Granite) ----

  async listSTTModels(): Promise<Array<InstalledModel & { active: boolean }>> {
    const activeSTT = this.config.activeSTT;
    const installed: Array<InstalledModel & { active: boolean }> = [];

    // Check for installed Sherpa-ONNX models
    const SHERPA_MODELS_DIR = process.env.SHERPA_MODELS_DIR || "/models/sherpa-onnx";
    try {
      const sherpaEntries = await fs.promises.readdir(SHERPA_MODELS_DIR, { withFileTypes: true });
      for (const entry of sherpaEntries) {
        if (entry.isDirectory() && entry.name.includes("sherpa")) {
          // Match catalog by partial name (directory may have date suffix like -2023-02-17)
          const catalogEntry = SHERPA_MODEL_CATALOG.find((c) => entry.name.startsWith(c.name));
          const displayName = catalogEntry?.name || entry.name;
          installed.push({
            name: displayName,
            size: catalogEntry?.size || "~20MB",
            provider: "sherpa",
            installedAt: "",
            active: activeSTT?.provider === "sherpa",
          });
        }
      }
    } catch {
      // Sherpa directory may not exist yet
    }

    // Check for installed Vosk models
    try {
      const voskEntries = await fs.promises.readdir(VOSK_MODELS_DIR, { withFileTypes: true });
      for (const entry of voskEntries) {
        if (entry.isDirectory() && entry.name.startsWith("vosk-model")) {
          const catalogEntry = VOSK_MODEL_CATALOG.find((c) => c.name === entry.name);
          installed.push({
            name: entry.name,
            size: catalogEntry?.size || "unknown",
            provider: "vosk",
            installedAt: "",
            active: activeSTT?.provider === "vosk" && activeSTT?.model === entry.name,
          });
        }
      }
    } catch {
      // Directory may not exist yet
    }

    // Check for installed Granite models
    try {
      const graniteEntries = await fs.promises.readdir(GRANITE_MODELS_DIR, { withFileTypes: true });
      for (const entry of graniteEntries) {
        if (entry.isDirectory()) {
          const catalogEntry = GRANITE_MODEL_CATALOG.find((c) => entry.name.includes(c.name.split("/").pop()!));
          const modelName = catalogEntry?.name || `ibm-granite/${entry.name}`;
          installed.push({
            name: modelName,
            size: catalogEntry?.size || "unknown",
            provider: "granite",
            installedAt: "",
            active: activeSTT?.provider === "granite" && activeSTT?.model === modelName,
          });
        }
      }
    } catch {
      // Directory may not exist yet
    }

    // Also check for Granite via Python (model cached by HuggingFace transformers)
    if (installed.filter((m) => m.provider === "granite").length === 0) {
      try {
        const { stdout } = await runCommand(
          `python3 -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('ibm-granite/granite-4.0-1b-speech', local_files_only=True); print('ok')" 2>/dev/null`
        );
        if (stdout.includes("ok")) {
          installed.push({
            name: "ibm-granite/granite-4.0-1b-speech",
            size: "~2GB",
            provider: "granite",
            installedAt: "",
            active: activeSTT?.provider === "granite" && activeSTT?.model === "ibm-granite/granite-4.0-1b-speech",
          });
        }
      } catch {
        // Granite not installed via transformers cache
      }
    }

    // Sync the config's installed list
    this.config.installedModels.stt = installed.map((m) => ({
      name: m.name,
      size: m.size,
      provider: m.provider as STTProviderName,
      installedAt: m.installedAt,
    }));
    await this.saveConfig();

    return installed;
  }

  getSTTCatalog(): STTModelInfo[] {
    return [
      ...SHERPA_MODEL_CATALOG,
      ...VOSK_MODEL_CATALOG,
      ...GRANITE_MODEL_CATALOG,
    ];
  }

  async installSTTModel(
    name: string,
    provider?: STTProviderName
  ): Promise<{ success: boolean; error?: string }> {
    // Auto-detect provider from model name
    const isSherpa = provider === "sherpa" || name.includes("sherpa");
    const isGranite = provider === "granite" || name.includes("granite");
    const isVosk = provider === "vosk" || name.includes("vosk");

    if (isSherpa) {
      // Sherpa-ONNX models are pre-installed on the GPU server — just activate
      return { success: true };
    }
    if (isVosk) {
      return this.installVoskModel(name);
    }
    if (isGranite) {
      return this.installGraniteModel(name);
    }
    // Default to Vosk
    return this.installVoskModel(name);
  }

  private async installGraniteModel(name: string): Promise<{ success: boolean; error?: string }> {
    const modelId = name.includes("/") ? name : "ibm-granite/granite-4.0-1b-speech";

    try {
      console.log(`[model-manager] Installing Granite STT model: ${modelId}`);

      // Download model via HuggingFace transformers (caches automatically)
      const cmd = `python3 -c "
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
print('Downloading processor...')
AutoProcessor.from_pretrained('${modelId}', trust_remote_code=True)
print('Downloading model...')
AutoModelForSpeechSeq2Seq.from_pretrained('${modelId}', trust_remote_code=True)
print('ok')
"`;
      await runCommand(cmd);

      console.log(`[model-manager] Granite STT model ${modelId} installed successfully`);

      const existing = this.config.installedModels.stt.find((m) => m.name === modelId);
      if (!existing) {
        this.config.installedModels.stt.push({
          name: modelId,
          size: "~2GB",
          provider: "granite",
          installedAt: new Date().toISOString(),
        });
      }
      await this.saveConfig();
      return { success: true };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`[model-manager] Granite STT install failed for ${modelId}:`, msg);
      return { success: false, error: msg };
    }
  }

  private async installVoskModel(name: string): Promise<{ success: boolean; error?: string }> {
    const validModel = VOSK_MODEL_CATALOG.find((m) => m.name === name);
    if (!validModel) {
      return {
        success: false,
        error: `Unknown Vosk model "${name}". Valid models: ${VOSK_MODEL_CATALOG.map((m) => m.name).join(", ")}`,
      };
    }

    try {
      console.log(`[model-manager] Installing Vosk STT model: ${name}`);

      await fs.promises.mkdir(VOSK_MODELS_DIR, { recursive: true });

      // Download and extract Vosk model from alphacephei
      const cmd = `python3 -c "
import urllib.request, zipfile, io, os
model_dir = '${VOSK_MODELS_DIR}/${name}'
if os.path.isdir(model_dir):
    print('already installed')
else:
    url = 'https://alphacephei.com/vosk/models/${name}.zip'
    print(f'Downloading from {url}...')
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall('${VOSK_MODELS_DIR}')
    print('ok')
"`;
      await runCommand(cmd);

      console.log(`[model-manager] Vosk STT model ${name} installed successfully`);

      const existing = this.config.installedModels.stt.find((m) => m.name === name);
      if (!existing) {
        this.config.installedModels.stt.push({
          name,
          size: validModel.size,
          provider: "vosk",
          installedAt: new Date().toISOString(),
        });
      }
      await this.saveConfig();
      return { success: true };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`[model-manager] Vosk STT install failed for ${name}:`, msg);
      return { success: false, error: msg };
    }
  }

  async deleteSTTModel(name: string): Promise<{ success: boolean; error?: string }> {
    const isGranite = name.includes("granite");
    const isVosk = name.includes("vosk");

    try {
      let deleted = false;

      if (isVosk) {
        // Delete Vosk model directory
        try {
          const modelPath = path.join(VOSK_MODELS_DIR, name);
          await fs.promises.rm(modelPath, { recursive: true, force: true });
          console.log(`[model-manager] Deleted Vosk STT model directory: ${modelPath}`);
          deleted = true;
        } catch {
          // Directory may not exist
        }
      } else if (isGranite) {
        // For Granite, clear the HuggingFace transformers cache for this model
        try {
          const cmd = `python3 -c "
from huggingface_hub import scan_cache_dir
cache = scan_cache_dir()
for repo in cache.repos:
    if '${name.split('/').pop()}' in repo.repo_id:
        for revision in repo.revisions:
            print(f'Deleting {repo.repo_id}')
            revision.delete()
print('ok')
"`;
          await runCommand(cmd);
          deleted = true;
          console.log(`[model-manager] Deleted Granite STT model cache: ${name}`);
        } catch {
          // Also try deleting from GRANITE_MODELS_DIR if it exists
        }

        // Also check GRANITE_MODELS_DIR
        try {
          const entries = await fs.promises.readdir(GRANITE_MODELS_DIR, { withFileTypes: true });
          for (const entry of entries) {
            if (entry.isDirectory() && entry.name.includes(name.split("/").pop()!)) {
              const modelPath = path.join(GRANITE_MODELS_DIR, entry.name);
              await fs.promises.rm(modelPath, { recursive: true, force: true });
              deleted = true;
            }
          }
        } catch {
          // Directory may not exist
        }
      }

      if (!deleted) {
        return { success: false, error: `STT model "${name}" not found` };
      }

      // Clear active if it was the deleted model
      if (this.config.activeSTT?.model === name) {
        this.config.activeSTT = null;
      }

      // Remove from installed list
      this.config.installedModels.stt = this.config.installedModels.stt.filter(
        (m) => m.name !== name
      );
      await this.saveConfig();

      return { success: true };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return { success: false, error: msg };
    }
  }

  async activateSTT(
    name: string,
    provider?: STTProviderName
  ): Promise<{ success: boolean; error?: string }> {
    // Auto-detect provider from name
    const resolvedProvider = provider
      || (name.includes("sherpa") ? "sherpa" as const
        : name.includes("vosk") ? "vosk" as const
        : name.includes("granite") ? "granite" as const
        : "sherpa" as const);

    // Check if model is in catalog or installed
    const allCatalogs = [...SHERPA_MODEL_CATALOG, ...VOSK_MODEL_CATALOG, ...GRANITE_MODEL_CATALOG];
    const inCatalog = allCatalogs.some((m) => m.name === name);
    const inInstalled = this.config.installedModels.stt.some((m) => m.name === name);

    if (!inCatalog && !inInstalled) {
      return {
        success: false,
        error: `STT model "${name}" is not recognized. Available: ${allCatalogs.map((m) => m.name).join(", ")}`,
      };
    }

    this.config.activeSTT = { provider: resolvedProvider, model: name };
    await this.saveConfig();
    console.log(`[model-manager] Active STT set to ${resolvedProvider}: ${name}`);
    return { success: true };
  }

  // ---- TTS Management (Kokoro + Piper) ----

  async listTTSVoices(): Promise<Array<InstalledModel & { active: boolean }>> {
    const activeTTS = this.config.activeTTS;
    const installed: Array<InstalledModel & { active: boolean }> = [];

    // Check for installed Kokoro (kokoro Python package)
    try {
      const { stdout } = await runCommand(
        `python3 -c "import kokoro; print('ok')" 2>/dev/null`
      );
      if (stdout.includes("ok")) {
        // Kokoro is installed -- all voices are available
        for (const voice of KOKORO_VOICE_CATALOG) {
          installed.push({
            name: voice.name,
            size: "~200MB (shared)",
            provider: "kokoro",
            installedAt: "",
            active: activeTTS?.provider === "kokoro" && activeTTS?.voice === voice.name,
          });
        }
      }
    } catch {
      // Kokoro package not installed
    }

    // Check for installed Piper voices
    try {
      const entries = await fs.promises.readdir(PIPER_MODELS_DIR);
      const onnxFiles = entries.filter((e) => e.endsWith(".onnx"));

      for (const onnxFile of onnxFiles) {
        const voiceName = onnxFile.replace(".onnx", "");
        let size = "unknown";
        try {
          const stat = await fs.promises.stat(path.join(PIPER_MODELS_DIR, onnxFile));
          size = formatBytes(stat.size);
        } catch {
          // ignore
        }

        installed.push({
          name: voiceName,
          size,
          provider: "piper",
          installedAt: "",
          active: activeTTS?.provider === "piper" && activeTTS?.voice === voiceName,
        });
      }
    } catch {
      // Directory may not exist yet
    }

    // Check for KokoClone cloned voices
    try {
      const manifestPath = "/data/cloned-voices/manifest.json";
      const manifestData = await fs.promises.readFile(manifestPath, "utf-8");
      const clonedVoices = JSON.parse(manifestData);
      if (Array.isArray(clonedVoices)) {
        for (const v of clonedVoices) {
          installed.push({
            name: v.id,
            displayName: v.name,
            size: "cloned",
            provider: "kokoclone",
            installedAt: v.createdAt || "",
            active: activeTTS?.provider === "kokoclone" && activeTTS?.voice === v.id,
          });
        }
      }
    } catch {
      // No cloned voices manifest
    }

    // Check for Chatterbox cloned voices
    try {
      const { chatterboxVoiceManager: cvm } = await import("./providers/tts/chatterbox");
      const cbVoices = await cvm.listVoices();
      for (const v of cbVoices) {
        installed.push({
          name: v.id,
          size: "cloned",
          provider: "chatterbox",
          installedAt: v.createdAt,
          active: activeTTS?.provider === "chatterbox" && activeTTS?.voice === v.id,
        });
      }
    } catch {
      // Chatterbox module not available
    }

    // Sync config
    this.config.installedModels.tts = installed.map((m) => ({
      name: m.name,
      size: m.size,
      provider: m.provider as "kokoro" | "piper" | "chatterbox" | "kokoclone",
      installedAt: m.installedAt,
    }));
    await this.saveConfig();

    return installed;
  }

  getTTSCatalog(): TTSVoiceInfo[] {
    return [...KOKORO_VOICE_CATALOG, ...PIPER_VOICE_CATALOG];
  }

  async installTTSVoice(
    name: string,
    provider?: "kokoro" | "piper"
  ): Promise<{ success: boolean; error?: string }> {
    // Auto-detect provider
    const isKokoro = provider === "kokoro" || KOKORO_VOICE_CATALOG.some((v) => v.name === name);

    if (isKokoro) {
      return this.installKokoroTTS();
    }

    return this.installPiperVoice(name);
  }

  private async installKokoroTTS(): Promise<{ success: boolean; error?: string }> {
    try {
      console.log(`[model-manager] Installing Kokoro-82M TTS...`);

      // Install the kokoro Python package (includes model weights)
      const cmd = `pip3 install --no-cache-dir kokoro>=0.9 soundfile`;
      await runCommand(cmd);

      console.log(`[model-manager] Kokoro-82M TTS installed successfully`);

      // All Kokoro voices come with the package, add first one as representative
      const existing = this.config.installedModels.tts.find((m) => m.provider === "kokoro");
      if (!existing) {
        this.config.installedModels.tts.push({
          name: "kokoro-82m",
          size: "~200MB",
          provider: "kokoro",
          installedAt: new Date().toISOString(),
        });
      }
      await this.saveConfig();
      return { success: true };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`[model-manager] Kokoro TTS install failed:`, msg);
      return { success: false, error: msg };
    }
  }

  private async installPiperVoice(name: string): Promise<{ success: boolean; error?: string }> {
    try {
      console.log(`[model-manager] Installing Piper TTS voice: ${name}`);

      // Ensure piper models dir exists
      await fs.promises.mkdir(PIPER_MODELS_DIR, { recursive: true });

      // Parse voice name: e.g., "en_US-lessac-medium" -> parts
      const parts = name.split("-");
      if (parts.length < 3) {
        return {
          success: false,
          error: `Invalid voice name "${name}". Expected format like "en_US-lessac-medium".`,
        };
      }

      const locale = parts[0]; // e.g., "en_US"
      const voiceName = parts[1]; // e.g., "lessac"
      const quality = parts[parts.length - 1]; // e.g., "medium"
      const lang = locale.split("_")[0]; // e.g., "en"

      const baseUrl = `https://huggingface.co/rhasspy/piper-voices/resolve/main/${lang}/${locale}/${voiceName}/${quality}`;
      const onnxUrl = `${baseUrl}/${name}.onnx`;
      const jsonUrl = `${baseUrl}/${name}.onnx.json`;

      const onnxPath = path.join(PIPER_MODELS_DIR, `${name}.onnx`);
      const jsonPath = path.join(PIPER_MODELS_DIR, `${name}.onnx.json`);

      // Download .onnx file
      console.log(`[model-manager] Downloading ${onnxUrl}`);
      const onnxRes = await fetch(onnxUrl, {
        redirect: "follow",
        signal: AbortSignal.timeout(300_000), // 5 min timeout
      });
      if (!onnxRes.ok) {
        return {
          success: false,
          error: `Failed to download voice model: HTTP ${onnxRes.status} from ${onnxUrl}`,
        };
      }
      const onnxBuffer = Buffer.from(await onnxRes.arrayBuffer());
      await fs.promises.writeFile(onnxPath, onnxBuffer);

      // Download .onnx.json config file
      console.log(`[model-manager] Downloading ${jsonUrl}`);
      const jsonRes = await fetch(jsonUrl, {
        redirect: "follow",
        signal: AbortSignal.timeout(30_000),
      });
      if (!jsonRes.ok) {
        return {
          success: false,
          error: `Failed to download voice config: HTTP ${jsonRes.status} from ${jsonUrl}`,
        };
      }
      const jsonBuffer = Buffer.from(await jsonRes.arrayBuffer());
      await fs.promises.writeFile(jsonPath, jsonBuffer);

      console.log(`[model-manager] TTS voice ${name} installed successfully`);

      // Update config
      const existing = this.config.installedModels.tts.find((m) => m.name === name);
      if (!existing) {
        this.config.installedModels.tts.push({
          name,
          size: formatBytes(onnxBuffer.length),
          provider: "piper",
          installedAt: new Date().toISOString(),
        });
      }
      await this.saveConfig();

      return { success: true };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`[model-manager] TTS install failed for ${name}:`, msg);
      return { success: false, error: msg };
    }
  }

  async deleteTTSVoice(name: string): Promise<{ success: boolean; error?: string }> {
    const isKokoro = KOKORO_VOICE_CATALOG.some((v) => v.name === name) || name === "kokoro-82m";

    if (isKokoro) {
      try {
        // Uninstall kokoro Python package
        await runCommand(`pip3 uninstall -y kokoro`);

        // Clear active if Kokoro was active
        if (this.config.activeTTS?.provider === "kokoro") {
          this.config.activeTTS = null;
        }

        // Remove all Kokoro entries from installed list
        this.config.installedModels.tts = this.config.installedModels.tts.filter(
          (m) => m.provider !== "kokoro"
        );
        await this.saveConfig();

        console.log(`[model-manager] Deleted Kokoro TTS`);
        return { success: true };
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        return { success: false, error: msg };
      }
    }

    // Piper voice deletion
    try {
      const onnxPath = path.join(PIPER_MODELS_DIR, `${name}.onnx`);
      const jsonPath = path.join(PIPER_MODELS_DIR, `${name}.onnx.json`);

      let deleted = false;

      try {
        await fs.promises.unlink(onnxPath);
        deleted = true;
      } catch {
        // File may not exist
      }

      try {
        await fs.promises.unlink(jsonPath);
        deleted = true;
      } catch {
        // File may not exist
      }

      if (!deleted) {
        return { success: false, error: `Piper TTS voice "${name}" not found in ${PIPER_MODELS_DIR}` };
      }

      // Clear active if it was the deleted voice
      if (this.config.activeTTS?.voice === name) {
        this.config.activeTTS = null;
      }

      // Remove from installed list
      this.config.installedModels.tts = this.config.installedModels.tts.filter(
        (m) => m.name !== name
      );
      await this.saveConfig();

      console.log(`[model-manager] Deleted Piper TTS voice: ${name}`);
      return { success: true };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return { success: false, error: msg };
    }
  }

  async activateTTS(
    name: string,
    provider?: "kokoro" | "piper"
  ): Promise<{ success: boolean; error?: string }> {
    const isKokoro = provider === "kokoro" || KOKORO_VOICE_CATALOG.some((v) => v.name === name);

    if (isKokoro) {
      // Validate voice ID
      const validVoice = KOKORO_VOICE_CATALOG.some((v) => v.name === name);
      if (!validVoice) {
        return {
          success: false,
          error: `Unknown Kokoro voice "${name}". Use one of: ${KOKORO_VOICE_CATALOG.map((v) => v.name).join(", ")}`,
        };
      }

      this.config.activeTTS = { provider: "kokoro", voice: name };
      await this.saveConfig();
      console.log(`[model-manager] Active TTS set to Kokoro: ${name}`);
      return { success: true };
    }

    // Piper voice activation
    let found = false;
    try {
      await fs.promises.access(path.join(PIPER_MODELS_DIR, `${name}.onnx`));
      found = true;
    } catch {
      // Not found
    }

    if (!found) {
      return {
        success: false,
        error: `Piper TTS voice "${name}" is not installed. Install it first.`,
      };
    }

    this.config.activeTTS = { provider: "piper", voice: name };
    await this.saveConfig();
    console.log(`[model-manager] Active TTS set to Piper: ${name}`);
    return { success: true };
  }

  // ---- HuggingFace Search ----

  async searchHuggingFace(
    query: string,
    type: "llm" | "stt" | "tts"
  ): Promise<HuggingFaceSearchResult[]> {
    if (type === "stt") {
      // Return Vosk + Deepgram + Granite model lists
      const voskResults = VOSK_MODEL_CATALOG.map((m) => ({
        id: m.name,
        name: m.name,
        downloads: 0,
        likes: 0,
        description: `[VOSK] ${m.description} (${m.size})`,
      }));
      const deepgramResults = [
        { id: "deepgram/flux-general-en", name: "flux-general-en", downloads: 0, likes: 0, description: "[DEEPGRAM] Conversational STT with native end-of-turn detection (recommended for voice agents)" },
        { id: "deepgram/nova-3-general", name: "nova-3-general", downloads: 0, likes: 0, description: "[DEEPGRAM] Latest general-purpose model, highest accuracy" },
        { id: "deepgram/nova-2-general", name: "nova-2-general", downloads: 0, likes: 0, description: "[DEEPGRAM] Previous generation, stable" },
      ];
      const graniteResults = GRANITE_MODEL_CATALOG.map((m) => ({
        id: m.name,
        name: m.name.split("/").pop()!,
        downloads: 0,
        likes: 0,
        description: `[GRANITE] ${m.description} (${m.size})`,
      }));
      return [...voskResults, ...deepgramResults, ...graniteResults];
    }

    try {
      let searchUrl: string;

      if (type === "llm") {
        searchUrl = `https://huggingface.co/api/models?search=${encodeURIComponent(query)}&filter=gguf&sort=downloads&direction=-1&limit=20`;
      } else {
        // TTS -- return Kokoro + Piper catalog first, then search HuggingFace
        const kokoroResults = KOKORO_VOICE_CATALOG
          .filter((v) => v.name.includes(query.toLowerCase()) || v.description.toLowerCase().includes(query.toLowerCase()))
          .map((v) => ({
            id: `kokoro/${v.name}`,
            name: v.name,
            downloads: 0,
            likes: 0,
            description: `[KOKORO] ${v.description}`,
          }));

        if (kokoroResults.length > 0) {
          return kokoroResults;
        }

        searchUrl = `https://huggingface.co/api/models?search=${encodeURIComponent("piper " + query)}&author=rhasspy&sort=downloads&direction=-1&limit=20`;
      }

      const res = await fetch(searchUrl, {
        signal: AbortSignal.timeout(10_000),
        headers: { Accept: "application/json" },
      });

      if (!res.ok) {
        console.error(`[model-manager] HuggingFace search failed: HTTP ${res.status}`);
        return [];
      }

      const data = (await res.json()) as Array<{
        id?: string;
        modelId?: string;
        downloads?: number;
        likes?: number;
        pipeline_tag?: string;
        cardData?: { description?: string };
      }>;

      return data.map((m) => ({
        id: m.id || m.modelId || "",
        name: (m.id || m.modelId || "").split("/").pop() || "",
        downloads: m.downloads || 0,
        likes: m.likes || 0,
        description: m.cardData?.description || m.pipeline_tag || "",
      }));
    } catch (err) {
      console.error("[model-manager] HuggingFace search error:", err);
      return [];
    }
  }
}

// ---- Singleton ----

export const modelManager = new ModelManager();
