/**
 * Model Manager -- Manages LLM (Ollama), STT (Granite/Whisper), and TTS (Piper) models.
 *
 * Default STT: IBM Granite 4.0 1B Speech (#1 OpenASR, keyword biasing, Apache 2.0)
 * Fallback STT: faster-whisper (can be enabled if Granite is not preferred)
 *
 * Tracks installed models, active selections, and provides install/remove/activate operations.
 * Persists state to DATA_DIR/model-config.json.
 */

import * as fs from "fs";
import * as path from "path";
import { exec } from "child_process";

// ---- Configuration ----

const DATA_DIR = process.env.DATA_DIR || "/data";
const WHISPER_MODELS_DIR = process.env.WHISPER_MODELS_DIR || "/models/whisper";
const GRANITE_MODELS_DIR = process.env.GRANITE_MODELS_DIR || "/models/granite";
const PIPER_MODELS_DIR = process.env.PIPER_MODELS_DIR || "/models/piper";
const OLLAMA_URL = (process.env.OLLAMA_URL || "http://localhost:11434/v1").replace(/\/v1\/?$/, "");
const CONFIG_PATH = path.join(DATA_DIR, "model-config.json");

// ---- Types ----

export interface InstalledModel {
  name: string;
  size: string;
  provider: string;
  installedAt: string;
}

export interface ModelConfig {
  activeLLM: { provider: "ollama"; model: string } | null;
  activeSTT: { provider: "granite" | "whisper"; model: string } | null;
  activeTTS: { provider: "piper"; voice: string } | null;
  installedModels: {
    llm: Array<{ name: string; size: string; provider: "ollama"; installedAt: string }>;
    stt: Array<{ name: string; size: string; provider: "granite" | "whisper"; installedAt: string }>;
    tts: Array<{ name: string; size: string; provider: "piper"; installedAt: string }>;
  };
}

export interface STTModelInfo {
  name: string;
  size: string;
  description: string;
  provider: "granite" | "whisper";
}

export interface WhisperModelInfo {
  name: string;
  size: string;
  description: string;
}

export interface HuggingFaceSearchResult {
  id: string;
  name: string;
  downloads: number;
  likes: number;
  description: string;
}

// ---- Hardcoded Whisper model catalog ----

const WHISPER_MODEL_CATALOG: WhisperModelInfo[] = [
  { name: "tiny", size: "39MB", description: "Fastest, multilingual, lowest accuracy" },
  { name: "tiny.en", size: "39MB", description: "Fastest, English only, lower accuracy" },
  { name: "base", size: "74MB", description: "Good balance, multilingual" },
  { name: "base.en", size: "74MB", description: "Good balance, English only" },
  { name: "small", size: "244MB", description: "Better accuracy, multilingual" },
  { name: "small.en", size: "244MB", description: "Better accuracy, English only" },
  { name: "medium", size: "769MB", description: "High accuracy, multilingual" },
  { name: "medium.en", size: "769MB", description: "High accuracy, English only" },
  { name: "large-v3", size: "1.5GB", description: "Best accuracy, multilingual" },
  { name: "turbo", size: "809MB", description: "Fast + accurate, multilingual" },
];

// ---- Hardcoded Granite STT model catalog ----

const GRANITE_MODEL_CATALOG: STTModelInfo[] = [
  {
    name: "ibm-granite/granite-4.0-1b-speech",
    size: "~2GB",
    description: "#1 OpenASR leaderboard, keyword biasing, Apache 2.0 (DEFAULT)",
    provider: "granite",
  },
];

// ---- Common Piper voices catalog ----

const PIPER_VOICE_CATALOG = [
  { name: "en_US-lessac-medium", description: "Female, natural", quality: "medium" },
  { name: "en_US-lessac-high", description: "Female, high quality", quality: "high" },
  { name: "en_US-amy-medium", description: "Female", quality: "medium" },
  { name: "en_US-ryan-medium", description: "Male", quality: "medium" },
  { name: "en_US-arctic-medium", description: "Multiple speakers", quality: "medium" },
  { name: "en_US-libritts_r-medium", description: "Multiple speakers", quality: "medium" },
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

  private defaultConfig(): ModelConfig {
    return {
      activeLLM: null,
      activeSTT: { provider: "granite", model: "ibm-granite/granite-4.0-1b-speech" },
      activeTTS: null,
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
      tts: typeof PIPER_VOICE_CATALOG;
    };
  }> {
    // Sync with Ollama to get the real list
    await this.syncLLMModels();

    const activeLLMName = this.config.activeLLM?.model ?? null;
    const activeSTTName = this.config.activeSTT?.model ?? null;
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
        stt: this.config.installedModels.stt.map((m) => ({
          ...m,
          active: m.name === activeSTTName,
        })),
        tts: this.config.installedModels.tts.map((m) => ({
          ...m,
          active: m.name === activeTTSName,
        })),
      },
      available: {
        stt: [
          ...GRANITE_MODEL_CATALOG,
          ...WHISPER_MODEL_CATALOG.map((m) => ({ ...m, provider: "whisper" as const })),
        ],
        tts: PIPER_VOICE_CATALOG,
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

  // ---- STT Management (Granite + Whisper) ----

  async listSTTModels(): Promise<Array<InstalledModel & { active: boolean }>> {
    const activeSTT = this.config.activeSTT;
    const installed: Array<InstalledModel & { active: boolean }> = [];

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

    // Check for installed Whisper models
    try {
      const entries = await fs.promises.readdir(WHISPER_MODELS_DIR, { withFileTypes: true });
      for (const entry of entries) {
        if (entry.isDirectory()) {
          const catalogEntry = WHISPER_MODEL_CATALOG.find((c) => entry.name.includes(c.name));
          const modelName = catalogEntry?.name || entry.name;

          let size = catalogEntry?.size || "unknown";
          try {
            const stat = await fs.promises.stat(path.join(WHISPER_MODELS_DIR, entry.name));
            if (!catalogEntry) {
              size = formatBytes(stat.size);
            }
          } catch {
            // ignore
          }

          installed.push({
            name: modelName,
            size,
            provider: "whisper",
            installedAt: "",
            active: activeSTT?.provider === "whisper" && activeSTT?.model === modelName,
          });
        }
      }
    } catch {
      // Directory may not exist yet
    }

    // Sync the config's installed list
    this.config.installedModels.stt = installed.map((m) => ({
      name: m.name,
      size: m.size,
      provider: m.provider as "granite" | "whisper",
      installedAt: m.installedAt,
    }));
    await this.saveConfig();

    return installed;
  }

  getSTTCatalog(): STTModelInfo[] {
    return [
      ...GRANITE_MODEL_CATALOG,
      ...WHISPER_MODEL_CATALOG.map((m) => ({ ...m, provider: "whisper" as const })),
    ];
  }

  async installSTTModel(
    name: string,
    provider?: "granite" | "whisper"
  ): Promise<{ success: boolean; error?: string }> {
    // Auto-detect provider from model name
    const isGranite = provider === "granite" || name.includes("granite");

    if (isGranite) {
      return this.installGraniteModel(name);
    }
    return this.installWhisperModel(name);
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

  private async installWhisperModel(name: string): Promise<{ success: boolean; error?: string }> {
    const validModel = WHISPER_MODEL_CATALOG.find((m) => m.name === name);
    if (!validModel) {
      return {
        success: false,
        error: `Unknown Whisper model "${name}". Valid models: ${WHISPER_MODEL_CATALOG.map((m) => m.name).join(", ")}`,
      };
    }

    try {
      console.log(`[model-manager] Installing Whisper STT model: ${name}`);

      await fs.promises.mkdir(WHISPER_MODELS_DIR, { recursive: true });

      const cmd = `python3 -c "from faster_whisper import WhisperModel; WhisperModel('${name}', download_root='${WHISPER_MODELS_DIR}')"`;
      await runCommand(cmd);

      console.log(`[model-manager] Whisper STT model ${name} installed successfully`);

      const existing = this.config.installedModels.stt.find((m) => m.name === name);
      if (!existing) {
        this.config.installedModels.stt.push({
          name,
          size: validModel.size,
          provider: "whisper",
          installedAt: new Date().toISOString(),
        });
      }
      await this.saveConfig();
      return { success: true };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`[model-manager] Whisper STT install failed for ${name}:`, msg);
      return { success: false, error: msg };
    }
  }

  async deleteSTTModel(name: string): Promise<{ success: boolean; error?: string }> {
    const isGranite = name.includes("granite");

    try {
      let deleted = false;

      if (isGranite) {
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
      } else {
        // Whisper model deletion
        const entries = await fs.promises.readdir(WHISPER_MODELS_DIR, { withFileTypes: true });
        for (const entry of entries) {
          if (entry.isDirectory() && (entry.name === name || entry.name.includes(name))) {
            const modelPath = path.join(WHISPER_MODELS_DIR, entry.name);
            await fs.promises.rm(modelPath, { recursive: true, force: true });
            console.log(`[model-manager] Deleted Whisper STT model directory: ${modelPath}`);
            deleted = true;
          }
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
    provider?: "granite" | "whisper"
  ): Promise<{ success: boolean; error?: string }> {
    const isGranite = provider === "granite" || name.includes("granite");

    if (isGranite) {
      // Check if Granite model is available (in catalog or installed)
      const inCatalog = GRANITE_MODEL_CATALOG.some((m) => m.name === name);
      const inInstalled = this.config.installedModels.stt.some(
        (m) => m.name === name && m.provider === "granite"
      );

      if (!inCatalog && !inInstalled) {
        return {
          success: false,
          error: `Granite STT model "${name}" is not recognized. Available: ${GRANITE_MODEL_CATALOG.map((m) => m.name).join(", ")}`,
        };
      }

      this.config.activeSTT = { provider: "granite", model: name };
      await this.saveConfig();
      console.log(`[model-manager] Active STT set to Granite: ${name}`);
      return { success: true };
    }

    // Whisper activation
    let found = false;
    try {
      const entries = await fs.promises.readdir(WHISPER_MODELS_DIR, { withFileTypes: true });
      found = entries.some(
        (e) => e.isDirectory() && (e.name === name || e.name.includes(name))
      );
    } catch {
      // Directory doesn't exist
    }

    const inCatalog = WHISPER_MODEL_CATALOG.some((m) => m.name === name);

    if (!found && !inCatalog) {
      return {
        success: false,
        error: `STT model "${name}" is not installed and not a recognized Whisper model.`,
      };
    }

    this.config.activeSTT = { provider: "whisper", model: name };
    await this.saveConfig();
    console.log(`[model-manager] Active STT set to Whisper: ${name}`);
    return { success: true };
  }

  // ---- TTS Management (Piper) ----

  async listTTSVoices(): Promise<Array<InstalledModel & { active: boolean }>> {
    const activeTTSName = this.config.activeTTS?.voice ?? null;
    const installed: Array<InstalledModel & { active: boolean }> = [];

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
          active: voiceName === activeTTSName,
        });
      }
    } catch {
      // Directory may not exist yet
    }

    // Sync config
    this.config.installedModels.tts = installed.map((m) => ({
      name: m.name,
      size: m.size,
      provider: "piper" as const,
      installedAt: m.installedAt,
    }));
    await this.saveConfig();

    return installed;
  }

  getTTSCatalog(): typeof PIPER_VOICE_CATALOG {
    return PIPER_VOICE_CATALOG;
  }

  async installTTSVoice(name: string): Promise<{ success: boolean; error?: string }> {
    try {
      console.log(`[model-manager] Installing TTS voice: ${name}`);

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
        return { success: false, error: `TTS voice "${name}" not found in ${PIPER_MODELS_DIR}` };
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

      console.log(`[model-manager] Deleted TTS voice: ${name}`);
      return { success: true };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return { success: false, error: msg };
    }
  }

  async activateTTS(name: string): Promise<{ success: boolean; error?: string }> {
    // Check if voice is installed
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
        error: `TTS voice "${name}" is not installed. Install it first.`,
      };
    }

    this.config.activeTTS = { provider: "piper", voice: name };
    await this.saveConfig();
    console.log(`[model-manager] Active TTS set to: ${name}`);
    return { success: true };
  }

  // ---- HuggingFace Search ----

  async searchHuggingFace(
    query: string,
    type: "llm" | "stt" | "tts"
  ): Promise<HuggingFaceSearchResult[]> {
    if (type === "stt") {
      // Return Granite + Whisper model lists
      const graniteResults = GRANITE_MODEL_CATALOG.map((m) => ({
        id: m.name,
        name: m.name.split("/").pop() || m.name,
        downloads: 0,
        likes: 0,
        description: `[GRANITE] ${m.description} (${m.size})`,
      }));
      const whisperResults = WHISPER_MODEL_CATALOG.map((m) => ({
        id: `openai/whisper-${m.name}`,
        name: m.name,
        downloads: 0,
        likes: 0,
        description: `[WHISPER] ${m.description} (${m.size})`,
      }));
      return [...graniteResults, ...whisperResults];
    }

    try {
      let searchUrl: string;

      if (type === "llm") {
        searchUrl = `https://huggingface.co/api/models?search=${encodeURIComponent(query)}&filter=gguf&sort=downloads&direction=-1&limit=20`;
      } else {
        // TTS
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
