import type { STTConfig, STTProvider } from "./stt/interface";
import type { TTSConfig, TTSProvider } from "./tts/interface";
import type { LLMConfig, LLMProvider } from "./llm/interface";
import { SherpaSTT } from "./stt/sherpa";
import { VoskSTT } from "./stt/vosk";
import { DeepgramSTT, DEEPGRAM_MODELS } from "./stt/deepgram";
import { GraniteSTT } from "./stt/granite";
import { OpenAICompatLLM } from "./llm/openai-compat";
import { PiperTTS } from "./tts/piper";
import { KokoroTTS } from "./tts/kokoro";
import { KokoCloneTTS } from "./tts/kokoclone";
import { ChatterboxTurboTTS } from "./tts/chatterbox";
import { Qwen3TTS } from "./tts/qwen3";
import { FishSpeechTTS } from "./tts/fish-speech";
import { checkKokoroHealth } from "./tts/kokoro";
import { checkQwen3Health } from "./tts/qwen3";
import { checkFishSpeechHealth } from "./tts/fish-speech";
import { modelManager } from "../model-manager";

/**
 * Provider factory functions — V2 optimized for 10+ concurrent calls.
 *
 * STT providers (in order of recommendation):
 *   Sherpa-ONNX — self-hosted CPU, streaming, native concurrency, best self-hosted accuracy
 *   Deepgram Flux — cloud, paid, native end-of-turn detection
 *   Vosk — self-hosted CPU, lightweight fallback
 *   Granite 4.0 1B — self-hosted GPU, multimodal (limited concurrency)
 *
 * LLM: Ollama with FLASH_ATTENTION + NUM_PARALLEL=8 (llama3.2:3b default)
 * TTS: Kokoro-82M (Piper as fallback)
 */

export function createSTTProvider(config: STTConfig): STTProvider {
  // Use active STT from model-manager as default if not specified in config
  if (!config.model && !config.provider) {
    const activeSTT = modelManager.getActiveSTT();
    const activeProvider = modelManager.getActiveSTTProvider();
    if (activeSTT) {
      config = { ...config, model: activeSTT, provider: activeProvider || "vosk" };
      console.log(`[providers] Using model-manager active STT: ${activeProvider}/${activeSTT}`);
    }
  }

  switch (config.provider) {
    case "deepgram": {
      const apiKey = process.env.DEEPGRAM_API_KEY;
      if (!apiKey) {
        console.warn("[providers] Deepgram requested but DEEPGRAM_API_KEY not set, falling back to Vosk");
        return new VoskSTT({ ...config, model: config.model || "vosk-model-en-us-0.22" });
      }
      const validDGModels = DEEPGRAM_MODELS.map(m => m.id);
      if (config.model && !validDGModels.includes(config.model)) {
        console.warn(`[providers] Invalid Deepgram model "${config.model}", defaulting to flux-general-en`);
        config = { ...config, model: "flux-general-en" };
      }
      if (!config.model) {
        config = { ...config, model: "flux-general-en" };
      }
      console.log(`[providers] Using Deepgram STT (model=${config.model})`);
      return new DeepgramSTT(config, apiKey);
    }
    case "sherpa":
    case "sherpa-onnx":
      // Sherpa-ONNX 20M model is too inaccurate for phone audio (mulaw 8kHz).
      // Fall through to Vosk which handles telephony audio well.
      console.log(`[providers] Sherpa-ONNX requested but too inaccurate for phone audio, using Vosk instead`);
      return new VoskSTT({ ...config, model: "vosk-model-en-us-0.22" });
    case "vosk":
      console.log(`[providers] Using Vosk STT (model=${config.model || "vosk-model-en-us-0.22"})`);
      return new VoskSTT({ ...config, model: config.model || "vosk-model-en-us-0.22" });
    case "granite":
      console.log(`[providers] Using Granite STT (model=${config.model || "ibm-granite/granite-4.0-1b-speech"})`);
      return new GraniteSTT(config);
    default:
      // Default to Vosk large model (accurate on phone audio, CPU-only, proven in V1)
      console.log(`[providers] Defaulting to Vosk STT (large model)`);
      return new VoskSTT({ ...config, model: "vosk-model-en-us-0.22" });
  }
}

/**
 * Create a TTS provider with fallback awareness.
 * For HTTP-based providers (kokoro, qwen3), the health check happens at synthesis time --
 * if the service is down, the HTTP request will fail and the error propagates to call-session
 * which handles it gracefully (logs error, signals onDone so session doesn't hang).
 *
 * Fallback chain: qwen3 -> kokoro -> piper (CPU, always available)
 */
export function createTTSProvider(config: TTSConfig): TTSProvider {
  // Use active TTS from model-manager as default if not specified in config
  if (!config.voiceId && !config.provider) {
    const activeTTS = modelManager.getActiveTTS();
    const activeTTSProvider = modelManager.getActiveTTSProvider();
    if (activeTTS) {
      config = { ...config, voiceId: activeTTS, provider: activeTTSProvider || "kokoro" };
      console.log(`[providers] Using model-manager active TTS: ${activeTTSProvider}/${activeTTS}`);
    }
  }

  switch (config.provider) {
    case "elevenlabs":
    case "11labs": {
      const apiKey = process.env.ELEVENLABS_API_KEY;
      if (!apiKey) {
        console.warn("[providers] ElevenLabs requested but ELEVENLABS_API_KEY not set, falling back to Kokoro");
        return new KokoroTTS(config);
      }
      try {
        const { ElevenLabsTTS } = require("./tts/elevenlabs");
        return new ElevenLabsTTS(config, apiKey);
      } catch {
        console.warn("[providers] ElevenLabs module not available, falling back to Kokoro");
        return new KokoroTTS(config);
      }
    }
    case "kokoro":
    case "kokoro-82m":
      return new KokoroTTS(config);
    case "kokoclone":
    case "clone":
      return new KokoCloneTTS(config);
    case "chatterbox":
    case "chatterbox-turbo":
      return new ChatterboxTurboTTS(config);
    case "qwen3":
    case "qwen3-tts":
      return new Qwen3TTS(config);
    case "fish-speech":
    case "fish":
      return new FishSpeechTTS(config);
    case "piper":
      return new PiperTTS(config);
    default:
      // Default to Kokoro (free, #1 TTS Arena, near-human quality)
      return new KokoroTTS(config);
  }
}

export function createLLMProvider(config: LLMConfig): LLMProvider {
  // Use active LLM model from model-manager as default if provider is ollama and model not specified
  if ((!config.model || config.provider === "ollama") && !config.model) {
    const activeLLM = modelManager.getActiveLLM();
    if (activeLLM) {
      config = { ...config, model: activeLLM };
      console.log(`[providers] Using model-manager active LLM model: ${activeLLM}`);
    }
  }

  const apiKeyMap: Record<string, string> = {
    openai: process.env.OPENAI_API_KEY || "",
    deepseek: process.env.DEEPSEEK_API_KEY || "",
    "deep-seek": process.env.DEEPSEEK_API_KEY || "",
    cerebras: process.env.CEREBRAS_API_KEY || "",
    groq: process.env.GROQ_API_KEY || "",
    deepinfra: process.env.DEEPINFRA_API_KEY || "",
    openrouter: process.env.OPENROUTER_API_KEY || "",
    ollama: "ollama", // Ollama doesn't require an API key
  };

  const baseUrlMap: Record<string, string> = {
    deepseek: process.env.DEEPSEEK_BASE_URL || "https://api.deepseek.com/v1",
    "deep-seek": process.env.DEEPSEEK_BASE_URL || "https://api.deepseek.com/v1",
    cerebras: "https://api.cerebras.ai/v1",
    groq: "https://api.groq.com/openai/v1",
    deepinfra: "https://api.deepinfra.com/v1/openai",
    openrouter: "https://openrouter.ai/api/v1",
    ollama: process.env.OLLAMA_URL || "http://localhost:11434/v1",
  };

  const apiKey = apiKeyMap[config.provider] || process.env.OPENAI_API_KEY || "ollama";
  const resolvedConfig = {
    ...config,
    baseUrl: config.baseUrl || baseUrlMap[config.provider],
  };

  // Default to Ollama if no provider specified or provider is unknown
  if (!resolvedConfig.baseUrl && !apiKeyMap[config.provider]) {
    resolvedConfig.baseUrl = process.env.OLLAMA_URL || "http://localhost:11434/v1";
  }

  return new OpenAICompatLLM(resolvedConfig, apiKey);
}
