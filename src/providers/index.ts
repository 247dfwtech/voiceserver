import type { STTConfig, STTProvider } from "./stt/interface";
import type { TTSConfig, TTSProvider } from "./tts/interface";
import type { LLMConfig, LLMProvider } from "./llm/interface";
import { ParakeetSTT } from "./stt/parakeet";
import { VoskSTT } from "./stt/vosk";
import { DeepgramSTT, DEEPGRAM_MODELS } from "./stt/deepgram";
import { GraniteSTT } from "./stt/granite";
import { OpenAICompatLLM } from "./llm/openai-compat";
import { PiperTTS } from "./tts/piper";
import { KokoroTTS } from "./tts/kokoro";
import { KokoCloneTTS } from "./tts/kokoclone";
import { ChatterboxTurboTTS } from "./tts/chatterbox";
import { modelManager } from "../model-manager";

/**
 * Provider factory functions.
 *
 * Free-first defaults:
 *   STT: Parakeet TDT 0.6b v2 (6.32% WER, best self-hosted English accuracy)
 *        Deepgram Flux as paid cloud option with native end-of-turn detection
 *        Vosk as CPU-only fallback (no GPU needed)
 *   LLM: Ollama (qwen3.5:9b default, auto-pulled on first boot)
 *   TTS: Kokoro-82M (Piper as fallback)
 * Optional paid providers are supported if API keys are present.
 *
 * NOTE: Whisper was removed — it hallucinates on 8kHz Twilio phone audio.
 */

export function createSTTProvider(config: STTConfig): STTProvider {
  // Use active STT from model-manager as default if not specified in config
  if (!config.model && !config.provider) {
    const activeSTT = modelManager.getActiveSTT();
    const activeProvider = modelManager.getActiveSTTProvider();
    if (activeSTT) {
      config = { ...config, model: activeSTT, provider: activeProvider || "parakeet" };
      console.log(`[providers] Using model-manager active STT: ${activeProvider}/${activeSTT}`);
    }
  }

  switch (config.provider) {
    case "deepgram": {
      const apiKey = process.env.DEEPGRAM_API_KEY;
      if (!apiKey) {
        console.warn("[providers] Deepgram requested but DEEPGRAM_API_KEY not set, falling back to Parakeet");
        return new ParakeetSTT(config);
      }
      // Validate model name — reject non-Deepgram model names sent by mistake
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
    case "parakeet":
    case "nvidia-parakeet":
      console.log(`[providers] Using Parakeet TDT STT (model=${config.model || "nvidia/parakeet-tdt-0.6b-v2"})`);
      return new ParakeetSTT(config);
    case "vosk":
      console.log(`[providers] Using Vosk STT (CPU fallback, model=${config.model || "vosk-model-small-en-us-0.15"})`);
      return new VoskSTT(config);
    case "granite":
      console.log(`[providers] Using Granite STT (model=${config.model || "ibm-granite/granite-4.0-1b-speech"})`);
      return new GraniteSTT(config);
    default:
      // Default to Parakeet TDT (free, local GPU, best self-hosted accuracy)
      console.log("[providers] Defaulting to Parakeet TDT STT");
      return new ParakeetSTT(config);
  }
}

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
    ollama: "ollama", // Ollama doesn't require an API key
  };

  const baseUrlMap: Record<string, string> = {
    deepseek: process.env.DEEPSEEK_BASE_URL || "https://api.deepseek.com/v1",
    "deep-seek": process.env.DEEPSEEK_BASE_URL || "https://api.deepseek.com/v1",
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
