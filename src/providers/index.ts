import type { STTConfig, STTProvider } from "./stt/interface";
import type { TTSConfig, TTSProvider } from "./tts/interface";
import type { LLMConfig, LLMProvider } from "./llm/interface";
import { WhisperSTT } from "./stt/whisper";
import { GraniteSTT } from "./stt/granite";
import { OpenAICompatLLM } from "./llm/openai-compat";
import { PiperTTS } from "./tts/piper";
import { KokoroTTS } from "./tts/kokoro";
import { modelManager } from "../model-manager";

/**
 * Provider factory functions.
 *
 * Free-first defaults:
 *   STT: IBM Granite 4.0 1B Speech (faster-whisper as fallback option)
 *   LLM: Ollama
 *   TTS: Kokoro-82M (Piper as fallback)
 * Optional paid providers are supported if API keys are present.
 */

export function createSTTProvider(config: STTConfig): STTProvider {
  // Use active STT from model-manager as default if not specified in config
  if (!config.model && !config.provider) {
    const activeSTT = modelManager.getActiveSTT();
    const activeProvider = modelManager.getActiveSTTProvider();
    if (activeSTT) {
      config = { ...config, model: activeSTT, provider: activeProvider || "granite" };
      console.log(`[providers] Using model-manager active STT: ${activeProvider}/${activeSTT}`);
    }
  }

  switch (config.provider) {
    case "deepgram": {
      const apiKey = process.env.DEEPGRAM_API_KEY;
      if (!apiKey) {
        console.warn("[providers] Deepgram requested but no API key, falling back to Granite");
        return new GraniteSTT(config);
      }
      try {
        const { DeepgramSTT } = require("./stt/deepgram");
        return new DeepgramSTT(config, apiKey);
      } catch {
        console.warn("[providers] @deepgram/sdk not installed, falling back to Granite");
        return new GraniteSTT(config);
      }
    }
    case "whisper":
    case "faster-whisper":
      return new WhisperSTT(config);
    case "granite":
    case "granite-speech":
    case "ibm-granite":
      return new GraniteSTT(config);
    default:
      // Default to Granite (free, #1 OpenASR, keyword biasing for names)
      return new GraniteSTT(config);
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
