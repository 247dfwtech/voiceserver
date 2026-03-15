import type { STTConfig, STTProvider } from "./stt/interface";
import type { TTSConfig, TTSProvider } from "./tts/interface";
import type { LLMConfig, LLMProvider } from "./llm/interface";
import { WhisperSTT } from "./stt/whisper";
import { OpenAICompatLLM } from "./llm/openai-compat";
import { PiperTTS } from "./tts/piper";
import { modelManager } from "../model-manager";

/**
 * Provider factory functions.
 *
 * Free-first: defaults to Whisper (STT), Ollama (LLM), and Piper (TTS).
 * Optional paid providers are supported if API keys are present.
 */

export function createSTTProvider(config: STTConfig): STTProvider {
  // Use active STT model from model-manager as default if not specified in config
  if (!config.model) {
    const activeSTT = modelManager.getActiveSTT();
    if (activeSTT) {
      config = { ...config, model: activeSTT };
      console.log(`[providers] Using model-manager active STT model: ${activeSTT}`);
    }
  }

  switch (config.provider) {
    case "deepgram": {
      // Dynamically import Deepgram only if requested and API key is available
      const apiKey = process.env.DEEPGRAM_API_KEY;
      if (!apiKey) {
        console.warn("[providers] Deepgram requested but DEEPGRAM_API_KEY not set, falling back to Whisper");
        return new WhisperSTT(config);
      }
      // Lazy require to avoid needing the SDK installed when not used
      try {
        const { DeepgramSTT } = require("./stt/deepgram");
        return new DeepgramSTT(config, apiKey);
      } catch {
        console.warn("[providers] @deepgram/sdk not installed, falling back to Whisper");
        return new WhisperSTT(config);
      }
    }
    case "whisper":
    case "faster-whisper":
      return new WhisperSTT(config);
    default:
      // Default to Whisper (free)
      return new WhisperSTT(config);
  }
}

export function createTTSProvider(config: TTSConfig): TTSProvider {
  // Use active TTS voice from model-manager as default if not specified in config
  if (!config.voiceId) {
    const activeTTS = modelManager.getActiveTTS();
    if (activeTTS) {
      config = { ...config, voiceId: activeTTS };
      console.log(`[providers] Using model-manager active TTS voice: ${activeTTS}`);
    }
  }

  switch (config.provider) {
    case "elevenlabs":
    case "11labs": {
      const apiKey = process.env.ELEVENLABS_API_KEY;
      if (!apiKey) {
        console.warn("[providers] ElevenLabs requested but ELEVENLABS_API_KEY not set, falling back to Piper");
        return new PiperTTS(config);
      }
      try {
        const { ElevenLabsTTS } = require("./tts/elevenlabs");
        return new ElevenLabsTTS(config, apiKey);
      } catch {
        console.warn("[providers] ElevenLabs module not available, falling back to Piper");
        return new PiperTTS(config);
      }
    }
    case "piper":
      return new PiperTTS(config);
    default:
      // Default to Piper (free)
      return new PiperTTS(config);
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
