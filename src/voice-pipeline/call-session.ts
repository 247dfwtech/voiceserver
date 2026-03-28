import { EventEmitter } from "events";
import {
  mulawToPcm16k,
  pcm16kToMulaw,
  calculateRMS,
  decodeBase64Audio,
  encodeBase64Audio,
  detectBeep,
} from "./audio-utils";
import { createSTTProvider, createTTSProvider, createLLMProvider } from "../providers";
import type { STTProvider } from "../providers/stt/interface";
import type { TTSProvider } from "../providers/tts/interface";
import type { LLMProvider, LLMMessage, LLMToolDefinition, LLMToolCall } from "../providers/llm/interface";
import { executeTool, type ToolDefinition, type ToolResult } from "./tool-executor";
import { incrementOllama, decrementOllama, getOllamaActiveRequests, getOllamaMaxParallel } from "../ollama-concurrency";

// ---- Inline utilities to avoid external dependencies ----

/** Simple {{variable}} substitution (no Liquid dependency needed) */
function substituteVariables(text: string, variables: Record<string, string>): string {
  if (!text) return text;
  const context: Record<string, string> = {
    ...variables,
    now: new Date().toISOString(),
  };
  return text.replace(/\{\{(\w+)\}\}/g, (_, key) => context[key] || "");
}

/** Cost breakdown for a call */
export interface CostBreakdown {
  stt: number;
  llm: number;
  tts: number;
  transport: number;
  analysis: number;
  overflowLlm: number;
  total: number;
}

/** Per-call cost tracking */
class CostTracker {
  private sttMinutes = 0;
  private llmInputTokens = 0;
  private llmOutputTokens = 0;
  private overflowLlmInputTokens = 0;
  private overflowLlmOutputTokens = 0;
  private overflowLlmModel: string | null = null;
  private ttsCharacters = 0;
  private transportMinutes = 0;

  private sttProvider: string;
  private llmModel: string;
  private ttsProvider: string;

  private static readonly PRICING = {
    stt: {
      deepgram: parseFloat(process.env.COST_STT_DEEPGRAM || "0.0059"),
      vosk: 0,
    } as Record<string, number>,
    llm: {
      "gpt-4o": { input: 0.0025, output: 0.01 },
      "gpt-4o-mini": { input: 0.00015, output: 0.0006 },
      "deepseek-chat": { input: 0.00014, output: 0.00028 },
      "llama-3.3-70b-versatile": { input: 0.00059, output: 0.00079 },
      "llama-3.1-8b-instant": { input: 0.00005, output: 0.00008 },
      "llama-3.3-70b": { input: 0.0001, output: 0.0001 },
      "meta-llama/Llama-3.3-70B-Instruct": { input: 0.00035, output: 0.0004 },
      "meta-llama/Meta-Llama-3.1-8B-Instruct": { input: 0.00003, output: 0.00005 },
    } as Record<string, { input: number; output: number }>,
    tts: {
      piper: 0.0,
      kokoro: 0.0,
      elevenlabs: parseFloat(process.env.COST_TTS_ELEVENLABS || "0.18"),
    } as Record<string, number>,
    transport: { twilio: parseFloat(process.env.COST_TRANSPORT_TWILIO || "0.014") },
  };

  constructor(providers: { stt: string; llm: string; tts: string }) {
    this.sttProvider = providers.stt;
    this.llmModel = providers.llm;
    this.ttsProvider = providers.tts;
  }

  addSTTUsage(durationSeconds: number): void {
    this.sttMinutes += durationSeconds / 60;
  }

  addLLMUsage(inputTokens: number, outputTokens: number): void {
    this.llmInputTokens += inputTokens;
    this.llmOutputTokens += outputTokens;
  }

  addOverflowLLMUsage(inputTokens: number, outputTokens: number, model: string): void {
    this.overflowLlmInputTokens += inputTokens;
    this.overflowLlmOutputTokens += outputTokens;
    this.overflowLlmModel = model;
  }

  addTTSUsage(characterCount: number): void {
    this.ttsCharacters += characterCount;
  }

  addTransportUsage(durationSeconds: number): void {
    this.transportMinutes += durationSeconds / 60;
  }

  getCost(): CostBreakdown {
    const sttRate = CostTracker.PRICING.stt[this.sttProvider] ?? 0;
    const sttCost = this.sttMinutes * sttRate;

    const llmPricing = CostTracker.PRICING.llm[this.llmModel] ?? { input: 0, output: 0 };
    const llmCost =
      (this.llmInputTokens / 1000) * llmPricing.input +
      (this.llmOutputTokens / 1000) * llmPricing.output;

    const overflowPricing = this.overflowLlmModel
      ? (CostTracker.PRICING.llm[this.overflowLlmModel] ?? { input: 0.001, output: 0.002 })
      : { input: 0, output: 0 };
    const overflowLlmCost =
      (this.overflowLlmInputTokens / 1000) * overflowPricing.input +
      (this.overflowLlmOutputTokens / 1000) * overflowPricing.output;

    const ttsRate = CostTracker.PRICING.tts[this.ttsProvider] ?? 0;
    const ttsCost = (this.ttsCharacters / 1000) * ttsRate;

    const transportCost = this.transportMinutes * CostTracker.PRICING.transport.twilio;

    const total = sttCost + llmCost + overflowLlmCost + ttsCost + transportCost;

    return {
      stt: Math.round(sttCost * 10000) / 10000,
      llm: Math.round(llmCost * 10000) / 10000,
      tts: Math.round(ttsCost * 10000) / 10000,
      transport: Math.round(transportCost * 10000) / 10000,
      analysis: 0, // Set post-call by notifyCallEnded
      overflowLlm: Math.round(overflowLlmCost * 10000) / 10000,
      total: Math.round(total * 10000) / 10000,
    };
  }
}

/** Voicemail/IVR detector — hybrid audio + transcript scoring */
type VMDetectionResult = "human" | "voicemail" | "unknown";

/** Expanded VM keyword list for STT fallback and real-time transcript detection */
const VM_KEYWORDS = [
  "leave a message", "leave your message", "record your message",
  "at the tone", "can't take your call", "not available",
  "voicemail", "after the beep", "record a message",
  "press pound", "hang up or press", "leave me a message",
  // VM menu/system phrases
  "press one", "press two", "press three",
  "you have reached", "not available right now",
  "leave your name", "leave a brief message",
  "at the beep", "is not available",
  "please leave", "the person you are calling",
  "the mailbox is full", "to replay your message",
  "to continue recording", "when you have finished recording",
  "record your name", "to send a fax",
  "your call has been forwarded",
  "no one is available to take your call",
];

class VoicemailDetector {
  // Audio analysis state
  private analysisWindowSeconds: number;
  private speechThreshold: number;
  private continuousSpeechFramesThreshold: number;
  private initialDelayFrames: number;
  private audioFrames: number[] = [];
  private speechFrameCount = 0;
  private silenceFrameCount = 0;
  private totalFrames = 0;
  private continuousSpeechRun = 0;
  private maxContinuousSpeech = 0;

  // Resolution state
  private resolved = false;
  private result: VMDetectionResult = "unknown";
  private resolveCallbacks: ((result: VMDetectionResult) => void)[] = [];

  // AMD / beep detection
  private attemptsRemaining: number;
  private pendingMachineStart = false;
  private amdBeepReceived = false;
  private beepWaitTimer: ReturnType<typeof setTimeout> | null = null;
  private silenceAfterSpeechFrames = 0;
  private waitingForBeepAfterSilence = false;
  private beepWaitAfterSilenceFrames = 0;
  private greetingEnded = false;
  private maxVoicemailWaitSeconds: number;

  // Transcript-based detection (Tasks 3, 4, 6)
  private transcriptWordCount = 0;
  private firstTranscriptTime = 0;
  private transcriptContainsVmKeyword = false;

  // Configurable thresholds (Task 7)
  private noSpeechTimeoutFrames: number;
  private wordCountThreshold: number;
  private confidenceThreshold: number;

  private callId: string;

  constructor(config: {
    analysisWindowSeconds?: number;
    speechThreshold?: number;
    continuousSpeechFramesThreshold?: number;
    twilioAmdResult?: string;
    initialDelaySeconds?: number;
    maxRetries?: number;
    provider?: string;
    maxVoicemailWaitSeconds?: number;
    noSpeechTimeoutSeconds?: number;
    wordCountThreshold?: number;
    confidenceThreshold?: number;
    callId?: string;
  } = {}) {
    this.analysisWindowSeconds = config.analysisWindowSeconds ?? 20; // Task 1: extended from 7 → 20
    this.speechThreshold = config.speechThreshold ?? 300;
    this.continuousSpeechFramesThreshold = config.continuousSpeechFramesThreshold ?? 60;
    this.initialDelayFrames = Math.round(((config.initialDelaySeconds ?? 1) * 1000) / 20);
    this.attemptsRemaining = config.maxRetries ?? 3;
    this.maxVoicemailWaitSeconds = config.maxVoicemailWaitSeconds ?? 25;
    this.noSpeechTimeoutFrames = Math.round(((config.noSpeechTimeoutSeconds ?? 5) * 1000) / 20); // Task 1: 5s default
    this.wordCountThreshold = config.wordCountThreshold ?? 6; // Task 3
    this.confidenceThreshold = config.confidenceThreshold ?? 40; // Task 6
    this.callId = config.callId ?? "unknown";

    // Skip audio analysis if provider is AMD-only (twilio or signalwire)
    const useAudio = config.provider !== "twilio" && config.provider !== "signalwire";

    if (config.twilioAmdResult) {
      const amd = config.twilioAmdResult.toLowerCase();
      if (amd === "machine_end_beep" || amd === "machine_end_silence" || amd === "machine_end_other") {
        this.result = "voicemail";
        this.resolved = true;
        this.amdBeepReceived = true;
      } else if (amd === "machine_start") {
        this.pendingMachineStart = true;
      } else if (amd === "human") {
        this.result = "human";
        this.resolved = true;
      }
    }

    // If provider is twilio-only and no AMD result yet, we'll wait for it to be forced later
    if (!useAudio && !this.resolved) {
      // Will rely on forceResult() being called from status callback
    }
  }

  /** Task 3: Feed real-time transcript to detector for word-count and keyword analysis */
  feedTranscript(text: string, isFinal: boolean): void {
    if (this.resolved) return;
    const words = text.trim().split(/\s+/).filter(w => w.length > 0).length;
    if (words > this.transcriptWordCount) {
      this.transcriptWordCount = words;
    }
    if (this.firstTranscriptTime === 0 && words > 0) {
      this.firstTranscriptTime = Date.now();
    }
    // Word count trigger: 6+ words in a single utterance within first 5s = likely VM
    const elapsed = this.firstTranscriptTime > 0 ? Date.now() - this.firstTranscriptTime : 0;
    if (words >= this.wordCountThreshold && elapsed < 5000 && !this.pendingMachineStart) {
      console.log(`[vm-detect:${this.callId}] Word count trigger: ${words} words in ${elapsed}ms — entering beep-wait`);
      this.pendingMachineStart = true;
    }
    // Task 6: Check for VM keywords in real-time transcripts
    if (isFinal && !this.transcriptContainsVmKeyword) {
      const lower = text.toLowerCase();
      if (VM_KEYWORDS.some(kw => lower.includes(kw))) {
        this.transcriptContainsVmKeyword = true;
        console.log(`[vm-detect:${this.callId}] VM keyword detected in transcript: "${text}"`);
        if (!this.pendingMachineStart) {
          this.pendingMachineStart = true;
        }
      }
    }
  }

  analyzeFrame(pcmAudio: Buffer): void {
    if (this.resolved) return;

    const rms = calculateRMS(pcmAudio);
    this.audioFrames.push(rms);
    this.totalFrames++;

    // If we got machine_start from AMD or audio/transcript analysis, wait for beep
    if (this.pendingMachineStart) {
      const isSpeech = rms > this.speechThreshold;

      // Task 5: Direct beep frequency detection
      // Only consider beep valid after 100+ speech frames (~2s of speech heard) —
      // prevents false positives from ring tones / early call noise.
      // Real voicemail beeps come AFTER the greeting (5-15s of speech).
      if (!this.amdBeepReceived && this.speechFrameCount >= 100 && detectBeep(pcmAudio)) {
        this.amdBeepReceived = true;
        console.log(`[vm-detect:${this.callId}] Beep tone detected via frequency analysis (after ${this.speechFrameCount} speech frames)`);
      }

      // If beep detected + now silence → resolve immediately
      if (this.amdBeepReceived && rms < this.speechThreshold) {
        this.resolve("voicemail");
        return;
      }

      if (isSpeech) {
        this.silenceAfterSpeechFrames = 0;
        this.speechFrameCount++;
        // If speech resumes during beep-wait, reset the post-silence beep watch
        if (this.waitingForBeepAfterSilence) {
          this.waitingForBeepAfterSilence = false;
          this.beepWaitAfterSilenceFrames = 0;
          console.log(`[vm-detect:${this.callId}] Speech resumed during beep-wait — resetting`);
        }
      } else {
        this.silenceAfterSpeechFrames++;
        // After hearing speech then 125+ frames of silence (2.5s) → greeting likely ended
        // Was 60 frames (1.2s) but mid-greeting pauses (1-2s) caused premature resolution
        // Don't resolve immediately — enter beep-watch phase to wait for the actual beep
        if (this.speechFrameCount > 50 && this.silenceAfterSpeechFrames > 125 && !this.waitingForBeepAfterSilence) {
          this.waitingForBeepAfterSilence = true;
          this.beepWaitAfterSilenceFrames = 0;
          console.log(`[vm-detect:${this.callId}] Greeting ended (2.5s silence) — waiting up to 5s for beep before speaking`);
        }

        // During beep-watch: keep listening for beep for up to 5s (250 frames)
        if (this.waitingForBeepAfterSilence) {
          this.beepWaitAfterSilenceFrames++;
          // Beep detected during watch → resolve immediately (beep + silence already confirmed)
          if (this.amdBeepReceived) {
            console.log(`[vm-detect:${this.callId}] Beep detected during post-silence watch — resolving`);
            this.resolve("voicemail");
            return;
          }
          // 5s (250 frames) elapsed with no beep — resolve anyway
          if (this.beepWaitAfterSilenceFrames >= 250) {
            console.log(`[vm-detect:${this.callId}] No beep after 5s post-silence — resolving as voicemail`);
            this.resolve("voicemail");
            return;
          }
        }
      }

      // Safety timeout: after maxVoicemailWaitSeconds (default 25s), just speak
      const maxWaitFrames = (this.maxVoicemailWaitSeconds * 1000) / 20;
      if (this.totalFrames >= maxWaitFrames) {
        this.resolve("voicemail");
      }
      return;
    }

    // Skip analysis during initial delay
    if (this.totalFrames < this.initialDelayFrames) return;

    const isSpeech = rms > this.speechThreshold;

    if (isSpeech) {
      this.speechFrameCount++;
      this.continuousSpeechRun++;
      this.silenceFrameCount = 0;
      if (this.continuousSpeechRun > this.maxContinuousSpeech) {
        this.maxContinuousSpeech = this.continuousSpeechRun;
      }
      if (this.continuousSpeechRun >= this.continuousSpeechFramesThreshold) {
        // Don't resolve immediately — mark as detected but wait for beep/silence
        this.pendingMachineStart = true;
        this.speechFrameCount = 0;
        this.silenceAfterSpeechFrames = 0;
        return;
      }
    } else {
      this.silenceFrameCount++;
      if (this.silenceFrameCount > 5) {
        this.continuousSpeechRun = 0;
      }
      // Early human detection: short speech burst followed by 1s+ silence
      if (this.speechFrameCount > 5 && this.silenceFrameCount >= 50) {
        // Task 4: Check if transcript suggests VM despite short audio pattern
        if (this.transcriptWordCount >= this.wordCountThreshold) {
          console.log(`[vm-detect:${this.callId}] Early human overridden: ${this.transcriptWordCount} words detected`);
          this.pendingMachineStart = true;
          return;
        }
        this.resolve("human");
        return;
      }
    }

    // Task 1: No-speech timeout — if 5s elapsed with ZERO speech, resolve as human
    // A silent answer is more likely human than a silent voicemail
    if (this.totalFrames >= this.initialDelayFrames + this.noSpeechTimeoutFrames && this.speechFrameCount === 0) {
      console.log(`[vm-detect:${this.callId}] No speech after ${this.noSpeechTimeoutFrames * 20}ms — resolving human`);
      this.resolve("human");
      return;
    }

    // Analysis window timeout — make final decision
    const maxFrames = (this.analysisWindowSeconds * 1000) / 20;
    if (this.totalFrames >= maxFrames) {
      this.makeDecision();
    }
  }

  /** Task 6: Hybrid confidence scoring replacing binary decision */
  private makeDecision(): void {
    if (this.resolved) return;
    const speechRatio = this.speechFrameCount / Math.max(this.totalFrames, 1);

    let score = 0; // 0 = definitely human, 100 = definitely voicemail

    // Audio features (max 60 points)
    if (speechRatio > 0.65) score += 20;
    if (speechRatio > 0.45) score += 10;
    if (this.maxContinuousSpeech > 60) score += 20; // 1.2s+ continuous
    if (this.maxContinuousSpeech > 30) score += 10; // 0.6s+ continuous

    // Transcript features (max 40 points)
    if (this.transcriptWordCount >= 10) score += 25;
    else if (this.transcriptWordCount >= this.wordCountThreshold) score += 15;

    // Keyword match
    if (this.transcriptContainsVmKeyword) score += 30;

    // Beep detected
    if (this.amdBeepReceived) score += 30;

    const decision = score >= this.confidenceThreshold ? "voicemail" : "human";
    console.log(`[vm-detect:${this.callId}] Decision: ${decision} (score=${score}/${this.confidenceThreshold}, ` +
      `speechRatio=${speechRatio.toFixed(2)}, maxSpeech=${this.maxContinuousSpeech}, ` +
      `words=${this.transcriptWordCount}, beep=${this.amdBeepReceived}, keyword=${this.transcriptContainsVmKeyword})`);

    this.resolve(decision);
  }

  private resolve(result: VMDetectionResult): void {
    if (this.resolved) return;
    this.resolved = true;
    this.result = result;
    console.log(`[vm-detect:${this.callId}] Resolved: ${result}`);
    for (const cb of this.resolveCallbacks) cb(result);
    this.resolveCallbacks = [];
  }

  getResult(): Promise<VMDetectionResult> {
    if (this.resolved) return Promise.resolve(this.result);
    return new Promise((resolve) => { this.resolveCallbacks.push(resolve); });
  }

  isResolved(): boolean { return this.resolved; }
  getCurrentResult(): VMDetectionResult { return this.result; }
  forceResult(result: VMDetectionResult): void { this.resolve(result); }

  /** Handle AMD result that arrives after construction (via /amd-result endpoint) */
  forceAmdResult(answeredBy: string): void {
    if (this.resolved) return;
    const amd = answeredBy.toLowerCase();
    if (amd === "machine_end_beep" || amd === "machine_end_silence" || amd === "machine_end_other") {
      this.resolve("voicemail");
    } else if (amd === "machine_start") {
      this.pendingMachineStart = true;
      console.log(`[vm-detect:${this.callId}] AMD machine_start received — waiting for beep/silence`);
    } else if (amd === "human") {
      this.resolve("human");
    }
  }
}

// ---- Session types ----

type SessionState =
  | "initializing"
  | "waiting_for_speech"
  | "listening"
  | "processing"
  | "speaking"
  | "transferring"
  | "ended";

export interface CallSessionConfig {
  callId: string;
  assistantId: string;
  systemPrompt: string;
  firstMessage?: string;
  firstMessageMode?: string;
  model: { provider: string; model: string; temperature?: number; maxTokens?: number; baseUrl?: string };
  voice: { provider: string; voiceId: string; model?: string; stability?: number };
  transcriber: { provider: string; model?: string; language?: string; keywords?: string[] };
  fallbackTranscriber?: { provider: string; model?: string; language?: string; keywords?: string[] };
  tools: ToolDefinition[];
  toolMode?: string; // "tools" | "trigger-phrases"
  triggerPhrases?: { phrase: string; toolName: string }[];
  endCallPhrases: string[];
  maxDuration: number;
  silenceTimeout: number;
  voicemailMessage?: string;
  voicemailDetectionEnabled?: boolean;
  voicemailDetectionConfig?: {
    provider?: string; // "twilio" | "audio" | "both"
    initialDelaySeconds?: number;
    analysisWindowSeconds?: number;
    maxRetries?: number;
    maxVoicemailWaitSeconds?: number;
    speechThreshold?: number;
    noSpeechTimeoutSeconds?: number;
    wordCountThreshold?: number;
    confidenceThreshold?: number;
  };
  customerNumber: string;
  customerName?: string;
  variableValues?: Record<string, string>;
  metadata?: Record<string, unknown>;
  fallbackDestination?: string;
  serverUrl?: string;
  backgroundDenoising?: boolean;
  startSpeakingPlan?: { waitSeconds?: number; smartEndpointingEnabled?: boolean };
  stopSpeakingPlan?: { numWords?: number; voiceSeconds?: number; backoffSeconds?: number };
  provider?: "twilio" | "signalwire";
  signalwireSpaceUrl?: string;
  twilioAmdResult?: string;
  analysisConfig?: {
    summaryEnabled?: boolean;
    summaryPrompt?: string;
    successEvaluationEnabled?: boolean;
    successEvaluationPrompt?: string;
    successEvaluationRubric?: string;
  };
  /** Pre-synthesized first message PCM (16kHz 16-bit mono) — synthesized at register-call time */
  preSynthesizedFirstMessage?: Buffer;
  /** Pre-synthesized first message as raw mulaw 8kHz — from audio cache file */
  preSynthesizedFirstMessageMulaw?: Buffer;
  /** Use cached audio first message (non-interruptable, instant playback) */
  audioFirstMessage?: boolean;
}

export class CallSession extends EventEmitter {
  private state: SessionState = "initializing";
  private config: CallSessionConfig;
  private stt: STTProvider | null = null;
  private tts: TTSProvider | null = null;
  private llm: LLMProvider | null = null;

  private conversationHistory: LLMMessage[] = [];
  private currentTranscript = "";
  private fullTranscript: { role: string; content: string }[] = [];

  private silenceTimer: ReturnType<typeof setTimeout> | null = null;
  private maxDurationTimer: ReturnType<typeof setTimeout> | null = null;
  private endpointingTimer: ReturnType<typeof setTimeout> | null = null;

  private currentTTSCancel: (() => void) | null = null;
  private currentLLMCancel: (() => void) | null = null;
  private isSpeaking = false;
  private deliveringVoicemail = false;
  private isEndingCall = false;

  private costTracker: CostTracker;
  public voicemailDetector: VoicemailDetector | null = null;
  private startedAt: Date | null = null;
  private audioChunkCount = 0;
  private consecutiveSTTErrors = 0;
  private sttFallbackAttempted = false;

  // Speaking plan state
  private speechStartedAt: number | null = null;
  private interimWordCount = 0;
  private utteranceDelayTimer: ReturnType<typeof setTimeout> | null = null;
  private playingFirstMessage = false; // true while first message TTS is playing — ignore user speech
  private audioFirstMessagePlaying = false; // true during cached audio playback — fully non-interruptable
  private pendingAudioDurationMs = 0; // tracks audio duration queued but not yet played by Twilio
  private lastBargeInAt: number | null = null; // timestamp of last barge-in for backoff

  // Overflow LLM tracking
  private overflowCount = 0; // how many LLM turns used overflow this call
  private overflowProvider: string | null = null;
  private overflowModel: string | null = null;
  private currentlyUsingOverflow = false; // true if the current/last LLM request used overflow

  // Ollama slot tracking — ensures slot is always released even if LLM onDone never fires
  private pendingOllamaRelease: (() => void) | null = null;

  constructor(config: CallSessionConfig) {
    super();
    this.config = config;
    this.costTracker = new CostTracker({
      stt: config.transcriber.provider,
      llm: config.model.model,
      tts: config.voice.provider,
    });
  }

  async start(): Promise<void> {
    this.startedAt = new Date();
    this.state = "initializing";

    // Initialize providers
    this.stt = createSTTProvider(this.config.transcriber);
    this.tts = createTTSProvider(this.config.voice);
    this.llm = createLLMProvider(this.config.model);

    // Build system prompt with variable substitution
    let systemPrompt = this.config.systemPrompt;
    if (this.config.variableValues) {
      systemPrompt = substituteVariables(systemPrompt, this.config.variableValues);
    }

    this.conversationHistory = [{ role: "system", content: systemPrompt }];

    // Set up STT event handlers
    this.stt.on("transcript", (data: { text: string; isFinal: boolean }) => {
      this.handleTranscript(data.text, data.isFinal);
    });

    this.stt.on("speech_started", () => {
      this.handleSpeechStarted();
    });

    this.stt.on("utterance_end", () => {
      this.handleUtteranceEnd();
    });

    this.stt.on("error", (err: Error) => {
      this.consecutiveSTTErrors++;
      console.error(`[session:${this.config.callId}] STT error (${this.consecutiveSTTErrors}/3):`, err.message);
      if (this.consecutiveSTTErrors >= 3) {
        // Attempt fallback to secondary STT provider before ending call
        if (!this.sttFallbackAttempted && this.config.fallbackTranscriber) {
          this.switchToFallbackSTT();
        } else {
          console.error(`[session:${this.config.callId}] STT errors exhausted (no fallback available), ending call`);
          this.endCall("stt-failure");
        }
      }
    });

    // Start STT in background — don't block first message on Deepgram WebSocket connect
    const sttReady = this.stt.start();

    // Set max duration timer
    this.maxDurationTimer = setTimeout(() => {
      this.endCall("max-duration-reached");
    }, this.config.maxDuration * 1000);

    // Initialize voicemail detection if enabled
    if (this.config.voicemailDetectionEnabled) {
      const vmConfig = this.config.voicemailDetectionConfig || {};
      this.voicemailDetector = new VoicemailDetector({
        twilioAmdResult: this.config.twilioAmdResult,
        analysisWindowSeconds: vmConfig.analysisWindowSeconds,
        initialDelaySeconds: vmConfig.initialDelaySeconds,
        maxRetries: vmConfig.maxRetries,
        maxVoicemailWaitSeconds: vmConfig.maxVoicemailWaitSeconds,
        provider: vmConfig.provider,
        speechThreshold: vmConfig.speechThreshold,
        noSpeechTimeoutSeconds: vmConfig.noSpeechTimeoutSeconds,
        wordCountThreshold: vmConfig.wordCountThreshold,
        confidenceThreshold: vmConfig.confidenceThreshold,
        callId: this.config.callId,
      });

      if (this.voicemailDetector.isResolved() && this.voicemailDetector.getCurrentResult() === "voicemail") {
        await sttReady; // Ensure STT is ready before returning
        this.handleVoicemailDetected();
        return;
      }

      if (!this.voicemailDetector.isResolved()) {
        this.voicemailDetector.getResult().then((result) => {
          if (result === "voicemail" && this.state !== "ended") {
            this.handleVoicemailDetected();
          }
        });
      }
    }

    // Handle first message — fire TTS immediately (parallel with STT connect)
    if (
      this.config.firstMessage &&
      this.config.firstMessageMode !== "assistant-waits-for-user"
    ) {
      let firstMsg = this.config.firstMessage;
      if (this.config.variableValues) {
        firstMsg = substituteVariables(firstMsg, this.config.variableValues);
      }

      // Helper: deliver first message using cached mulaw, pre-synth PCM, or live TTS
      const deliverFirstMessage = () => {
        this.playingFirstMessage = true;
        if (this.config.preSynthesizedFirstMessageMulaw) {
          // Cached audio file — fastest path, non-interruptable
          this.audioFirstMessagePlaying = true;
          console.log(`[session:${this.config.callId}] Using cached audio first message (mulaw, non-interruptable)`);
          this.speakPreSynthesizedMulaw(this.config.preSynthesizedFirstMessageMulaw);
        } else if (this.config.preSynthesizedFirstMessage) {
          console.log(`[session:${this.config.callId}] Using pre-synthesized first message (cached at register-call)`);
          this.speakPreSynthesized(this.config.preSynthesizedFirstMessage, firstMsg);
        } else {
          this.speak(firstMsg);
        }
      };

      // If voicemail detection is active and not yet resolved, hold first message
      if (this.voicemailDetector && !this.voicemailDetector.isResolved()) {
        console.log(`[session:${this.config.callId}] Holding first message — voicemail detection pending`);
        this.state = "waiting_for_speech";
        this.resetSilenceTimer();
        this.voicemailDetector.getResult().then((result) => {
          if (result === "human" && this.state !== "ended") {
            console.log(`[session:${this.config.callId}] Human confirmed — delivering first message`);
            deliverFirstMessage();
            this.conversationHistory.push({ role: "assistant", content: firstMsg });
            this.fullTranscript.push({ role: "AI", content: firstMsg });
            const estimatedPlaybackMs = Math.max(firstMsg.length * 60, 5000);
            setTimeout(() => {
              this.playingFirstMessage = false;
              this.audioFirstMessagePlaying = false;
              this.currentTranscript = "";
              this.state = "waiting_for_speech";
              this.resetSilenceTimer();
              console.log(`[session:${this.config.callId}] First message playback window ended (${estimatedPlaybackMs}ms)`);
            }, estimatedPlaybackMs);
          }
          // If voicemail — handleVoicemailDetected() already handles the voicemail message
        });
      } else {
      deliverFirstMessage();
      // Don't rely on TTS completion callback to clear playingFirstMessage —
      // TTS synthesis finishes in ~1s but audio plays for 10-15s on the phone.
      // Estimate playback: ~60ms per character is a rough TTS duration heuristic.
      const estimatedPlaybackMs = Math.max(firstMsg.length * 60, 5000);
      setTimeout(() => {
        this.playingFirstMessage = false;
        this.audioFirstMessagePlaying = false;
        this.currentTranscript = "";
        this.state = "waiting_for_speech";
        this.resetSilenceTimer();
        console.log(`[session:${this.config.callId}] First message playback window ended (${estimatedPlaybackMs}ms)`);
      }, estimatedPlaybackMs);
      this.conversationHistory.push({ role: "assistant", content: firstMsg });
      this.fullTranscript.push({ role: "AI", content: firstMsg });
      }
    } else {
      this.state = "waiting_for_speech";
      this.resetSilenceTimer();
    }

    // Ensure STT is connected before we start processing user audio
    await sttReady;
  }

  // Debug: capture first 5s of PCM audio to WAV for analysis
  private debugAudioChunks: Buffer[] = [];
  private debugAudioSaved = false;

  /** Feed raw Twilio mu-law base64 audio into the pipeline */
  handleAudio(base64Audio: string): void {
    if (this.state === "ended" || this.state === "initializing") return;

    const mulaw = decodeBase64Audio(base64Audio);

    this.audioChunkCount++;

    // Debug: capture audio at 15-20s mark (750-1000 chunks) — after first message, when user speaks
    const CAPTURE_START = 750;  // 15 seconds in
    const CAPTURE_END = 1000;   // 20 seconds in
    if (!this.debugAudioSaved && this.audioChunkCount >= CAPTURE_START && this.audioChunkCount <= CAPTURE_END) {
      this.debugAudioChunks.push(mulawToPcm16k(mulaw));
      // Also save raw mulaw for comparison
      if (this.audioChunkCount === CAPTURE_START) {
        console.log(`[session:${this.config.callId}] DEBUG: Starting audio capture at chunk ${CAPTURE_START} (${CAPTURE_START*20/1000}s into call)`);
        console.log(`[session:${this.config.callId}] DEBUG: mulaw chunk size=${mulaw.length}, mulaw bytes=[${Array.from(mulaw.slice(0,20)).join(',')}]`);
      }
      if (this.audioChunkCount === CAPTURE_END) {
        this.debugAudioSaved = true;
        const allPcm = Buffer.concat(this.debugAudioChunks);
        // Write WAV header + PCM data
        const wavHeader = Buffer.alloc(44);
        wavHeader.write("RIFF", 0);
        wavHeader.writeUInt32LE(36 + allPcm.length, 4);
        wavHeader.write("WAVE", 8);
        wavHeader.write("fmt ", 12);
        wavHeader.writeUInt32LE(16, 16); // chunk size
        wavHeader.writeUInt16LE(1, 20); // PCM
        wavHeader.writeUInt16LE(1, 22); // mono
        wavHeader.writeUInt32LE(16000, 24); // sample rate
        wavHeader.writeUInt32LE(32000, 28); // byte rate
        wavHeader.writeUInt16LE(2, 32); // block align
        wavHeader.writeUInt16LE(16, 34); // bits per sample
        wavHeader.write("data", 36);
        wavHeader.writeUInt32LE(allPcm.length, 40);
        const wav = Buffer.concat([wavHeader, allPcm]);
        require("fs").writeFileSync(`/tmp/debug_call_audio_${this.config.callId.slice(0,8)}.wav`, wav);
        console.log(`[session:${this.config.callId}] DEBUG: Saved 5s audio to /tmp/debug_call_audio_${this.config.callId.slice(0,8)}.wav (${allPcm.length} bytes PCM, ${allPcm.length/32000}s)`);
        this.debugAudioChunks = [];
      }
    }

    // Track STT cost: each chunk is 20ms of audio
    this.costTracker.addSTTUsage(0.02);

    // If STT accepts raw mulaw (e.g. Deepgram), send it directly — better quality, no conversion
    if (this.stt && (this.stt as any).config?.acceptsMulaw) {
      this.stt.send(mulaw);

      // Voicemail detector still needs PCM
      if (this.voicemailDetector && !this.voicemailDetector.isResolved()) {
        const pcm = mulawToPcm16k(mulaw);
        this.voicemailDetector.analyzeFrame(pcm);
      }
      return;
    }

    // Default path: convert to PCM 16kHz for local STT
    const pcm = mulawToPcm16k(mulaw);

    // Feed to voicemail detector if active
    if (this.voicemailDetector && !this.voicemailDetector.isResolved()) {
      this.voicemailDetector.analyzeFrame(pcm);
    }

    // Feed to STT
    if (this.stt) {
      this.stt.send(pcm);
    }
  }

  /** Switch to fallback STT provider mid-call when primary fails */
  private switchToFallbackSTT(): void {
    if (!this.config.fallbackTranscriber || this.sttFallbackAttempted) return;
    this.sttFallbackAttempted = true;

    const fallbackConfig = this.config.fallbackTranscriber;
    console.warn(
      `[session:${this.config.callId}] Switching to fallback STT: ${fallbackConfig.provider}/${fallbackConfig.model || "default"}`
    );

    // Close current STT — remove listeners first to prevent stale handler accumulation
    if (this.stt) {
      this.stt.removeAllListeners();
      this.stt.close();
    }

    // Reset error counter
    this.consecutiveSTTErrors = 0;

    // Create new STT provider from fallback config
    this.stt = createSTTProvider(fallbackConfig);

    // Re-attach event handlers
    this.stt.on("transcript", (data: { text: string; isFinal: boolean }) => {
      this.handleTranscript(data.text, data.isFinal);
    });

    this.stt.on("speech_started", () => {
      this.handleSpeechStarted();
    });

    this.stt.on("utterance_end", () => {
      this.handleUtteranceEnd();
    });

    this.stt.on("error", (err: Error) => {
      this.consecutiveSTTErrors++;
      console.error(`[session:${this.config.callId}] Fallback STT error (${this.consecutiveSTTErrors}/3):`, err.message);
      if (this.consecutiveSTTErrors >= 3) {
        console.error(`[session:${this.config.callId}] Fallback STT also failed, ending call`);
        this.endCall("stt-failure");
      }
    });

    // Start the fallback STT
    this.stt.start().catch((err) => {
      console.error(`[session:${this.config.callId}] Failed to start fallback STT:`, err.message);
      this.endCall("stt-failure");
    });

    // Update cost tracker provider
    this.costTracker = new CostTracker({
      stt: fallbackConfig.provider,
      llm: this.config.model.model,
      tts: this.config.voice.provider,
    });
  }

  private handleSpeechStarted(): void {
    // Record when speech started for stopSpeakingPlan thresholds
    this.speechStartedAt = Date.now();
    this.interimWordCount = 0;

    // Cached audio first message — fully non-interruptable, ignore all speech
    if (this.audioFirstMessagePlaying) return;

    // During first message: always defer to handleTranscript for threshold-based barge-in
    // Short acknowledgements ("right", "yes") will be discarded; 3+ words will barge in
    if (this.playingFirstMessage) return;

    // Don't immediately cancel TTS — wait for enough speech to confirm barge-in
    // The actual barge-in decision happens in handleTranscript when we have word counts
    const minWords = this.config.stopSpeakingPlan?.numWords ?? 0;
    const minVoiceSeconds = this.config.stopSpeakingPlan?.voiceSeconds ?? 0;
    if (!this.isSpeaking || (minWords === 0 && minVoiceSeconds === 0)) {
      // No stopSpeakingPlan configured or not speaking — use legacy behavior
      if (this.isSpeaking) {
        this.cancelSpeaking();
      }
    }
    // Otherwise: defer barge-in to handleTranscript where we check thresholds

    this.resetSilenceTimer();
  }

  private handleTranscript(text: string, isFinal: boolean): void {
    if (this.state === "ended") return;

    // Cached audio first message — fully non-interruptable, suppress all transcripts
    if (this.audioFirstMessagePlaying) return;

    // Feed transcript to VM detector for word-count and keyword analysis (before suppression)
    if (this.voicemailDetector && !this.voicemailDetector.isResolved()) {
      this.voicemailDetector.feedTranscript(text, isFinal);
    }

    // Suppress transcripts while voicemail detection is pending OR after voicemail confirmed.
    // Without this, the greeting audio gets transcribed and the LLM responds to it.
    if (this.voicemailDetector && (!this.voicemailDetector.isResolved() || this.voicemailDetector.getCurrentResult() === "voicemail")) {
      return;
    }

    // Reset STT error counter on successful transcription
    this.consecutiveSTTErrors = 0;

    // Fallback: if detection resolved as "human" but transcript contains VM keywords,
    // retroactively treat as voicemail (catches short greetings the audio analyzer missed)
    if (this.voicemailDetector && this.voicemailDetector.getCurrentResult() === "human" &&
        this.playingFirstMessage && isFinal) {
      const lower = text.toLowerCase();
      const vmKeywords = VM_KEYWORDS;
      if (vmKeywords.some(kw => lower.includes(kw))) {
        console.log(`[session:${this.config.callId}] VM keyword detected in transcript: "${text}" — switching to voicemail`);
        this.playingFirstMessage = false;
        this.cancelSpeaking();
        this.voicemailDetector.forceResult("voicemail");
        this.handleVoicemailDetected();
        return;
      }
    }

    // During first message: only allow barge-in if user says enough words (stopSpeakingPlan threshold)
    // Short acknowledgements like "right", "yes", "hello" are discarded
    if (this.playingFirstMessage) {
      if (!isFinal) return; // Only check on final transcripts

      const testAccumulated = (this.currentTranscript + " " + text).trim();
      const wordCount = testAccumulated.split(/\s+/).filter(Boolean).length;
      const minWords = this.config.stopSpeakingPlan?.numWords ?? 3;

      if (wordCount >= minWords) {
        // User said enough — barge in, stop first message, treat as real speech
        this.playingFirstMessage = false;
        this.currentTranscript += (this.currentTranscript ? " " : "") + text;
        this.cancelSpeaking();
        console.log(`[session:${this.config.callId}] First message barge-in: "${testAccumulated}" (${wordCount} words)`);
      }
      // Otherwise: discard — don't accumulate short words for later
      return;
    }

    // Check stopSpeakingPlan barge-in thresholds while assistant is speaking
    if (this.isSpeaking) {
      const accumulated = (this.currentTranscript + " " + text).trim();
      this.interimWordCount = accumulated.split(/\s+/).filter(Boolean).length;

      const minWords = this.config.stopSpeakingPlan?.numWords ?? 0;
      const minVoiceMs = (this.config.stopSpeakingPlan?.voiceSeconds ?? 0) * 1000;
      const speechDuration = this.speechStartedAt ? Date.now() - this.speechStartedAt : 0;

      if (this.interimWordCount >= minWords && speechDuration >= minVoiceMs) {
        this.cancelSpeaking();
      }
    }

    if (isFinal) {
      this.currentTranscript += (this.currentTranscript ? " " : "") + text;

      // Check for end call phrases
      const lower = this.currentTranscript.toLowerCase();
      for (const phrase of this.config.endCallPhrases) {
        if (lower.includes(phrase.toLowerCase())) {
          this.endCall("assistant-ended-call");
          return;
        }
      }
    }

    this.emit("transcript", { text, isFinal, accumulated: this.currentTranscript });
    this.resetSilenceTimer();
  }

  private handleUtteranceEnd(): void {
    if (this.state === "ended" || !this.currentTranscript.trim()) return;
    // Suppress while voicemail detection pending
    if (this.voicemailDetector && !this.voicemailDetector.isResolved()) return;

    const waitSeconds = this.config.startSpeakingPlan?.waitSeconds ?? 0;

    // Clear any existing utterance delay timer (resets if more speech arrives)
    if (this.utteranceDelayTimer) {
      clearTimeout(this.utteranceDelayTimer);
      this.utteranceDelayTimer = null;
    }

    if (waitSeconds > 0) {
      // Delay before processing — gives user time to continue speaking
      this.utteranceDelayTimer = setTimeout(() => {
        this.utteranceDelayTimer = null;
        this.dispatchUtterance();
      }, waitSeconds * 1000);
    } else {
      this.dispatchUtterance();
    }
  }

  private dispatchUtterance(): void {
    if (this.state === "ended" || !this.currentTranscript.trim()) return;

    const userText = this.currentTranscript.trim();
    this.currentTranscript = "";
    this.speechStartedAt = null;
    this.interimWordCount = 0;

    this.state = "processing";
    this.fullTranscript.push({ role: "User", content: userText });
    this.conversationHistory.push({ role: "user", content: userText });

    this.emit("user_speech", userText);

    // Apply backoff delay after a barge-in before assistant responds
    const backoffSeconds = this.config.stopSpeakingPlan?.backoffSeconds ?? 0;
    if (backoffSeconds > 0 && this.lastBargeInAt) {
      const elapsed = (Date.now() - this.lastBargeInAt) / 1000;
      const remaining = backoffSeconds - elapsed;
      if (remaining > 0) {
        console.log(`[session:${this.config.callId}] Backoff: waiting ${remaining.toFixed(1)}s after barge-in before responding`);
        setTimeout(() => {
          if (this.state !== "ended") this.processWithLLM();
        }, remaining * 1000);
        return;
      }
    }

    this.processWithLLM();
  }

  /** Check AI response text for trigger phrases and execute matching tools */
  private checkTriggerPhrases(text: string): void {
    if (!this.config.triggerPhrases || this.config.triggerPhrases.length === 0) return;
    const lower = text.toLowerCase();
    // Find the trigger phrase that appears earliest in the response text
    // (not earliest in array) so "we'll give you a callback. Have a wonderful day"
    // correctly triggers callback, not endCall
    let bestMatch: typeof this.config.triggerPhrases[0] | null = null;
    let bestIndex = lower.length;
    for (const tp of this.config.triggerPhrases) {
      const idx = lower.indexOf(tp.phrase.toLowerCase());
      if (idx >= 0 && idx < bestIndex) {
        bestIndex = idx;
        bestMatch = tp;
      }
    }
    if (bestMatch) {
      const tp = bestMatch;
      {
        console.log(`[session:${this.config.callId}] Trigger phrase matched: "${tp.phrase}" → ${tp.toolName} (position ${bestIndex})`);
        // Find the tool definition
        const tool = this.config.tools.find(t => t.name === tp.toolName || t.type === tp.toolName);
        if (tool) {
          // Synthesize a tool call — wait for TTS to finish so caller hears the phrase
          const toolCall: LLMToolCall = {
            id: `trigger-${Date.now()}`,
            type: "function",
            function: { name: tool.name || tp.toolName, arguments: "{}" },
          };
          this.waitForTTSFinish().then(async () => {
            if (this.isEndingCall) return;
            // Extra buffer for network transit — audio may still be in Twilio/SignalWire buffer
            await new Promise(r => setTimeout(r, 1500));
            if (this.isEndingCall) return;
            this.handleToolCall(toolCall);
          });
        } else if (tp.toolName === "end_call" || tp.toolName === "endCall") {
          // Built-in end call — wait for full TTS playback before hanging up
          this.waitForTTSFinish().then(async () => {
            if (this.isEndingCall) return;
            // Wait for pending audio to play on the phone + network buffer
            const audioWait = Math.max(this.pendingAudioDurationMs + 2000, 1500);
            await new Promise(r => setTimeout(r, audioWait));
            if (this.isEndingCall) return;
            this.endCall("assistant-ended-call");
          });
        } else if (tp.toolName === "transfer" || tp.toolName === "transferCall") {
          // Built-in transfer with fallback destination
          this.waitForTTSFinish().then(async () => {
            if (this.isEndingCall || this.state === "transferring") return;
            await new Promise(r => setTimeout(r, 1500));
            if (this.isEndingCall) return;
            this.state = "transferring";
            this.emit("transfer", { destination: this.config.fallbackDestination });
          });
        }
      }
    }
  }

  private processWithLLM(): void {
    if (!this.llm) return;

    // --- Overflow LLM routing ---
    const isOllama = this.config.model.provider === "ollama" || !this.config.model.provider;
    let llmToUse: LLMProvider = this.llm;
    let usingOverflow = false;
    let decrementOnComplete = false;

    if (isOllama) {
      const active = getOllamaActiveRequests();
      const max = getOllamaMaxParallel();
      const overflowProvider = process.env.OVERFLOW_LLM_PROVIDER;
      const overflowModel = process.env.OVERFLOW_LLM_MODEL;

      if (active >= max && overflowProvider && overflowModel) {
        // Overflow to cloud API
        console.log(`[session:${this.config.callId}] Overflow triggered: ollama ${active}/${max} → ${overflowProvider}/${overflowModel}`);
        llmToUse = createLLMProvider({
          provider: overflowProvider,
          model: overflowModel,
          temperature: this.config.model.temperature,
          maxTokens: this.config.model.maxTokens,
        });
        usingOverflow = true;
        this.overflowCount++;
        this.overflowProvider = overflowProvider;
        this.overflowModel = overflowModel;
        this.currentlyUsingOverflow = true;
      } else {
        // Use Ollama — track the slot
        incrementOllama();
        decrementOnComplete = true;
        this.currentlyUsingOverflow = false;
      }
    }

    // In trigger-phrases mode, don't send tools to the LLM
    const useTriggerPhrases = this.config.toolMode === "trigger-phrases";
    let deferredToolCall: LLMToolCall | null = null;
    console.log(`[session:${this.config.callId}] LLM dispatch: toolMode=${this.config.toolMode || "not-set"}, tools=${this.config.tools.length}, triggerPhrases=${(this.config.triggerPhrases || []).length}, useTriggerPhrases=${useTriggerPhrases}`);

    const llmTools: LLMToolDefinition[] = useTriggerPhrases ? [] : this.config.tools
      .filter((t) => t.functionDefinition)
      .map((t) => ({
        type: "function" as const,
        function: {
          name: t.functionDefinition!.name,
          description: t.functionDefinition!.description,
          parameters: t.functionDefinition!.parameters,
        },
      }));

    // Only add built-in tool definitions when NOT using trigger phrases
    if (!useTriggerPhrases) for (const t of this.config.tools) {
      if (t.type === "endCall" && !t.functionDefinition) {
        llmTools.push({
          type: "function",
          function: {
            name: "end_call",
            description: "End the phone call",
            parameters: { type: "object", properties: {} },
          },
        });
      }
      if (t.type === "transferCall" && !t.functionDefinition) {
        llmTools.push({
          type: "function",
          function: {
            name: t.name,
            description: t.description || "Transfer the call",
            parameters: { type: "object", properties: {} },
          },
        });
      }
      if (t.type === "dtmf" && !t.functionDefinition) {
        llmTools.push({
          type: "function",
          function: {
            name: t.name,
            description: t.description || "Send DTMF tones",
            parameters: { type: "object", properties: {} },
          },
        });
      }
    }

    let fullResponse = "";
    let ignoredToolCall = false;
    let decremented = false;
    const releaseOllamaSlot = () => {
      if (decrementOnComplete && !decremented) {
        decremented = true;
        decrementOllama();
        this.pendingOllamaRelease = null;
      }
    };
    // Track so endCall() can release even if onDone never fires
    if (decrementOnComplete) this.pendingOllamaRelease = releaseOllamaSlot;

    const { cancel } = llmToUse.streamCompletion({
      messages: this.conversationHistory,
      tools: llmTools.length > 0 ? llmTools : undefined,
      onUsage: usingOverflow
        ? (input, output) => this.costTracker.addOverflowLLMUsage(input, output, this.overflowModel!)
        : (input, output) => this.costTracker.addLLMUsage(input, output),
      onToken: (token: string) => {
        fullResponse += token;
        if (this.shouldStartTTS(fullResponse)) {
          const sentence = this.extractCompleteSentence(fullResponse);
          if (sentence) {
            fullResponse = fullResponse.slice(sentence.length).trimStart();
            this.speak(sentence);
          }
        }
      },
      onToolCall: (toolCall: LLMToolCall) => {
        // In trigger-phrases mode, IGNORE all LLM tool calls — actions come from phrase matching only
        if (useTriggerPhrases) {
          console.log(`[session:${this.config.callId}] Ignoring LLM tool call in trigger-phrases mode: ${toolCall.function.name}`);
          ignoredToolCall = true;
          return;
        }
        // Flush any remaining text to TTS before handling the tool call
        // This ensures "I'll transfer you now" is spoken before ff_transfer fires
        const remaining = fullResponse.trim();
        if (remaining) {
          fullResponse = "";
          this.speak(remaining);
          this.handleToolCall(toolCall);
        } else {
          // LLM produced tool call with NO text — defer execution
          // Will retry for spoken text in onDone, then execute
          console.log(`[session:${this.config.callId}] LLM tool call with no text — deferring: ${toolCall.function.name}`);
          deferredToolCall = toolCall;
        }
      },
      onDone: (text: string) => {
        releaseOllamaSlot();
        const remaining = (fullResponse || text).trim();

        // If LLM produced only a tool call with no text in trigger-phrases mode,
        // retry with a clean conversation that has NO tool references
        if (useTriggerPhrases && ignoredToolCall && !remaining) {
          console.log(`[session:${this.config.callId}] LLM produced only tool call, no text — retrying with clean messages`);
          // Build clean messages without any tool-related entries
          const cleanMessages = this.conversationHistory.filter(
            (m: any) => m.role !== "tool" && !m.tool_calls && !m.tool_call_id
          );
          cleanMessages.push({
            role: "system",
            content: "Respond naturally to the customer. Do not attempt to call any functions or tools. Just speak.",
          });
          // Direct LLM call with NO tools parameter at all
          const retryResult = this.llm!.streamCompletion({
            messages: cleanMessages,
            onToken: (token: string) => {
              fullResponse += token;
              if (this.shouldStartTTS(fullResponse)) {
                const sentence = this.extractCompleteSentence(fullResponse);
                if (sentence) {
                  fullResponse = fullResponse.slice(sentence.length).trimStart();
                  this.speak(sentence);
                }
              }
            },
            onToolCall: () => {
              // Ignore any tool calls on retry
            },
            onDone: (retryText: string) => {
              const retryRemaining = (fullResponse || retryText).trim();
              if (retryRemaining && !this.isSpeaking) {
                this.speak(retryRemaining);
              }
              if (retryText) {
                this.conversationHistory.push({ role: "assistant", content: retryText });
                this.fullTranscript.push({ role: "AI", content: retryText });
                this.checkTriggerPhrases(retryText);
              }
              this.state = "waiting_for_speech";
              this.resetSilenceTimer();
            },
          });
          // Store cancel so endCall() can abort this retry
          this.currentLLMCancel = () => retryResult.cancel();
          return;
        }

        // If LLM produced only a tool call with no text in tools mode,
        // retry to get spoken text, then execute the deferred tool
        if (!useTriggerPhrases && deferredToolCall && !remaining) {
          console.log(`[session:${this.config.callId}] LLM produced only tool call, no text — retrying for speech before executing ${deferredToolCall.function.name}`);
          const savedToolCall = deferredToolCall;
          const cleanMessages = this.conversationHistory.filter(
            (m: any) => m.role !== "tool" && !m.tool_calls && !m.tool_call_id
          );
          cleanMessages.push({
            role: "system",
            content: "Respond naturally to the customer. Do not attempt to call any functions or tools. Just speak your response out loud.",
          });
          const retryResult2 = this.llm!.streamCompletion({
            messages: cleanMessages,
            onToken: (token: string) => {
              fullResponse += token;
              if (this.shouldStartTTS(fullResponse)) {
                const sentence = this.extractCompleteSentence(fullResponse);
                if (sentence) {
                  fullResponse = fullResponse.slice(sentence.length).trimStart();
                  this.speak(sentence);
                }
              }
            },
            onToolCall: () => { /* Ignore tool calls on retry */ },
            onDone: (retryText: string) => {
              const retryRemaining = (fullResponse || retryText).trim();
              if (retryRemaining && !this.isSpeaking) {
                this.speak(retryRemaining);
              }
              if (retryText) {
                this.conversationHistory.push({ role: "assistant", content: retryText });
                this.fullTranscript.push({ role: "AI", content: retryText });
              }
              // NOW execute the deferred tool (after text has been queued for TTS)
              if (this.state !== "ended") this.handleToolCall(savedToolCall);
            },
          });
          // Store cancel so endCall() can abort this retry
          this.currentLLMCancel = () => retryResult2.cancel();
          return;
        }

        if (remaining && !this.isSpeaking) {
          this.speak(remaining);
        }
        if (text) {
          this.conversationHistory.push({ role: "assistant", content: text });
          this.fullTranscript.push({ role: "AI", content: text });
        }
        // In trigger-phrases mode, scan the AI's full response for trigger phrases
        if (useTriggerPhrases && text) {
          this.checkTriggerPhrases(text);
        }
        this.state = "waiting_for_speech";
        this.resetSilenceTimer();
      },
    });

    this.currentLLMCancel = () => {
      releaseOllamaSlot();
      cancel();
    };
  }

  private shouldStartTTS(text: string): boolean {
    // Prefer splitting on sentence-ending punctuation (.!?) with min length
    if (/[.!?]/.test(text) && text.length > 20) return true;
    // For long accumulated text (80+ chars), also split on commas/semicolons to flush
    if (/[,;:]/.test(text) && text.length > 80) return true;
    return false;
  }

  private extractCompleteSentence(text: string): string | null {
    // First try sentence-ending punctuation
    const sentenceMatch = text.match(/^(.*?[.!?])\s*/);
    if (sentenceMatch && sentenceMatch[1].length >= 20) return sentenceMatch[1];
    // For long text, fall back to comma/semicolon split
    if (text.length > 80) {
      const commaMatch = text.match(/^(.*?[,;:])\s*/);
      if (commaMatch) return commaMatch[1];
    }
    return null;
  }

  /** Stream pre-synthesized PCM audio (16kHz 16-bit mono) without calling TTS */
  private speakPreSynthesized(pcm16k: Buffer, text: string): void {
    if (this.state === "ended") return;

    this.isSpeaking = true;
    this.state = "speaking";
    this.costTracker.addTTSUsage(text.length);

    // Emit in 100ms chunks (3200 bytes at 16kHz 16-bit mono)
    const CHUNK_SIZE = 3200;
    let offset = 0;

    const emitChunks = () => {
      while (offset < pcm16k.length) {
        const end = Math.min(offset + CHUNK_SIZE, pcm16k.length);
        const chunk = pcm16k.subarray(offset, end);
        const mulaw = pcm16kToMulaw(chunk);
        const base64 = encodeBase64Audio(mulaw);
        this.pendingAudioDurationMs += (mulaw.length / 8000) * 1000;
        this.emit("audio", base64);
        offset = end;
      }
      this.isSpeaking = false;
      if (this.state !== "ended") {
        this.state = "waiting_for_speech";
        this.resetSilenceTimer();
      }
    };

    // Use setImmediate to avoid blocking the event loop
    setImmediate(emitChunks);
    console.log(`[session:${this.config.callId}] Streaming pre-synthesized first message (${pcm16k.length} bytes, ${(pcm16k.length / 2 / 16000).toFixed(2)}s)`);
  }

  /** Stream raw mulaw audio from cached file — zero conversion, fastest possible playback */
  private speakPreSynthesizedMulaw(mulaw: Buffer): void {
    this.isSpeaking = true;
    // Emit in 800-byte chunks (100ms at 8kHz mulaw)
    const CHUNK_SIZE = 800;
    let offset = 0;

    const emitChunks = () => {
      while (offset < mulaw.length) {
        const end = Math.min(offset + CHUNK_SIZE, mulaw.length);
        const chunk = mulaw.subarray(offset, end);
        const base64 = encodeBase64Audio(chunk);
        this.pendingAudioDurationMs += (chunk.length / 8000) * 1000;
        this.emit("audio", base64);
        offset = end;
      }
      this.isSpeaking = false;
    };

    setImmediate(emitChunks);
    const durationSec = (mulaw.length / 8000).toFixed(2);
    console.log(`[session:${this.config.callId}] Streaming cached audio first message (${mulaw.length} bytes, ${durationSec}s, non-interruptable)`);
  }

  private speak(text: string): void {
    if (!this.tts || this.state === "ended") return;

    this.isSpeaking = true;
    this.state = "speaking";

    this.costTracker.addTTSUsage(text.length);

    const { cancel } = this.tts.synthesizeStream(
      text,
      (pcmChunk: Buffer) => {
        const mulaw = pcm16kToMulaw(pcmChunk);
        const base64 = encodeBase64Audio(mulaw);
        // Track audio duration: mulaw 8kHz = 8000 bytes/sec
        this.pendingAudioDurationMs += (mulaw.length / 8000) * 1000;
        this.emit("audio", base64);
      },
      () => {
        this.isSpeaking = false;
        if (this.state !== "ended") {
          this.state = "waiting_for_speech";
          this.resetSilenceTimer();
        }
      },
      (err: Error) => {
        console.error(`[session:${this.config.callId}] TTS error:`, err.message);
        this.isSpeaking = false;
        if (this.state !== "ended") {
          this.state = "waiting_for_speech";
          this.resetSilenceTimer();
        }
      }
    );

    this.currentTTSCancel = cancel;
  }

  private cancelSpeaking(): void {
    if (this.currentTTSCancel) {
      this.currentTTSCancel();
      this.currentTTSCancel = null;
    }
    if (this.currentLLMCancel) {
      this.currentLLMCancel();
      this.currentLLMCancel = null;
    }
    this.isSpeaking = false;
    this.pendingAudioDurationMs = 0;
    this.lastBargeInAt = Date.now();
    this.emit("clear_audio");
  }

  private waitForTTSFinish(timeoutMs: number = 30000, maxPlaybackMs: number = 10000): Promise<void> {
    // Wait for TTS synthesis to complete, then wait for Twilio to play the audio.
    // pendingAudioDurationMs tracks how much audio was queued — we wait that long
    // after synthesis finishes so the caller actually hears the message.
    const startPending = this.pendingAudioDurationMs;
    return new Promise((resolve) => {
      if (!this.isSpeaking) {
        // TTS just finished or hasn't started — wait for any pending audio to play
        const waitMs = Math.min(this.pendingAudioDurationMs, maxPlaybackMs);
        this.pendingAudioDurationMs = 0;
        setTimeout(resolve, waitMs > 0 ? waitMs : 500);
        return;
      }
      const checkInterval = setInterval(() => {
        if (!this.isSpeaking || this.state === "ended") {
          clearInterval(checkInterval);
          clearTimeout(timeout);
          // Synthesis done — now wait for Twilio to play the audio
          const audioMs = this.pendingAudioDurationMs - startPending;
          const waitMs = Math.min(Math.max(audioMs, 1000), maxPlaybackMs);
          this.pendingAudioDurationMs = 0;
          setTimeout(resolve, waitMs);
        }
      }, 100);
      const timeout = setTimeout(() => {
        clearInterval(checkInterval);
        this.pendingAudioDurationMs = 0;
        resolve();
      }, timeoutMs);
    });
  }

  private async handleToolCall(toolCall: LLMToolCall): Promise<void> {
    const result: ToolResult = await executeTool(toolCall, this.config.tools, {
      callId: this.config.callId,
      customerNumber: this.config.customerNumber,
      customerName: this.config.customerName,
      assistantId: this.config.assistantId,
      metadata: this.config.metadata,
      fallbackDestination: this.config.fallbackDestination,
    });

    this.emit("tool_call", { toolCall, result });
    this.fullTranscript.push({ role: "Tool", content: `[${toolCall.function.name}] ${result.action || "executed"}` });

    if (result.action === "endCall") {
      await this.waitForTTSFinish();
      // Extra buffer for network transit — audio may still be in Twilio/SignalWire buffer
      await new Promise(r => setTimeout(r, 1500));
      this.endCall("assistant-ended-call");
      return;
    }

    if (result.action === "transfer") {
      if (this.state === "transferring") return; // Guard: prevent double transfer
      await this.waitForTTSFinish();
      await new Promise(r => setTimeout(r, 1500));
      this.state = "transferring";
      this.emit("transfer", result.actionData);
      return;
    }

    // Feed tool result back to LLM
    this.conversationHistory.push({
      role: "tool",
      content: result.result,
      tool_call_id: toolCall.id,
      name: toolCall.function.name,
    });

    this.processWithLLM();
  }

  private resetSilenceTimer(): void {
    if (this.deliveringVoicemail) return;
    if (this.silenceTimer) clearTimeout(this.silenceTimer);
    this.silenceTimer = setTimeout(() => {
      if (this.state !== "ended" && this.state !== "speaking" && !this.playingFirstMessage) {
        this.endCall("silence-timed-out");
      } else if (this.playingFirstMessage) {
        // First message still playing — restart silence timer, don't end call
        this.resetSilenceTimer();
      }
    }, this.config.silenceTimeout * 1000);
  }

  private handleVoicemailDetected(): void {
    console.log(`[session:${this.config.callId}] Voicemail detected — delivering message after short pause`);
    this.emit("voicemail_detected");

    // Cancel any ongoing speech (first message playing over greeting)
    this.cancelSpeaking();

    // Disable silence timer — voicemail playback can be 30s+ and must not be interrupted
    this.deliveringVoicemail = true;
    if (this.silenceTimer) clearTimeout(this.silenceTimer);

    if (this.config.voicemailMessage) {
      // 1500ms pause after beep detected before speaking — ensures beep has fully passed
      setTimeout(async () => {
        if (this.state === "ended") return;
        console.log(`[session:${this.config.callId}] Delivering voicemail message`);
        this.speak(this.config.voicemailMessage!);
        this.fullTranscript.push({ role: "AI", content: `[Voicemail] ${this.config.voicemailMessage}` });

        // Wait for TTS synthesis to finish AND for the audio to actually play on the phone.
        // speak() sets isSpeaking=false when synthesis completes (~0.08s for Kokoro),
        // but Twilio needs the full playback duration. pendingAudioDurationMs tracks the
        // exact audio length — no artificial cap so any message length works.
        // 2s buffer added for network latency between voiceserver and Twilio/SignalWire.
        await this.waitForTTSFinish(60000, 60000);
        // Extra 2s buffer for network/jitter — audio may still be in transit
        await new Promise(r => setTimeout(r, 2000));
        console.log(`[session:${this.config.callId}] Voicemail message playback complete`);

        // endCall guards against double-call internally
        this.endCall("voicemail");
      }, 1500);
    } else {
      this.endCall("voicemail");
    }
  }

  endCall(reason: string): void {
    if (this.state === "ended" || this.isEndingCall) return;
    this.isEndingCall = true;
    this.state = "ended";

    if (this.silenceTimer) clearTimeout(this.silenceTimer);
    if (this.maxDurationTimer) clearTimeout(this.maxDurationTimer);
    if (this.endpointingTimer) clearTimeout(this.endpointingTimer);
    if (this.utteranceDelayTimer) clearTimeout(this.utteranceDelayTimer);

    this.cancelSpeaking();

    // Release Ollama slot if LLM was mid-stream (onDone never fired)
    if (this.pendingOllamaRelease) {
      this.pendingOllamaRelease();
      this.pendingOllamaRelease = null;
    }

    if (this.stt) {
      this.stt.removeAllListeners();
      this.stt.close();
    }

    const transcriptStr = this.fullTranscript
      .map((t) => `${t.role}: ${t.content}`)
      .join("\n");

    const duration = this.startedAt
      ? Math.round((Date.now() - this.startedAt.getTime()) / 1000)
      : 0;

    this.costTracker.addTransportUsage(duration);

    this.emit("ended", {
      callId: this.config.callId,
      endedReason: reason,
      transcript: transcriptStr,
      duration,
      cost: this.costTracker.getCost(),
      overflow: this.overflowCount > 0
        ? { used: true, count: this.overflowCount, provider: this.overflowProvider, model: this.overflowModel }
        : { used: false },
    });
  }

  getState(): SessionState {
    return this.state;
  }

  getTranscript(): string {
    return this.fullTranscript.map((t) => `${t.role}: ${t.content}`).join("\n");
  }

  isUsingOverflow(): boolean {
    return this.currentlyUsingOverflow;
  }
}
