import { EventEmitter } from "events";
import {
  mulawToPcm16k,
  pcm16kToMulaw,
  calculateRMS,
  decodeBase64Audio,
  encodeBase64Audio,
} from "./audio-utils";
import { createSTTProvider, createTTSProvider, createLLMProvider } from "../providers";
import type { STTProvider } from "../providers/stt/interface";
import type { TTSProvider } from "../providers/tts/interface";
import type { LLMProvider, LLMMessage, LLMToolDefinition, LLMToolCall } from "../providers/llm/interface";
import { executeTool, type ToolDefinition, type ToolResult } from "./tool-executor";

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
  total: number;
}

/** Per-call cost tracking */
class CostTracker {
  private sttMinutes = 0;
  private llmInputTokens = 0;
  private llmOutputTokens = 0;
  private ttsCharacters = 0;
  private transportMinutes = 0;

  private sttProvider: string;
  private llmModel: string;
  private ttsProvider: string;

  private static readonly PRICING = {
    stt: {
      deepgram: parseFloat(process.env.COST_STT_DEEPGRAM || "0.0059"),
      whisper: parseFloat(process.env.COST_STT_WHISPER || "0"),
    } as Record<string, number>,
    llm: {
      "gpt-4o": { input: 0.0025, output: 0.01 },
      "gpt-4o-mini": { input: 0.00015, output: 0.0006 },
      "deepseek-chat": { input: 0.00014, output: 0.00028 },
    } as Record<string, { input: number; output: number }>,
    tts: {
      elevenlabs: parseFloat(process.env.COST_TTS_ELEVENLABS || "0.18"),
      piper: 0.0,
      kokoro: 0.0,
      chatterbox: 0.0,
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

    const ttsRate = CostTracker.PRICING.tts[this.ttsProvider] ?? 0;
    const ttsCost = (this.ttsCharacters / 1000) * ttsRate;

    const transportCost = this.transportMinutes * CostTracker.PRICING.transport.twilio;

    const total = sttCost + llmCost + ttsCost + transportCost;

    return {
      stt: Math.round(sttCost * 10000) / 10000,
      llm: Math.round(llmCost * 10000) / 10000,
      tts: Math.round(ttsCost * 10000) / 10000,
      transport: Math.round(transportCost * 10000) / 10000,
      total: Math.round(total * 10000) / 10000,
    };
  }
}

/** Voicemail/IVR detector */
type VMDetectionResult = "human" | "voicemail" | "unknown";

class VoicemailDetector {
  private analysisWindowSeconds: number;
  private speechThreshold: number;
  private continuousSpeechFramesThreshold: number;
  private audioFrames: number[] = [];
  private speechFrameCount = 0;
  private silenceFrameCount = 0;
  private totalFrames = 0;
  private continuousSpeechRun = 0;
  private maxContinuousSpeech = 0;
  private resolved = false;
  private result: VMDetectionResult = "unknown";
  private resolveCallback: ((result: VMDetectionResult) => void) | null = null;

  constructor(config: { analysisWindowSeconds?: number; speechThreshold?: number; continuousSpeechFramesThreshold?: number; twilioAmdResult?: string } = {}) {
    this.analysisWindowSeconds = config.analysisWindowSeconds ?? 5;
    this.speechThreshold = config.speechThreshold ?? 300;
    this.continuousSpeechFramesThreshold = config.continuousSpeechFramesThreshold ?? 150; // 150 * 20ms = 3s of unbroken speech

    if (config.twilioAmdResult) {
      const amd = config.twilioAmdResult.toLowerCase();
      if (amd === "machine_start" || amd === "machine_end_beep" || amd === "machine_end_silence" || amd === "machine_end_other") {
        this.result = "voicemail";
        this.resolved = true;
      } else if (amd === "human") {
        this.result = "human";
        this.resolved = true;
      }
    }
  }

  analyzeFrame(pcmAudio: Buffer): void {
    if (this.resolved) return;

    const rms = calculateRMS(pcmAudio);
    this.audioFrames.push(rms);
    this.totalFrames++;

    const isSpeech = rms > this.speechThreshold;

    if (isSpeech) {
      this.speechFrameCount++;
      this.continuousSpeechRun++;
      this.silenceFrameCount = 0;
      if (this.continuousSpeechRun > this.maxContinuousSpeech) {
        this.maxContinuousSpeech = this.continuousSpeechRun;
      }
      if (this.continuousSpeechRun >= this.continuousSpeechFramesThreshold) {
        this.resolve("voicemail");
        return;
      }
    } else {
      this.silenceFrameCount++;
      if (this.silenceFrameCount > 5) {
        this.continuousSpeechRun = 0;
      }
    }

    const maxFrames = (this.analysisWindowSeconds * 1000) / 20;
    if (this.totalFrames >= maxFrames) {
      this.makeDecision();
    }
  }

  private makeDecision(): void {
    if (this.resolved) return;
    const speechRatio = this.speechFrameCount / this.totalFrames;
    // Require high speech ratio (80%), long continuous speech (40+ frames = 800ms),
    // and minimum total speech frames (100+) to avoid false positives on talkative humans
    if (speechRatio > 0.8 && this.maxContinuousSpeech > 40 && this.speechFrameCount > 100) {
      this.resolve("voicemail");
    } else {
      this.resolve("human");
    }
  }

  private resolve(result: VMDetectionResult): void {
    if (this.resolved) return;
    this.resolved = true;
    this.result = result;
    if (this.resolveCallback) {
      this.resolveCallback(result);
    }
  }

  getResult(): Promise<VMDetectionResult> {
    if (this.resolved) return Promise.resolve(this.result);
    return new Promise((resolve) => { this.resolveCallback = resolve; });
  }

  isResolved(): boolean { return this.resolved; }
  getCurrentResult(): VMDetectionResult { return this.result; }
  forceResult(result: VMDetectionResult): void { this.resolve(result); }
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
  tools: ToolDefinition[];
  endCallPhrases: string[];
  maxDuration: number;
  silenceTimeout: number;
  voicemailMessage?: string;
  voicemailDetectionEnabled?: boolean;
  customerNumber: string;
  customerName?: string;
  variableValues?: Record<string, string>;
  metadata?: Record<string, unknown>;
  fallbackDestination?: string;
  serverUrl?: string;
  backgroundDenoising?: boolean;
  startSpeakingPlan?: { waitSeconds?: number; smartEndpointingEnabled?: boolean };
  stopSpeakingPlan?: { numWords?: number; voiceSeconds?: number; backoffSeconds?: number };
  twilioAmdResult?: string;
  analysisConfig?: {
    summaryEnabled?: boolean;
    summaryPrompt?: string;
    successEvaluationEnabled?: boolean;
    successEvaluationPrompt?: string;
    successEvaluationRubric?: string;
  };
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

  private costTracker: CostTracker;
  private voicemailDetector: VoicemailDetector | null = null;
  private startedAt: Date | null = null;
  private audioChunkCount = 0;
  private consecutiveSTTErrors = 0;

  // Speaking plan state
  private speechStartedAt: number | null = null;
  private interimWordCount = 0;
  private utteranceDelayTimer: ReturnType<typeof setTimeout> | null = null;
  private playingFirstMessage = false; // true while first message TTS is playing — ignore user speech
  private pendingAudioDurationMs = 0; // tracks audio duration queued but not yet played by Twilio

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
        console.error(`[session:${this.config.callId}] 3 consecutive STT errors, ending call`);
        this.endCall("stt-failure");
      }
    });

    // Start STT
    await this.stt.start();

    // Set max duration timer
    this.maxDurationTimer = setTimeout(() => {
      this.endCall("max-duration-reached");
    }, this.config.maxDuration * 1000);

    // Initialize voicemail detection if enabled
    if (this.config.voicemailDetectionEnabled) {
      this.voicemailDetector = new VoicemailDetector({
        twilioAmdResult: this.config.twilioAmdResult,
      });

      if (this.voicemailDetector.isResolved() && this.voicemailDetector.getCurrentResult() === "voicemail") {
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

    // Handle first message
    if (
      this.config.firstMessage &&
      this.config.firstMessageMode !== "assistant-waits-for-user"
    ) {
      let firstMsg = this.config.firstMessage;
      if (this.config.variableValues) {
        firstMsg = substituteVariables(firstMsg, this.config.variableValues);
      }
      this.playingFirstMessage = true;
      this.speak(firstMsg);
      // Don't rely on TTS completion callback to clear playingFirstMessage —
      // TTS synthesis finishes in ~1s but audio plays for 10-15s on the phone.
      // Estimate playback: ~60ms per character is a rough TTS duration heuristic.
      const estimatedPlaybackMs = Math.max(firstMsg.length * 60, 5000);
      setTimeout(() => {
        this.playingFirstMessage = false;
        // Discard any sub-threshold words that were said during first message
        this.currentTranscript = "";
      }, estimatedPlaybackMs);
      this.conversationHistory.push({ role: "assistant", content: firstMsg });
      this.fullTranscript.push({ role: "AI", content: firstMsg });
    } else {
      this.state = "waiting_for_speech";
      this.resetSilenceTimer();
    }
  }

  /** Feed raw Twilio mu-law base64 audio into the pipeline */
  handleAudio(base64Audio: string): void {
    if (this.state === "ended") return;

    const mulaw = decodeBase64Audio(base64Audio);

    this.audioChunkCount++;

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

    // Default path: convert to PCM 16kHz for Whisper and other local STT
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

  private handleSpeechStarted(): void {
    // Record when speech started for stopSpeakingPlan thresholds
    this.speechStartedAt = Date.now();
    this.interimWordCount = 0;

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

    // Reset STT error counter on successful transcription
    this.consecutiveSTTErrors = 0;

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
    this.processWithLLM();
  }

  private processWithLLM(): void {
    if (!this.llm) return;

    const llmTools: LLMToolDefinition[] = this.config.tools
      .filter((t) => t.functionDefinition)
      .map((t) => ({
        type: "function" as const,
        function: {
          name: t.functionDefinition!.name,
          description: t.functionDefinition!.description,
          parameters: t.functionDefinition!.parameters,
        },
      }));

    for (const t of this.config.tools) {
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

    const { cancel } = this.llm.streamCompletion({
      messages: this.conversationHistory,
      tools: llmTools.length > 0 ? llmTools : undefined,
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
        // Flush any remaining text to TTS before handling the tool call
        // This ensures "I'll transfer you now" is spoken before ff_transfer fires
        const remaining = fullResponse.trim();
        if (remaining) {
          fullResponse = "";
          this.speak(remaining);
        }
        this.handleToolCall(toolCall);
      },
      onDone: (text: string) => {
        const remaining = (fullResponse || text).trim();
        if (remaining && !this.isSpeaking) {
          this.speak(remaining);
        }
        if (text) {
          this.conversationHistory.push({ role: "assistant", content: text });
          this.fullTranscript.push({ role: "AI", content: text });
        }
        this.state = "waiting_for_speech";
        this.resetSilenceTimer();
      },
    });

    this.currentLLMCancel = cancel;
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
    this.emit("clear_audio");
  }

  private waitForTTSFinish(timeoutMs: number = 30000): Promise<void> {
    // Wait for TTS synthesis to complete, then wait for Twilio to play the audio.
    // pendingAudioDurationMs tracks how much audio was queued — we wait that long
    // after synthesis finishes so the caller actually hears the message.
    const startPending = this.pendingAudioDurationMs;
    return new Promise((resolve) => {
      if (!this.isSpeaking) {
        // TTS just finished or hasn't started — wait for any pending audio to play
        const waitMs = Math.min(this.pendingAudioDurationMs, 10000);
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
          const waitMs = Math.min(Math.max(audioMs, 1000), 10000);
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
      this.endCall("assistant-ended-call");
      return;
    }

    if (result.action === "transfer") {
      await this.waitForTTSFinish();
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
    if (this.silenceTimer) clearTimeout(this.silenceTimer);
    this.silenceTimer = setTimeout(() => {
      if (this.state !== "ended" && this.state !== "speaking") {
        this.endCall("silence-timed-out");
      }
    }, this.config.silenceTimeout * 1000);
  }

  private handleVoicemailDetected(): void {
    console.log(`[session:${this.config.callId}] Voicemail detected`);
    this.emit("voicemail_detected");

    if (this.config.voicemailMessage) {
      this.speak(this.config.voicemailMessage);
      this.fullTranscript.push({ role: "AI", content: `[Voicemail] ${this.config.voicemailMessage}` });

      const checkDone = setInterval(() => {
        if (!this.isSpeaking && this.state !== "ended") {
          clearInterval(checkDone);
          this.endCall("voicemail");
        }
      }, 500);

      setTimeout(() => {
        clearInterval(checkDone);
        if (this.state !== "ended") {
          this.endCall("voicemail");
        }
      }, 15000);
    } else {
      this.endCall("voicemail");
    }
  }

  endCall(reason: string): void {
    if (this.state === "ended") return;
    this.state = "ended";

    if (this.silenceTimer) clearTimeout(this.silenceTimer);
    if (this.maxDurationTimer) clearTimeout(this.maxDurationTimer);
    if (this.endpointingTimer) clearTimeout(this.endpointingTimer);
    if (this.utteranceDelayTimer) clearTimeout(this.utteranceDelayTimer);

    this.cancelSpeaking();

    if (this.stt) this.stt.close();

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
    });
  }

  getState(): SessionState {
    return this.state;
  }

  getTranscript(): string {
    return this.fullTranscript.map((t) => `${t.role}: ${t.content}`).join("\n");
  }
}
