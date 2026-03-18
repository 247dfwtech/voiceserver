import { EventEmitter } from "events";

export interface STTConfig {
  provider: string;
  model?: string;
  language?: string;
  keywords?: string[];
  /** If true, the provider accepts raw mulaw 8kHz audio (no PCM conversion needed) */
  acceptsMulaw?: boolean;
}

export interface STTProvider extends EventEmitter {
  /** Start a streaming session -- call send() with audio chunks, listen for 'transcript' events */
  start(): Promise<void>;
  /** Send raw PCM audio (16-bit, 16kHz mono) */
  send(audio: Buffer): void;
  /** Signal end of audio stream */
  finish(): Promise<void>;
  /** Close the connection and clean up */
  close(): void;
}

/** Emitted events: 'transcript' { text, isFinal }, 'error' Error */
export type STTTranscriptEvent = {
  text: string;
  isFinal: boolean;
  confidence?: number;
};
