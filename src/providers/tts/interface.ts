export interface TTSConfig {
  provider: string;
  voiceId: string;
  model?: string;
  speed?: number;
  stability?: number;
  baseVoice?: string;
}

export interface TTSProvider {
  /** Synthesize text to audio. Returns raw PCM audio (16-bit, 16kHz mono) or streams chunks via callback. */
  synthesize(
    text: string,
    onChunk?: (chunk: Buffer) => void
  ): Promise<Buffer>;

  /** Synthesize with streaming -- returns immediately, calls onChunk for each audio chunk */
  synthesizeStream(
    text: string,
    onChunk: (chunk: Buffer) => void,
    onDone: () => void,
    onError?: (err: Error) => void
  ): { cancel: () => void };
}
