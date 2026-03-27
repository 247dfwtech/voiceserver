/**
 * Audio conversion utilities for Twilio Media Streams.
 * Twilio sends mu-law 8kHz mono audio. Our STT providers need linear PCM 16kHz.
 * TTS providers return PCM 16kHz. We need to convert back to mu-law 8kHz for Twilio.
 */

// mu-law decompression table
const MULAW_DECODE_TABLE = new Int16Array(256);
(function initMulawTable() {
  for (let i = 0; i < 256; i++) {
    let mu = ~i & 0xff;
    const sign = mu & 0x80;
    mu &= 0x7f;
    mu = (mu << 1) | 1;
    const exponent = (mu >> 4) & 0x07;
    const mantissa = mu & 0x0f;
    let sample = ((mantissa << 3) + 0x84) << exponent;
    sample -= 0x84;
    MULAW_DECODE_TABLE[i] = sign ? -sample : sample;
  }
})();

// Linear to mu-law compression
const MULAW_BIAS = 33;
const MULAW_CLIP = 32635;

function linearToMulaw(sample: number): number {
  const sign = sample < 0 ? 0x80 : 0;
  if (sample < 0) sample = -sample;
  if (sample > MULAW_CLIP) sample = MULAW_CLIP;
  sample += MULAW_BIAS;

  let exponent = 7;
  let mask = 0x4000;
  while (exponent > 0 && (sample & mask) === 0) {
    exponent--;
    mask >>= 1;
  }

  const mantissa = (sample >> (exponent + 3)) & 0x0f;
  const mulaw = ~(sign | (exponent << 4) | mantissa) & 0xff;
  return mulaw;
}

/**
 * Convert mu-law 8kHz audio to linear PCM 16-bit 16kHz (for STT)
 * Upsamples by duplicating samples (simple but effective for speech)
 */
export function mulawToPcm16k(mulaw: Buffer): Buffer {
  const pcm = Buffer.alloc(mulaw.length * 4); // 2x for 16-bit, 2x for upsample
  for (let i = 0; i < mulaw.length; i++) {
    const sample = MULAW_DECODE_TABLE[mulaw[i]];
    // Upsample 8kHz -> 16kHz by duplicating each sample
    pcm.writeInt16LE(sample, i * 4);
    pcm.writeInt16LE(sample, i * 4 + 2);
  }
  return pcm;
}

/**
 * Convert mu-law 8kHz audio to linear PCM 16-bit 8kHz (keeping same rate)
 */
export function mulawToPcm8k(mulaw: Buffer): Buffer {
  const pcm = Buffer.alloc(mulaw.length * 2);
  for (let i = 0; i < mulaw.length; i++) {
    pcm.writeInt16LE(MULAW_DECODE_TABLE[mulaw[i]], i * 2);
  }
  return pcm;
}

/**
 * Convert linear PCM 16-bit 16kHz to mu-law 8kHz (for Twilio)
 * Downsamples by taking every other sample
 */
export function pcm16kToMulaw(pcm: Buffer): Buffer {
  const numSamples = pcm.length / 2;
  const mulaw = Buffer.alloc(Math.floor(numSamples / 2));
  for (let i = 0; i < mulaw.length; i++) {
    const sample = pcm.readInt16LE(i * 4); // skip every other sample
    mulaw[i] = linearToMulaw(sample);
  }
  return mulaw;
}

/**
 * Convert linear PCM 16-bit 8kHz to mu-law 8kHz (same rate)
 */
export function pcm8kToMulaw(pcm: Buffer): Buffer {
  const numSamples = pcm.length / 2;
  const mulaw = Buffer.alloc(numSamples);
  for (let i = 0; i < numSamples; i++) {
    const sample = pcm.readInt16LE(i * 2);
    mulaw[i] = linearToMulaw(sample);
  }
  return mulaw;
}

/**
 * Calculate RMS energy of PCM audio buffer (for VAD/silence detection)
 */
export function calculateRMS(pcm: Buffer): number {
  const numSamples = pcm.length / 2;
  if (numSamples === 0) return 0;
  let sum = 0;
  for (let i = 0; i < numSamples; i++) {
    const sample = pcm.readInt16LE(i * 2);
    sum += sample * sample;
  }
  return Math.sqrt(sum / numSamples);
}

/**
 * Goertzel algorithm — efficient single-frequency magnitude detector.
 * Returns magnitude (power) at the target frequency.
 */
export function goertzelMagnitude(pcm: Buffer, sampleRate: number, targetFreq: number): number {
  const numSamples = pcm.length / 2;
  if (numSamples === 0) return 0;
  const k = Math.round(numSamples * targetFreq / sampleRate);
  const w = (2 * Math.PI * k) / numSamples;
  const coeff = 2 * Math.cos(w);
  let s1 = 0, s2 = 0;
  for (let i = 0; i < numSamples; i++) {
    const sample = pcm.readInt16LE(i * 2) / 32768;
    const s0 = sample + coeff * s1 - s2;
    s2 = s1;
    s1 = s0;
  }
  return Math.sqrt(s1 * s1 + s2 * s2 - coeff * s1 * s2);
}

/**
 * Detect if audio contains a voicemail beep tone (900-1200Hz).
 * Uses Goertzel to check if beep-range frequency is dominant vs other bands.
 */
export function detectBeep(pcm: Buffer, sampleRate: number = 16000): boolean {
  const rms = calculateRMS(pcm);
  if (rms < 200) return false; // Too quiet to be a beep

  const mag1000 = goertzelMagnitude(pcm, sampleRate, 1000);
  const mag440 = goertzelMagnitude(pcm, sampleRate, 440);
  const mag2000 = goertzelMagnitude(pcm, sampleRate, 2000);

  const beepPower = mag1000;
  const noisePower = (mag440 + mag2000) / 2;

  return beepPower > 0.1 && (noisePower === 0 || beepPower / noisePower > 3);
}

/**
 * Twilio sends base64-encoded mu-law audio. Decode it.
 */
export function decodeBase64Audio(base64: string): Buffer {
  return Buffer.from(base64, "base64");
}

/**
 * Encode mu-law audio to base64 for sending back to Twilio
 */
export function encodeBase64Audio(mulaw: Buffer): string {
  return mulaw.toString("base64");
}
