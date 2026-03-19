#!/usr/bin/env python3
"""
Test script for Vosk STT (CPU fallback) on GPU server.
Run: python3 scripts/test-vosk.py

Tests:
1. vosk package installation
2. Model download (auto-downloads ~40MB small English model)
3. Transcription of a synthetic WAV file
4. Streaming protocol test (length-prefixed PCM)
"""
import sys
import time
import os
import struct
import wave
import math
import tempfile
import json

VOSK_MODELS_DIR = os.environ.get("VOSK_MODELS_DIR", "/models/vosk")
MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_PATH = os.path.join(VOSK_MODELS_DIR, MODEL_NAME)

def generate_test_wav(filepath, duration_s=2, sample_rate=16000):
    """Generate a WAV with silence + tone"""
    n_samples = int(duration_s * sample_rate)
    with wave.open(filepath, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            sample = int(16000 * math.sin(2 * math.pi * 440 * i / sample_rate))
            wf.writeframes(struct.pack('<h', sample))

def generate_speech_like_wav(filepath, duration_s=3, sample_rate=16000):
    """Generate a WAV with varying frequencies (simulates speech-like audio)"""
    n_samples = int(duration_s * sample_rate)
    with wave.open(filepath, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            t = i / sample_rate
            # Mix of frequencies to simulate speech formants
            sample = int(8000 * (
                math.sin(2 * math.pi * 200 * t) +
                0.5 * math.sin(2 * math.pi * 800 * t) +
                0.3 * math.sin(2 * math.pi * 2500 * t)
            ))
            sample = max(-32767, min(32767, sample))
            wf.writeframes(struct.pack('<h', sample))

print("=" * 60)
print("VOSK STT (CPU FALLBACK) TEST SCRIPT")
print("=" * 60)

# Step 1: Check vosk installation
print("\n[1/5] Checking vosk installation...")
try:
    from vosk import Model, KaldiRecognizer, SetLogLevel
    SetLogLevel(-1)
    print("  OK: vosk imported successfully")
except ImportError as e:
    print(f"  FAIL: {e}")
    print("  FIX: pip install vosk")
    sys.exit(1)

# Step 2: Check/download model
print(f"\n[2/5] Checking model at {MODEL_PATH}...")
if os.path.isdir(MODEL_PATH):
    print(f"  OK: Model already downloaded")
else:
    print(f"  Downloading {MODEL_NAME}...")
    os.makedirs(VOSK_MODELS_DIR, exist_ok=True)
    import urllib.request
    import zipfile
    import io

    url = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"
    print(f"  URL: {url}")
    t0 = time.time()
    try:
        with urllib.request.urlopen(url) as resp:
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(VOSK_MODELS_DIR)
        dl_time = time.time() - t0
        print(f"  OK: Downloaded and extracted in {dl_time:.1f}s")
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

# Step 3: Load model
print("\n[3/5] Loading Vosk model...")
t0 = time.time()
try:
    model = Model(MODEL_PATH)
    load_time = time.time() - t0
    print(f"  OK: Model loaded in {load_time:.1f}s (CPU only)")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# Step 4: Test file-based transcription
print("\n[4/5] Testing file-based transcription...")
tmp_wav = tempfile.mktemp(suffix=".wav")
generate_test_wav(tmp_wav, duration_s=2)

t0 = time.time()
try:
    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(True)

    wf = wave.open(tmp_wav, 'rb')
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        rec.AcceptWaveform(data)
    wf.close()

    result = json.loads(rec.FinalResult())
    latency = time.time() - t0
    text = result.get("text", "")
    print(f"  OK: Transcription in {latency:.3f}s")
    print(f"  Output: '{text}' (tone — expected empty or noise)")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()
finally:
    os.unlink(tmp_wav)

# Step 5: Test streaming protocol (simulates what vosk.ts does)
print("\n[5/5] Testing streaming PCM protocol...")
tmp_wav2 = tempfile.mktemp(suffix=".wav")
generate_speech_like_wav(tmp_wav2, duration_s=3)

t0 = time.time()
try:
    rec2 = KaldiRecognizer(model, 16000)
    rec2.SetWords(True)

    wf2 = wave.open(tmp_wav2, 'rb')
    chunk_count = 0
    partial_count = 0
    final_count = 0

    while True:
        data = wf2.readframes(800)  # 50ms chunks (800 samples at 16kHz)
        if len(data) == 0:
            break
        chunk_count += 1

        if rec2.AcceptWaveform(data):
            result = json.loads(rec2.Result())
            if result.get("text"):
                final_count += 1
        else:
            partial = json.loads(rec2.PartialResult())
            if partial.get("partial"):
                partial_count += 1

    final_result = json.loads(rec2.FinalResult())
    wf2.close()
    latency = time.time() - t0

    print(f"  OK: Streaming test in {latency:.3f}s")
    print(f"  Chunks processed: {chunk_count}")
    print(f"  Partial results: {partial_count}")
    print(f"  Final results: {final_count}")
    print(f"  Final text: '{final_result.get('text', '')}'")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()
finally:
    os.unlink(tmp_wav2)

# Check for real audio files
test_files = [
    "/tmp/test-audio.wav",
    "/data/test-audio.wav",
]
for tf in test_files:
    if os.path.exists(tf):
        print(f"\n  Bonus: Testing with real audio {tf}...")
        try:
            rec3 = KaldiRecognizer(model, 16000)
            wf3 = wave.open(tf, 'rb')
            while True:
                data = wf3.readframes(4000)
                if len(data) == 0:
                    break
                rec3.AcceptWaveform(data)
            wf3.close()
            result = json.loads(rec3.FinalResult())
            print(f"  Result: '{result.get('text', '')}'")
        except Exception as e:
            print(f"  FAIL: {e}")

print("\n" + "=" * 60)
print("VOSK TEST COMPLETE")
print("=" * 60)
