#!/usr/bin/env python3
"""
Test script for Parakeet TDT 0.6B v2 STT on GPU server.
Run: python3 scripts/test-parakeet.py

Tests:
1. Model download/load
2. Transcription of a synthetic WAV file
3. Latency measurement
"""
import sys
import time
import os
import struct
import wave
import math
import tempfile

def generate_test_wav(filepath, duration_s=2, sample_rate=16000):
    """Generate a simple WAV file with a tone (for testing pipeline, not accuracy)"""
    n_samples = int(duration_s * sample_rate)
    with wave.open(filepath, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        # Generate a 440Hz sine wave
        for i in range(n_samples):
            sample = int(16000 * math.sin(2 * math.pi * 440 * i / sample_rate))
            wf.writeframes(struct.pack('<h', sample))

print("=" * 60)
print("PARAKEET TDT 0.6B v2 TEST SCRIPT")
print("=" * 60)

# Step 1: Check if nemo is installed
print("\n[1/4] Checking nemo_toolkit installation...")
try:
    import nemo.collections.asr as nemo_asr
    print("  OK: nemo.collections.asr imported successfully")
except ImportError as e:
    print(f"  FAIL: {e}")
    print("  FIX: pip install 'nemo_toolkit[asr]'")
    sys.exit(1)

# Step 2: Check CUDA
print("\n[2/4] Checking CUDA availability...")
import torch
if torch.cuda.is_available():
    print(f"  OK: CUDA available — {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  WARN: No CUDA — will use CPU (slower)")

# Step 3: Load model
print("\n[3/4] Loading Parakeet TDT model (first run downloads ~1.2GB)...")
t0 = time.time()
try:
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    load_time = time.time() - t0
    print(f"  OK: Model loaded in {load_time:.1f}s on {device}")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# Step 4: Test transcription
print("\n[4/4] Testing transcription...")

# First test: synthetic tone (just tests the pipeline works)
tmp_wav = tempfile.mktemp(suffix=".wav")
generate_test_wav(tmp_wav, duration_s=2)

t0 = time.time()
try:
    transcriptions = model.transcribe([tmp_wav])
    latency = time.time() - t0

    if transcriptions and len(transcriptions) > 0:
        result = transcriptions[0]
        if hasattr(result, 'text'):
            text = result.text
        elif isinstance(result, str):
            text = result
        else:
            text = str(result)
        print(f"  OK: Transcription returned in {latency:.3f}s")
        print(f"  Output: '{text}' (tone — expected empty or noise)")
        print(f"  Return type: {type(result).__name__}")
    else:
        print(f"  WARN: Empty transcription result (type={type(transcriptions)})")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()
finally:
    os.unlink(tmp_wav)

# Second test: check if there's a real audio file to test with
test_files = [
    "/tmp/test-audio.wav",
    "/data/test-audio.wav",
]
for tf in test_files:
    if os.path.exists(tf):
        print(f"\n  Testing with real audio: {tf}")
        t0 = time.time()
        try:
            result = model.transcribe([tf])
            latency = time.time() - t0
            text = result[0].text if hasattr(result[0], 'text') else str(result[0])
            print(f"  OK: '{text}' ({latency:.3f}s)")
        except Exception as e:
            print(f"  FAIL: {e}")

print("\n" + "=" * 60)
print("PARAKEET TEST COMPLETE")
print("=" * 60)
