#!/bin/bash
set -e

echo "Installing piper-tts..."
pip install piper-tts

echo "Downloading default English voice (en_US-lessac-medium)..."
mkdir -p /models/piper

wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" \
  -O /models/piper/en_US-lessac-medium.onnx

wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" \
  -O /models/piper/en_US-lessac-medium.onnx.json

echo "Piper installation complete!"
echo "Test with: echo 'Hello world' | piper --model /models/piper/en_US-lessac-medium.onnx --output_raw | aplay -r 22050 -f S16_LE"
