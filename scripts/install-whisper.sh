#!/bin/bash
set -e

echo "Installing faster-whisper..."
pip install faster-whisper

echo "Downloading base.en model..."
python3 -c "from faster_whisper import WhisperModel; WhisperModel('base.en')"

echo "Whisper installation complete!"
echo "Test with: faster-whisper --help"
