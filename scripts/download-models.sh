#!/bin/bash
set -e

echo "=== Downloading Granite 4.0 1B Speech STT model (default) ==="
python3 -c "
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
print('Downloading Granite STT processor...')
AutoProcessor.from_pretrained('ibm-granite/granite-4.0-1b-speech', trust_remote_code=True)
print('Downloading Granite STT model...')
AutoModelForSpeechSeq2Seq.from_pretrained('ibm-granite/granite-4.0-1b-speech', trust_remote_code=True)
print('Granite STT ready!')
"

echo ""
echo "=== Pulling Ollama LLM models ==="
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull qwen2.5:7b
ollama pull phi3:mini

echo ""
echo "Done! Available LLM models:"
ollama list
