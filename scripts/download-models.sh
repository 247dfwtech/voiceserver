#!/bin/bash
set -e

echo "Pulling Ollama models..."
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull qwen2.5:7b
ollama pull phi3:mini

echo ""
echo "Done! Available models:"
ollama list
