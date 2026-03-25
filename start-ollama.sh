#!/bin/bash
# Source voiceserver .env for dynamic settings
if [ -f /opt/voiceserverV2/.env ]; then
  export $(grep -E '^OLLAMA_' /opt/voiceserverV2/.env | xargs)
fi

# Defaults (only if not set from .env)
export OLLAMA_ORIGINS="*"
export OLLAMA_HOST="0.0.0.0"
export OLLAMA_FLASH_ATTENTION=${OLLAMA_FLASH_ATTENTION:-1}
export OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-8}

echo "[ollama] Starting with NUM_PARALLEL=$OLLAMA_NUM_PARALLEL FLASH_ATTENTION=$OLLAMA_FLASH_ATTENTION"
exec ollama serve 2>&1
