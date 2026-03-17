#!/bin/bash
###############################################################################
# setup-gpu-server.sh — Automated bare metal setup for voiceserver on GPU VPS
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/247dfwtech/voiceserver/main/scripts/setup-gpu-server.sh | bash
#   — OR —
#   chmod +x setup-gpu-server.sh && ./setup-gpu-server.sh
#
# Tested on: Ubuntu 22.04 / 24.04 with NVIDIA RTX 4090
# Expected VRAM usage: ~8.5GB / 24GB
#   - Granite STT:  ~2.0GB
#   - Kokoro TTS:   ~0.5GB
#   - Qwen3.5 9B:   ~6.0GB
#   - Headroom:     ~15.5GB
###############################################################################

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; }
info() { echo -e "${BLUE}[→]${NC} $1"; }

VOICESERVER_DIR="/opt/voiceserver"
MODELS_DIR="/models"
DATA_DIR="/data"
REPO_URL="https://github.com/247dfwtech/voiceserver.git"

###############################################################################
# Pre-flight checks
###############################################################################

if [ "$EUID" -ne 0 ]; then
  err "Please run as root: sudo bash setup-gpu-server.sh"
  exit 1
fi

echo ""
echo "============================================================"
echo "  voiceserver — GPU Bare Metal Setup"
echo "  STT: Granite 4.0 1B Speech (default) + Whisper (fallback)"
echo "  TTS: Kokoro-82M (default) + Piper (fallback)"
echo "  LLM: Qwen3.5 9B via Ollama (auto-pulled)"
echo "============================================================"
echo ""

# Check for NVIDIA GPU
if ! command -v nvidia-smi &>/dev/null; then
  warn "nvidia-smi not found. Checking if NVIDIA drivers need to be installed..."
  if lspci | grep -i nvidia &>/dev/null; then
    info "NVIDIA GPU detected. Installing drivers..."
    apt-get update
    apt-get install -y nvidia-driver-535 nvidia-utils-535
    warn "NVIDIA drivers installed. A REBOOT is required."
    warn "After reboot, run this script again."
    read -p "Reboot now? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      reboot
    fi
    exit 1
  else
    err "No NVIDIA GPU detected. This script requires an NVIDIA GPU."
    exit 1
  fi
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
log "GPU detected: ${GPU_NAME} (${GPU_VRAM} MiB VRAM)"

if [ "${GPU_VRAM}" -lt 16000 ]; then
  warn "Less than 16GB VRAM detected. Qwen3.5 9B may not fit alongside STT/TTS."
  warn "Consider using a smaller LLM model."
fi

###############################################################################
# 1. System dependencies
###############################################################################

info "Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y \
  curl wget git build-essential \
  python3 python3-pip python3-venv \
  ffmpeg sox libsndfile1 \
  htop tmux unzip jq \
  2>&1 | tail -1

log "System dependencies installed"

###############################################################################
# 2. Node.js 22 (via NodeSource)
###############################################################################

if command -v node &>/dev/null && [[ "$(node -v)" == v22* ]]; then
  log "Node.js $(node -v) already installed"
else
  info "Installing Node.js 22..."
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
  apt-get install -y nodejs
  log "Node.js $(node -v) installed"
fi

# Install pm2 globally
if ! command -v pm2 &>/dev/null; then
  info "Installing pm2..."
  npm install -g pm2
  log "pm2 installed"
else
  log "pm2 already installed"
fi

###############################################################################
# 3. Ollama (LLM inference server)
###############################################################################

if command -v ollama &>/dev/null; then
  log "Ollama already installed"
else
  info "Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
  log "Ollama installed"
fi

# Ensure Ollama service is running
info "Starting Ollama service..."
if systemctl is-active --quiet ollama 2>/dev/null; then
  log "Ollama service already running"
else
  # Try systemd first, fall back to manual
  if systemctl start ollama 2>/dev/null; then
    systemctl enable ollama 2>/dev/null || true
    log "Ollama service started via systemd"
  else
    info "Starting Ollama manually..."
    nohup ollama serve > /var/log/ollama.log 2>&1 &
    sleep 3
    log "Ollama started manually"
  fi
fi

# Wait for Ollama to be ready
info "Waiting for Ollama to be ready..."
for i in $(seq 1 30); do
  if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    log "Ollama is ready"
    break
  fi
  sleep 1
  if [ "$i" -eq 30 ]; then
    err "Ollama failed to start after 30 seconds"
    exit 1
  fi
done

# Pull default LLM model
info "Pulling Qwen3.5 9B (default LLM, ~6GB)... this may take a few minutes"
if ollama list 2>/dev/null | grep -q "qwen3.5:9b"; then
  log "Qwen3.5 9B already downloaded"
else
  ollama pull qwen3.5:9b
  log "Qwen3.5 9B downloaded"
fi

###############################################################################
# 4. Python AI packages
###############################################################################

info "Installing Python AI packages..."

# Upgrade pip
python3 -m pip install --upgrade pip 2>&1 | tail -1

# faster-whisper (STT fallback)
info "Installing faster-whisper..."
pip3 install --no-cache-dir faster-whisper 2>&1 | tail -1
log "faster-whisper installed"

# Piper TTS (fallback)
info "Installing piper-tts..."
pip3 install --no-cache-dir piper-tts 2>&1 | tail -1
log "piper-tts installed"

# Kokoro-82M TTS (default)
info "Installing Kokoro TTS..."
pip3 install --no-cache-dir "kokoro>=0.9" soundfile 2>&1 | tail -1
log "Kokoro TTS installed"

# Granite STT dependencies (transformers + torch + torchaudio + huggingface_hub)
info "Installing Granite STT dependencies (transformers, torch, torchaudio)... this may take a few minutes"
pip3 install --no-cache-dir transformers torch torchaudio huggingface_hub soundfile 2>&1 | tail -1
log "Granite STT dependencies installed"

###############################################################################
# 5. Pre-download AI models
###############################################################################

mkdir -p "${MODELS_DIR}/piper" "${MODELS_DIR}/granite" "${MODELS_DIR}/kokoro"
mkdir -p "${DATA_DIR}/cloned-voices"

# Granite STT model (~2GB)
info "Downloading Granite 4.0 1B Speech STT model..."
python3 -c "
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
print('Downloading Granite STT model...')
AutoProcessor.from_pretrained('ibm-granite/granite-4.0-1b-speech', trust_remote_code=True)
AutoModelForSpeechSeq2Seq.from_pretrained('ibm-granite/granite-4.0-1b-speech', trust_remote_code=True)
print('Done')
" 2>&1
log "Granite STT model downloaded"

# Kokoro-82M TTS model
info "Downloading Kokoro-82M TTS model..."
python3 -c "
from kokoro import KPipeline
print('Downloading Kokoro-82M model...')
pipeline = KPipeline(lang_code='a')
print('Done')
" 2>&1
log "Kokoro TTS model downloaded"

# Whisper base.en (fallback STT)
info "Downloading Whisper base.en model (fallback)..."
python3 -c "
from faster_whisper import WhisperModel
print('Downloading Whisper base.en...')
WhisperModel('base.en')
print('Done')
" 2>&1
log "Whisper model downloaded"

# Piper default voice (fallback TTS)
info "Downloading Piper default voice..."
PIPER_VOICE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
if [ ! -f "${MODELS_DIR}/piper/en_US-lessac-medium.onnx" ]; then
  wget -q "${PIPER_VOICE_URL}/en_US-lessac-medium.onnx" -O "${MODELS_DIR}/piper/en_US-lessac-medium.onnx"
  wget -q "${PIPER_VOICE_URL}/en_US-lessac-medium.onnx.json" -O "${MODELS_DIR}/piper/en_US-lessac-medium.onnx.json"
  log "Piper default voice downloaded"
else
  log "Piper default voice already exists"
fi

###############################################################################
# 6. Clone and build voiceserver
###############################################################################

info "Setting up voiceserver..."

if [ -d "${VOICESERVER_DIR}/.git" ]; then
  info "Updating existing voiceserver..."
  cd "${VOICESERVER_DIR}"
  git pull origin main
else
  info "Cloning voiceserver..."
  git clone "${REPO_URL}" "${VOICESERVER_DIR}"
  cd "${VOICESERVER_DIR}"
fi

info "Installing Node.js dependencies..."
npm install 2>&1 | tail -3

info "Building TypeScript..."
npm run build 2>&1 | tail -3
log "voiceserver built successfully"

###############################################################################
# 7. Environment configuration
###############################################################################

ENV_FILE="${VOICESERVER_DIR}/.env"

if [ ! -f "${ENV_FILE}" ]; then
  info "Creating .env file..."
  cat > "${ENV_FILE}" <<'ENVEOF'
# voiceserver environment configuration
# Generated by setup-gpu-server.sh

# Server ports
WS_PORT=8765
IPC_PORT=8766

# Model directories
PIPER_MODELS_DIR=/models/piper
WHISPER_MODEL=base.en
GRANITE_MODELS_DIR=/models/granite
KOKORO_MODELS_DIR=/models/kokoro
KOKORO_VOICE=af_heart
CLONED_VOICES_DIR=/data/cloned-voices

# Ollama (local LLM)
OLLAMA_URL=http://localhost:11434/v1
DEFAULT_LLM=qwen3.5:9b

# Optional: Paid provider API keys (uncomment and fill in if needed)
# DEEPGRAM_API_KEY=
# ELEVENLABS_API_KEY=
# OPENAI_API_KEY=
# DEEPSEEK_API_KEY=

NODE_ENV=production
ENVEOF
  log ".env file created at ${ENV_FILE}"
else
  warn ".env file already exists, skipping creation"
fi

###############################################################################
# 8. pm2 process management
###############################################################################

info "Setting up pm2..."

# Create pm2 ecosystem file
cat > "${VOICESERVER_DIR}/ecosystem.config.cjs" <<'PM2EOF'
module.exports = {
  apps: [
    {
      name: "voiceserver",
      script: "dist/index.js",
      cwd: "/opt/voiceserver",
      env: {
        NODE_ENV: "production",
      },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "2G",
      error_file: "/var/log/voiceserver/error.log",
      out_file: "/var/log/voiceserver/out.log",
      merge_logs: true,
      time: true,
    },
  ],
};
PM2EOF

# Create log directory
mkdir -p /var/log/voiceserver

# Stop existing instance if running
pm2 stop voiceserver 2>/dev/null || true
pm2 delete voiceserver 2>/dev/null || true

# Start voiceserver with pm2
cd "${VOICESERVER_DIR}"
pm2 start ecosystem.config.cjs
pm2 save

# Set up pm2 to start on boot
pm2 startup systemd -u root --hp /root 2>/dev/null || pm2 startup 2>/dev/null || true
pm2 save

log "voiceserver started with pm2"

###############################################################################
# 9. Firewall (optional, allow WS + IPC ports)
###############################################################################

if command -v ufw &>/dev/null; then
  info "Configuring firewall..."
  ufw allow 8765/tcp comment "voiceserver WebSocket" 2>/dev/null || true
  ufw allow 8766/tcp comment "voiceserver IPC" 2>/dev/null || true
  ufw allow 22/tcp comment "SSH" 2>/dev/null || true
  log "Firewall rules added (8765, 8766, 22)"
fi

###############################################################################
# 10. Verification
###############################################################################

echo ""
echo "============================================================"
echo "  Verifying installation..."
echo "============================================================"
echo ""

# Check all components
PASS=true

check() {
  if eval "$2" &>/dev/null; then
    log "$1"
  else
    err "$1 — FAILED"
    PASS=false
  fi
}

check "Node.js $(node -v)"          "node -v"
check "npm $(npm -v)"               "npm -v"
check "pm2 installed"               "pm2 -v"
check "Ollama running"              "curl -s http://localhost:11434/api/tags"
check "Qwen3.5 9B available"        "ollama list | grep -q qwen3.5"
check "Python3 available"           "python3 --version"
check "faster-whisper installed"    "python3 -c 'import faster_whisper'"
check "piper-tts installed"         "python3 -c 'import piper'"
check "Kokoro TTS installed"        "python3 -c 'from kokoro import KPipeline'"
check "Transformers installed"      "python3 -c 'import transformers'"
check "Torch installed"             "python3 -c 'import torch; print(f\"CUDA: {torch.cuda.is_available()}\")"
check "Piper voice exists"          "test -f /models/piper/en_US-lessac-medium.onnx"
check "voiceserver built"           "test -f /opt/voiceserver/dist/index.js"

# Check if voiceserver is actually running
sleep 2
if pm2 list | grep -q "voiceserver.*online"; then
  log "voiceserver is running (pm2)"
else
  warn "voiceserver may still be starting up — check: pm2 logs voiceserver"
fi

echo ""
echo "============================================================"

if [ "$PASS" = true ]; then
  echo -e "  ${GREEN}Setup complete!${NC}"
else
  echo -e "  ${YELLOW}Setup finished with some warnings — check above${NC}"
fi

echo ""
echo "  GPU:  ${GPU_NAME} (${GPU_VRAM} MiB)"
echo ""
echo "  Services:"
echo "    WebSocket:  ws://0.0.0.0:8765"
echo "    IPC API:    http://0.0.0.0:8766"
echo "    Ollama:     http://localhost:11434"
echo ""
echo "  Useful commands:"
echo "    pm2 logs voiceserver     — view live logs"
echo "    pm2 restart voiceserver  — restart voiceserver"
echo "    pm2 monit                — monitor CPU/memory"
echo "    nvidia-smi               — check GPU usage"
echo "    ollama list              — list downloaded models"
echo "    ollama pull <model>      — download additional models"
echo ""
echo "  Config: ${VOICESERVER_DIR}/.env"
echo "  Logs:   /var/log/voiceserver/"
echo ""
echo "  Next steps:"
echo "    1. Point vapiclone's VOICESERVER_URL to this server's IP:8766"
echo "    2. Point dialer4clone's WS URL to this server's IP:8765"
echo "    3. Make a test call to verify the pipeline works end-to-end"
echo ""
echo "============================================================"
