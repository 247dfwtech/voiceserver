#!/bin/bash
###############################################################################
# setup-gpu-server.sh — Automated bare metal setup for voiceserver on GPU VPS
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/247dfwtech/voiceserver/main/scripts/setup-gpu-server.sh | bash
#   — OR —
#   chmod +x setup-gpu-server.sh && ./setup-gpu-server.sh
#
# Tested on: Ubuntu 22.04 / 24.04 with NVIDIA RTX 4090, CUDA 12.x
# Also works when copying/cloning an existing Vast.ai instance.
#
# Expected VRAM usage: ~7GB baseline / ~11GB with Chatterbox / 24GB total
#   - Whisper STT (small.en): ~0.5GB (persistent subprocess, loads once)
#   - Kokoro TTS:              ~0.5GB (persistent subprocess, loads once)
#   - Qwen3.5:9b:              ~6.0GB (via Ollama)
#   - Chatterbox Turbo:        ~4.2GB (persistent subprocess, 5-min idle timeout)
#   - Deepgram STT:            0GB (cloud API, no GPU — needs DEEPGRAM_API_KEY)
#   - Headroom:                ~13GB
#
# ⚠️  VAST.AI INSTANCE COPY NOTE:
#   When you copy a Vast.ai instance, /etc/environment contains:
#     HF_HOME="/workspace/.hf_home"
#   The /workspace volume is STALE and causes Kokoro/Granite to fail with:
#     OSError: [Errno 116] Stale file handle
#   This script fixes that automatically.
#
# WHAT THIS SCRIPT SETS UP:
#   1. System deps (python3, ffmpeg, libsndfile1, etc.)
#   2. Node.js (NVM if present, else NodeSource) + PM2
#   3. Ollama + default LLM model (qwen3.5:9b — recommended for call quality)
#   4. PyTorch with CUDA (correct wheel for detected CUDA version)
#   5. Kokoro TTS + faster-whisper + transformers Python packages
#   6. KokoClone voice cloning dependencies (kokoro-onnx, kanade-tokenizer, torchaudio)
#   7. Miniconda + Chatterbox Turbo conda env (python 3.11) + model download
#   8. Pre-downloads ALL AI models to HuggingFace cache:
#      - Whisper small.en (~244MB) + base.en (~74MB) — primary STT
#      - hexgrad/Kokoro-82M (~313MB) + all 4 voice packs (af_heart, af_nicole, am_adam, af_sarah)
#      - ResembleAI/chatterbox-turbo (~4GB) — voice cloning
#   9. Clones/updates voiceserver repo, npm install, tsc build
#  10. Creates .env with all required keys (including DEEPGRAM_API_KEY placeholder)
#  11. Creates cloudflared tunnel scripts
#  12. Creates PM2 ecosystem config (fork mode) with all 5 processes
#  13. Starts all processes, saves PM2 config for boot persistence
#  14. Verifies GPU monitoring endpoints (/health with GPU data, /metrics/history)
#  15. Prints tunnel URLs for Railway env var setup
#
# POST-SETUP: API keys (Deepgram, OpenAI, etc.) can be set via:
#   - VapiClone Settings page (auto-syncs to GPU server via /settings API)
#   - GitHub Secrets (auto-synced on deploy via GitHub Actions)
#   - Manual: echo 'DEEPGRAM_API_KEY=xxx' >> /opt/voiceserver/.env
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
HF_HOME_DIR="/root/.cache/huggingface"

# IPC/auth secrets — these MUST match what you set in vapiclone Railway env vars
# VAPICLONE_API_KEY = the API_KEY value from Railway vapiclone service
# IPC_SECRET = the IPC_SECRET value from Railway vapiclone service
# Update these before running if your values differ:
VAPICLONE_API_KEY_DEFAULT="vs-internal-secret-2026"
IPC_SECRET_DEFAULT="vs-ipc-2026"
VAPICLONE_API_URL_DEFAULT="https://vapiclone-production.up.railway.app"

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
echo "  STT: Whisper small.en (FREE default) + Deepgram Flux (PAID cloud)"
echo "  TTS: Kokoro-82M (persistent subprocess, default)"
echo "  TTS: Chatterbox Turbo (voice cloning, conda env)"
echo "  TTS: KokoClone (voice cloning, system Python)"
echo "  LLM: Qwen3.5:9b via Ollama (auto-pulled)"
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
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}')
log "GPU detected: ${GPU_NAME} (${GPU_VRAM} MiB VRAM, CUDA ${CUDA_VERSION})"

if [ "${GPU_VRAM}" -lt 8000 ]; then
  warn "Less than 8GB VRAM detected. Whisper STT + Kokoro TTS + qwen3.5:9b require ~7GB total."
  warn "Consider using Deepgram STT (cloud, 0 VRAM) and a smaller LLM model."
fi

###############################################################################
# 0. Fix Vast.ai HF_HOME stale file handle (CRITICAL for copied instances)
###############################################################################

info "Fixing HuggingFace cache directory..."

# Vast.ai sets HF_HOME=/workspace/.hf_home in /etc/environment.
# When copying an instance, /workspace is a stale NFS mount that causes:
#   OSError: [Errno 116] Stale file handle
# Fix: redirect HF_HOME to /root/.cache/huggingface (always writable).

mkdir -p "${HF_HOME_DIR}"

if grep -q 'HF_HOME' /etc/environment 2>/dev/null; then
  sed -i "s|HF_HOME=.*|HF_HOME=\"${HF_HOME_DIR}\"|g" /etc/environment
  log "Fixed HF_HOME in /etc/environment → ${HF_HOME_DIR}"
else
  echo "HF_HOME=\"${HF_HOME_DIR}\"" >> /etc/environment
  log "Added HF_HOME to /etc/environment → ${HF_HOME_DIR}"
fi

# Also export for current session
export HF_HOME="${HF_HOME_DIR}"

###############################################################################
# 1. System dependencies
###############################################################################

info "Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
apt-get install -y \
  curl wget git build-essential \
  python3 python3-pip python3-venv \
  ffmpeg sox libsndfile1 \
  htop tmux unzip jq \
  2>&1 | tail -1

log "System dependencies installed"

###############################################################################
# 2. Node.js (via NVM if already present, else NodeSource)
###############################################################################

# Check if NVM-installed node exists (common on Vast.ai)
NVM_NODE="/opt/nvm/versions/node/v24.12.0/bin"
if [ -d "${NVM_NODE}" ]; then
  export PATH="${NVM_NODE}:${PATH}"
  log "Node.js $(node -v) found via NVM"
elif command -v node &>/dev/null; then
  log "Node.js $(node -v) already installed"
else
  info "Installing Node.js 22 via NodeSource..."
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
  apt-get install -y nodejs
  log "Node.js $(node -v) installed"
fi

# Ensure PATH is set for PM2 in current session
export PATH="/opt/nvm/versions/node/v24.12.0/bin:/opt/instance-tools/bin:${PATH}"

# Install pm2 globally
if ! command -v pm2 &>/dev/null; then
  info "Installing pm2..."
  npm install -g pm2
  log "pm2 installed"
else
  log "pm2 $(pm2 -v) already installed"
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

# Start Ollama with required env vars for cloudflared tunnel access
info "Starting Ollama (with OLLAMA_ORIGINS=* for tunnel access)..."
pkill ollama 2>/dev/null || true
sleep 1
OLLAMA_ORIGINS="*" OLLAMA_HOST="0.0.0.0" nohup ollama serve > /var/log/ollama.log 2>&1 &
sleep 3

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

# Pull default LLM model — qwen3.5:9b recommended for call quality
info "Pulling qwen3.5:9b (recommended LLM, ~6GB)... this may take a few minutes"
if ollama list 2>/dev/null | grep -q "qwen3.5:9b"; then
  log "qwen3.5:9b already downloaded"
else
  ollama pull qwen3.5:9b
  log "qwen3.5:9b downloaded"
fi

###############################################################################
# 4. Python AI packages (with CUDA-enabled PyTorch)
###############################################################################

info "Installing Python AI packages..."

# Upgrade pip
python3 -m pip install --upgrade pip -q

# Detect CUDA version for correct PyTorch wheel
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)

# PyTorch supports: cu118, cu121, cu124, cu126 — use closest available
if [ "${CUDA_MAJOR}" -ge 12 ] && [ "${CUDA_MINOR}" -ge 6 ]; then
  TORCH_CUDA_TAG="cu126"
elif [ "${CUDA_MAJOR}" -ge 12 ] && [ "${CUDA_MINOR}" -ge 4 ]; then
  TORCH_CUDA_TAG="cu124"
elif [ "${CUDA_MAJOR}" -ge 12 ] && [ "${CUDA_MINOR}" -ge 1 ]; then
  TORCH_CUDA_TAG="cu121"
else
  TORCH_CUDA_TAG="cu118"
fi

TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"
info "Installing PyTorch with CUDA ${TORCH_CUDA_TAG} (CUDA ${CUDA_VERSION} detected)..."
pip3 install --no-cache-dir torch torchaudio --index-url "${TORCH_INDEX}" -q 2>&1 | tail -2
log "PyTorch + torchaudio installed (CUDA-enabled)"

# Verify CUDA is available in torch
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in torch!'; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}')"
log "PyTorch CUDA verified"

# Kokoro-82M TTS (default) — requires torch
info "Installing Kokoro TTS..."
pip3 install --no-cache-dir "kokoro>=0.9" soundfile -q 2>&1 | tail -1
log "Kokoro TTS installed"

# faster-whisper (PRIMARY STT — free, local GPU, keyword biasing)
info "Installing faster-whisper (primary STT)..."
pip3 install --no-cache-dir faster-whisper -q 2>&1 | tail -1
log "faster-whisper installed"

# huggingface_hub for model downloads
pip3 install --no-cache-dir huggingface_hub soundfile -q 2>&1 | tail -1

# Piper TTS (fallback for CPU-only)
info "Installing piper-tts (TTS fallback)..."
pip3 install --no-cache-dir piper-tts -q 2>&1 | tail -1
log "piper-tts installed"

# KokoClone voice cloning dependencies (uses system Python)
info "Installing KokoClone voice cloning dependencies..."
pip3 install --no-cache-dir kokoro-onnx torchaudio soundfile -q 2>&1 | tail -1
pip3 install --no-cache-dir git+https://github.com/frothywater/kanade-tokenizer -q 2>&1 | tail -1
log "KokoClone dependencies installed (kokoro-onnx, kanade-tokenizer)"

###############################################################################
# 5. Miniconda + Chatterbox Turbo conda env
###############################################################################

CONDA_BIN="/root/miniconda3/bin/conda"

if [ -x "${CONDA_BIN}" ]; then
  log "Miniconda already installed at /root/miniconda3/"
else
  info "Installing Miniconda..."
  curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p /root/miniconda3
  rm -f /tmp/miniconda.sh
  log "Miniconda installed"
fi

# Add conda to PATH for current session and future shells
export PATH="/root/miniconda3/bin:${PATH}"
if ! grep -q 'miniconda3/bin' /root/.bashrc 2>/dev/null; then
  echo 'export PATH=/root/miniconda3/bin:$PATH' >> /root/.bashrc
fi

# Create chatterbox conda env if it doesn't exist
if ${CONDA_BIN} env list 2>/dev/null | grep -q chatterbox; then
  log "Chatterbox conda env already exists"
else
  info "Creating Chatterbox conda env (Python 3.11)..."
  ${CONDA_BIN} create -yn chatterbox python=3.11 -q 2>&1 | tail -3
  log "Chatterbox conda env created"
fi

# Install Chatterbox Turbo into conda env
info "Installing Chatterbox Turbo into conda env..."

# Clone Chatterbox repo if not present
if [ -d "/opt/chatterbox-repo/.git" ]; then
  info "Updating existing Chatterbox repo..."
  cd /opt/chatterbox-repo && git pull origin main 2>&1 | tail -2
else
  info "Cloning Chatterbox repo..."
  git clone https://github.com/resemble-ai/chatterbox.git /opt/chatterbox-repo 2>&1 | tail -2
fi

cd /opt/chatterbox-repo
${CONDA_BIN} run -n chatterbox pip install -e . -q 2>&1 | tail -3
log "Chatterbox Turbo package installed"

# Fix setuptools for resemble-perth watermarker (pkg_resources removed in v72+)
info "Fixing setuptools for resemble-perth compatibility..."
${CONDA_BIN} run -n chatterbox pip install 'setuptools<71' -q 2>&1 | tail -1
log "setuptools pinned for perth compatibility"

# Verify Chatterbox import works
info "Verifying Chatterbox Turbo import..."
if ${CONDA_BIN} run -n chatterbox python3 -c "from chatterbox.tts_turbo import ChatterboxTurboTTS; print('ok')" 2>&1 | grep -q "ok"; then
  log "Chatterbox Turbo import verified"
else
  err "Chatterbox Turbo import failed — check conda env"
fi

###############################################################################
# 6. Pre-download AI models (with HF_HOME set correctly)
###############################################################################

mkdir -p "${MODELS_DIR}/piper" "${MODELS_DIR}/kokoro"
mkdir -p "${DATA_DIR}/cloned-voices" "${DATA_DIR}/chatterbox-voices/references"

# Kokoro-82M TTS model + voice packs
# The KPipeline downloads lazily per voice on first synthesis. Pre-downloading
# all voice packs avoids 404 errors during live calls.
info "Pre-downloading Kokoro-82M model and all voice packs (~400MB total)..."
HF_HOME="${HF_HOME_DIR}" python3 -c "
from kokoro import KPipeline
import sys

print('Loading Kokoro-82M model...')
# Initialize with American English (downloads base model)
pipeline = KPipeline(lang_code='a')

# Pre-download all commonly used voice packs by running a silent synthesis
print('Pre-downloading voice packs (af_heart, af_nicole, am_adam, af_sarah)...')
for voice in ['af_heart', 'af_nicole', 'am_adam', 'af_sarah']:
    try:
        list(pipeline('test', voice=voice, speed=1.0))
        print(f'  voice pack {voice}: OK')
    except Exception as e:
        print(f'  voice pack {voice}: {e}', file=sys.stderr)

print('Kokoro model and voice packs ready')
" 2>&1
log "Kokoro TTS model and voice packs downloaded"

# Whisper small.en (PRIMARY STT — recommended for phone audio quality)
info "Pre-downloading Whisper small.en (primary STT, ~244MB)..."
HF_HOME="${HF_HOME_DIR}" python3 -c "
from faster_whisper import WhisperModel
print('Downloading Whisper small.en...')
WhisperModel('small.en', device='cpu', compute_type='int8')
print('Whisper small.en ready')
" 2>&1
log "Whisper small.en downloaded"

# Also download base.en as fast fallback option
info "Pre-downloading Whisper base.en (fallback, ~74MB)..."
HF_HOME="${HF_HOME_DIR}" python3 -c "
from faster_whisper import WhisperModel
WhisperModel('base.en', device='cpu', compute_type='int8')
print('Whisper base.en ready')
" 2>&1
log "Whisper base.en downloaded"

# Granite STT (OPTIONAL / EXPERIMENTAL — non-standard API, not recommended as default)
# Uncomment if you want Granite available as an option:
# info "Pre-downloading Granite 4.0 1B Speech STT model (~2GB)..."
# HF_HOME="${HF_HOME_DIR}" python3 -c "
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
# model_id = 'ibm-granite/granite-4.0-1b-speech'
# AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
# AutoModelForSpeechSeq2Seq.from_pretrained(model_id, trust_remote_code=True)
# print('Granite STT model ready')
# " 2>&1
# log "Granite STT model downloaded"

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

# Chatterbox Turbo model (voice cloning TTS — downloads from HuggingFace)
info "Pre-downloading Chatterbox Turbo model (~4GB)..."
HF_HOME="${HF_HOME_DIR}" ${CONDA_BIN} run -n chatterbox python3 -c "
from huggingface_hub import snapshot_download
import os
path = snapshot_download(
    repo_id='ResembleAI/chatterbox-turbo',
    token=False,
    allow_patterns=['*.safetensors', '*.json', '*.txt', '*.pt', '*.model']
)
print(f'Chatterbox Turbo model downloaded to: {path}')
files = os.listdir(path)
print(f'Files: {len(files)} ({", ".join(files[:5])}...)')
" 2>&1
log "Chatterbox Turbo model downloaded"

# Verify Chatterbox model loads on GPU
info "Verifying Chatterbox Turbo model loads on GPU..."
HF_HOME="${HF_HOME_DIR}" HF_TOKEN=skip ${CONDA_BIN} run -n chatterbox python3 -c "
import os, torch
os.environ['HF_TOKEN'] = 'skip'
from chatterbox.tts_turbo import ChatterboxTurboTTS
model = ChatterboxTurboTTS.from_pretrained(device='cuda')
print(f'Chatterbox Turbo loaded: sr={model.sr}, GPU={torch.cuda.memory_allocated()//1024//1024}MB')
del model
torch.cuda.empty_cache()
print('Model unloaded, GPU memory freed')
" 2>&1
log "Chatterbox Turbo model verified on GPU"

###############################################################################
# 7. Clone and build voiceserver
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
npm install --silent 2>&1 | tail -2

info "Building TypeScript..."
npm run build 2>&1 | tail -3
log "voiceserver built successfully"

###############################################################################
# 8. Environment configuration
###############################################################################

ENV_FILE="${VOICESERVER_DIR}/.env"

if [ ! -f "${ENV_FILE}" ]; then
  info "Creating .env file..."
  cat > "${ENV_FILE}" <<ENVEOF
# voiceserver environment configuration
# Generated by setup-gpu-server.sh on $(date)

# Server ports
WS_PORT=8765
IPC_PORT=8766

# Model directories
PIPER_MODELS_DIR=/models/piper
KOKORO_MODELS_DIR=/models/kokoro
KOKORO_VOICE=af_heart
CLONED_VOICES_DIR=/data/cloned-voices
CHATTERBOX_VOICES_DIR=/data/chatterbox-voices
CONDA_PATH=/root/miniconda3/bin/conda

# Ollama (local LLM)
OLLAMA_URL=http://localhost:11434/v1
DEFAULT_LLM=qwen3.5:9b

# HuggingFace cache (fixes stale /workspace mount on Vast.ai copies)
HF_HOME=${HF_HOME_DIR}

# VapiClone app URL and auth
# VAPICLONE_API_KEY must match the API_KEY env var in your Railway vapiclone service
# IPC_SECRET must match the IPC_SECRET env var in your Railway vapiclone service
VAPICLONE_API_URL=${VAPICLONE_API_URL_DEFAULT}
VAPICLONE_API_KEY=${VAPICLONE_API_KEY_DEFAULT}
IPC_SECRET=${IPC_SECRET_DEFAULT}

# Paid provider API keys (can also be set via VapiClone Settings UI)
# DEEPGRAM_API_KEY=
# ELEVENLABS_API_KEY=
# OPENAI_API_KEY=
# DEEPSEEK_API_KEY=

NODE_ENV=production
ENVEOF
  log ".env file created at ${ENV_FILE}"
else
  warn ".env file already exists — checking for missing keys..."
  if ! grep -q 'HF_HOME' "${ENV_FILE}"; then
    echo "HF_HOME=${HF_HOME_DIR}" >> "${ENV_FILE}"
    log "Added HF_HOME to existing .env"
  fi
  if ! grep -q 'VAPICLONE_API_KEY' "${ENV_FILE}"; then
    echo "VAPICLONE_API_KEY=${VAPICLONE_API_KEY_DEFAULT}" >> "${ENV_FILE}"
    log "Added VAPICLONE_API_KEY to existing .env"
  fi
  if ! grep -q 'IPC_SECRET' "${ENV_FILE}"; then
    echo "IPC_SECRET=${IPC_SECRET_DEFAULT}" >> "${ENV_FILE}"
    log "Added IPC_SECRET to existing .env"
  fi
  if ! grep -q 'VAPICLONE_API_URL' "${ENV_FILE}"; then
    echo "VAPICLONE_API_URL=${VAPICLONE_API_URL_DEFAULT}" >> "${ENV_FILE}"
    log "Added VAPICLONE_API_URL to existing .env"
  fi
  if ! grep -q 'CHATTERBOX_VOICES_DIR' "${ENV_FILE}"; then
    echo "CHATTERBOX_VOICES_DIR=/data/chatterbox-voices" >> "${ENV_FILE}"
    log "Added CHATTERBOX_VOICES_DIR to existing .env"
  fi
  if ! grep -q 'CONDA_PATH' "${ENV_FILE}"; then
    echo "CONDA_PATH=/root/miniconda3/bin/conda" >> "${ENV_FILE}"
    log "Added CONDA_PATH to existing .env"
  fi
fi

###############################################################################
# 9. Cloudflared tunnel scripts
###############################################################################

CLOUDFLARED_BIN=""
for p in /opt/instance-tools/bin/cloudflared /usr/local/bin/cloudflared $(which cloudflared 2>/dev/null); do
  if [ -x "$p" ]; then
    CLOUDFLARED_BIN="$p"
    break
  fi
done

if [ -z "$CLOUDFLARED_BIN" ]; then
  info "Installing cloudflared..."
  curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    -o /usr/local/bin/cloudflared
  chmod +x /usr/local/bin/cloudflared
  CLOUDFLARED_BIN="/usr/local/bin/cloudflared"
  log "cloudflared installed"
else
  log "cloudflared found at ${CLOUDFLARED_BIN}"
fi

# Create tunnel scripts
cat > "${VOICESERVER_DIR}/start-tunnel-ipc.sh" <<EOF
#!/bin/bash
# Cloudflared quick tunnel for IPC HTTP (port 8766)
exec ${CLOUDFLARED_BIN} tunnel --no-tls-verify --url http://localhost:8766
EOF

cat > "${VOICESERVER_DIR}/start-tunnel-ws.sh" <<EOF
#!/bin/bash
# Cloudflared quick tunnel for WebSocket (port 8765)
exec ${CLOUDFLARED_BIN} tunnel --no-tls-verify --url http://localhost:8765
EOF

cat > "${VOICESERVER_DIR}/start-tunnel-ollama.sh" <<EOF
#!/bin/bash
# Cloudflared quick tunnel for Ollama API (port 11434)
exec ${CLOUDFLARED_BIN} tunnel --no-tls-verify --url http://localhost:11434
EOF

cat > "${VOICESERVER_DIR}/start-ollama.sh" <<EOF
#!/bin/bash
# Start Ollama with CORS open for cloudflared tunnel access
# OLLAMA_ORIGINS=* is REQUIRED — without it, Ollama returns 403 to tunnel requests
export OLLAMA_ORIGINS="*"
export OLLAMA_HOST="0.0.0.0"
exec ollama serve
EOF

chmod +x \
  "${VOICESERVER_DIR}/start-tunnel-ipc.sh" \
  "${VOICESERVER_DIR}/start-tunnel-ws.sh" \
  "${VOICESERVER_DIR}/start-tunnel-ollama.sh" \
  "${VOICESERVER_DIR}/start-ollama.sh"

log "Cloudflared tunnel scripts created"

###############################################################################
# 10. PM2 process management
###############################################################################

info "Setting up pm2..."

mkdir -p /var/log/voiceserver

# Create pm2 ecosystem file
# CRITICAL NOTES:
#   - HF_HOME must be set here so Python subprocesses (Kokoro/Granite) use the
#     correct cache dir, not /workspace/.hf_home (stale on Vast.ai copies)
#   - VAPICLONE_API_KEY and IPC_SECRET are passed through so voiceserver can
#     authenticate when notifying vapiclone of call events
#   - instances: 1 is required (cluster mode causes issues with persistent
#     Python subprocesses shared at module level)
cat > "${VOICESERVER_DIR}/ecosystem.config.cjs" <<PM2EOF
module.exports = {
  apps: [
    {
      name: "ollama",
      script: "${VOICESERVER_DIR}/start-ollama.sh",
      autorestart: true,
      error_file: "/var/log/voiceserver/ollama-error.log",
      out_file: "/var/log/voiceserver/ollama.log",
      merge_logs: true,
      time: true,
    },
    {
      name: "voiceserver",
      script: "dist/index.js",
      cwd: "${VOICESERVER_DIR}",
      env: {
        NODE_ENV: "production",
        HF_HOME: "${HF_HOME_DIR}",
      },
      instances: 1,
      exec_mode: "fork",
      autorestart: true,
      watch: false,
      max_memory_restart: "3G",
      error_file: "/var/log/voiceserver/error.log",
      out_file: "/var/log/voiceserver/out.log",
      merge_logs: true,
      time: true,
    },
    {
      name: "tunnel-ipc",
      script: "${VOICESERVER_DIR}/start-tunnel-ipc.sh",
      autorestart: true,
      error_file: "/var/log/voiceserver/tunnel-ipc-error.log",
      out_file: "/var/log/voiceserver/tunnel-ipc.log",
      merge_logs: true,
      time: true,
    },
    {
      name: "tunnel-ws",
      script: "${VOICESERVER_DIR}/start-tunnel-ws.sh",
      autorestart: true,
      error_file: "/var/log/voiceserver/tunnel-ws-error.log",
      out_file: "/var/log/voiceserver/tunnel-ws.log",
      merge_logs: true,
      time: true,
    },
    {
      name: "tunnel-ollama",
      script: "${VOICESERVER_DIR}/start-tunnel-ollama.sh",
      autorestart: true,
      error_file: "/var/log/voiceserver/tunnel-ollama-error.log",
      out_file: "/var/log/voiceserver/tunnel-ollama.log",
      merge_logs: true,
      time: true,
    },
  ],
};
PM2EOF

# Kill any stale ollama process started earlier in this script
pkill ollama 2>/dev/null || true
sleep 1

# Stop/delete existing pm2 processes
pm2 stop all 2>/dev/null || true
pm2 delete all 2>/dev/null || true

# Start everything from ecosystem file
cd "${VOICESERVER_DIR}"
pm2 start ecosystem.config.cjs
pm2 save

# Set up pm2 to start on boot
pm2 startup systemd -u root --hp /root 2>/dev/null || pm2 startup 2>/dev/null || true
pm2 save

log "All PM2 processes started (voiceserver + ollama + 3 cloudflared tunnels)"

###############################################################################
# 11. Wait for tunnels and show URLs
###############################################################################

info "Waiting for cloudflared tunnels to start (15s)..."
sleep 15

echo ""
echo "  ┌─────────────────────────────────────────────────────────────────┐"
echo "  │  Cloudflared Tunnel URLs — copy these to Railway env vars       │"
echo "  └─────────────────────────────────────────────────────────────────┘"

IPC_URL=$(grep -a 'trycloudflare.com' /var/log/voiceserver/tunnel-ipc.log 2>/dev/null | grep -o 'https://[^ ]*trycloudflare.com' | tail -1 || echo "starting — check: pm2 logs tunnel-ipc")
WS_URL=$(grep -a 'trycloudflare.com' /var/log/voiceserver/tunnel-ws.log 2>/dev/null | grep -o 'https://[^ ]*trycloudflare.com' | tail -1 || echo "starting — check: pm2 logs tunnel-ws")
OLLAMA_TUNNEL=$(grep -a 'trycloudflare.com' /var/log/voiceserver/tunnel-ollama.log 2>/dev/null | grep -o 'https://[^ ]*trycloudflare.com' | tail -1 || echo "starting — check: pm2 logs tunnel-ollama")

echo ""
echo "    Railway env var                  Value"
echo "    ─────────────────────────────────────────────────────────────────"
echo "    VOICE_SERVER_IPC_URL           = ${IPC_URL}"
echo "    VOICE_SERVER_URL               = ${WS_URL/https/wss}"
echo "    OLLAMA_URL                     = ${OLLAMA_TUNNEL}/v1"
echo ""
echo "    Also set in Railway vapiclone:"
echo "    IPC_SECRET                     = ${IPC_SECRET_DEFAULT}"
echo "    VASTAI_INSTANCE_ID             = <your new instance ID>"
echo ""

###############################################################################
# 12. Verification
###############################################################################

echo "============================================================"
echo "  Verifying installation..."
echo "============================================================"
echo ""

PASS=true

check() {
  if eval "$2" &>/dev/null; then
    log "$1"
  else
    err "$1 — FAILED"
    PASS=false
  fi
}

check "Node.js $(node -v)"                    "node -v"
check "npm $(npm -v)"                         "npm -v"
check "pm2 installed"                         "pm2 -v"
check "Ollama running"                        "curl -s http://localhost:11434/api/tags"
check "qwen3.5:9b available"                  "ollama list | grep -q 'qwen3.5:9b'"
check "Python3 available"                     "python3 --version"
check "PyTorch installed (CUDA)"              "python3 -c 'import torch; assert torch.cuda.is_available()'"
check "Kokoro TTS installed"                  "python3 -c 'from kokoro import KPipeline'"
check "faster-whisper installed"              "python3 -c 'import faster_whisper'"
check "soundfile installed"                   "python3 -c 'import soundfile'"
check "KokoClone deps (kokoro-onnx)"          "python3 -c 'import kokoro_onnx'"
check "KokoClone deps (kanade-tokenizer)"     "python3 -c 'from kanade_tokenizer import KanadeModel'"
check "Miniconda installed"                   "test -x /root/miniconda3/bin/conda"
check "Chatterbox conda env exists"           "${CONDA_BIN} env list | grep -q chatterbox"
check "Chatterbox Turbo importable"           "${CONDA_BIN} run -n chatterbox python3 -c 'from chatterbox.tts_turbo import ChatterboxTurboTTS; print(\"ok\")' 2>&1 | grep -q ok"
check "Chatterbox Turbo model cached"         "test -d ${HF_HOME_DIR}/hub/models--ResembleAI--chatterbox-turbo"
check "Piper voice exists"                    "test -f /models/piper/en_US-lessac-medium.onnx"
check "voiceserver built"                     "test -f /opt/voiceserver/dist/index.js"
check "HF_HOME set correctly"                 "test -d ${HF_HOME_DIR}"
check "Kokoro model cached"                   "test -d ${HF_HOME_DIR}/hub/models--hexgrad--Kokoro-82M"
check "Whisper small.en cached"               "test -d ${HF_HOME_DIR}/hub/models--Systran--faster-whisper-small.en"
check "cloudflared available"                 "test -x ${CLOUDFLARED_BIN}"
check ".env has VAPICLONE_API_KEY"            "grep -q VAPICLONE_API_KEY ${ENV_FILE}"
check ".env has IPC_SECRET"                   "grep -q IPC_SECRET ${ENV_FILE}"
check ".env has CONDA_PATH"                   "grep -q CONDA_PATH ${ENV_FILE}"
check "Chatterbox voices dir exists"          "test -d /data/chatterbox-voices/references"
check "KokoClone voices dir exists"           "test -d /data/cloned-voices"

# Check if voiceserver is actually running
sleep 3
if pm2 list | grep -q "voiceserver.*online"; then
  log "voiceserver is running (pm2)"
else
  warn "voiceserver may still be starting up — check: pm2 logs voiceserver"
fi

# Test health endpoint (now includes GPU data)
sleep 3
if curl -s http://localhost:8766/health | grep -q '"ok":true'; then
  log "Voice server health check passed"
  # Verify GPU monitoring is working
  if curl -s http://localhost:8766/health | grep -q '"gpu":{'; then
    log "GPU monitoring active (nvidia-smi integration)"
  else
    warn "GPU data not in health response — nvidia-smi may not be available"
  fi
  # Verify metrics history endpoint
  if curl -s http://localhost:8766/metrics/history?range=1h | grep -q '"snapshots"'; then
    log "Metrics history endpoint working (3-day ring buffer)"
  fi
else
  warn "Voice server health check not ready yet — may still be loading models"
fi

echo ""
echo "============================================================"

if [ "$PASS" = true ]; then
  echo -e "  ${GREEN}Setup complete!${NC}"
else
  echo -e "  ${YELLOW}Setup finished with some warnings — check above${NC}"
fi

echo ""
echo "  GPU:  ${GPU_NAME} (${GPU_VRAM} MiB VRAM)"
echo ""
echo "  Services (PM2):"
echo "    voiceserver   — WebSocket ws://0.0.0.0:8765 + IPC http://0.0.0.0:8766"
echo "    ollama        — http://localhost:11434 (OLLAMA_ORIGINS=*)"
echo "    tunnel-ipc    — cloudflared → localhost:8766"
echo "    tunnel-ws     — cloudflared → localhost:8765"
echo "    tunnel-ollama — cloudflared → localhost:11434"
echo ""
echo "  ⚠️  IMPORTANT: Set these in Railway vapiclone env vars:"
echo "    VOICE_SERVER_IPC_URL = ${IPC_URL:-<check pm2 logs tunnel-ipc>}"
echo "    VOICE_SERVER_URL     = ${WS_URL/https/wss:-<check pm2 logs tunnel-ws>}"
echo "    OLLAMA_URL           = ${OLLAMA_TUNNEL:-<check pm2 logs tunnel-ollama>}/v1"
echo "    IPC_SECRET           = ${IPC_SECRET_DEFAULT}"
echo "    VASTAI_INSTANCE_ID   = <new instance ID>"
echo ""
echo "  ⚠️  Tunnel URLs are EPHEMERAL — they change when PM2 restarts tunnels."
echo "     After any server reboot or tunnel restart, get new URLs and update Railway."
echo ""
echo "  Deploy new voiceserver code:"
echo "    export PATH=\"/opt/nvm/versions/node/v24.12.0/bin:/opt/instance-tools/bin:\$PATH\""
echo "    cd /opt/voiceserver && git pull && npm run build && pm2 restart voiceserver --update-env"
echo ""
echo "  Check logs:"
echo "    pm2 logs voiceserver                     — live logs"
echo "    pm2 logs tunnel-ipc --lines 5 --nostream — get IPC tunnel URL"
echo "    pm2 logs tunnel-ws --lines 5 --nostream  — get WS tunnel URL"
echo "    pm2 monit                                — monitor CPU/memory"
echo "    nvidia-smi                               — check GPU usage"
echo "    ollama list                              — list downloaded models"
echo ""
echo "  Config:  ${VOICESERVER_DIR}/.env"
echo "  Logs:    /var/log/voiceserver/"
echo "  Models:  ${HF_HOME_DIR}/hub/"
echo ""
echo "============================================================"
