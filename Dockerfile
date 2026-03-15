FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    python3 \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 22
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install faster-whisper CLI
RUN pip3 install --no-cache-dir faster-whisper

# Install piper-tts (fallback TTS)
RUN pip3 install --no-cache-dir piper-tts

# Install Kokoro-82M TTS (default TTS -- #1 TTS Arena, near-human quality)
RUN pip3 install --no-cache-dir kokoro>=0.9 soundfile

# Install Granite STT dependencies (transformers + torch + soundfile)
RUN pip3 install --no-cache-dir transformers torch huggingface_hub

# Pre-download the default Granite STT model (ibm-granite/granite-4.0-1b-speech)
RUN python3 -c "\
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor; \
AutoProcessor.from_pretrained('ibm-granite/granite-4.0-1b-speech', trust_remote_code=True); \
AutoModelForSpeechSeq2Seq.from_pretrained('ibm-granite/granite-4.0-1b-speech', trust_remote_code=True); \
print('Granite STT model downloaded')"

# Pre-download Kokoro-82M model weights (so first TTS call is instant)
RUN python3 -c "\
from kokoro import KPipeline; \
pipeline = KPipeline(lang_code='a'); \
print('Kokoro-82M TTS model downloaded')"

# Pre-download the default Whisper model (kept as fallback)
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('base.en')"

# Download default Piper voice (kept as fallback)
RUN mkdir -p /models/piper && \
    wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" \
      -O /models/piper/en_US-lessac-medium.onnx && \
    wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" \
      -O /models/piper/en_US-lessac-medium.onnx.json

# Set up app directory
WORKDIR /app

# Copy package files and install dependencies
COPY package.json package-lock.json* ./
RUN npm install --production=false

# Copy source code
COPY . .

# Build TypeScript
RUN npm run build

# Clean dev dependencies
RUN npm prune --production

# Expose WebSocket and IPC ports
EXPOSE 8765 8766

# Set default environment variables
ENV WS_PORT=8765
ENV IPC_PORT=8766
ENV PIPER_MODELS_DIR=/models/piper
ENV WHISPER_MODEL=base.en
ENV GRANITE_MODELS_DIR=/models/granite
ENV KOKORO_MODELS_DIR=/models/kokoro
ENV KOKORO_VOICE=af_heart
ENV NODE_ENV=production

CMD ["node", "dist/index.js"]
