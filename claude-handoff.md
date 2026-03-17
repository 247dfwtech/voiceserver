# VoiceServer — Claude Handoff Document

**Last updated:** 2026-03-17 (rev 2)
**GitHub:** https://github.com/247dfwtech/voiceserver
**Local path:** /Users/adriansanchez/Desktop/voiceserver
**Running on:** Vast.ai Reserved GPU Instance #33032104 (Quebec, CA — RTX 4090)
**SSH:** `ssh -p 45194 root@70.29.210.33` (direct IP — Vast.ai proxy `ssh5.vast.ai:32104` may have routing issues)
**Server path:** /opt/voiceserver/

---

## What Is This?

VoiceServer is the GPU-powered voice processing engine that handles real-time phone calls. It receives audio from Twilio via WebSocket, transcribes speech (STT), generates AI responses (LLM), synthesizes speech (TTS), and streams audio back to Twilio. All AI models run locally on a dedicated RTX 4090 GPU — no paid API calls for core voice processing.

It's the backend engine for VapiClone. VapiClone handles orchestration (assistants, phone numbers, call logs). VoiceServer handles the actual voice processing.

---

## Current State (What's Working)

- **WebSocket server** (port 8765) — Accepts Twilio Media Stream connections, processes audio in real-time
- **IPC HTTP server** (port 8766) — Health checks, model management, call registration, TTS testing, voice cloning
- **IBM Granite 4.0 1B Speech** — Default STT, #1 on OpenASR leaderboard, supports keyword biasing
- **Kokoro-82M TTS** — Default TTS, #1 in TTS Spaces Arena, 54 voices, sub-0.3s latency on RTX 4090
- **Ollama with qwen3:4b** — Default LLM for conversation and post-call analysis
- **KokoClone voice cloning** — Clone voices from 3-10s reference audio samples
- **Model manager** — Install/activate/remove LLM/STT/TTS models via IPC API
- **Post-call analysis** — Summary generation and success evaluation after each call
- **Tool execution** — Function webhooks, call transfer, DTMF, end call (with SSRF protection)
- **Voicemail detection** — Detects sustained speech patterns in first 5 seconds
- **Cost tracking** — Per-call breakdown (STT, LLM, TTS, transport). Local models are $0.
- **PM2 managed** — voiceserver + 2 cloudflared tunnels, auto-restart, boot persistence
- **Cloudflared tunnels** — Public HTTPS/WSS URLs for IPC and WebSocket access from Railway
- **Health endpoint** — Reports sessions, memory, uptime, Ollama status, installed models

## What's Not Working / Known Issues

- **Cloudflared tunnel URLs are ephemeral** — Quick tunnels generate random `trycloudflare.com` URLs. If tunnel processes restart (PM2 auto-restart, server reboot), URLs change and Railway env vars need manual updating. A proper Cloudflare named tunnel would fix this.
- **Ollama is now PM2-managed** — Runs via `start-ollama.sh` with `OLLAMA_ORIGINS=*` and `OLLAMA_HOST=0.0.0.0` to allow cloudflared tunnel access.
- **GitHub Actions auto-deploy needs SSH secret update** — The deploy workflow (`.github/workflows/deploy.yml`) has secrets for the old Vast.ai instance. SSH host (174.92.170.239), port (45194), and SSH key need updating in GitHub repo settings.
- **Granite STT not verified on new instance** — The Granite STT Python dependencies (transformers, torch, torchaudio) were not explicitly installed on the new instance. Kokoro TTS Python package was installed but Granite STT may need: `pip3 install transformers torch torchaudio soundfile`.
- **faster-whisper not installed on new instance** — If using Whisper as STT fallback, need: `pip3 install faster-whisper`.
- **Piper TTS not installed on new instance** — Piper binary and models not set up. Only Kokoro is available.
- **qwen3:4b instead of qwen3.5:9b** — The new instance has qwen3:4b pulled (smaller model). The .env references `qwen3.5:9b` as DEFAULT_LLM but the model manager log says it found 1 model already installed and skipped auto-pull. May need to pull qwen3.5:9b manually if better quality is needed: `ollama pull qwen3.5:9b`.

---

## What Was Just Completed (March 2026)

1. **Migrated to new reserved Vast.ai instance** — Old on-demand instance (32982226, Oregon) lost GPU availability overnight. New reserved instance (33032104, Quebec CA) set up from scratch.
2. **Fresh install on new instance:**
   - Installed Ollama + pulled qwen3:4b model
   - Installed PM2 globally via npm
   - Cloned voiceserver repo from GitHub, ran `npm install` + `npm run build`
   - Downloaded Kokoro TTS model files (kokoro-v1.0.onnx, voices-v1.0.bin) to /models/kokoro/
   - Installed PyTorch CUDA wheel (`pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu124`)
   - Installed kokoro Python package (`pip3 install kokoro soundfile`)
   - Fixed HF_HOME stale file handle: changed `/etc/environment` `HF_HOME` from `/workspace/.hf_home` to `/root/.cache/huggingface`
   - Created .env with production config (including `HF_HOME=/root/.cache/huggingface`)
   - Created PM2 ecosystem config with voiceserver + 2 cloudflared tunnels, HF_HOME in env
   - Configured PM2 startup persistence (`pm2 save` + `pm2 startup`)
3. **Cloudflared tunnels running via PM2:**
   - IPC tunnel: `https://plaintiff-databases-brave-isolation.trycloudflare.com` → localhost:8766
   - WS tunnel: `wss://cup-delegation-shake-stickers.trycloudflare.com` → localhost:8765
   - Ollama tunnel: `https://occupational-briefly-flag-cas.trycloudflare.com` → localhost:11434
4. **Ollama added to PM2** — Managed via `start-ollama.sh` with `OLLAMA_ORIGINS=*` and `OLLAMA_HOST=0.0.0.0` to allow cloudflared tunnel access. Without these env vars, Ollama returns 403 to non-localhost requests.
5. **VapiClone Railway env vars updated** to point to new tunnel URLs (IPC, WS, and Ollama).
6. **Voice server confirmed Connected** in VapiClone sidebar.
7. **Fixed Kokoro TTS timeouts** — Increased model load timeout 60s→120s and synthesis timeout 30s→90s in `src/providers/tts/kokoro.ts`. GPU cold start takes 60-90s on first call.
8. **Fixed /tts/test spinning forever** — Added a module-level KokoroTTS singleton in `src/index.ts`. Previously each Play button click spawned a new Python subprocess (60-90s cold start, always timed out). Now the singleton is reused across preview requests.
9. **Updated setup-gpu-server.sh** — Added HF_HOME fix, CUDA-aware PyTorch install, ollama PM2 management, all 5 PM2 processes, automatic verification checks.

---

## Key Decisions and Why

| Decision | Why |
|---|---|
| RTX 4090 on Vast.ai | Best price/performance for GPU inference. ~$0.30/hr reserved. Runs all 3 AI models (STT + LLM + TTS) with room to spare (~8.5GB of 24GB VRAM used). |
| Reserved instance instead of on-demand | On-demand lost the GPU overnight when stopped. Reserved guarantees availability. At ~$17-22/month for 24/7, it's very affordable. |
| Leave running 24/7, don't stop | Stopping risks losing the GPU (even reserved can have issues), and the cost is negligible. Avoids tunnel URL changes and PM2 restart hassles. |
| IBM Granite STT over Whisper | Better accuracy (#1 OpenASR), supports keyword biasing (important for sales — names, product terms), smaller model (2GB vs Whisper large-v3). |
| Kokoro TTS over ElevenLabs/Piper | #1 in TTS quality rankings, near-human naturalness, only 82M params, sub-0.3s latency, free (Apache 2.0). Piper is fallback for CPU-only environments. |
| Ollama for LLM | Runs locally on GPU, no API costs. OpenAI-compatible API makes it a drop-in replacement. Qwen3 models are strong for conversation. |
| Cloudflared quick tunnels | Vast.ai's Docker networking blocks direct port access from outside. Only 6 explicitly mapped ports work, and 8765/8766 aren't among them. Cloudflared creates public HTTPS/WSS URLs that work through Docker. |
| Separate voiceserver repo | Decouples voice processing from the web app. Can be deployed independently, version-controlled separately, and run on different hardware. |

---

## Next Steps (In Order)

1. **Install Granite STT dependencies** — `pip3 install transformers torch torchaudio soundfile` on the Vast.ai instance. Without this, STT falls back to Whisper (which also isn't installed). Critical for actual phone calls.
2. **Install faster-whisper** — `pip3 install faster-whisper` as STT fallback.
3. **Pull qwen3.5:9b model** — `ollama pull qwen3.5:9b` for better LLM quality (4b is smaller/faster but less capable).
4. **Update GitHub Actions SSH secrets** — In GitHub repo settings (247dfwtech/voiceserver), update `SSH_HOST`, `SSH_PORT`, `SSH_PRIVATE_KEY` for the new Vast.ai instance (174.92.170.239, port 45194).
5. **Set up Cloudflare named tunnel** — Replace quick tunnels with permanent named tunnel to get stable URLs that don't change on restart. Currently 3 tunnels (IPC, WS, Ollama) all use ephemeral URLs.
6. **Test end-to-end call** — Make a test call through VapiClone to verify the full audio pipeline works on the new instance.

---

## Gotchas and Important Context

- **HF_HOME stale file handle (Critical)** — When copying a Vast.ai instance, `/etc/environment` retains `HF_HOME="/workspace/.hf_home"` but `/workspace` is a stale NFS mount. This causes `OSError: [Errno 116] Stale file handle` in any Python code downloading HuggingFace models (Kokoro, Granite, Whisper). Fix: `sed -i 's|HF_HOME=.*|HF_HOME="/root/.cache/huggingface"|g' /etc/environment`, then `pm2 delete voiceserver && pm2 start ecosystem.config.cjs --only voiceserver` (restart alone doesn't pick up /etc/environment).
- **PyTorch CUDA wheel (Critical)** — `pip install torch` installs CPU-only version. Must use `pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu124` for GPU support. Without this, Kokoro fails with `ModuleNotFoundError: No module named 'torch'`.
- **PM2 needs PATH set** — Node.js is at `/opt/nvm/versions/node/v24.12.0/bin`. Always set PATH before running PM2 commands: `export PATH="/opt/nvm/versions/node/v24.12.0/bin:/opt/instance-tools/bin:$PATH"`.
- **`pm2 restart voiceserver --update-env`** vs full delete/start** — `--update-env` picks up PM2 ecosystem env changes but NOT `/etc/environment` changes. For `/etc/environment` changes, must do `pm2 delete voiceserver && pm2 start ecosystem.config.cjs --only voiceserver`.
- **Cloudflared is at `/opt/instance-tools/bin/cloudflared`** — Pre-installed on Vast.ai instances. No need to install separately.
- **Vast.ai "200 ports" is misleading** — Only 6 explicitly Docker-mapped ports are externally accessible (22→45194, 1111→45143, 6006→45087, 8080→45164, 8384→45035, 72299→45086). Internal ports like 8765/8766 are NOT accessible from outside despite the "200 ports" label.
- **Tunnel URLs change on restart** — If voice server shows "Offline" in VapiClone, check `pm2 logs tunnel-ipc --lines 5 --nostream` and `pm2 logs tunnel-ws --lines 5 --nostream` for new URLs. Update Railway env vars `VOICE_SERVER_IPC_URL` and `VOICE_SERVER_URL`.
- **Python packages are system-level** — No virtualenv on Vast.ai. `pip3 install` goes to system Python. The `kokoro-onnx` package is installed but Granite STT dependencies may be missing.
- **Model files location** — Kokoro models at `/models/kokoro/`, cloned voices at `/data/cloned-voices/`, Ollama models managed by Ollama internally (`~/.ollama/models/`).
- **Max 20 concurrent sessions** — Hardcoded in the server config. Adjustable via `MAX_CONCURRENT_CALLS` env var.
- **Cost tracking: local models = $0** — Ollama, Granite, Kokoro, Whisper, Piper are all free. Only Deepgram, OpenAI, ElevenLabs incur API costs if configured.
- **SSRF protection in tool executor** — Function tool webhooks block requests to localhost, private IP ranges (10.x, 172.16-31.x, 192.168.x), and cloud metadata endpoints (169.254.169.254).
- **Audio format conversion** — Twilio sends/receives mu-law 8kHz mono. Voice server converts to PCM 16-bit 16kHz for STT and back to mu-law for Twilio output.
- **Ollama needs OLLAMA_ORIGINS=*** — Without this env var, Ollama returns 403 to cloudflared tunnel requests. The `start-ollama.sh` script sets this. If Ollama is restarted manually, make sure to include this env var.

---

## Relevant File Paths

### Local (Desktop) / GitHub

| Path | What's In It |
|---|---|
| `src/index.ts` | Main entry — WebSocket server (8765) + IPC HTTP server (8766), session management |
| `src/model-manager.ts` | Model lifecycle — install/activate/remove LLM/STT/TTS, persist config |
| `src/providers/stt/granite.ts` | IBM Granite 4.0 1B Speech STT (Python subprocess) |
| `src/providers/stt/whisper.ts` | faster-whisper STT fallback |
| `src/providers/tts/kokoro.ts` | Kokoro-82M TTS (persistent Python subprocess, GPU) |
| `src/providers/tts/kokoclone.ts` | KokoClone voice cloning (reference audio → cloned voice) |
| `src/providers/tts/piper.ts` | Piper TTS fallback (CPU-friendly) |
| `src/providers/llm/openai-compat.ts` | OpenAI-compatible LLM client (works with Ollama, OpenAI, DeepSeek) |
| `src/providers/index.ts` | Provider factory — auto-selects based on env/config with fallbacks |
| `src/voice-pipeline/call-session.ts` | Core call handler — STT→LLM→TTS loop, tool execution, analysis |
| `src/voice-pipeline/analysis-runner.ts` | Post-call summary + success evaluation |
| `src/voice-pipeline/tool-executor.ts` | Tool execution with SSRF protection |
| `src/voice-pipeline/audio-utils.ts` | Audio format conversion (mu-law ↔ PCM, resampling) |
| `src/voice-pipeline/voicemail-detector.ts` | Voicemail detection from audio patterns |
| `package.json` | Dependencies, scripts (build: tsc, start: node dist/index.js) |
| `tsconfig.json` | TypeScript config (ES2022, Node module resolution) |
| `Dockerfile` | Multi-stage GPU build (nvidia/cuda base, pre-installs all models) |
| `docker-compose.yml` | Ollama + voiceserver services with GPU access |
| `.github/workflows/deploy.yml` | GitHub Actions — SSH into Vast.ai, git pull, build, pm2 restart |
| `scripts/setup-gpu-server.sh` | One-click bare metal setup script |

### Vast.ai GPU Server (70.29.210.33, SSH port 45194)

| Path | What's In It |
|---|---|
| `/opt/voiceserver/` | Cloned repo with built dist/ |
| `/opt/voiceserver/.env` | Production environment config |
| `/opt/voiceserver/ecosystem.config.cjs` | PM2 config: voiceserver + tunnel-ipc + tunnel-ws |
| `/opt/voiceserver/dist/index.js` | Compiled entry point (PM2 runs this) |
| `/opt/voiceserver/start-tunnel-ipc.sh` | Cloudflared tunnel script for IPC (port 8766) |
| `/opt/voiceserver/start-tunnel-ws.sh` | Cloudflared tunnel script for WebSocket (port 8765) |
| `/opt/voiceserver/start-tunnel-ollama.sh` | Cloudflared tunnel script for Ollama (port 11434) |
| `/opt/voiceserver/start-ollama.sh` | Ollama startup script with OLLAMA_ORIGINS=* |
| `/models/kokoro/kokoro-v1.0.onnx` | Kokoro TTS model (311MB) |
| `/models/kokoro/voices-v1.0.bin` | Kokoro voice data (27MB) |
| `/data/cloned-voices/` | KokoClone cloned voice storage |
| `/data/model-config.json` | Model manager persistent config (active models) |
| `/var/log/voiceserver/out.log` | Voice server stdout log |
| `/var/log/voiceserver/error.log` | Voice server stderr log |
| `/var/log/voiceserver/tunnel-ipc.log` | IPC tunnel log (contains tunnel URL) |
| `/var/log/voiceserver/tunnel-ws.log` | WS tunnel log (contains tunnel URL) |

### Current .env on Vast.ai

```env
WS_PORT=8765
IPC_PORT=8766
KOKORO_MODELS_DIR=/models/kokoro
KOKORO_VOICE=af_heart
CLONED_VOICES_DIR=/data/cloned-voices
OLLAMA_URL=http://localhost:11434/v1
DEFAULT_LLM=qwen3:4b
NODE_ENV=production
VAPICLONE_API_URL=https://vapiclone-production.up.railway.app
```

### Current PM2 Processes

| Name | Script | Purpose |
|---|---|---|
| ollama | start-ollama.sh | Ollama LLM server (with OLLAMA_ORIGINS=* for tunnel access) |
| voiceserver | dist/index.js | Main voice processing server |
| tunnel-ipc | start-tunnel-ipc.sh | Cloudflared tunnel for IPC HTTP (port 8766) |
| tunnel-ws | start-tunnel-ws.sh | Cloudflared tunnel for WebSocket (port 8765) |
| tunnel-ollama | start-tunnel-ollama.sh | Cloudflared tunnel for Ollama API (port 11434) |

### Current Cloudflared Tunnel URLs

| Service | URL |
|---|---|
| IPC (health, models, call registration) | `https://plaintiff-databases-brave-isolation.trycloudflare.com` |
| WebSocket (Twilio audio stream) | `wss://cup-delegation-shake-stickers.trycloudflare.com` |
| Ollama API (LLM inference from Railway) | `https://occupational-briefly-flag-cas.trycloudflare.com` |

---

## Voice Pipeline Flow

```
Twilio (phone call audio)
    |
    v  mu-law 8kHz mono (WebSocket port 8765)
    |
[Audio Conversion] → PCM 16-bit 16kHz
    |
    v
[STT: IBM Granite / Whisper / Deepgram]
    |
    v  text transcript
    |
[LLM: Ollama qwen3 / OpenAI / DeepSeek]
    |  (includes system prompt, conversation history, tools)
    v  response text (+ optional tool calls)
    |
[Tool Executor] → function webhooks, call transfer, DTMF, end call
    |
    v  final response text
    |
[TTS: Kokoro / Piper / ElevenLabs]
    |
    v  PCM 16-bit audio
    |
[Audio Conversion] → mu-law 8kHz mono
    |
    v  (WebSocket back to Twilio)
    |
Twilio → Caller hears AI response
```

---

## Tech Stack Summary

| Layer | Technology |
|---|---|
| Runtime | Node.js 22+ (TypeScript) |
| WebSocket | ws 8.18 |
| STT | IBM Granite 4.0 1B (default), faster-whisper (fallback), Deepgram (optional paid) |
| TTS | Kokoro-82M (default), Piper (fallback), ElevenLabs (optional paid) |
| LLM | Ollama + Qwen3 (default), OpenAI/DeepSeek (optional paid) |
| Voice Cloning | KokoClone (Kokoro + Kanade) |
| GPU | NVIDIA RTX 4090 (24GB VRAM) on Vast.ai |
| Process Manager | PM2 (auto-restart, boot persistence) |
| Tunneling | Cloudflared quick tunnels |
| CI/CD | GitHub Actions → SSH deploy to Vast.ai |
| Audio | mu-law ↔ PCM conversion, 8kHz ↔ 16kHz resampling |
