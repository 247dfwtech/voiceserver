# VoiceServer — Claude Handoff Document

**Last updated:** 2026-03-17 (rev 4)
**GitHub:** https://github.com/247dfwtech/voiceserver
**Local path:** /Users/adriansanchez/Desktop/voiceserver
**Running on:** Vast.ai Reserved GPU Instance #33032104 (Quebec, CA — RTX 4090)
**SSH:** `ssh -p 45194 root@70.29.210.33`
**Server path:** /opt/voiceserver/

---

## What Is This?

VoiceServer is the GPU-powered voice processing engine that handles real-time phone calls. It receives audio from Twilio via WebSocket, transcribes speech (STT), generates AI responses (LLM), synthesizes speech (TTS), and streams audio back to Twilio. All AI models run locally on a dedicated RTX 4090 GPU — no paid API calls for core voice processing.

---

## Current State (What's Working)

- **WebSocket server** (port 8765) — Accepts Twilio Media Stream connections
- **IPC HTTP server** (port 8766) — Health checks, model management, call registration, TTS testing
- **IBM Granite 4.0 1B Speech STT** ✅ — Persistent Python subprocess (model loads once, stays in memory). Model pre-downloaded at `/root/.cache/huggingface/`. No more timeout failures.
- **Kokoro-82M TTS** ✅ — Persistent Python subprocess, 54 voices, sub-0.3s latency on RTX 4090. All voice packs (af_heart, af_nicole, am_adam, af_sarah) pre-downloaded.
- **Ollama with qwen3:4b** ✅ — LLM for conversation, post-call analysis. `<think>` blocks handled with 2000-token minimum.
- **KokoClone voice cloning** — Clone voices from 3-10s reference audio samples
- **Model manager** — Install/activate/remove LLM/STT/TTS models via IPC API
- **Post-call analysis** — Summary generation and success evaluation after each call
- **Tool execution** — Function webhooks, call transfer, DTMF, end call (with SSRF protection)
- **Voicemail detection** — Detects sustained speech patterns in first 5 seconds
- **Cost tracking** — Per-call breakdown (STT, LLM, TTS, transport). Local models = $0.
- **PM2 managed** — voiceserver + ollama + 3 cloudflared tunnels, auto-restart, boot persistence
- **Health endpoint** — Reports sessions, memory, uptime, Ollama status
- **`/logs` endpoint** — `GET /logs?lines=N` returns recent PM2 log lines; requires `x-ipc-secret` header
- **IPC auth** — All IPC calls require either `Authorization: Bearer <VAPICLONE_API_KEY>` or `x-ipc-secret: <IPC_SECRET>` header

## What's Not Working / Known Issues

- **Cloudflared tunnel URLs are ephemeral** — Quick tunnels generate random `trycloudflare.com` URLs. If tunnel processes restart (PM2 auto-restart, server reboot), URLs change and Railway env vars need manual updating. A proper Cloudflare named tunnel would fix this.
- **faster-whisper not verified on current instance** — If Granite has issues, `pip3 install faster-whisper` and switch STT provider to "whisper" in assistant config.
- **Piper TTS not installed on current instance** — Only Kokoro is available. Install with `pip3 install piper-tts` if needed.
- **qwen3:4b instead of qwen3.5:9b** — Smaller model. Run `ollama pull qwen3.5:9b` for better quality if VRAM allows.

---

## What Was Just Completed (March 2026)

**Session 4 (latest):**
- **Fixed Granite STT persistent subprocess (Critical)** — Granite STT was spawning a NEW Python process per utterance, loading a 2GB model from disk every time (15-30s per load, 15s timeout → timeout every call). Completely rewrote `src/providers/stt/granite.ts` to use a module-level singleton persistent Python process (same pattern as KokoroTTS). Model loads ONCE on first call, stays in GPU memory. Per-utterance latency now 0.5-2s. Committed and deployed.
- **Pre-downloaded all Kokoro voice packs** — `af_heart.pt`, `af_nicole.pt`, `am_adam.pt`, `af_sarah.pt` downloaded to `/root/.cache/huggingface/`. No more 404 on voice pack lazy download.
- **Pre-downloaded Granite model** — `ibm-granite/granite-4.0-1b-speech` cached to `/root/.cache/huggingface/`. No cold-start download needed.
- **Deployed latest code to GPU** — `git pull && npm run build && pm2 restart voiceserver --update-env` confirmed deployed.

**Session 3 fixes:**
- **Fixed Kokoro TTS stdout corruption (Critical)** — `torch` and `kokoro` imports print warnings to stdout, corrupting the binary PCM protocol. Fixed by redirecting `sys.stdout = sys.stderr` before all imports.
- **Fixed Ollama max_tokens for thinking models** — `qwen3:4b` uses ~1000 tokens for chain-of-thought. Fixed by enforcing 2000-token minimum for Ollama provider in `openai-compat.ts`.
- **GitHub Actions SSH secrets updated** — All 4 secrets updated for current Vast.ai instance. Auto-deploy on push works.
- **Added `/logs` IPC endpoint** — Remote debugging without SSH.
- **IPC_SECRET auth fixed** — Added `VAPICLONE_API_KEY` and `IPC_SECRET` to voiceserver `.env`. `notifyCallEnded` now sends both `Authorization: Bearer` and `x-ipc-secret` headers.
- **Added `flux-general-en` model name validation** — Vapi-imported assistants have Deepgram model names. Added `config.model.includes('/')` check to reject non-HuggingFace IDs.

**Session 2 fixes:**
- Migrated to new reserved Vast.ai instance (33032104, Quebec CA, RTX 4090)
- Full fresh install: Ollama, PM2, kokoro, torch CUDA, cloudflared tunnels
- Fixed HF_HOME stale file handle from Vast.ai instance copy
- Configured PM2 with all 5 processes + boot persistence

---

## Key Decisions and Why

| Decision | Why |
|---|---|
| RTX 4090 on Vast.ai | Best price/performance. ~$0.30/hr reserved. Runs all 3 AI models with ~15GB headroom. |
| Reserved instance | On-demand lost the GPU overnight when stopped. Reserved guarantees availability. |
| Leave running 24/7 | Stopping risks losing the GPU. ~$17-22/month is negligible. Avoids tunnel URL churn. |
| IBM Granite STT over Whisper | Better accuracy (#1 OpenASR), keyword biasing for names/sales terms, ~2GB vs Whisper large-v3. |
| Persistent Python subprocess for STT/TTS | Models load once into GPU VRAM. Per-request subprocess was loading 2GB model every utterance (15-30s). Persistent process makes it 0.5-2s. |
| Kokoro TTS over ElevenLabs/Piper | #1 quality, near-human naturalness, 82M params, sub-0.3s on GPU, Apache 2.0. |
| Ollama for LLM | Free, local, no API costs. OpenAI-compatible. |

---

## Next Steps (In Order)

1. **Test a live call end-to-end** — All known issues fixed. Make a call and verify caller hears firstMessage (Kokoro TTS) and conversation flows (Granite STT → Ollama LLM → Kokoro TTS).
2. **Set up Cloudflare named tunnel** — Replace quick tunnels with permanent named tunnel for stable URLs.
3. **Pull qwen3.5:9b model** — `ollama pull qwen3.5:9b` for better conversation quality.
4. **Implement Deepgram STT option** — For production scale, Deepgram is faster and more reliable than self-hosted. Add `DEEPGRAM_API_KEY` to `.env` and set transcriber provider in assistant config.

---

## Gotchas and Important Context

- **Granite STT persistent process (Critical)** — `granite.ts` uses a module-level singleton Python process. The process loads `ibm-granite/granite-4.0-1b-speech` ONCE and handles all transcriptions via stdin/stdout. If the process crashes, it self-resets on the next call. Never revert to per-request subprocess spawning — it always times out.
- **Kokoro stdout redirect (Critical)** — The Python Kokoro subprocess redirects `sys.stdout = sys.stderr` before importing torch/kokoro. This prevents library warnings from corrupting the binary PCM protocol. If you ever rewrite the Python script, keep this redirect.
- **Ollama max_tokens minimum** — `openai-compat.ts` enforces a 2000-token minimum for Ollama because qwen3 thinking models use ~1000 tokens for CoT first.
- **HF_HOME stale file handle (Critical)** — When copying a Vast.ai instance, `/etc/environment` retains `HF_HOME="/workspace/.hf_home"` but that path is stale. Causes `OSError: [Errno 116] Stale file handle`. Fix: `sed -i 's|HF_HOME=.*|HF_HOME="/root/.cache/huggingface"|g' /etc/environment`, then delete and restart voiceserver PM2 process.
- **PyTorch CUDA wheel (Critical)** — `pip install torch` installs CPU-only. Must use `pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu124`.
- **PM2 needs PATH set** — Node.js at `/opt/nvm/versions/node/v24.12.0/bin`. Always `export PATH="/opt/nvm/versions/node/v24.12.0/bin:/opt/instance-tools/bin:$PATH"`.
- **`pm2 restart --update-env`** vs full delete/start** — `--update-env` picks up ecosystem env changes but NOT `/etc/environment` changes. For `/etc/environment` changes, must `pm2 delete voiceserver && pm2 start ecosystem.config.cjs --only voiceserver`.
- **Cloudflared at `/opt/instance-tools/bin/cloudflared`** — Pre-installed on Vast.ai.
- **Vast.ai "200 ports" is misleading** — Only 6 Docker-mapped ports accessible externally. 8765/8766 are NOT accessible. Always use cloudflared tunnels.
- **Tunnel URLs change on restart** — If voice server shows "Offline" in VapiClone, check `pm2 logs tunnel-ipc --lines 5 --nostream` for new URL. Update Railway `VOICE_SERVER_IPC_URL` and `VOICE_SERVER_URL`.
- **IPC auth headers** — IPC endpoints require either `Authorization: Bearer <VAPICLONE_API_KEY>` or `x-ipc-secret: <IPC_SECRET>`. Both are set in `.env`. Current values: `VAPICLONE_API_KEY=vs-internal-secret-2026`, `IPC_SECRET=vs-ipc-2026`.
- **Audio format** — Twilio sends/receives mu-law 8kHz mono. Voice server converts to PCM 16-bit 16kHz for STT and back to mu-law.
- **Ollama needs OLLAMA_ORIGINS=*** — Without this, returns 403 to cloudflared tunnel requests.

---

## Relevant File Paths

### Local (Desktop) / GitHub

| Path | What's In It |
|---|---|
| `src/index.ts` | Main entry — WebSocket (8765) + IPC HTTP (8766), session management, `/logs` endpoint |
| `src/model-manager.ts` | Model lifecycle — install/activate/remove |
| `src/providers/stt/granite.ts` | Granite STT — **persistent Python subprocess**, module-level singleton, stdin/stdout protocol |
| `src/providers/stt/whisper.ts` | faster-whisper STT fallback |
| `src/providers/tts/kokoro.ts` | Kokoro-82M TTS — persistent Python subprocess, length-prefixed binary PCM protocol |
| `src/providers/tts/kokoclone.ts` | KokoClone voice cloning |
| `src/providers/tts/piper.ts` | Piper TTS fallback (CPU-friendly) |
| `src/providers/llm/openai-compat.ts` | OpenAI-compatible LLM client — 2000-token min for Ollama |
| `src/providers/index.ts` | Provider factory — auto-selects based on config with fallbacks |
| `src/voice-pipeline/call-session.ts` | Core call handler — STT→LLM→TTS loop, tool execution |
| `src/voice-pipeline/analysis-runner.ts` | Post-call summary + success evaluation |
| `src/voice-pipeline/tool-executor.ts` | Tool execution with SSRF protection |
| `src/voice-pipeline/audio-utils.ts` | Audio format conversion (mu-law ↔ PCM, resampling) |
| `scripts/setup-gpu-server.sh` | **One-click bare metal setup** — installs all deps, pre-downloads models, configures PM2 |
| `.github/workflows/deploy.yml` | GitHub Actions — SSH into Vast.ai, git pull, build, pm2 restart |

### Vast.ai GPU Server (70.29.210.33, SSH port 45194)

| Path | What's In It |
|---|---|
| `/opt/voiceserver/` | Cloned repo with built dist/ |
| `/opt/voiceserver/.env` | Production environment config (see below) |
| `/opt/voiceserver/ecosystem.config.cjs` | PM2 config: voiceserver + ollama + 3 tunnels |
| `/opt/voiceserver/dist/index.js` | Compiled entry point (PM2 runs this) |
| `/opt/voiceserver/start-tunnel-ipc.sh` | Cloudflared tunnel → localhost:8766 |
| `/opt/voiceserver/start-tunnel-ws.sh` | Cloudflared tunnel → localhost:8765 |
| `/opt/voiceserver/start-tunnel-ollama.sh` | Cloudflared tunnel → localhost:11434 |
| `/opt/voiceserver/start-ollama.sh` | Ollama startup with OLLAMA_ORIGINS=* |
| `/root/.cache/huggingface/` | HuggingFace model cache (Granite STT + Kokoro voice packs) |
| `/var/log/voiceserver/` | PM2 log files (out.log, error.log, tunnel-*.log) |

### Current .env on Vast.ai (`/opt/voiceserver/.env`)

```env
WS_PORT=8765
IPC_PORT=8766
KOKORO_MODELS_DIR=/models/kokoro
KOKORO_VOICE=af_heart
CLONED_VOICES_DIR=/data/cloned-voices
OLLAMA_URL=http://localhost:11434/v1
DEFAULT_LLM=qwen3:4b
NODE_ENV=production
HF_HOME=/root/.cache/huggingface
VAPICLONE_API_URL=https://vapiclone-production.up.railway.app
VAPICLONE_API_KEY=vs-internal-secret-2026
IPC_SECRET=vs-ipc-2026
```

### Current PM2 Processes

| Name | Script | Purpose |
|---|---|---|
| ollama | start-ollama.sh | Ollama LLM server (OLLAMA_ORIGINS=* for tunnel) |
| voiceserver | dist/index.js | Main voice processing server |
| tunnel-ipc | start-tunnel-ipc.sh | Cloudflared tunnel for IPC HTTP (port 8766) |
| tunnel-ws | start-tunnel-ws.sh | Cloudflared tunnel for WebSocket (port 8765) |
| tunnel-ollama | start-tunnel-ollama.sh | Cloudflared tunnel for Ollama (port 11434) |

### Current Cloudflared Tunnel URLs

| Service | URL |
|---|---|
| IPC (VOICE_SERVER_IPC_URL) | `https://plaintiff-databases-brave-isolation.trycloudflare.com` |
| WebSocket (VOICE_SERVER_URL) | `wss://cup-delegation-shake-stickers.trycloudflare.com` |
| Ollama (OLLAMA_URL) | `https://occupational-briefly-flag-cas.trycloudflare.com/v1` |

---

## Debugging Commands

```bash
# Check voiceserver logs (from Mac terminal)
curl -s -H "x-ipc-secret: vs-ipc-2026" \
  "https://plaintiff-databases-brave-isolation.trycloudflare.com/logs?lines=50" \
  | python3 -c "import sys,json; [print(l) for l in json.load(sys.stdin)['lines']]"

# SSH into GPU
ssh -p 45194 root@70.29.210.33

# After SSH: deploy latest code
export PATH="/opt/nvm/versions/node/v24.12.0/bin:/opt/instance-tools/bin:$PATH"
cd /opt/voiceserver && git pull && npm run build && pm2 restart voiceserver --update-env

# Get current tunnel URLs (after restart)
pm2 logs tunnel-ipc --lines 5 --nostream
pm2 logs tunnel-ws --lines 5 --nostream
pm2 logs tunnel-ollama --lines 5 --nostream

# Check GPU usage
nvidia-smi

# Check Ollama models
ollama list

# Watch live logs
pm2 logs voiceserver
```

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
[STT: IBM Granite persistent subprocess]
  - Python process lives across all calls
  - Sends audio WAV path via stdin
  - Gets transcript via stdout
    |
    v  text transcript
    |
[LLM: Ollama qwen3 / OpenAI / DeepSeek]
  - system prompt + conversation history + tools
  - 2000-token minimum for Ollama (qwen3 thinking models)
    |
    v  response text (+ optional tool calls)
    |
[Tool Executor] → function webhooks, transfer, DTMF, end call
    |
    v  final response text
    |
[TTS: Kokoro persistent subprocess]
  - Python process lives across all calls
  - Sends JSON command via stdin
  - Gets length-prefixed PCM via stdout
    |
    v  PCM 16-bit 24kHz → resampled to 16kHz
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
| STT | IBM Granite 4.0 1B (persistent subprocess, default), faster-whisper (fallback), Deepgram (paid) |
| TTS | Kokoro-82M (persistent subprocess, default), Piper (fallback), ElevenLabs (paid) |
| LLM | Ollama + Qwen3 (default), OpenAI/DeepSeek (paid) |
| GPU | NVIDIA RTX 4090 (24GB VRAM) on Vast.ai |
| Process Manager | PM2 (auto-restart, boot persistence) |
| Tunneling | Cloudflared quick tunnels |
| CI/CD | GitHub Actions → SSH deploy to Vast.ai |
| Audio | mu-law ↔ PCM conversion, 8kHz ↔ 16kHz resampling |
