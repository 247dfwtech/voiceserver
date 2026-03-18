# VoiceServer — Claude Handoff Document

**Last updated:** 2026-03-18 (rev 5)
**GitHub:** https://github.com/247dfwtech/voiceserver
**Local path:** /Users/adriansanchez/Desktop/voiceserver
**Running on:** Vast.ai Reserved GPU Instance #33032104 (Quebec, CA — RTX 4090)
**SSH:** `ssh -p 45194 root@70.29.210.33`
**Server path:** /opt/voiceserver/

---

## What Is This?

VoiceServer is the GPU-powered voice processing engine that handles real-time phone calls. It receives audio from Twilio via WebSocket, transcribes speech (STT), generates AI responses (LLM), synthesizes speech (TTS), and streams audio back to Twilio. All AI models run locally on a dedicated RTX 4090 GPU — no paid API calls for core voice processing.

---

## Current State (What's Working) ✅

- **Full voice pipeline CONFIRMED WORKING** — STT → LLM → TTS end-to-end in live calls
- **WebSocket server** (port 8765) — Accepts Twilio Media Stream connections
- **IPC HTTP server** (port 8766) — Health checks, model management, call registration, TTS testing
- **Whisper STT (small.en)** ✅ — Persistent Python subprocess using `faster-whisper` package. Model loads once into GPU VRAM. Keyword biasing via `initial_prompt` for domain terms. 1000ms silence threshold for natural phone pauses.
- **Kokoro-82M TTS** ✅ — Module-level singleton persistent Python subprocess shared across ALL call sessions. 54 voices, sub-0.3s latency on RTX 4090. All voice packs pre-downloaded.
- **Ollama with qwen3.5:9b** ✅ — LLM for conversation + tool calling. `<think>` blocks handled with 2000-token minimum.
- **Tool execution** ✅ — Function webhooks, call transfer, DTMF, end_call_tool confirmed working in live calls
- **KokoClone voice cloning** — Clone voices from 3-10s reference audio samples
- **Model manager** — Install/activate/remove LLM/STT/TTS models via IPC API
- **Post-call analysis** — Summary generation and success evaluation (needs fix: hardcoded model name)
- **Voicemail detection** — Detects sustained speech patterns in first 5 seconds (disabled by default for testing)
- **Cost tracking** — Per-call breakdown (STT, LLM, TTS, transport). Local models = $0.
- **PM2 managed** — voiceserver + ollama + 3 cloudflared tunnels, auto-restart, boot persistence
- **GitHub Actions auto-deploy** — Push to main → SSH into Vast.ai → build → restart
- **`/logs` endpoint** — `GET /logs?lines=N` returns recent PM2 log lines; requires `x-ipc-secret` header
- **IPC auth** — All IPC calls require either `Authorization: Bearer <VAPICLONE_API_KEY>` or `x-ipc-secret: <IPC_SECRET>` header

## What's Not Working / Known Issues

- **Cloudflared tunnel URLs are ephemeral** — Quick tunnels generate random `trycloudflare.com` URLs. If tunnel processes restart, URLs change and Railway env vars need manual updating. Need Cloudflare named tunnel.
- **Post-call analysis hardcoded model** — `analysis-runner.ts` still references `qwen3:4b` which was deleted. Needs to use the active LLM from model-manager. Quick fix.
- **Transcription accuracy still improving** — Whisper `small.en` is better than `base.en` but phone audio (8kHz mulaw → 16kHz PCM) is inherently low quality. Consider `medium.en` or Deepgram for production.
- **Granite STT experimental** — IBM Granite 4.0 1B Speech has a non-standard multimodal chat API (not a standard ASR pipeline). Works but had 4+ different API errors during development. Whisper is the recommended default. Granite code is kept as experimental option.

---

## What Was Just Completed (Session 5 — March 2026)

### MILESTONE: Full voice pipeline working end-to-end in live calls 🎉

1. **Fixed Granite STT — 4 attempts, multiple API issues:**
   - Attempt 1: `AutoProcessor(audio_array)` → routes to text tokenizer ("Invalid text provided")
   - Attempt 2: `hf_pipeline(file_path)` → passes `sampling_rate` kwarg Granite rejects
   - Attempt 3: `AutoProcessor(audio, return_tensors)` without sampling_rate → still routes to text tokenizer
   - Attempt 4: Correct multimodal chat API from HuggingFace model card (requires BOTH text prompt with `<|audio|>` token + audio tensor)
   - Final: `device_map` parameter requires `accelerate` pip package (not installed)
   - **Decision: Switched default STT to Whisper** — battle-tested, simple API, works out of the box

2. **Rewrote Whisper STT as persistent Python subprocess:**
   - Old: spawned `faster-whisper` CLI binary per utterance (binary wasn't installed → ENOENT)
   - New: module-level singleton Python subprocess using `faster-whisper` package (same pattern as Kokoro)
   - Model loads once into GPU VRAM, handles all transcriptions via stdin/stdout JSON protocol
   - Supports keyword biasing via `initial_prompt` parameter

3. **Fixed firstMessage never being spoken:**
   - Root cause: assistant had `firstMessageMode: "assistant-waits-for-user"` → assistant never spoke first
   - For outbound calls, must be `"assistant-speaks-first"`
   - Changed via API, added UI selector in vapiclone Behavior tab

4. **Fixed voicemail detection false positive:**
   - Voicemail detector triggered after 3 seconds on a real human answering
   - `continuousSpeechFramesThreshold: 60` (1.2s) too aggressive for phone calls
   - Disabled voicemail detection on test assistant
   - Fixed non-working UI toggle (replaced shadcn Switch with select dropdown)

5. **Tuned for better quality:**
   - Upgraded Whisper from `base.en` (74MB) to `small.en` (244MB) — much better phone audio accuracy
   - Increased silence threshold from 500ms to 1000ms — stops cutting off mid-sentence
   - Added keyword biasing: solar, Freedom Forever, TXU, electric bill, financing, etc.
   - Upgraded LLM from `qwen3:4b` to `qwen3.5:9b` — smarter responses
   - Deleted `qwen3:4b` from Ollama to free resources

6. **UI fixes in vapiclone:**
   - STT model dropdown now shows correct Whisper models per provider
   - Keywords input: comma-separated single-line (was broken multi-line textarea)
   - Voicemail detection: working On/Off dropdown (was non-functional Switch)
   - First Message Mode selector added to Behavior tab
   - Improved text inputs (char counts, placeholders, better sizing)
   - Bushido Pros branding (samurai logo + name in sidebar)

---

## Key Decisions and Why

| Decision | Why |
|---|---|
| RTX 4090 on Vast.ai | Best price/performance. ~$0.30/hr reserved. Runs all 3 AI models with headroom. |
| **Whisper over Granite for STT** | Granite has non-standard multimodal API that caused 4+ different errors. Whisper is battle-tested, simple, reliable. Granite kept as experimental. |
| Persistent Python subprocesses | Models load once into GPU VRAM. Per-request subprocess was loading models every utterance (15-30s). Persistent process: 0.3-2s. |
| `small.en` Whisper model | Best balance of accuracy and speed for real-time phone calls. `base.en` too inaccurate, `medium.en` too slow. |
| Keyword biasing via initial_prompt | Whisper hears "Freedom Forever" correctly instead of "freedom for ever" when initial_prompt contains the term. |
| 1000ms silence threshold | Phone conversations have natural pauses. 500ms was cutting speakers off mid-sentence. |
| `qwen3.5:9b` over `qwen3:4b` | 2x smarter, better conversation flow, proper tool calling. 6.1GB fits easily on RTX 4090. |
| Kokoro TTS module-level singleton | ALL call sessions share one Python process. Without this, each call spawned a new 30-60s cold start → caller heard silence. |

---

## Active Model Configuration

| Component | Model | VRAM | Status |
|-----------|-------|------|--------|
| **STT** | Whisper `small.en` | ~500MB | ✅ Active (persistent subprocess) |
| **LLM** | qwen3.5:9b | ~5-6GB | ✅ Active (Ollama) |
| **TTS** | Kokoro-82M | ~500MB | ✅ Active (persistent subprocess) |
| STT (experimental) | IBM Granite 4.0 1B Speech | ~2GB | ❌ Inactive (code exists, not recommended) |
| LLM (deleted) | qwen3:4b | — | ❌ Deleted from Ollama |
| **Total VRAM during call** | | **~7GB** | Well within 24GB RTX 4090 |

---

## Gotchas and Important Context

### Critical — Will break calls if wrong

- **Kokoro module-level singleton** — `kokoro.ts` uses module-level variables (`_proc`, `_ready`, `_sharedQueue`). ALL KokoroTTS instances share one Python process. Without this, each call spawns a 30-60s cold start. `destroy()` is a no-op. Never make the process per-instance.
- **Kokoro stdout redirect** — Python subprocess redirects `sys.stdout = sys.stderr` before importing torch/kokoro. Prevents library warnings from corrupting binary PCM protocol.
- **Whisper persistent subprocess** — `whisper.ts` uses same module-level singleton pattern. Sends JSON `{"path": "/tmp/audio.wav", "keywords": ["solar"]}` via stdin. Gets transcript line via stdout.
- **firstMessageMode must be "assistant-speaks-first" for outbound calls** — If set to "assistant-waits-for-user", caller hears silence because STT must transcribe first but caller doesn't know to speak.
- **Ollama max_tokens minimum** — `openai-compat.ts` enforces 2000-token minimum because qwen3.5 thinking models use ~1000 tokens for chain-of-thought before the actual response.
- **PyTorch CUDA wheel** — `pip install torch` installs CPU-only. Must use `pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu124`.
- **PM2 needs PATH** — Node.js at `/opt/nvm/versions/node/v24.12.0/bin`. Always export PATH before PM2 commands.
- **`pm2 restart --update-env`** — Picks up ecosystem env changes but NOT `/etc/environment` changes. For those: `pm2 delete voiceserver && pm2 start ecosystem.config.cjs --only voiceserver`.

### Important — Will cause confusion if forgotten

- **Cloudflared URLs change on restart** — Check `pm2 logs tunnel-ipc --lines 5 --nostream` for new URL. Update Railway env vars.
- **Vast.ai "200 ports" is misleading** — Only 6 Docker-mapped ports accessible externally. 8765/8766 NOT accessible. Always use cloudflared tunnels.
- **IPC auth headers** — IPC endpoints require `x-ipc-secret: <IPC_SECRET>` header. Current: `vs-ipc-2026`.
- **Audio format** — Twilio sends mu-law 8kHz mono. Voice server converts to PCM 16-bit 16kHz for STT. Kokoro outputs 24kHz, resampled to 16kHz, then to mu-law 8kHz for Twilio.
- **Ollama needs OLLAMA_ORIGINS=*** — Without this, returns 403 to cloudflared tunnel requests.
- **HF_HOME stale file handle** — When copying Vast.ai instance, `/etc/environment` has stale HF_HOME path. Fix: `sed -i 's|HF_HOME=.*|HF_HOME="/root/.cache/huggingface"|g' /etc/environment`.
- **Voicemail detection false positives** — The audio-based detector (not Twilio AMD) uses `continuousSpeechFramesThreshold: 60` (1.2s of continuous speech). This is aggressive for phone calls. Recommend disabling for initial testing.

---

## Relevant File Paths

### Local (Desktop) / GitHub

| Path | What's In It |
|---|---|
| `src/index.ts` | Main entry — WebSocket (8765) + IPC HTTP (8766), session management, `/logs` endpoint |
| `src/model-manager.ts` | Model lifecycle — install/activate/remove. Default STT: whisper/small.en |
| `src/providers/stt/whisper.ts` | **Whisper STT — persistent Python subprocess**, module-level singleton, JSON stdin protocol with keyword biasing |
| `src/providers/stt/granite.ts` | Granite STT — experimental, non-standard multimodal chat API, not recommended |
| `src/providers/tts/kokoro.ts` | **Kokoro-82M TTS — module-level singleton**, shared across all calls, binary PCM protocol |
| `src/providers/tts/kokoclone.ts` | KokoClone voice cloning |
| `src/providers/tts/piper.ts` | Piper TTS fallback (CPU-friendly) |
| `src/providers/llm/openai-compat.ts` | OpenAI-compatible LLM client — 2000-token min for Ollama |
| `src/providers/index.ts` | Provider factory — auto-selects based on config |
| `src/voice-pipeline/call-session.ts` | Core call handler — STT→LLM→TTS loop, tool execution, firstMessage |
| `src/voice-pipeline/analysis-runner.ts` | Post-call summary + success evaluation (⚠️ hardcoded qwen3:4b — needs fix) |
| `src/voice-pipeline/tool-executor.ts` | Tool execution with SSRF protection |
| `src/voice-pipeline/audio-utils.ts` | Audio format conversion (mu-law ↔ PCM, resampling) |
| `scripts/setup-gpu-server.sh` | **One-click bare metal setup** — installs all deps, pre-downloads models, configures PM2 |
| `.github/workflows/deploy.yml` | GitHub Actions — SSH deploy, installs Python deps, builds, restarts PM2 |

### Current .env on Vast.ai (`/opt/voiceserver/.env`)

```env
WS_PORT=8765
IPC_PORT=8766
KOKORO_MODELS_DIR=/models/kokoro
KOKORO_VOICE=af_heart
CLONED_VOICES_DIR=/data/cloned-voices
OLLAMA_URL=http://localhost:11434/v1
DEFAULT_LLM=qwen3.5:9b
NODE_ENV=production
HF_HOME=/root/.cache/huggingface
VAPICLONE_API_URL=https://vapiclone-production.up.railway.app
VAPICLONE_API_KEY=vs-internal-secret-2026
IPC_SECRET=vs-ipc-2026
```

### Current Cloudflared Tunnel URLs

| Service | URL |
|---|---|
| IPC (VOICE_SERVER_IPC_URL) | `https://plaintiff-databases-brave-isolation.trycloudflare.com` |
| WebSocket (VOICE_SERVER_URL) | `wss://cup-delegation-shake-stickers.trycloudflare.com` |
| Ollama (OLLAMA_URL) | `https://occupational-briefly-flag-cas.trycloudflare.com/v1` |

---

## Debugging Commands

```bash
# Check voiceserver logs remotely (no SSH needed)
curl -s -H "x-ipc-secret: vs-ipc-2026" \
  "https://plaintiff-databases-brave-isolation.trycloudflare.com/logs?lines=50" \
  | python3 -c "import sys,json; [print(l) for l in json.load(sys.stdin)['lines']]"

# Check model status
curl -s -H "x-ipc-secret: vs-ipc-2026" \
  "https://plaintiff-databases-brave-isolation.trycloudflare.com/models/status" \
  | python3 -m json.tool

# Pre-warm Kokoro TTS
curl -s -H "x-ipc-secret: vs-ipc-2026" \
  "https://plaintiff-databases-brave-isolation.trycloudflare.com/tts/test" \
  -X POST -H "Content-Type: application/json" \
  -d '{"text":"Test.","voice":"af_heart"}'

# Switch active STT/LLM
curl -s -X POST -H "x-ipc-secret: vs-ipc-2026" -H "Content-Type: application/json" \
  "https://plaintiff-databases-brave-isolation.trycloudflare.com/models/stt/activate" \
  -d '{"name":"small.en"}'

# SSH into GPU
ssh -p 45194 root@70.29.210.33

# After SSH: deploy latest code
export PATH="/opt/nvm/versions/node/v24.12.0/bin:/opt/instance-tools/bin:$PATH"
cd /opt/voiceserver && git pull && npm run build && pm2 restart voiceserver --update-env

# Check GPU usage
nvidia-smi

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
[STT: Whisper small.en persistent subprocess]
  - Python process lives across all calls
  - Sends JSON {"path", "keywords"} via stdin
  - Gets transcript line via stdout
  - Keywords bias recognition toward domain terms
    |
    v  text transcript
    |
[LLM: Ollama qwen3.5:9b]
  - system prompt + conversation history + tools
  - 2000-token minimum for thinking models
  - Tool calling (transfer, end_call, webhooks)
    |
    v  response text (+ optional tool calls)
    |
[Tool Executor] → function webhooks, transfer, DTMF, end call
    |
    v  final response text
    |
[TTS: Kokoro module-level singleton subprocess]
  - One Python process shared by ALL calls
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

## Next Steps (In Order)

1. **Fix post-call analysis model reference** — `analysis-runner.ts` hardcodes `qwen3:4b` (deleted). Should use active LLM from model-manager.
2. **Tune transcription accuracy** — Consider `medium.en` if latency is acceptable, or Deepgram for production scale.
3. **Set up Cloudflare named tunnel** — Permanent URLs that survive restarts.
4. **Tune voicemail detection thresholds** — Increase `continuousSpeechFramesThreshold` from 60 to 150+ before re-enabling.
5. **Full UI redesign** — Bushido Pros branding across vapiclone and dialer4clone (separate session).

---

## Tech Stack Summary

| Layer | Technology |
|---|---|
| Runtime | Node.js 22+ (TypeScript) |
| WebSocket | ws 8.18 |
| STT | **Whisper small.en** (persistent subprocess, default), IBM Granite (experimental), Deepgram (paid) |
| TTS | **Kokoro-82M** (module-level singleton, default), Piper (fallback), ElevenLabs (paid) |
| LLM | **Ollama + qwen3.5:9b** (default), OpenAI/DeepSeek (paid) |
| GPU | NVIDIA RTX 4090 (24GB VRAM) on Vast.ai |
| Process Manager | PM2 (auto-restart, boot persistence) |
| Tunneling | Cloudflared quick tunnels (ephemeral URLs) |
| CI/CD | GitHub Actions → SSH deploy to Vast.ai |
| Audio | mu-law ↔ PCM conversion, 8kHz ↔ 16kHz ↔ 24kHz resampling |
