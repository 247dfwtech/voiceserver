# VoiceServer — Claude Handoff Document

**Last updated:** 2026-03-18 (rev 9)
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

- **Full voice pipeline CONFIRMED WORKING** — STT → LLM → TTS end-to-end in live calls. Both Whisper and Deepgram Flux STT tested successfully on real phone calls.
- **WebSocket server** (port 8765) — Accepts Twilio Media Stream connections
- **IPC HTTP server** (port 8766) — Health checks, model management, call registration, TTS testing
- **Whisper STT (small.en)** ✅ — Persistent Python subprocess using `faster-whisper` package. Model loads once into GPU VRAM. Keyword biasing via `initial_prompt` for domain terms. 1000ms silence threshold for natural phone pauses. FIFO transcription queue (no race condition).
- **Deepgram Flux STT** ✅ — Cloud-based turn-based streaming via `wss://api.deepgram.com/v2/listen`. Native end-of-turn detection (no manual VAD needed). Supports `flux-general-en` (recommended for voice agents), `nova-3-general`, `nova-2-general`. Keyterm prompting for domain terms. Requires `DEEPGRAM_API_KEY`.
- **Kokoro-82M TTS** ✅ — Module-level singleton persistent Python subprocess shared across ALL call sessions. 54 voices, sub-0.3s latency on RTX 4090. All voice packs pre-downloaded.
- **Ollama with qwen3.5:9b** ✅ — LLM for conversation + tool calling. `<think>` blocks handled with 2000-token minimum.
- **Tool execution** ✅ — Function webhooks, call transfer, DTMF, end_call_tool confirmed working in live calls
- **KokoClone voice cloning** — Clone voices from 3-10s reference audio samples using Kokoro + Kanade voice conversion
- **Chatterbox Turbo voice cloning** ✅ — Clone voices from ~10s reference audio using Resemble AI's Chatterbox Turbo model. Persistent Python subprocess (conda env `chatterbox`), 24kHz output resampled to 16kHz, ~4.2GB VRAM. IPC endpoints at `/chatterbox/*`.
- **Model manager** — Install/activate/remove LLM/STT/TTS models via IPC API
- **Settings API** ✅ — `GET/PUT /settings` endpoint for runtime API key management. Keys persist to `.env` + `process.env` (no restart needed). VapiClone Settings page auto-syncs keys to GPU server.
- **Post-call analysis** — Summary generation and success evaluation (uses DEFAULT_LLM env var)
- **Voicemail detection** — Detects sustained speech patterns in first 5 seconds (disabled by default for testing)
- **Cost tracking** — Per-call breakdown (STT, LLM, TTS, transport). Local models = $0.
- **PM2 managed** — voiceserver + ollama + 3 cloudflared tunnels, auto-restart, boot persistence
- **GitHub Actions auto-deploy** — Push to main → SSH into Vast.ai → build → restart
- **GPU/CPU/Memory monitoring** ✅ — `gpu-monitor.ts` collects metrics every 30s via nvidia-smi + os module. 3-day in-memory ring buffer (8640 entries, ~1.7MB). Endpoints: `/health` (enhanced with GPU data inline), `/health/gpu` (fresh snapshot), `/metrics/history?range=1h|6h|24h|3d` (historical data for charts)
- **`/logs` endpoint** — `GET /logs?lines=N` returns recent PM2 log lines; requires `x-ipc-secret` header
- **IPC auth** — All IPC calls require either `Authorization: Bearer <VAPICLONE_API_KEY>` or `x-ipc-secret: <IPC_SECRET>` header

## What's Not Working / Known Issues

- **Cloudflared tunnel URLs are ephemeral** — Quick tunnels generate random `trycloudflare.com` URLs. If tunnel processes restart, URLs change and Railway env vars need manual updating. Need Cloudflare named tunnel.
- **Transcription accuracy still improving** — Whisper `small.en` is better than `base.en` but phone audio (8kHz mulaw → 16kHz PCM) is inherently low quality. Deepgram Flux is now available as a cloud alternative with native end-of-turn detection. Consider `medium.en` for higher local accuracy.

---

## What Was Just Completed (Session 9 — March 18, 2026)

### Call Transfer Fix + GitHub Actions Auto-Deploy Fix

**MILESTONE: Call transfer fully working.** LLM invokes `ff_transfer` tool → voiceserver executes Twilio `<Dial>` → customer is transferred to fallback number. End-to-end confirmed.

1. **Fixed transferCall/dtmf tools invisible to LLM** — `call-session.ts` only sent tools with `functionDefinition` to the LLM. transferCall/dtmf tools had `functionDefinition: null`. Added auto-generated function schemas for these types (mirroring existing endCall fallback).
2. **Fixed transfer destination always `undefined`** — UI stored destination as `config.url` but executor read `config.destination`. Fixed UI to use correct field. Also added `fallbackDestination` from phone number config as fallback when tool has no explicit destination.
3. **Added Twilio credentials to voiceserver `.env`** — Transfer requires `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN` on the GPU server. Added to `.env`, GitHub Secrets, deploy workflow, and settings sync.
4. **Fixed GitHub Actions auto-deploy (was no-op since creation)** — The `deploy.yml` had `script:` nested under `env:` instead of `with:`, so SSH connected but executed an empty command. Every deploy showed "success" but did nothing. GPU server was stuck on session 7 code. Fixed YAML structure; deploy now pulls, builds, restarts, and logs the deployed commit.
5. **Added Twilio creds to vapiclone settings sync** — `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN` now sync from vapiclone Settings page to voiceserver via IPC `/settings` PUT endpoint.
6. **Fixed `DEFAULT_LLM=qwen3:4b` on VPS** — Was causing all post-call analysis to fail with 404. Updated to `qwen3.5:9b`.
7. **Added detailed transfer logging** — Transfer events now log destination, callSid, success/failure, and error details for debugging.

---

## Session 8 — March 2026

### Deepgram STT + Bug Fixes + Settings API + Error Handling

**MILESTONE: Deepgram Flux STT confirmed working in live phone call.** Assistant spoke first message, caller responded naturally, LLM understood context and responded correctly. Both Whisper and Deepgram now work as selectable STT options.

1. **Deepgram Flux STT provider** — New `src/providers/stt/deepgram.ts`. Turn-based streaming via WebSocket (`wss://api.deepgram.com/v2/listen`). Native end-of-turn detection — no manual VAD/silence thresholds needed. Supports `flux-general-en` (voice agent optimized), `nova-3-general`, `nova-2-general`. Keyterm prompting, auto-reconnect, 10s keepalive. **Tested and confirmed working on real phone call.**
2. **Whisper STT race condition fix** — Replaced single `pendingResolve`/`pendingReject` with proper FIFO queue. Concurrent transcription requests now serialize safely instead of dropping callbacks.
3. **TTS stuck-state fix** — Added `onError` callback to `TTSProvider.synthesizeStream()` interface. All TTS providers (Kokoro, Chatterbox, Piper) now call `onError` on failure. Call session resets `isSpeaking` + state on TTS error.
4. **LLM error recovery** — `onDone("")` now fires on stream error so sessions transition back to `waiting_for_speech` instead of hanging in `processing` state.
5. **Settings API** — New `GET/PUT /settings` IPC endpoint. Allows VapiClone to push API keys (Deepgram, OpenAI, ElevenLabs, DeepSeek) at runtime. Persists to `.env` file + `process.env`. No restart needed.
6. **Deploy workflow API key sync** — `.github/workflows/deploy.yml` now syncs API keys from GitHub Secrets into `.env` on every deploy.
7. **Sentence chunking tuning** — TTS now splits on sentence-ending punctuation (`.!?`) at 20+ chars instead of eagerly splitting on commas. Long text (80+) still splits on commas.
8. **Voicemail detection tuning** — Threshold increased 60→150 frames (1.2s→3s continuous speech), speech ratio 0.7→0.8, added min 100 speech frames check.
9. **STT error counter** — 3 consecutive STT errors auto-end the call with reason `stt-failure`.
10. **IPC error logging** — `ipcError()` helper replaces 11 silent catch blocks. All errors now logged with method + path context.
11. **Pending config TTL** — 30s interval sweeps configs older than 60s (replaces per-call setTimeout).
12. **Default STT changed** — Granite removed from provider factory. Default is now Whisper `small.en`. Granite code still exists but won't load/use GPU.
13. **Test chat blank bubble fix** — Strip unclosed `<think>` tags, return fallback when LLM produces only thinking. UI shows error instead of blank bubble.

---

## Session 7 — March 2026

### GPU Monitoring + Server Debugging

1. **GPU/CPU/Memory metrics backend** — New `src/gpu-monitor.ts`: nvidia-smi integration, CPU utilization via `os.cpus()`, system RAM. Collects snapshots every 30 seconds into an in-memory ring buffer (8640 entries = 3 days, ~1.7MB).
2. **Enhanced `/health` endpoint** — Now includes `gpu` (util%, VRAM, temp, power, name), `cpu` (util%), `systemMemory` (used/total MB) inline. Backward-compatible.
3. **New `/health/gpu` endpoint** — Fresh GPU snapshot on demand (runs nvidia-smi immediately).
4. **New `/metrics/history` endpoint** — Historical data for charts. Params: `?range=1h|6h|24h|3d&maxPoints=300`. Downsamples automatically.
5. **Fixed PM2 exec_mode** — Was `cluster` (causing 25+ restarts). Fixed to `fork` in ecosystem.config.cjs. Persistent Python subprocesses require fork mode.
6. **Killed orphan processes** — Stale Kokoro test processes (PIDs 9544/9545) from previous session consuming 1.6GB RAM.
7. **Fixed analysis-runner.ts** — Already uses `DEFAULT_LLM` env var (was wrongly listed as broken in previous handoff).
8. **Updated tunnel URLs** — Tunnels regenerated after PM2 restart. Railway env vars updated.

---

## Session 6 — March 2026

### Added Chatterbox Turbo Voice Cloning

1. **New TTS provider: Chatterbox Turbo** — Resemble AI's voice cloning model. Uses persistent Python subprocess running in conda env `chatterbox` (Python 3.11) on GPU. Model: `ResembleAI/chatterbox-turbo` from HuggingFace (public, no auth needed). Sample rate 24kHz, resampled to 16kHz. ~4.2GB VRAM.
2. **Persistent subprocess with idle timeout** — Same singleton pattern as Kokoro. Model loads once, serves all requests. 5-minute idle timeout kills process to free GPU memory. Auto-respawns on next request.
3. **Voice management** — `ChatterboxVoiceManager` stores reference audio in `/data/chatterbox-voices/references/` with `manifest.json`. Same CRUD pattern as KokoClone's `VoiceCloneManager`.
4. **IPC endpoints** — `/chatterbox/status`, `/chatterbox/voices`, `/chatterbox/create`, `/chatterbox/voices/:id` (DELETE), `/chatterbox/test`. Same patterns as `/voice-clone/*`.
5. **Fixed KokoClone import bug** — `from kanade import KanadeModel` → `from kanade_tokenizer import KanadeModel`. The pip package `kanade-tokenizer` exports as `kanade_tokenizer`, not `kanade`.
6. **GPU server setup** — Installed miniconda (`/root/miniconda3/`), created conda env `chatterbox` at `/venv/chatterbox/`, cloned repo to `/opt/chatterbox-repo/`, installed `setuptools<71` (required for `pkg_resources` used by `resemble-perth` watermarker). Model downloaded and verified generating audio.

### Gotchas discovered:
- `conda run` on this server does NOT support `--no-banner` flag (conda 26.1.1)
- `HF_TOKEN` env var must be set to any non-empty string (e.g. "skip") to bypass auth check in Chatterbox's `from_pretrained()` — model is public but code has `token=os.getenv("HF_TOKEN") or True`
- `resemble-perth` watermarker needs `setuptools<71` for `pkg_resources` module (removed in setuptools 72+)
- Chatterbox Turbo repo is `ResembleAI/chatterbox-turbo` (NOT `ResembleAI/chatterbox` which is multilingual)

---

## Session 5 — March 2026

### MILESTONE: Full voice pipeline working end-to-end in live calls

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
| **Whisper as default STT, Deepgram as paid option** | Whisper is free/local/reliable. Deepgram Flux has native end-of-turn detection (best for voice agents) but costs $0.0059/min. Granite removed (non-standard API, 4+ errors). |
| Persistent Python subprocesses | Models load once into GPU VRAM. Per-request subprocess was loading models every utterance (15-30s). Persistent process: 0.3-2s. |
| `small.en` Whisper model | Best balance of accuracy and speed for real-time phone calls. `base.en` too inaccurate, `medium.en` too slow. |
| Keyword biasing via initial_prompt | Whisper hears "Freedom Forever" correctly instead of "freedom for ever" when initial_prompt contains the term. |
| 1000ms silence threshold | Phone conversations have natural pauses. 500ms was cutting speakers off mid-sentence. |
| `qwen3.5:9b` over `qwen3:4b` | 2x smarter, better conversation flow, proper tool calling. 6.1GB fits easily on RTX 4090. |
| Kokoro TTS module-level singleton | ALL call sessions share one Python process. Without this, each call spawned a new 30-60s cold start → caller heard silence. |
| Chatterbox Turbo persistent subprocess | Same singleton pattern as Kokoro but with 5-min idle timeout. GPU model loading takes 10-30s cold start. Idle timeout frees ~4.2GB VRAM when not in use. |
| Separate `/chatterbox/*` endpoints (not reusing `/voice-clone/*`) | KokoClone and Chatterbox are distinct systems with different dependencies, conda envs, and storage dirs. Separate namespaces prevent confusion. |
| Conda env for Chatterbox (not system Python) | Chatterbox has conflicting dependencies with the system Python packages (different torch versions, etc). Conda isolates it cleanly. |

---

## Active Model Configuration

| Component | Model | VRAM | Status |
|-----------|-------|------|--------|
| **STT** | Whisper `small.en` | ~500MB | ✅ Active (persistent subprocess, default) |
| **STT** | Deepgram Flux `flux-general-en` | 0 (cloud) | ✅ Available (needs `DEEPGRAM_API_KEY`) |
| **LLM** | qwen3.5:9b | ~5-6GB | ✅ Active (Ollama) |
| **TTS** | Kokoro-82M | ~500MB | ✅ Active (persistent subprocess) |
| **TTS (cloning)** | Chatterbox Turbo | ~4.2GB | ✅ Available (persistent subprocess, idle timeout) |
| TTS (cloning) | KokoClone | ~200MB | ✅ Available (one-off subprocess per synthesis) |
| STT (deprecated) | IBM Granite 4.0 1B Speech | ~2GB | ❌ Removed from factory (code exists, not loaded) |
| **Total VRAM during call** | | **~7GB** (up to ~11GB with Chatterbox) | Within 24GB RTX 4090 |

---

## Gotchas and Important Context

### Critical — Will break calls if wrong

- **Kokoro module-level singleton** — `kokoro.ts` uses module-level variables (`_proc`, `_ready`, `_sharedQueue`). ALL KokoroTTS instances share one Python process. Without this, each call spawns a 30-60s cold start. `destroy()` is a no-op. Never make the process per-instance.
- **Chatterbox Turbo persistent subprocess** — `chatterbox.ts` uses same module-level singleton pattern. Runs in conda env `chatterbox` via `conda run -n chatterbox python3`. Has 5-min idle timeout to free GPU memory. Must set `HF_TOKEN` env to non-empty string to bypass auth check.
- **KokoClone uses `kanade_tokenizer` not `kanade`** — The pip package `kanade-tokenizer` installs as Python module `kanade_tokenizer`. Previous import `from kanade import KanadeModel` was wrong.
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
| `src/index.ts` | Main entry — WebSocket (8765) + IPC HTTP (8766), session management, `/logs`, `/health/gpu`, `/metrics/history` endpoints |
| `src/gpu-monitor.ts` | **GPU/CPU/Memory metrics** — nvidia-smi integration, ring buffer (3-day, 30s interval), snapshot collection |
| `src/model-manager.ts` | Model lifecycle — install/activate/remove. Default STT: whisper/small.en |
| `src/providers/stt/whisper.ts` | **Whisper STT — persistent Python subprocess**, module-level singleton, FIFO queue, keyword biasing |
| `src/providers/stt/deepgram.ts` | **Deepgram Flux STT** — turn-based WebSocket streaming, native end-of-turn, keyterms |
| `src/providers/stt/granite.ts` | Granite STT — deprecated, not loaded by provider factory |
| `src/providers/tts/kokoro.ts` | **Kokoro-82M TTS — module-level singleton**, shared across all calls, binary PCM protocol |
| `src/providers/tts/kokoclone.ts` | KokoClone voice cloning (Kokoro + Kanade voice conversion) |
| `src/providers/tts/chatterbox.ts` | **Chatterbox Turbo voice cloning** — persistent subprocess in conda env, ChatterboxVoiceManager + ChatterboxTurboTTS |
| `src/providers/tts/piper.ts` | Piper TTS fallback (CPU-friendly) |
| `src/providers/llm/openai-compat.ts` | OpenAI-compatible LLM client — 2000-token min for Ollama |
| `src/providers/index.ts` | Provider factory — auto-selects based on config |
| `src/voice-pipeline/call-session.ts` | Core call handler — STT→LLM→TTS loop, tool execution, firstMessage |
| `src/voice-pipeline/analysis-runner.ts` | Post-call summary + success evaluation (uses DEFAULT_LLM env var) |
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
CHATTERBOX_VOICES_DIR=/data/chatterbox-voices
CONDA_PATH=/root/miniconda3/bin/conda
OLLAMA_URL=http://localhost:11434/v1
DEFAULT_LLM=qwen3.5:9b
NODE_ENV=production
HF_HOME=/root/.cache/huggingface
VAPICLONE_API_URL=https://vapiclone-production.up.railway.app
VAPICLONE_API_KEY=vs-internal-secret-2026
IPC_SECRET=vs-ipc-2026
TWILIO_ACCOUNT_SID=***
TWILIO_AUTH_TOKEN=***
DEEPGRAM_API_KEY=***
```

### Current Cloudflared Tunnel URLs

| Service | URL |
|---|---|
| IPC (VOICE_SERVER_IPC_URL) | `https://additionally-recovery-rice-deposit.trycloudflare.com` |
| WebSocket (VOICE_SERVER_URL) | `wss://diy-brakes-ping-collaboration.trycloudflare.com` |
| Ollama (OLLAMA_URL) | `https://health-selling-moment-oregon.trycloudflare.com/v1` |

---

## Debugging Commands

```bash
# Check voiceserver logs remotely (no SSH needed)
curl -s -H "x-ipc-secret: vs-ipc-2026" \
  "https://additionally-recovery-rice-deposit.trycloudflare.com/logs?lines=50" \
  | python3 -c "import sys,json; [print(l) for l in json.load(sys.stdin)['lines']]"

# Check model status
curl -s -H "x-ipc-secret: vs-ipc-2026" \
  "https://additionally-recovery-rice-deposit.trycloudflare.com/models/status" \
  | python3 -m json.tool

# Pre-warm Kokoro TTS
curl -s -H "x-ipc-secret: vs-ipc-2026" \
  "https://additionally-recovery-rice-deposit.trycloudflare.com/tts/test" \
  -X POST -H "Content-Type: application/json" \
  -d '{"text":"Test.","voice":"af_heart"}'

# Switch active STT/LLM
curl -s -X POST -H "x-ipc-secret: vs-ipc-2026" -H "Content-Type: application/json" \
  "https://additionally-recovery-rice-deposit.trycloudflare.com/models/stt/activate" \
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

1. **Set up Cloudflare named tunnel** — Permanent URLs that survive restarts. Currently tunnel URLs change on every PM2 restart.
2. **Tune voicemail detection thresholds** — Now at 150 frames (3s). Test with voicemail detection enabled.
3. **Full UI redesign** — Bushido Pros branding across vapiclone and dialer4clone (separate session).

---

## Tech Stack Summary

| Layer | Technology |
|---|---|
| Runtime | Node.js 22+ (TypeScript) |
| WebSocket | ws 8.18 |
| STT | **Whisper small.en** (persistent subprocess, default), **Deepgram Flux** (cloud, native end-of-turn, paid) |
| TTS | **Kokoro-82M** (module-level singleton, default), **Chatterbox Turbo** (voice cloning, conda env), KokoClone (voice cloning), Piper (fallback), ElevenLabs (paid) |
| LLM | **Ollama + qwen3.5:9b** (default), OpenAI/DeepSeek (paid) |
| GPU | NVIDIA RTX 4090 (24GB VRAM) on Vast.ai |
| Process Manager | PM2 (auto-restart, boot persistence) |
| Tunneling | Cloudflared quick tunnels (ephemeral URLs) |
| CI/CD | GitHub Actions → SSH deploy to Vast.ai |
| Monitoring | nvidia-smi + os module → in-memory ring buffer (3 days) → `/metrics/history` API |
| Audio | mu-law ↔ PCM conversion, 8kHz ↔ 16kHz ↔ 24kHz resampling |
