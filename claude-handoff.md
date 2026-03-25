# VoiceServer V2 — Claude Handoff Document

**Last updated:** 2026-03-19 (rev 18 — V2 upgrade)
**GitHub:** https://github.com/247dfwtech/voiceserver (V2 not pushed yet)
**Local path:** /Users/adriansanchez/Desktop/voiceserverV2
**Running on:** Vast.ai Reserved GPU Instance #33032104 (Quebec, CA — RTX 4090)
**SSH:** `ssh -p 45194 root@70.29.210.33`
**Server path:** /opt/voiceserverV2/

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
- **PM2 managed** — voiceserver + ollama + kokoro-fastapi, auto-restart, boot persistence (Cloudflare tunnels removed 2026-03-25)
- **GitHub Actions auto-deploy** ✅ — Push to main → SSH into Vast.ai → pull, build, restart. Deploy logs deployed commit hash. Syncs Twilio + Deepgram + OpenAI keys from GitHub Secrets to `.env`.
- **GPU/CPU/Memory monitoring** ✅ — `gpu-monitor.ts` collects metrics every 30s via nvidia-smi + os module. 3-day in-memory ring buffer (8640 entries, ~1.7MB). Endpoints: `/health` (enhanced with GPU data inline), `/health/gpu` (fresh snapshot), `/metrics/history?range=1h|6h|24h|3d` (historical data for charts)
- **`/logs` endpoint** — `GET /logs?lines=N` returns recent PM2 log lines; requires `x-ipc-secret` header
- **IPC auth** — All IPC calls require either `Authorization: Bearer <VAPICLONE_API_KEY>` or `x-ipc-secret: <IPC_SECRET>` header

## What's Not Working / Known Issues

- ~~**Cloudflared tunnel URLs are ephemeral**~~ — RESOLVED 2026-03-25: Cloudflare tunnels removed entirely. Now using Caddy reverse proxy + **TLS (Let's Encrypt)** on Vast.ai mapped ports. WebSocket URL: `wss://gpu.prosbookings.com:45087`. Caddy config in `/etc/Caddyfile`, certs at `/etc/certs/`. Port mapping: ext 45087→Caddy :6006 [TLS]→WS :8765, ext 45164→Caddy :8080→IPC :8766, ext 45035→Caddy :8384→Ollama :11434. **IMPORTANT: Twilio/SignalWire require `wss://` — plain `ws://` silently fails.** See CLAUDE.md "New Instance Setup" for TLS cert instructions.
- **Deepgram WebSocket drops every ~5 seconds** — FIXED (rev 13). Root cause: KeepAlive sent as text data frames interleaved with binary audio caused Deepgram's audio processor to choke. Fix: switched to `ws.ping()` control frames.
- **Whisper hallucinations on 8kHz phone audio** — Whisper `small.en` produces severe hallucinations on Twilio phone audio (mulaw 8kHz resampled to 16kHz): "Thanks for watching!", "$100,000,000,000,000,000", "I'll get them ready" when user said something completely different. **Deepgram Flux is strongly recommended for production phone calls.** Whisper may work better with `medium.en` or `turbo` but untested.
- **PM2 doesn't auto-reload .env on file change** — If you add/change API keys in `.env`, you must run `pm2 restart voiceserver --update-env`. The `dotenv/config` import only reads `.env` at process startup.

---

## What Was Just Completed (Session 14 — March 19, 2026)

### Debugging & Fixes — STT, LLM, Tunnels, Call Pipeline

1. **Sherpa-ONNX abandoned** — 20M model too inaccurate for mulaw phone audio. Removed from server (model + pip package). Code stays as placeholder in provider factory (routes to Vosk).
2. **Vosk tested** — works on clean audio but fails on real phone audio (mulaw artifacts). Not production-viable.
3. **Deepgram confirmed as production STT** — see CRITICAL section below.
4. **stt.finish() bug fixed** — was destroying user transcripts at first message end. Removed the finish() call.
5. **Silence timer bug fixed** — was firing during first message playback. Added playingFirstMessage guard.
6. **Ollama OLLAMA_ORIGINS=* fix** — PM2 entry recreated to use start-ollama.sh script (not bare command). Fixes 403 on tunnel requests.
7. **Tunnel PM2 entries fixed** — were pointing to deleted V1 path `/opt/voiceserver/`. Recreated with V2 paths.
8. **qwen3.5:9b references purged** — removed hardcoded fallbacks from test-chat route and analysis-runner.ts.
9. **LLM warmup on boot** — llama3.2:3b loaded into GPU VRAM at startup.

---

## CRITICAL: Deepgram Flux STT — Proven Working Config (DO NOT CHANGE)

These settings were proven in V1 Sessions 8-12. If testing other STT options, NEVER alter these:

### The 4 Rules
1. **acceptsMulaw: true** — raw Twilio mulaw → Deepgram directly, NO PCM conversion
2. **ws.ping() for keepalive** — NOT `ws.send(JSON.stringify({type: "KeepAlive"}))` (causes audio choke)
3. **v2 endpoint** — `wss://api.deepgram.com/v2/listen` with `flux-general-en`
4. **DEEPGRAM_API_KEY in process.env** — must restart PM2 with `--update-env` after .env changes

### Protected Files (do not modify for other STT experiments)
- `src/providers/stt/deepgram.ts` — ws.ping() keepalive, acceptsMulaw, event mapping
- `src/providers/stt/interface.ts` — acceptsMulaw flag definition
- `call-session.ts` lines 423-432 — acceptsMulaw audio routing bypass

### Connection URL
```
wss://api.deepgram.com/v2/listen?model=flux-general-en&encoding=mulaw&sample_rate=8000&eot_threshold=0.75&eot_timeout_ms=1800
```

### Event Mapping
| Deepgram Event | STT Event | Action |
|---|---|---|
| StartOfTurn | speech_started | Barge-in if speaking |
| Update | transcript (isFinal: false) | Interim text |
| EndOfTurn | transcript (isFinal: true) + utterance_end | → LLM processing |

### Known Gotchas
- `stt.finish()` sends Deepgram `Finalize` which splits active utterances — NEVER call during live speech
- Text KeepAlive frames interleaved with binary audio cause Deepgram to choke — always use ws.ping()
- Assistant must have provider: "deepgram" and model: "flux-general-en" saved in VapiClone

---

## Session 13 — March 19, 2026

### V2 Infrastructure Upgrade
- V2 folders, node_modules, GPU server prep, Ollama NUM_PARALLEL=8 + FLASH_ATTENTION=1
- 8 concurrent LLM requests in 4.8s (stress tested)
- VapiClone Settings page (server config card), assistant page (STT/LLM dropdowns)
- Security hardening, LLM warmup, model manager updates
- Deployed to /opt/voiceserverV2/ and Railway

---

## Session 12 — March 18, 2026

### MILESTONE: Full Voice Pipeline Perfect — Transfer with Spoken Message

**Historic milestone:** Complete end-to-end call confirmed perfect. Deepgram Flux transcribes → LLM responds → Kokoro speaks transfer message → caller hears full message → Twilio transfers. Tool invocations visible in transcript.

1. **Fixed Deepgram WebSocket dropping every ~5 seconds** — Root cause: `ws.send(JSON.stringify({type: "KeepAlive"}))` sent text data frames interleaved with binary audio. Deepgram docs warn this "will cause audio processing to choke." Fix: switched to `ws.ping()` WebSocket control frames (RFC-6455). Connection now stays stable.
2. **Configurable Deepgram Flux end-of-turn settings** — `eot_threshold` (was hardcoded 0.7, now configurable, default 0.75) and `eot_timeout_ms` (was hardcoded 5000, now configurable, default 1800).
3. **Confidence threshold filtering** — Filters low-confidence EndOfTurn transcripts before emitting.
4. **Whisper VAD configurable** — `endOfTurnTimeoutMs` → silence frames, `confidenceThreshold` → RMS threshold.
5. **startSpeakingPlan** — `waitSeconds` (default 0.6s) delay after utterance_end before LLM. Timer resets on more speech.
6. **stopSpeakingPlan** — Barge-in requires `numWords` (default 3) AND `voiceSeconds` (default 0.5s). Prevents "yes" interrupting mid-sentence.
7. **TTS-before-tool (3 fixes):**
   - Transfer/endCall wait for TTS to finish before executing
   - LLM response text flushed to TTS before tool call fires (was stuck in buffer)
   - `waitForTTSFinish` tracks actual audio duration and waits for Twilio playback
8. **Natural first message barge-in** — `playingFirstMessage` flag active for estimated audio playback duration (~60ms/char). During greeting: short acknowledgements (< numWords threshold) are discarded entirely, not accumulated. If user says 4+ words, assistant stops greeting and responds. Sub-threshold words cleared when greeting finishes. **Confirmed working in live test calls** — user spoke two words during greeting, both discarded, assistant only responded to post-greeting speech.
9. **Tool calls in transcript** — `fullTranscript.push({role: "Tool", content: "[ff_transfer] transfer"})` so call logs show when tools fired.
10. **STT provider in end-of-call report** — `sttProvider` and `sttModel` now sent to vapiclone so call logs show which transcriber was used.
11. **Configurable cost rates** — CostTracker reads from env vars (`COST_STT_DEEPGRAM`, `COST_TRANSPORT_TWILIO`, etc.) with sensible defaults. Set via vapiclone Settings sync. Kokoro/Whisper/Ollama hardcoded at $0 (self-hosted).
12. **distil-large-v3 model downloaded** — Pre-downloaded on GPU for testing. 6x faster than large-v3, similar accuracy, 756MB. Select in assistant Transcriber tab.
13. **NVIDIA Canary + Parakeet** — Added as provider options in vapiclone UI dropdown (not yet implemented server-side, placeholders for future).

---

## Session 11 — March 18, 2026

### MILESTONE: Call Transfer Working End-to-End

**Full pipeline confirmed:** Deepgram Flux transcribes → LLM invokes `ff_transfer` → Twilio transfers call to fallback number. Live test call successfully transferred to +16824413056.

1. **Fixed audio format for Deepgram** — Was converting Twilio's mulaw 8kHz to PCM 16kHz before sending to Deepgram, but telling Deepgram `encoding=mulaw`. Now sends raw mulaw bytes with correct `encoding=mulaw&sample_rate=8000`. Deepgram handles conversion internally.
2. **Kokoro TTS pre-warm on server startup** — Added startup pre-warm that synthesizes a short test phrase during server boot. First call no longer has 6-second cold start — Kokoro model is already loaded in GPU VRAM.
3. **Tools tab in vapiclone UI** — Added 7th tab to assistant editor for attaching tools. Critical discovery: tools MUST be attached to the assistant via `toolIds` — mentioning them in the system prompt is not sufficient for the LLM to invoke them.

---

## Session 10 — March 18, 2026

### Deepgram Model Validation + STT Debugging

1. **Fixed Deepgram HTTP 400 — wrong model name** — When assistant config had `provider: "deepgram"` with `model: "small.en"` (a Whisper model name left over from provider switch), Deepgram rejected the WebSocket connection with HTTP 400. Provider factory (`src/providers/index.ts`) now validates model names against `DEEPGRAM_MODELS` list and defaults to `flux-general-en` if invalid.
2. **Discovered PM2 env not auto-loading .env changes** — `DEEPGRAM_API_KEY` was in `/opt/voiceserver/.env` but the PM2-managed process didn't have it in `process.env` because PM2 started before the key was added. Fixed with `pm2 restart voiceserver --update-env`. Documented as gotcha.
3. **Confirmed Deepgram Flux v2 WebSocket works** — Direct test from GPU server: WebSocket connects to `wss://api.deepgram.com/v2/listen?model=flux-general-en` with stored API key. Connection succeeds.
4. **Discovered Whisper hallucination severity** — Whisper `small.en` transcribing real phone calls produced: `"Thanks for watching!"`, `"$100,000,000,000,000,000"`, `"I'll get them ready"`, `"electric, and solar panels, and solar, and solar, holding, dollars"`. All completely wrong. Root cause: 8kHz mulaw phone audio resampled to 16kHz is too low quality for Whisper. Deepgram Flux recommended for production.

---

## Session 9 — March 18, 2026

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

## Active Model Configuration (V2)

| Component | Model | VRAM | Status |
|-----------|-------|------|--------|
| **STT** | Deepgram Flux `flux-general-en` | 0 (cloud) | ✅ PRODUCTION DEFAULT (needs `DEEPGRAM_API_KEY`) |
| **STT** | Vosk `en-us-0.22` | 0 (CPU) | ⚠️ Installed but inaccurate on phone audio |
| **STT** | Sherpa-ONNX | — | ❌ REMOVED from server (too small for phone audio) |
| **LLM** | llama3.2:3b | ~2GB + KV cache | ✅ Active (Ollama, NUM_PARALLEL=8, FLASH_ATTENTION=1) |
| **LLM** | qwen3:1.7b | ~1.4GB | ✅ Installed (switchable) |
| **TTS** | Kokoro-82M | ~500MB | ✅ Active (persistent subprocess, 96x realtime) |
| **TTS (cloning)** | KokoClone | ~200MB | ✅ Ready (kokoro-onnx + kanade-tokenizer) |
| **TTS (cloning)** | Chatterbox Turbo | ~4.2GB | ❌ NOT installed (needs conda env) |
| **Total VRAM during call** | | **~10GB** | 14GB headroom on 24GB RTX 4090 |

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
- **PM2 does NOT auto-reload .env on file change** — `dotenv/config` reads `.env` once at process startup. If you add/change keys in `.env`, you MUST run `pm2 restart voiceserver --update-env`. Without this, the running process won't see new keys even though they're in the file.
- **Deepgram model name validation** — Provider factory validates `config.model` against known Deepgram models. If a Whisper model name (e.g. `small.en`) is passed when Deepgram is selected, it auto-defaults to `flux-general-en`. This catches stale model names from UI provider switches.
- **Twilio credentials required on GPU VPS** — `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN` must be in voiceserver `.env` for call transfers to work. Without them, `ff_transfer` tool fires but transfer silently fails. Set via GitHub Secrets (auto-synced on deploy), VapiClone Settings page, or manually.
- **GitHub Actions deploy.yml `script:` must be under `with:`, NOT `env:`** — Previously the `script:` was nested under `env:` which made YAML treat it as an env var. Result: SSH connected but executed empty command, every deploy showed "success" but did nothing. The `env:` block must be a sibling of `with:`, not nested inside it.

### Important — Will cause confusion if forgotten

- **Vast.ai port mapping** — Only Docker-mapped ports are accessible externally. Voiceserver ports (8765/8766/11434) are exposed via Caddy reverse proxy on mapped ports (6006→8765, 8080→8766, 8384→11434). Caddy config: `/etc/Caddyfile`.
- **IPC auth headers** — IPC endpoints require `x-ipc-secret: <IPC_SECRET>` header. Current: `vs-ipc-2026`.
- **Audio format** — Twilio sends mu-law 8kHz mono. For Deepgram: raw mulaw bytes sent directly with `encoding=mulaw&sample_rate=8000` (Deepgram handles conversion). For Whisper: converted to PCM 16-bit 16kHz. Kokoro outputs 24kHz, resampled to 16kHz, then to mu-law 8kHz for Twilio.
- **Ollama needs OLLAMA_ORIGINS=*** — Without this, returns 403 to requests from external IPs (Caddy proxy or otherwise).
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
| `src/providers/stt/sherpa.ts` | **Sherpa-ONNX STT — persistent Python subprocess**, streaming, CPU-only, DEFAULT |
| `src/providers/stt/whisper.ts` | Whisper STT — persistent Python subprocess, FIFO queue, keyword biasing (hallucinates on phone audio) |
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
| `.github/workflows/deploy.yml` | GitHub Actions — SSH deploy to Vast.ai: syncs API keys (Twilio, Deepgram, etc.), pulls, builds, restarts PM2. Logs deployed commit hash. |

### Current .env on Vast.ai (`/opt/voiceserverV2/.env`)

```env
WS_PORT=8765
IPC_PORT=8766
KOKORO_MODELS_DIR=/models/kokoro
KOKORO_VOICE=af_heart
CLONED_VOICES_DIR=/data/cloned-voices
CHATTERBOX_VOICES_DIR=/data/chatterbox-voices
CONDA_PATH=/root/miniconda3/bin/conda
OLLAMA_URL=http://localhost:11434/v1
DEFAULT_LLM=llama3.2:3b
NODE_ENV=production
HF_HOME=/root/.cache/huggingface
VAPICLONE_API_URL=https://vapiclone-production.up.railway.app
VAPICLONE_API_KEY=vs-internal-secret-2026
IPC_SECRET=vs-ipc-2026
TWILIO_ACCOUNT_SID=***
TWILIO_AUTH_TOKEN=***
DEEPGRAM_API_KEY=***
```

### GitHub Secrets (for auto-deploy)

| Secret | Purpose |
|---|---|
| `SSH_HOST` | Vast.ai instance IP (70.29.210.33) |
| `SSH_PORT` | Vast.ai SSH port (45194) |
| `SSH_USER` | root |
| `SSH_PRIVATE_KEY` | SSH private key for Vast.ai |
| `TWILIO_ACCOUNT_SID` | Twilio account SID — synced to .env on deploy |
| `TWILIO_AUTH_TOKEN` | Twilio auth token — synced to .env on deploy |
| `DEEPGRAM_API_KEY` | Deepgram API key — synced to .env on deploy |

### Direct Connection URLs (Caddy Reverse Proxy — set 2026-03-25)

| Service | URL | Route |
|---|---|---|
| IPC (VOICE_SERVER_IPC_URL) | `http://70.29.210.33:45164` | ext 45164 → Caddy :8080 → localhost:8766 |
| WebSocket (VOICE_SERVER_URL) | `ws://70.29.210.33:45087` | ext 45087 → Caddy :6006 → localhost:8765 |
| Ollama (OLLAMA_URL) | `http://70.29.210.33:45035/v1` | ext 45035 → Caddy :8384 → localhost:11434 |

**Preference: Always use direct connections over Cloudflare tunnels.** Lower latency, static URLs, fewer processes.

---

## Debugging Commands

```bash
# Check voiceserver logs remotely (no SSH needed)
curl -s -H "x-ipc-secret: vs-ipc-2026" \
  "http://70.29.210.33:45164/logs?lines=50" \
  | python3 -c "import sys,json; [print(l) for l in json.load(sys.stdin)['lines']]"

# Check model status
curl -s -H "x-ipc-secret: vs-ipc-2026" \
  "http://70.29.210.33:45164/models/status" \
  | python3 -m json.tool

# Pre-warm Kokoro TTS
curl -s -H "x-ipc-secret: vs-ipc-2026" \
  "http://70.29.210.33:45164/tts/test" \
  -X POST -H "Content-Type: application/json" \
  -d '{"text":"Test.","voice":"af_heart"}'

# Switch active STT/LLM
curl -s -X POST -H "x-ipc-secret: vs-ipc-2026" -H "Content-Type: application/json" \
  "http://70.29.210.33:45164/models/stt/activate" \
  -d '{"name":"small.en"}'

# SSH into GPU
ssh -p 45194 root@70.29.210.33

# After SSH: deploy latest code (normally auto-deploys via GitHub Actions)
export PATH="/usr/bin:/opt/nvm/versions/node/v24.12.0/bin:/opt/instance-tools/bin:$PATH"
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
[STT: Sherpa-ONNX streaming (CPU, persistent subprocess)]
  - Zipformer 20M English model
  - Streams base64 PCM chunks via JSON stdin
  - Gets real-time partial + final transcripts via stdout
  - Native endpoint detection (no manual VAD)
    |
    v  text transcript
    |
[LLM: Ollama llama3.2:3b (NUM_PARALLEL=8, FLASH_ATTENTION=1)]
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

1. ~~**Set up Cloudflare named tunnel**~~ — DONE (2026-03-25): Removed tunnels entirely. Using direct Caddy reverse proxy on Vast.ai mapped ports. Static URLs, lower latency.
3. **Tune voicemail detection thresholds** — Now at 150 frames (3s). Test with voicemail detection enabled.
4. **Full UI redesign** — Bushido Pros branding across vapiclone and dialer4clone (separate session).

---

## Tech Stack Summary

| Layer | Technology |
|---|---|
| Runtime | Node.js 22+ (TypeScript) |
| WebSocket | ws 8.18 |
| STT | **Sherpa-ONNX Zipformer** (persistent subprocess, CPU, default), **Deepgram Flux** (cloud, paid), Vosk (fallback) |
| TTS | **Kokoro-82M** (module-level singleton, default), **Chatterbox Turbo** (voice cloning, conda env), KokoClone (voice cloning), Piper (fallback), ElevenLabs (paid) |
| LLM | **Ollama + llama3.2:3b** (default, NUM_PARALLEL=8), qwen3:1.7b (backup), OpenAI/DeepSeek (paid) |
| GPU | NVIDIA RTX 4090 (24GB VRAM) on Vast.ai |
| Process Manager | PM2 (auto-restart, boot persistence) |
| External Access | Caddy reverse proxy on Vast.ai mapped ports (direct, static URLs — no Cloudflare tunnels) |
| CI/CD | GitHub Actions → SSH deploy to Vast.ai |
| Monitoring | nvidia-smi + os module → in-memory ring buffer (3 days) → `/metrics/history` API |
| Audio | mu-law ↔ PCM conversion, 8kHz ↔ 16kHz ↔ 24kHz resampling |
