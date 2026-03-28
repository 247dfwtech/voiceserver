# VoiceServer V2

Node.js/TypeScript voice pipeline server running on Vast.ai GPU (RTX 4090).
Handles real-time STT → LLM → TTS for concurrent AI phone calls.
**LIVE — first successful calls on both Twilio + SignalWire 2026-03-23.**

**GitHub:** https://github.com/247dfwtech/voiceserver
**Last updated:** 2026-03-28 (rev 29 — production test run fixes)

---

## Current State

- **Full voice pipeline LIVE** — Production calls running. Deepgram Flux STT, Kokoro-82M TTS, Ollama qwen2.5:3b-4k LLM.
- **Dual provider** — Both Twilio + SignalWire verified (2026-03-23)
- **ElevenLabs TTS** — Cloud TTS via REST streaming API, pcm_16000 output (no resampling). Configurable model/stability/voice per assistant. Falls back to Kokoro if API key missing.
- **Audio First Message** — Cached mulaw per subuser at `/data/audio-first-messages/`. Instant non-interruptable playback (~5ms). Provider-agnostic (Kokoro, ElevenLabs, Groq, Piper, etc.) — cache key includes `{provider}_{voiceId}`.
- **Voicemail detection** — Audio-based (zero AMD cost) + STT keyword fallback. Thresholds tuned 2026-03-26.
- **Transfer detection** — `call-forwarded` vs `customer-ended-call` with double-transfer guard
- **Overflow LLM** — Auto-routes to Groq when Ollama slots full
- **Post-call analysis** — Groq llama-3.1-8b-instant
- **GPU monitoring** — nvidia-smi metrics, 3-day history ring buffer
- **Custom voices** — `am_adrian` (Kokoro voice clone)
- **VRAM**: ~7GB/24GB used (plenty of headroom)

---

## Architecture

- **Runtime**: Node.js + TypeScript, managed by PM2 on GPU server
- **Instance**: Vast.ai Reserved GPU #33032104 (Quebec, CA — RTX 4090), 63GB RAM
- **Ports**: 8765 (WebSocket for Twilio/SignalWire media streams), 8766 (IPC HTTP)
- **External Access**: Caddy reverse proxy + TLS on Vast.ai mapped ports (direct, no Cloudflare tunnels)
  - WS: ext 45087 → Caddy :6006 **[TLS]** → localhost:8765 → `wss://gpu.prosbookings.com:45087`
  - IPC: ext 45164 → Caddy :8080 → localhost:8766 → `http://70.29.210.33:45164`
  - Ollama: ext 45035 → Caddy :8384 → localhost:11434 → `http://70.29.210.33:45035/v1`
  - URLs configured in vapiclone Settings page (DB-backed). Static — only change on new Vast.ai instance.
  - **TLS is REQUIRED** on the WebSocket port — Twilio/SignalWire reject plain `ws://` for media streams.
  - Cert: Let's Encrypt via acme.sh (DNS challenge, manual TXT record in GoDaddy). **Expires 2026-06-23** (every 90 days).
  - Caddyfile: `/etc/Caddyfile`, certs at `/etc/certs/gpu.prosbookings.com.{crt,key}`
  - Domain: `gpu.prosbookings.com` → A record pointing to Vast.ai instance IP (GoDaddy DNS)

---

## Production Settings (current as of 2026-03-26)

- **LLM**: Ollama qwen2.5:3b-4k (NUM_PARALLEL=9, FLASH_ATTENTION=1, 4K context)
  - IMPORTANT: Use `-4k` model variants (created via Modelfile with `PARAMETER num_ctx 4096`)
  - Default 32K context wastes VRAM (52GB, spills to CPU). 4K is sufficient for phone calls.
  - Available models: `llama3.2:3b-4k`, `qwen2.5:3b-4k`, `phi4-mini-4k`
  - To create a 4K variant: `echo "FROM model:tag\nPARAMETER num_ctx 4096" | ollama create model:tag-4k`
- **STT**: Deepgram Flux (flux-general-en, v2 API, mulaw 8kHz direct)
- **TTS**: Kokoro-82M via FastAPI (voice: am_adrian, port 8880), ElevenLabs (cloud, Turbo v2.5)
- **AMD**: Audio-based voicemail detection (no Twilio/SignalWire AMD charges)
- **Tool mode**: trigger-phrases (5 phrases, 2 tools) — matched "transfer you" → ff_transfer in 3s
- **Post-call analysis**: Groq llama-3.1-8b-instant
- **Overflow LLM**: Groq (configured, activates when Ollama slots full)
- **Custom voices**: `am_adrian` (custom trained) at `/app/api/src/voices/v1_0/am_adrian.pt`

---

## CRITICAL: SignalWire streamSid (DO NOT REVERT)

SignalWire sends `streamSid` inside `msg.start.streamSid`, NOT at the top level like Twilio.
```typescript
// src/index.ts line 169 — MUST have this fallback
streamSid = msg.streamSid || msg.start?.streamSid || null;
```
Without this, ALL outbound audio to SignalWire callers is silently dropped (the `if (streamSid)` check fails).

---

## CRITICAL: Deepgram STT (DO NOT CHANGE)

### The 4 Rules
1. **acceptsMulaw: true** — raw mulaw 8kHz sent directly, NO PCM conversion
2. **ws.ping() for keepalive** — NOT `ws.send(JSON.stringify({type: "KeepAlive"}))`
3. **v2 endpoint** — `wss://api.deepgram.com/v2/listen` with `flux-general-en`
4. **DEEPGRAM_API_KEY in .env** — PM2 must be restarted with `--update-env`

### Connection Parameters
```
wss://api.deepgram.com/v2/listen?model=flux-general-en&encoding=mulaw&sample_rate=8000&eot_threshold=0.75&eot_timeout_ms=1800
```

**Never use Whisper for phone calls** — severe hallucinations on 8kHz mulaw audio.

---

## First Message Optimization

Three tiers of first message delivery, fastest first:

1. **Audio First Message (cached mulaw)** — When `audioFirstMessage: true` on the assistant, the first message is synthesized once per subuser (keyed by `{provider}_{voiceId}_{assistantId}_{agentName}_{textHash}`) and saved as raw mulaw 8kHz at `/data/audio-first-messages/`. Subsequent calls stream the file directly — zero TTS, zero PCM→mulaw conversion (~5ms). Fully non-interruptable. Provider-agnostic: uses `createTTSProvider()` factory — works with Kokoro, ElevenLabs, Groq, Piper, UnrealSpeech. Call metadata records `firstMessageSource: "audio-cache"` vs `"live-tts"`.

2. **Pre-synthesized PCM (per-call)** — When `/register-call` receives a config with `firstMessage` and Kokoro TTS, it synthesizes in the background. By the time Twilio connects (~1-2s later), PCM is cached. `speakPreSynthesized()` streams it with PCM→mulaw conversion (~50-100ms).

3. **Live TTS** — Fallback if neither cache exists. Kokoro synthesizes in real-time via `speak()` (~300-500ms).

STT starts in background (not blocking) in parallel with all three paths.

---

## Key Files

| Path | Purpose |
|---|---|
| `src/index.ts` | WebSocket + IPC server, session lifecycle, streamSid extraction (line 169), audio cache, register-call |
| `src/voice-pipeline/call-session.ts` | Core: STT↔LLM↔TTS, barge-in, voicemail detection, silence timer, cost tracking, overflow LLM, trigger-phrase matching |
| `src/voice-pipeline/call-transfer.ts` | Provider-aware transfer (Twilio/SignalWire) |
| `src/voice-pipeline/analysis-runner.ts` | Post-call analysis (Groq default) |
| `src/providers/tts/kokoro.ts` | Kokoro-82M via FastAPI (port 8880) |
| `src/providers/tts/elevenlabs.ts` | ElevenLabs cloud TTS via REST streaming (pcm_16000) |
| `src/providers/stt/deepgram.ts` | Deepgram Flux WebSocket client |
| `src/providers/llm/openai-compat.ts` | All LLM providers (Ollama, Groq, Cerebras, etc.) |
| `src/providers/index.ts` | Provider factory (instantiates STT/LLM/TTS by name — includes ElevenLabs) |
| `src/gpu-monitor.ts` | GPU/CPU/RAM metrics (PSS-based RAM, not RSS), 3-day ring buffer |
| `src/ollama-concurrency.ts` | Atomic counter for in-flight Ollama requests (overflow detection) |
| `src/kvoicewalk-manager.ts` | KVoiceWalk process manager — spawn, log parsing, completion flow, voice deployment, cleanup |

---

## PM2 Services on GPU

| Name | Port | Notes |
|------|------|-------|
| voiceserver | 8765/8766 | Main voice pipeline |
| ollama | 11434 | LLM inference, started via start-ollama.sh |
| kokoro-fastapi | 8880 | Primary TTS (~1.8GB VRAM) |

---

## IPC Endpoints

- `/health` — GPU stats, service health, sessions, disk, network
- `/sessions` — Active calls, Ollama slot utilization, overflow status
- `/services` — PM2 service statuses, start/stop
- `/amd-result/:callId` — AMD result from vapiclone (voicemail detection)
- `/register-call` — Register pending call config (60s TTL)
- `/settings` — GET/PUT server settings (.env persistence)
- `/tts/test` — TTS voice testing
- `/kvoicewalk/start` — POST: Start kvoicewalk voice cloning (receives base64 audio + transcript + name)
- `/kvoicewalk/status` — GET: Current step, score, ETA, score history (parses run.log)
- `/kvoicewalk/stop` — POST: Stop walk early, deploy best voice to Kokoro, cleanup
- `/kvoicewalk/best-sample` — GET: Returns current best .wav as audio/wav binary
- `/kvoicewalk/voices` — GET: List custom (non-stock) Kokoro voices
- `/kvoicewalk/voices/:name` — DELETE: Remove custom voice .pt, restart kokoro-fastapi

---

## Voicemail Detection — WORKING (Audio-Based, Zero AMD Cost)

**No Twilio/SignalWire AMD used** — all detection is built-in audio analysis in `call-session.ts`:
- `VoicemailDetector` class: RMS energy analysis on PCM frames, speech ratio, continuous speech detection
- Continuous speech (1.2s+) → machine detected → enters beep-wait mode → waits for 2.5s silence → enters beep-watch phase (up to 2s listening for Goertzel beep) → resolves as voicemail
- Thresholds (tuned 2026-03-26 LIVE1 Part 2): speech ratio >65%, max continuous >20 frames (400ms), min 50 speech frames, 7s analysis window
- **STT keyword fallback**: If audio analysis resolves as "human" but first transcript contains VM keywords ("leave a message", "at the tone", "can't take your call", etc.), retroactively switches to voicemail handling
- **No-response fallback**: if VM resolves as "human" but no user speech arrives within 8s of first message completion → retroactively switches to voicemail mode
- Beep detection: Goertzel checks 440Hz (T-Mobile), 850Hz (AT&T), 1000Hz (default), 1400Hz (VOIP) — requires 100+ speech frames AND 10 consecutive beep frames (200ms sustained tone) — filters false positives from ring tones and speech harmonics
- Post-silence beep-watch: after 2.5s silence, listens up to 2 more seconds for beep before resolving (was 5s)
- Post-resolve delay: 800ms before speaking voicemail message (was 1500ms)
- First message HELD while detection runs
- All STT transcripts suppressed while detection pending AND after voicemail confirmed
- **VM N/M ratio in voiceserver**: in-memory `vmDialerCounters` Map checks vmLeaveN/vmLeaveM from config at VM detection time — exact 1-in-M delivery based on actual VM outcomes, not async placement-time logic
- If `voicemailMessage` set AND N/M counter passes → TTS speaks it → `waitForTTSFinish` waits actual audio duration (up to 60s) → hangs up
- If no `voicemailMessage` or N/M counter skips → hangs up immediately
- dialer4clone always sends voicemailMessage + vmLeaveN/vmLeaveM in `assistantOverrides`; voiceserver decides leave/skip

---

## Transfer Detection — Distinct EndedReasons (2026-03-24)

Stream stop no longer always reports `twilio-stream-stopped`. `SessionEntry` tracks `transferInitiated` flag:
- Transfer tool fires → flag set → stream stops → `endedReason: "call-forwarded"`
- No transfer → stream stops → `endedReason: "customer-ended-call"`
- dialer4clone uses `call-forwarded` to trigger auto phone lookup (no more false positives)

### Double-Transfer Guard (2026-03-25 LIVE1)
Transfer could fire twice if LLM both said "transfer" (trigger phrase) AND invoked the transfer tool — causing two outbound legs to the fallback number. Fixed with two guards:
1. `index.ts` transfer event handler: skips if `transferInitiated` already true
2. `call-session.ts`: both trigger-phrase and tool-call paths check `this.state === "transferring"` before emitting

---

## Post-Call Analysis

- Runs after call ends for answered calls with real conversation
- Default provider: Groq llama-3.1-8b-instant
- Supported: local (Ollama), deepseek, openrouter, cerebras, groq, deepinfra
- Cost tracked per-call in CostBreakdown.analysis

---

## Overflow LLM

When Ollama slots full, excess requests auto-route to cloud API.
- Counter in `src/ollama-concurrency.ts`, checked in `processWithLLM()`
- Settings: `OVERFLOW_LLM_PROVIDER` + `OVERFLOW_LLM_MODEL` env vars
- Cost tracked separately in `CostBreakdown.overflowLlm`

---

## Build & Deploy

```bash
ssh -p 45194 root@70.29.210.33
export PATH=/opt/nvm/versions/node/v24.12.0/bin:$PATH
cd /opt/voiceserverV2 && git pull && npx tsc && pm2 restart voiceserver --update-env
```

---

## New Instance Setup — TLS + Caddy (NEVER use Cloudflare tunnels)

When setting up a new Vast.ai instance, follow this to get TLS working for Twilio/SignalWire media streams:

1. **Note the new Vast.ai port mappings** — check `env | grep VAST_TCP_PORT` for external→internal port map
2. **Update DNS** — change `gpu.prosbookings.com` A record in GoDaddy to the new Vast.ai IP
3. **Install acme.sh**: `curl -s https://get.acme.sh | sh -s email=247dfwtech@gmail.com`
4. **Issue cert** (manual DNS challenge — GoDaddy API permissions are limited):
   ```bash
   /root/.acme.sh/acme.sh --issue -d gpu.prosbookings.com --dns \
     --yes-I-know-dns-manual-mode-enough-go-ahead-please --server letsencrypt
   # It prints a TXT record value — add it in GoDaddy DNS as:
   #   Type: TXT, Name: _acme-challenge.gpu, Value: <the value>
   # Then complete:
   /root/.acme.sh/acme.sh --renew -d gpu.prosbookings.com \
     --yes-I-know-dns-manual-mode-enough-go-ahead-please --server letsencrypt
   mkdir -p /etc/certs
   cp /root/.acme.sh/gpu.prosbookings.com_ecc/fullchain.cer /etc/certs/gpu.prosbookings.com.crt
   cp /root/.acme.sh/gpu.prosbookings.com_ecc/gpu.prosbookings.com.key /etc/certs/gpu.prosbookings.com.key
   ```
5. **Configure Caddy** — add to `/etc/Caddyfile` (adjust internal ports to match Vast.ai mapping):
   ```
   :6006 {
       tls /etc/certs/gpu.prosbookings.com.crt /etc/certs/gpu.prosbookings.com.key
       reverse_proxy localhost:8765 { flush_interval -1 }
   }
   :8080 {
       reverse_proxy localhost:8766 { flush_interval -1 }
   }
   :8384 {
       reverse_proxy localhost:11434 { flush_interval -1 }
   }
   ```
6. **Reload Caddy**: `/opt/portal-aio/caddy_manager/caddy reload --config /etc/Caddyfile`
7. **Update vapiclone Settings** (DB-backed, not env vars):
   - `VOICE_SERVER_URL` = `wss://gpu.prosbookings.com:<external WS port>`
   - `VOICE_SERVER_IPC_URL` = `http://<new IP>:<external IPC port>`
   - `OLLAMA_URL` = `http://<new IP>:<external Ollama port>/v1`
8. **Cert renewal** — expires every 90 days, re-run step 4

---

## Known Issues

- **GitHub Actions deploy is a no-op** — must deploy manually via SSH
- **PM2 doesn't auto-reload .env** — `pm2 restart --update-env` captures shell env (may set empty vars that block dotenv). Safest: `pm2 delete voiceserver && pm2 start ecosystem.config.cjs --only voiceserver`
- **Vast.ai instance IP change** — update DNS A record, re-issue cert, update vapiclone Settings URLs
- **Stale src/call-session.ts on GPU** — moved to src/voice-pipeline/call-session.ts, old file causes harmless tsc warnings
- **Build path on GPU**: `export PATH=/opt/nvm/versions/node/v24.12.0/bin:$PATH` required before npm/npx commands
- **NEVER use Cloudflare tunnels** — always use Caddy reverse proxy + TLS. Lower latency, static URLs, fewer processes. Twilio/SignalWire require `wss://` for media streams.
- **Never use Whisper for phone calls** — severe hallucinations on 8kHz mulaw audio. Use Deepgram Flux.

---

## Recent Sessions

### Session 33 — 2026-03-28 (Production Test Run Analysis & Fixes)
Six fixes from production test run analysis:
- **Beep-watch timeout reduced** — 5s (250 frames) → 2s (100 frames) for faster "no beep" fallback path
- **Post-beep delay reduced** — 1500ms → 800ms before speaking voicemail message
- **Broadened Goertzel beep detection** — checks 440Hz (T-Mobile), 850Hz (AT&T), 1000Hz (default), 1400Hz (VOIP). Consecutive frame threshold lowered 15→10 (300ms→200ms). Catches more carrier beep frequencies.
- **VM N/M ratio moved to voiceserver** — in-memory `vmDialerCounters` Map (keyed by dialerId) checks vmLeaveN/vmLeaveM from sessionConfig at VM detection time. Gives exact 1-in-M VM message delivery based on actual VM outcomes, not async placement-time counter.
- **No-response fallback** — if VM detector resolves as "human" but no user speech arrives within 8s of first message completion → retroactively switches to voicemail mode (catches silent VMs with no greeting)
- **Audio first message cache provider-agnostic** — removed Kokoro-only gate, uses `createTTSProvider()` factory. Works with ElevenLabs, Groq, Piper, UnrealSpeech, and future providers. Cache key now `{provider}_{voiceId}_{assistantId}_{agentName}_{textHash}` to prevent cross-provider collisions.

### Session 31 — 2026-03-28 (VM Beep Detection Hardened — Verified on 2 phones)
Three fixes to stop assistant speaking before the voicemail beep:
- **Beep-watch phase** — after 2.5s silence (greeting ended), listens for Goertzel beep for up to 5 more seconds before resolving. Previously resolved immediately on silence.
- **Minimum speech gate** — beep detection requires 100+ speech frames (~2s of actual speech) before accepting any beep. Prevents false positives from ring tones / early call noise.
- **Consecutive frame requirement** — requires 15 consecutive beep frames (300ms sustained tone) instead of a single frame. Filters out transient speech harmonics that matched 1000Hz.
- Worst-case greeting-to-speak time: 2.5s silence + 5s beep-watch + 1.5s post-resolve delay = 9s
- If real beep detected during watch → resolves instantly (no unnecessary delay)
- Verified working on 2 different phones/carriers

### Session 30 — 2026-03-28 (VM Beep-Wait Tuning)
- Increased VM beep-wait silence threshold from 60 frames (1.2s) to 125 frames (2.5s) — mid-greeting pauses were causing premature voicemail message playback before the actual beep

### Session 29 — 2026-03-28 (Cross-Stack Bug Audit — 16 fixes across 3 apps)
**voiceserverV2 fixes (10):**
- IPC auth fail-closed: empty `IPC_SECRET` now rejects all requests (was silently unauthenticated)
- `/register-call` validates callId (string) and config (object) before storing
- `/end-call/:id` now also clears `pendingConfigs` (not just active sessions)
- Duplicate WS for same callId rejected (`sessions.has()` check before creating)
- `ws.close()` guarded with `readyState === OPEN` check (prevents double-close)
- Ollama slot leak fix: `pendingOllamaRelease` tracked on instance, released in `endCall()` even if `onDone` never fires
- LLM retry paths store cancel handle in `currentLLMCancel` so `endCall()` can abort them; deferred tool call guarded with `state !== "ended"`
- STT `removeAllListeners()` before `close()` on fallback switch and `endCall()` (prevents handler accumulation)
- `endCall()` made atomic with `isEndingCall` flag (prevents double-fire from concurrent async callbacks)
- Trigger phrase `.then()` callbacks check `isEndingCall` before executing post-call actions
- ElevenLabs TTS error handler: safe access `err instanceof Error ? err.message : String(err)` (prevents secondary throw)
- Deepgram `ws.close()` catch now logs at debug level instead of swallowing

### kvoicewalk2 — 2026-03-27
- ElevenLabs TTS provider (`src/providers/tts/elevenlabs.ts`) — REST streaming, pcm_16000 direct output
- Added to provider factory with Kokoro fallback if API key missing
- Cost tracking: $0.18/1K chars (configurable via `COST_TTS_ELEVENLABS`)
- PM2 env fix: stale empty `ELEVENLABS_API_KEY` was blocking dotenv — resolved by deleting and re-creating PM2 process

### LIVE1 Part 2 — 2026-03-26
**Voicemail Detection Overhaul** (21 production calls analyzed):
- Lowered continuous speech threshold: 2s → 1.2s (catches short carrier greetings)
- Extended analysis window: 5s → 7s, lowered decision thresholds
- Increased beep-wait silence: 800ms → 1.2s, post-beep delay: 500ms → 1500ms
- Added STT keyword fallback: if audio misses but transcript says "leave a message" etc., retroactively switches to voicemail
- Added pendingAudioDurationMs-aware wait on endCall trigger

**Audio First Message Cache:**
- `audioFirstMessage` toggle on assistants → cached mulaw per subuser
- File: `/data/audio-first-messages/{assistantId}_{agentName}_{textHash}.ulaw`
- Fully non-interruptable playback, call metadata records source

### kvoicewalk2 — 2026-03-27
**Voice Cloning Studio backend:**
- New `src/kvoicewalk-manager.ts` module — spawns kvoicewalk python process, parses run.log for progress/scores, completion flow (copy best .pt → Kokoro voices, write metadata, restart kokoro-fastapi, cleanup)
- 6 new IPC endpoints: `/kvoicewalk/{start,status,stop,best-sample,voices,voices/:name}`
- One job at a time (GPU constraint), idempotent completion flow, stock voice deletion protection
- Output subdir polling (kvoicewalk creates it mid-run), score-from-filename parsing
- First clone: ElevenLabs Jessica → jessica_11clone (initiated from UI)

### LIVE1 — 2026-03-25
- Double-transfer guard (trigger phrase + tool call firing simultaneously)
- Cloudflare tunnels → Caddy reverse proxy + TLS
- Custom Kokoro voice `am_adrian`, pre-synthesized first message

### Earlier Sessions (March 2026)
- Session 12: Full voice pipeline perfect — transfer with spoken message
- Session 11: Call transfer end-to-end
- Session 8: Deepgram Flux STT confirmed in live calls
- Session 7: GPU monitoring backend
