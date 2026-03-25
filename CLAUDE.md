# VoiceServer V2

Node.js/TypeScript voice pipeline server running on Vast.ai GPU (RTX 4090).
Handles real-time STT → LLM → TTS for concurrent AI phone calls.
**LIVE — first successful calls on both Twilio + SignalWire 2026-03-23.**

## Architecture

- **Runtime**: Node.js + TypeScript, managed by PM2 on GPU server
- **Deploy**: `ssh -p 45194 root@70.29.210.33` then `cd /opt/voiceserverV2 && git pull && npx tsc && pm2 restart voiceserver --update-env`
- **Ports**: 8765 (WebSocket for Twilio/SignalWire media streams), 8766 (IPC HTTP)
- **RAM**: 63GB total
- **External Access**: Caddy reverse proxy + TLS on Vast.ai mapped ports (direct, no Cloudflare tunnels)
  - WS: ext 45087 → Caddy :6006 **[TLS]** → localhost:8765 → `wss://gpu.prosbookings.com:45087`
  - IPC: ext 45164 → Caddy :8080 → localhost:8766 → `http://70.29.210.33:45164`
  - Ollama: ext 45035 → Caddy :8384 → localhost:11434 → `http://70.29.210.33:45035/v1`
  - URLs configured in vapiclone Settings page (DB-backed). Static — only change on new Vast.ai instance.
  - **TLS is REQUIRED** on the WebSocket port — Twilio/SignalWire reject plain `ws://` for media streams.
  - Cert: Let's Encrypt via acme.sh (DNS challenge, manual TXT record in GoDaddy). Expires every 90 days.
  - Caddyfile: `/etc/Caddyfile`, certs at `/etc/certs/gpu.prosbookings.com.{crt,key}`
  - Domain: `gpu.prosbookings.com` → A record pointing to Vast.ai instance IP (GoDaddy DNS)

## Verified Production Settings (2026-03-23)

These exact settings produced perfect calls on both Twilio and SignalWire:
- **LLM**: Ollama llama3.2:3b-4k (NUM_PARALLEL=9, FLASH_ATTENTION=1, 4K context)
  - IMPORTANT: Use `-4k` model variants (created via Modelfile with `PARAMETER num_ctx 4096`)
  - Default 32K context wastes VRAM (52GB, spills to CPU). 4K is sufficient for phone calls.
  - Available models: `llama3.2:3b-4k`, `qwen2.5:3b-4k`, `phi4-mini-4k`
  - To create a 4K variant: `echo "FROM model:tag\nPARAMETER num_ctx 4096" | ollama create model:tag-4k`
- **STT**: Deepgram Flux (flux-general-en, v2 API, mulaw 8kHz direct)
- **TTS**: Kokoro-82M via FastAPI (voice: am_fenrir, port 8880)
- **AMD**: Audio-based voicemail detection (no Twilio/SignalWire AMD charges)
- **Tool mode**: trigger-phrases (5 phrases, 2 tools) — matched "transfer you" → ff_transfer in 3s
- **Post-call analysis**: Groq llama-3.1-8b-instant
- **Overflow LLM**: Groq (configured, activates when Ollama slots full)
- **Custom voices**: `am_adrian` (custom trained) at `/app/api/src/voices/v1_0/am_adrian.pt`

## First Message Optimization

Two optimizations ensure the first message plays as fast as possible:

1. **STT starts in background** (not blocking) — `const sttReady = this.stt.start()` (no `await`), so Deepgram WebSocket connects in parallel with first message delivery. `await sttReady` only happens at end of `start()`.

2. **Pre-synthesized first message** — When `/register-call` receives a config with a `firstMessage` and Kokoro TTS, it immediately synthesizes the audio in the background. By the time Twilio's WebSocket connects (~1-2s later), the PCM is already cached. `CallSession.start()` checks for `config.preSynthesizedFirstMessage` and streams it directly via `speakPreSynthesized()` instead of calling Kokoro live. Falls back to live synthesis if pre-synth hasn't completed.

## CRITICAL: SignalWire streamSid (DO NOT REVERT)

SignalWire sends `streamSid` inside `msg.start.streamSid`, NOT at the top level like Twilio.
```typescript
// src/index.ts line 169 — MUST have this fallback
streamSid = msg.streamSid || msg.start?.streamSid || null;
```
Without this, ALL outbound audio to SignalWire callers is silently dropped (the `if (streamSid)` check fails).

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

## Key Files

- `src/index.ts` — HTTP/WebSocket server, IPC endpoints, streamSid extraction (line 169), session lifecycle
- `src/voice-pipeline/call-session.ts` — Core session: STT ↔ LLM ↔ TTS orchestration, barge-in, voicemail detection, silence timer, cost tracking, overflow LLM routing, trigger-phrase matching
- `src/ollama-concurrency.ts` — Atomic counter for in-flight Ollama requests (overflow detection)
- `src/voice-pipeline/analysis-runner.ts` — Post-call analysis via configurable LLM (Groq default)
- `src/voice-pipeline/call-transfer.ts` — Provider-aware transfer: Twilio or SignalWire based on config.provider
- `src/providers/tts/kokoro.ts` — Kokoro-82M via FastAPI HTTP (port 8880)
- `src/providers/stt/deepgram.ts` — Deepgram Flux WebSocket client (production STT)
- `src/providers/llm/openai-compat.ts` — Single class for ALL LLM providers (Ollama, Groq, Cerebras, etc.)
- `src/providers/index.ts` — Provider factory (instantiates STT/LLM/TTS by name)
- `src/gpu-monitor.ts` — GPU/system monitoring (PSS-based RAM, not RSS)

## PM2 Services on GPU

| Name | Port | Notes |
|------|------|-------|
| voiceserver | 8765/8766 | Main voice pipeline |
| ollama | 11434 | LLM inference, started via start-ollama.sh |
| kokoro-fastapi | 8880 | Primary TTS (~1.8GB VRAM) |
| ~~tunnel-ipc / tunnel-ws / tunnel-ollama~~ | — | REMOVED 2026-03-25 — replaced by Caddy reverse proxy |

## IPC Endpoints

- `/health` — GPU stats, service health, sessions, disk, network
- `/sessions` — Active calls, Ollama slot utilization, overflow status
- `/services` — PM2 service statuses, start/stop
- `/amd-result/:callId` — AMD result from vapiclone (voicemail detection)
- `/register-call` — Register pending call config (60s TTL)
- `/settings` — GET/PUT server settings (.env persistence)
- `/tts/test` — TTS voice testing

## Voicemail Detection — WORKING (Audio-Based, Zero AMD Cost)

**No Twilio/SignalWire AMD used** — all detection is built-in audio analysis in `call-session.ts`:
- `VoicemailDetector` class: RMS energy analysis on PCM frames, speech ratio, continuous speech detection
- Long continuous speech (2s+) → machine detected → enters beep-wait mode → waits for silence → resolves as voicemail
- Thresholds (tuned 2026-03-24): speech ratio >65%, max continuous >30 frames (600ms), min 80 speech frames
- Short speech with pauses → human → delivers first message
- First message HELD while detection runs (~5s analysis window)
- All STT transcripts suppressed while detection pending AND after voicemail confirmed
- `getResult()` supports multiple listeners (callback array, not single overwrite)
- If `voicemailMessage` set → TTS speaks it → `waitForTTSFinish` waits actual audio duration (up to 60s) → hangs up
- If no `voicemailMessage` → hangs up immediately
- dialer4clone controls whether voicemailMessage is sent per-call (N/M frequency)
- System prompt includes voicemail awareness as LLM-level fallback detection

## Transfer Detection — Distinct EndedReasons (2026-03-24)

Stream stop no longer always reports `twilio-stream-stopped`. `SessionEntry` tracks `transferInitiated` flag:
- Transfer tool fires → flag set → stream stops → `endedReason: "call-forwarded"`
- No transfer → stream stops → `endedReason: "customer-ended-call"`
- dialer4clone uses `call-forwarded` to trigger auto phone lookup (no more false positives)

### Double-Transfer Guard (2026-03-25 LIVE1)
Transfer could fire twice if LLM both said "transfer" (trigger phrase) AND invoked the transfer tool — causing two outbound legs to the fallback number. Fixed with two guards:
1. `index.ts` transfer event handler: skips if `transferInitiated` already true
2. `call-session.ts`: both trigger-phrase and tool-call paths check `this.state === "transferring"` before emitting

## Post-Call Analysis

- Runs after call ends for answered calls with real conversation
- Default provider: Groq llama-3.1-8b-instant
- Supported: local (Ollama), deepseek, openrouter, cerebras, groq, deepinfra
- Cost tracked per-call in CostBreakdown.analysis

## Overflow LLM

When Ollama slots full, excess requests auto-route to cloud API.
- Counter in `src/ollama-concurrency.ts`, checked in `processWithLLM()`
- Settings: `OVERFLOW_LLM_PROVIDER` + `OVERFLOW_LLM_MODEL` env vars
- Cost tracked separately in `CostBreakdown.overflowLlm`

## Build & Deploy

```bash
ssh -p 45194 root@70.29.210.33
export PATH=/opt/nvm/versions/node/v24.12.0/bin:$PATH
cd /opt/voiceserverV2 && git pull && npx tsc && pm2 restart voiceserver --update-env
```

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

## Known Issues

- **GitHub Actions deploy is a no-op** — must deploy manually via SSH
- **Vast.ai instance IP change** — update DNS A record, re-issue cert, update vapiclone Settings URLs
- **Stale src/call-session.ts on GPU** — moved to src/voice-pipeline/call-session.ts, old file causes harmless tsc warnings
- **Build path on GPU**: `export PATH=/opt/nvm/versions/node/v24.12.0/bin:$PATH` required before npm/npx commands
- **NEVER use Cloudflare tunnels** — always use Caddy reverse proxy + TLS. Lower latency, static URLs, fewer processes. Twilio/SignalWire require `wss://` for media streams.
