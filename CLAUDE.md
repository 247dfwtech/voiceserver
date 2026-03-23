# VoiceServer V2

Node.js/TypeScript voice pipeline server running on Vast.ai GPU (RTX 4090).
Handles real-time STT → LLM → TTS for concurrent AI phone calls.

## Architecture

- **Runtime**: Node.js + TypeScript, managed by PM2 on GPU server
- **Deploy**: `scp` files to GPU, `npx tsc` to compile, `pm2 restart voiceserver`
- **GPU**: `ssh -o StrictHostKeyChecking=no -p 45194 root@66.245.227.160` (IP is dynamic — check Vast.ai console)
- **Ports**: 8765 (WebSocket for Twilio/SignalWire media streams), 8766 (IPC HTTP)
- **RAM**: 63GB total — Ollama NUM_PARALLEL=10 uses ~38GB KV cache. Max safe: 12 for llama3.2:3b.
- **Tunnels**: Cloudflare quick tunnels (random URLs on restart). IPC/WS/Ollama URLs configured in vapiclone Settings page (DB-backed).

## Key Files

- `src/voice-pipeline/call-session.ts` — Core call session: STT ↔ LLM ↔ TTS orchestration, barge-in, voicemail detection, silence timer, cost tracking. CallSessionConfig includes `provider` ("twilio"|"signalwire") field.
- `src/voice-pipeline/analysis-runner.ts` — Post-call analysis: summary + success evaluation via configurable LLM provider. Only runs on answered calls with real conversation.
- `src/providers/tts/` — TTS providers: `kokoro.ts` (HTTP to port 8880), `qwen3.ts` (HTTP to port 8881, voice cloning), `piper.ts` (CPU fallback)
- `src/providers/stt/` — STT providers: `deepgram.ts` (cloud, production), `vosk.ts` (CPU fallback), `granite.ts`
- `src/providers/llm/` — LLM providers: `ollama.ts` (local GPU), `openai.ts`, `deepseek.ts`
- `src/providers/index.ts` — Provider factory (instantiates STT/LLM/TTS by name)
- `src/voice-pipeline/call-transfer.ts` — Provider-aware call transfer: `getProviderClient()` selects Twilio or SignalWire client based on `config.provider`. Supports Dial, Conference, and DTMF.
- `src/index.ts` — HTTP/WebSocket server, IPC endpoints (/register-call, /settings, /health, /tts/test, /qwen3/*, /services, /amd-result/:callId). Passes `config.provider` to transfer handlers.
- `src/gpu-monitor.ts` — GPU/system monitoring: per-process resources (PSS-based RAM, not RSS to avoid overcounting), disk, network, history snapshots
- `src/model-manager.ts` — Tracks active models and TTS services

## PM2 Services on GPU

- **voiceserver** — Node.js voice pipeline
- **ollama** — LLM inference (port 11434), started via `start-ollama.sh` which reads `.env`
- **kokoro-fastapi** — Primary TTS (port 8880), ~1.8GB VRAM
- **qwen3-tts** — Voice cloning TTS (port 8881), ~1.5GB VRAM
- **tunnel-ipc** / **tunnel-ws** / **tunnel-ollama** — Cloudflare quick tunnels
- All managed via `ecosystem.config.cjs`, start/stop controllable from vapiclone Settings page

## IPC Endpoints

- `/health` — GPU stats, per-process resources, service health, sessions, disk, network
- `/sessions` — Active call list with states
- `/services` — PM2 service statuses (GET), start/stop services (POST /services/:name/start|stop)
- `/amd-result/:callId` — Receives AMD result forwarded from vapiclone, resolves voicemail detection
- `/register-call` — Register pending call config
- `/settings` — GET/PUT server settings (.env). Includes SW_PROJECT_ID, SW_AUTH_TOKEN, SW_SPACE_URL for SignalWire.
- `/metrics/history` — 3-day historical snapshots

## Voicemail Detection (IN PROGRESS — Not fully working yet)

### Current Implementation
- Uses Twilio AMD `DetectMessageEnd` mode (switched from `Enable` — see history below)
- `asyncAmd: true` so calls connect immediately while AMD runs in background
- AMD result forwarded: Twilio → vapiclone `/api/webhooks/twilio/status` → voiceserver `/amd-result/:callId`
- VoicemailDetector class in call-session.ts with audio-based fallback detection
- First message HELD while detection pending (AI stays silent)
- ALL STT transcripts suppressed while detection pending (prevents LLM from acting on greeting audio)
- On voicemail: cancels speech, 500ms pause, speaks voicemailMessage, ends call

### What We've Tried (History)
1. **Original**: `machineDetection: "Enable"` → AMD fires `machine_start` immediately → AI spoke over greeting. FAILED.
2. **Added beep-wait**: On `machine_start`, entered beep-wait mode listening for 800ms silence. But AMD result wasn't reaching voiceserver (status webhook didn't forward it). FAILED.
3. **Added AMD forwarding**: vapiclone status webhook now forwards `AnsweredBy` to voiceserver `/amd-result/:callId`. But first message still played over greeting. FAILED.
4. **Held first message**: Added logic to hold first message until detection resolves, deliver on "human", skip on "voicemail". But `return` statement skipped silence timer setup → call hung in silence forever. FAILED.
5. **Fixed initialization**: Removed `return`, kept silence timer running. But STT transcribed voicemail greeting → LLM processed it → triggered ff_transfer tool. FAILED.
6. **Switched to DetectMessageEnd**: Changed from `Enable` to `DetectMessageEnd` so Twilio waits for greeting+beep before callback. Added `asyncAmd: true`. Reduced delay to 500ms. CURRENT.
7. **Suppressed transcripts**: Added guards in handleTranscript() and handleUtteranceEnd() to ignore ALL input while voicemail detection is pending. CURRENT.

### Known Issues
- AMD callback from `DetectMessageEnd` may not be arriving (needs verification with next test call)
- `forceAmdResult("machine_start")` enters beep-wait mode; `machine_end_beep` resolves immediately
- Audio-based fallback: detects 3s continuous speech → enters beep-wait → resolves after 800ms silence

## Post-Call Analysis

- Runs after call ends in `notifyCallEnded()` in index.ts
- Skipped for unanswered calls: voicemail, no-answer, busy, failed, machine-detected, stt-failure
- Supported providers: local (Ollama), deepseek, openrouter, cerebras, groq, deepinfra
- Cost tracked per-call via token usage, included in CostBreakdown.analysis

## Build

```bash
# On GPU server
export PATH=/opt/nvm/versions/node/v24.12.0/bin:$PATH
cd /opt/voiceserverV2
npx tsc --noEmit  # type-check
npx tsc           # compile (ignore src/call-session.ts errors — stale file)
pm2 restart voiceserver
```

## Known Issues

- Stale `src/call-session.ts` file on GPU causes tsc errors (moved to `src/voice-pipeline/call-session.ts`). Doesn't block compilation.
- Qwen3-TTS restarts frequently in pm2 — model loading issues with Base model
- Vast.ai instance IP is dynamic — changes on reboot. Tunnel URLs also change. Must update in vapiclone Settings.
- GPU RAM monitoring: uses PSS not RSS to avoid overcounting shared memory. Ollama runner child process (~38GB) is the main consumer.
