# VoiceServer V2

Node.js/TypeScript voice pipeline server running on Vast.ai GPU (RTX 4090).
Handles real-time STT → LLM → TTS for concurrent AI phone calls.

## Architecture

- **Runtime**: Node.js + TypeScript, managed by PM2 on GPU server
- **Deploy**: `scp` files to GPU, `npx tsc` to compile, `pm2 restart voiceserver`
- **GPU**: `ssh -p 45194 root@70.29.210.33`, code at `/opt/voiceserverV2/`
- **Ports**: 8765 (IPC HTTP), 8766 (WebSocket for Twilio media streams)

## Key Directories

- `src/voice-pipeline/call-session.ts` — Core call session: STT ↔ LLM ↔ TTS orchestration, barge-in, voicemail detection, silence timer, cost tracking
- `src/providers/tts/` — TTS providers: `kokoro.ts` (HTTP to port 8880), `qwen3.ts` (HTTP to port 8881, voice cloning), `piper.ts` (CPU fallback)
- `src/providers/stt/` — STT providers: `deepgram.ts` (cloud), `vosk.ts` (CPU fallback), `granite.ts`
- `src/providers/llm/` — LLM providers: `ollama.ts` (local GPU), `openai.ts`, `deepseek.ts`
- `src/providers/index.ts` — Provider factory (instantiates STT/LLM/TTS by name)
- `src/index.ts` — HTTP/WebSocket server, IPC endpoints (/register-call, /settings, /health, /tts/test, /qwen3/*)
- `src/model-manager.ts` — Tracks active models and TTS services

## External Services on Same GPU

- **Ollama** (port 11434) — LLM inference, config via `/opt/voiceserverV2/start-ollama.sh` which reads `.env`
- **Kokoro-FastAPI** (port 8880) — Primary TTS, `/opt/start-kokoro-fastapi.sh`
- **Qwen3-TTS** (port 8881) — Voice cloning TTS, `/opt/start-qwen3-tts.sh`, model `Qwen/Qwen3-TTS-12Hz-0.6B-Base`

## Settings

Settings come from two sources:
1. `.env` file at `/opt/voiceserverV2/.env` (persisted across restarts)
2. `PUT /settings` from vapiclone dashboard (updates `.env` + `process.env` in real-time)

Ollama-specific settings (NUM_PARALLEL, FLASH_ATTENTION) trigger `pm2 restart ollama` when changed.

## Per-Call Config

Each call receives its full assistant config via `POST /register-call` from vapiclone. The voiceserver is stateless — no assistant config is cached. Settings include: model, voice, transcriber, behavior (maxDuration, silenceTimeout, endCallPhrases), speaking plans, voicemail detection, tools, analysis config.

## Build

```bash
# On GPU server
export PATH=/opt/nvm/versions/node/v24.12.0/bin:$PATH
cd /opt/voiceserverV2
npx tsc --noEmit  # type-check
npx tsc           # compile
pm2 restart voiceserver
```
