/**
 * KVoiceWalk Manager — Manages the kvoicewalk voice-cloning training process.
 *
 * Encapsulates:
 *  - Spawning the kvoicewalk Python subprocess (via `uv run main.py`)
 *  - Piping stdout+stderr to /data/kvoicewalk/run.log
 *  - Parsing progress lines and score improvements from run.log
 *  - Completion flow: copy best .pt → Kokoro voices, write metadata, restart kokoro-fastapi
 *  - Cleanup: delete output dir (keep recordings)
 *  - Listing and deleting custom voices
 *
 * Only one kvoicewalk job may run at a time (GPU resource constraint).
 *
 * Paths (GPU server):
 *   kvoicewalk install:   /data/kvoicewalk/
 *   recordings:          /data/kvoicewalk/recordings/{name}/
 *   output:              /data/kvoicewalk/out/{name}_{suffix}_{timestamp}/
 *   run log:             /data/kvoicewalk/run.log
 *   Kokoro voices:       /app/api/src/voices/v1_0/
 */

import * as fs from "fs";
import * as fsp from "fs/promises";
import * as path from "path";
import { spawn, type ChildProcess } from "child_process";
import { exec } from "child_process";

// ---- Paths ----

const KVOICEWALK_DIR = "/data/kvoicewalk";
const RECORDINGS_DIR = path.join(KVOICEWALK_DIR, "recordings");
const OUTPUT_DIR = path.join(KVOICEWALK_DIR, "out");
const RUN_LOG = path.join(KVOICEWALK_DIR, "run.log");
const KOKORO_VOICES_DIR = "/app/api/src/voices/v1_0";

// ---- Stock voices — never delete ----

const STOCK_VOICES = new Set([
  "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
  "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
  "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
  "am_onyx", "am_puck", "am_santa",
  "bf_emma", "bf_isabella", "bm_george", "bm_lewis",
]);

// ---- Types ----

export interface KVoiceWalkScoreEntry {
  step: number;
  targetSim: number;
  featureSim: number;
  score: number;
}

export interface KVoiceWalkStatus {
  /** Whether the process is still running */
  running: boolean;
  /** Current step (from tqdm progress line) */
  step: number | null;
  /** Total steps */
  totalSteps: number | null;
  /** ETA string from tqdm (e.g. "1:52:09") */
  eta: string | null;
  /** Iterations per second */
  itsPerSec: number | null;
  /** Best score seen so far (highest score entry) */
  bestScore: number | null;
  /** All score-improvement entries parsed from log */
  scoreHistory: KVoiceWalkScoreEntry[];
  /** Voice name being trained */
  voiceName: string | null;
  /** Output subdirectory being written to (null until first .pt found) */
  outputSubdir: string | null;
}

export interface CustomVoice {
  name: string;
  /** Score embedded in the filename, if available */
  score: number | null;
  /** Created-at from metadata file, or null */
  createdAt: string | null;
  /** Path to .pt file */
  ptPath: string;
}

// ---- Module-level job state ----

interface JobState {
  process: ChildProcess;
  voiceName: string;
  recordingDir: string;
  outputSubdir: string | null;
  logStream: fs.WriteStream;
  completionTriggered: boolean;
}

let currentJob: JobState | null = null;

// ---- Helpers ----

function promisifyExec(cmd: string): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    exec(cmd, { env: { ...process.env, PATH: `/usr/local/bin:${process.env.PATH || ""}` } }, (err, stdout, stderr) => {
      if (err) reject(err);
      else resolve({ stdout, stderr });
    });
  });
}

/**
 * Find the latest timestamped output subdirectory for a given voice name.
 * kvoicewalk creates: /data/kvoicewalk/out/{name}_{suffix}_{YYYYMMDD_HHMMSS}/
 * We pick the most-recently-modified directory whose name starts with the voice name.
 */
async function findOutputSubdir(voiceName: string): Promise<string | null> {
  try {
    const entries = await fsp.readdir(OUTPUT_DIR, { withFileTypes: true });
    const candidates = entries
      .filter(e => e.isDirectory() && e.name.startsWith(voiceName))
      .map(e => path.join(OUTPUT_DIR, e.name));

    if (candidates.length === 0) return null;

    // Pick the most recently modified
    const stats = await Promise.all(
      candidates.map(async p => ({ p, mtime: (await fsp.stat(p)).mtimeMs }))
    );
    stats.sort((a, b) => b.mtime - a.mtime);
    return stats[0].p;
  } catch {
    return null;
  }
}

/**
 * Parse run.log and return the current KVoiceWalkStatus.
 * We read only the last ~200 lines to stay fast on large logs.
 */
function parseRunLog(logPath: string, voiceName: string | null, outputSubdir: string | null): KVoiceWalkStatus {
  const status: KVoiceWalkStatus = {
    running: currentJob !== null,
    step: null,
    totalSteps: null,
    eta: null,
    itsPerSec: null,
    bestScore: null,
    scoreHistory: [],
    voiceName,
    outputSubdir,
  };

  let raw: string;
  try {
    raw = fs.readFileSync(logPath, "utf-8");
  } catch {
    return status;
  }

  // Work with last ~500 lines to avoid parsing megabytes
  const lines = raw.split("\n");
  const tail = lines.slice(-500);

  // Regex: tqdm progress line
  // Example: " 32%|███▏      | 3200/10000 [52:50<1:52:09,  1.01it/s]"
  const progressRe = /(\d+)%\|[^|]*\|\s+(\d+)\/(\d+)\s+\[[\d:]+<([\d:]+),\s*([\d.]+)it\/s\]/;

  // Regex: score line
  // Example: "Step:1413 Target Sim:0.769 Self Sim:0.955 Feature Sim:0.302 Score:82.38"
  const scoreRe = /Step:(\d+)\s+Target Sim:([\d.]+)\s+Self Sim:[\d.]+\s+Feature Sim:([\d.]+)\s+Score:([\d.]+)/;

  for (const line of tail) {
    const pm = progressRe.exec(line);
    if (pm) {
      status.step = parseInt(pm[2], 10);
      status.totalSteps = parseInt(pm[3], 10);
      status.eta = pm[4];
      status.itsPerSec = parseFloat(pm[5]);
    }

    const sm = scoreRe.exec(line);
    if (sm) {
      const entry: KVoiceWalkScoreEntry = {
        step: parseInt(sm[1], 10),
        targetSim: parseFloat(sm[2]),
        featureSim: parseFloat(sm[3]),
        score: parseFloat(sm[4]),
      };
      status.scoreHistory.push(entry);
    }
  }

  if (status.scoreHistory.length > 0) {
    status.bestScore = Math.max(...status.scoreHistory.map(e => e.score));
  }

  return status;
}

/**
 * Extract the score from a kvoicewalk output filename.
 * Filename pattern: {name}_{step}_{score}_{simhash}.pt
 * e.g. my_new_voice_1413_82.38_0.77_crystal.pt
 */
function scoreFromFilename(filename: string): number | null {
  // Strip extension
  const base = filename.replace(/\.(pt|wav)$/, "");
  // Parts separated by underscore — score is typically 4th token (index 3) for simple names,
  // but voice names may contain underscores. kvoicewalk always appends: _step_score_sim_hash
  // We look for the last sequence of 4 tokens that match: integer, float, float, word
  const parts = base.split("_");
  for (let i = parts.length - 4; i >= 0; i--) {
    const step = parseInt(parts[i], 10);
    const score = parseFloat(parts[i + 1]);
    const sim = parseFloat(parts[i + 2]);
    if (!isNaN(step) && !isNaN(score) && !isNaN(sim) && parts[i + 3]) {
      return score;
    }
  }
  return null;
}

/**
 * Find the .pt file with the highest score in an output directory.
 */
async function findBestPt(subdir: string): Promise<{ ptPath: string; score: number | null; wavPath: string | null } | null> {
  let entries: string[];
  try {
    entries = await fsp.readdir(subdir);
  } catch {
    return null;
  }

  const ptFiles = entries.filter(e => e.endsWith(".pt"));
  if (ptFiles.length === 0) return null;

  let best: { file: string; score: number } | null = null;

  for (const f of ptFiles) {
    const score = scoreFromFilename(f);
    if (score !== null && (best === null || score > best.score)) {
      best = { file: f, score };
    }
  }

  // Fallback: just pick last alphabetically if no parseable score
  const chosenFile = best ? best.file : ptFiles[ptFiles.length - 1];
  const ptPath = path.join(subdir, chosenFile);

  // Try to find a matching .wav (same base name)
  const wavBase = chosenFile.replace(/\.pt$/, ".wav");
  const wavPath = entries.includes(wavBase) ? path.join(subdir, wavBase) : null;

  return { ptPath, score: best?.score ?? null, wavPath };
}

/**
 * Completion flow:
 * 1. Kill python process if still running
 * 2. Find best .pt in output dir
 * 3. Copy to Kokoro voices dir
 * 4. Write metadata JSON
 * 5. Restart kokoro-fastapi via pm2
 * 6. Delete output subdirectory
 * 7. Close log stream
 */
async function runCompletionFlow(job: JobState): Promise<void> {
  if (job.completionTriggered) return;
  job.completionTriggered = true;

  const tag = `[kvoicewalk:${job.voiceName}]`;

  // 1. Kill process if still alive
  if (job.process.exitCode === null) {
    try {
      job.process.kill("SIGTERM");
      console.log(`${tag} Sent SIGTERM to kvoicewalk process`);
    } catch (e) {
      console.warn(`${tag} Could not kill process:`, e);
    }
  }

  // 2. Find output subdir (may have been located during run)
  let subdir = job.outputSubdir;
  if (!subdir) {
    subdir = await findOutputSubdir(job.voiceName);
    job.outputSubdir = subdir;
  }

  if (!subdir) {
    console.warn(`${tag} No output subdirectory found — cannot install voice`);
    job.logStream.close();
    currentJob = null;
    return;
  }

  // 3. Find best .pt
  const best = await findBestPt(subdir);
  if (!best) {
    console.warn(`${tag} No .pt files in ${subdir} — cannot install voice`);
    job.logStream.close();
    currentJob = null;
    return;
  }

  const destPt = path.join(KOKORO_VOICES_DIR, `${job.voiceName}.pt`);
  const destMeta = path.join(KOKORO_VOICES_DIR, `${job.voiceName}.meta.json`);

  // 4. Copy .pt to Kokoro voices
  try {
    await fsp.mkdir(KOKORO_VOICES_DIR, { recursive: true });
    await fsp.copyFile(best.ptPath, destPt);
    console.log(`${tag} Installed voice → ${destPt} (score: ${best.score ?? "unknown"})`);
  } catch (e) {
    console.error(`${tag} Failed to copy .pt to Kokoro voices:`, e);
    job.logStream.close();
    currentJob = null;
    return;
  }

  // 5. Write metadata
  try {
    const meta = {
      name: job.voiceName,
      score: best.score,
      createdAt: new Date().toISOString(),
    };
    await fsp.writeFile(destMeta, JSON.stringify(meta, null, 2), "utf-8");
  } catch (e) {
    console.warn(`${tag} Could not write metadata:`, e);
  }

  // 6. Restart kokoro-fastapi
  try {
    await promisifyExec("pm2 restart kokoro-fastapi");
    console.log(`${tag} kokoro-fastapi restarted`);
  } catch (e) {
    console.warn(`${tag} pm2 restart kokoro-fastapi failed:`, e);
  }

  // 7. Delete output subdirectory (keep recordings)
  try {
    await fsp.rm(subdir, { recursive: true, force: true });
    console.log(`${tag} Deleted output dir: ${subdir}`);
  } catch (e) {
    console.warn(`${tag} Could not delete output dir ${subdir}:`, e);
  }

  job.logStream.close();
  currentJob = null;
  console.log(`${tag} Completion flow done.`);
}

// ---- Public API ----

/**
 * Returns true if a kvoicewalk job is currently running.
 */
export function isRunning(): boolean {
  return currentJob !== null;
}

/**
 * Save audio + transcript to disk, then spawn the kvoicewalk process.
 *
 * @param audioBuffer  Raw audio bytes (any format kvoicewalk accepts, typically WAV/mp3)
 * @param transcript   Text transcript of the audio
 * @param voiceName    Name for the trained voice (used as output filename)
 */
export async function startWalk(
  audioBuffer: Buffer,
  transcript: string,
  voiceName: string
): Promise<{ ok: boolean; error?: string }> {
  if (currentJob) {
    return { ok: false, error: "A kvoicewalk job is already running" };
  }

  // Sanitize name: only alphanumerics and underscores
  const safeName = voiceName.replace(/[^a-zA-Z0-9_]/g, "_");
  if (!safeName) {
    return { ok: false, error: "Invalid voice name" };
  }

  const recordingDir = path.join(RECORDINGS_DIR, safeName);
  const audioPath = path.join(recordingDir, `${safeName}.wav`);
  const transcriptPath = path.join(recordingDir, `${safeName}.txt`);

  // Save audio and transcript
  try {
    await fsp.mkdir(recordingDir, { recursive: true });
    await fsp.writeFile(audioPath, audioBuffer);
    await fsp.writeFile(transcriptPath, transcript, "utf-8");
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return { ok: false, error: `Failed to save recording: ${msg}` };
  }

  // Truncate run.log for this new run
  try {
    await fsp.mkdir(KVOICEWALK_DIR, { recursive: true });
    await fsp.writeFile(RUN_LOG, `=== kvoicewalk started: ${safeName} @ ${new Date().toISOString()} ===\n`, "utf-8");
  } catch (e) {
    console.warn("[kvoicewalk] Could not truncate run.log:", e);
  }

  // Open log stream (append mode — we already wrote the header above)
  const logStream = fs.createWriteStream(RUN_LOG, { flags: "a" });

  // Spawn kvoicewalk
  const args = [
    "run", "main.py",
    "--target_audio", audioPath,
    "--target_text", transcriptPath,
    "--interpolate_start",
    "--step_limit", "10000",
    "--output_name", safeName,
  ];

  let proc: ChildProcess;
  try {
    proc = spawn("uv", args, {
      cwd: KVOICEWALK_DIR,
      env: {
        ...process.env,
        PATH: `/usr/local/bin:${process.env.PATH || ""}`,
      },
      stdio: ["ignore", "pipe", "pipe"],
    });
  } catch (e) {
    logStream.close();
    const msg = e instanceof Error ? e.message : String(e);
    return { ok: false, error: `Failed to spawn kvoicewalk: ${msg}` };
  }

  const job: JobState = {
    process: proc,
    voiceName: safeName,
    recordingDir,
    outputSubdir: null,
    logStream,
    completionTriggered: false,
  };
  currentJob = job;

  // Pipe stdout and stderr to log file
  if (proc.stdout) proc.stdout.pipe(logStream, { end: false });
  if (proc.stderr) proc.stderr.pipe(logStream, { end: false });

  // Periodically try to locate the output subdir (it's created after training begins)
  const subdirPoller = setInterval(async () => {
    if (!currentJob || currentJob !== job) {
      clearInterval(subdirPoller);
      return;
    }
    if (!job.outputSubdir) {
      const found = await findOutputSubdir(safeName);
      if (found) {
        job.outputSubdir = found;
        console.log(`[kvoicewalk:${safeName}] Output subdir detected: ${found}`);
        clearInterval(subdirPoller);
      }
    } else {
      clearInterval(subdirPoller);
    }
  }, 10_000);

  // On process exit, trigger completion flow automatically
  proc.on("exit", async (code, signal) => {
    clearInterval(subdirPoller);
    console.log(`[kvoicewalk:${safeName}] Process exited — code=${code} signal=${signal}`);
    if (currentJob === job) {
      await runCompletionFlow(job);
    }
  });

  proc.on("error", async (err) => {
    clearInterval(subdirPoller);
    console.error(`[kvoicewalk:${safeName}] Process error:`, err);
    logStream.write(`\n[ERROR] ${err.message}\n`);
    if (currentJob === job) {
      await runCompletionFlow(job);
    }
  });

  console.log(`[kvoicewalk:${safeName}] Spawned (pid=${proc.pid})`);
  return { ok: true };
}

/**
 * Parse run.log and return current training status.
 * Returns null if no job has ever run (log does not exist).
 */
export function getStatus(): KVoiceWalkStatus | null {
  if (!fs.existsSync(RUN_LOG)) return null;

  const voiceName = currentJob?.voiceName ?? null;
  const outputSubdir = currentJob?.outputSubdir ?? null;
  return parseRunLog(RUN_LOG, voiceName, outputSubdir);
}

/**
 * Find the .wav sample with the highest score in the current job's output directory.
 * Returns null if no job is running or no samples exist yet.
 */
export async function getBestSample(): Promise<{ audio: Buffer; filename: string } | null> {
  const subdir = currentJob?.outputSubdir ?? null;
  if (!subdir) {
    // Try to find from voiceName
    if (!currentJob) return null;
    const found = await findOutputSubdir(currentJob.voiceName);
    if (!found) return null;
    currentJob.outputSubdir = found;
  }

  const targetSubdir = currentJob?.outputSubdir ?? null;
  if (!targetSubdir) return null;

  const best = await findBestPt(targetSubdir);
  if (!best || !best.wavPath) return null;

  try {
    const audio = await fsp.readFile(best.wavPath);
    return { audio, filename: path.basename(best.wavPath) };
  } catch {
    return null;
  }
}

/**
 * Kill the running kvoicewalk process and trigger the completion flow
 * (copy best .pt, restart kokoro-fastapi, clean output dir).
 */
export async function stopWalk(): Promise<{ ok: boolean; error?: string }> {
  if (!currentJob) {
    return { ok: false, error: "No kvoicewalk job is running" };
  }
  try {
    await runCompletionFlow(currentJob);
    return { ok: true };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return { ok: false, error: msg };
  }
}

/**
 * List all custom (non-stock) Kokoro .pt voices.
 */
export async function listCustomVoices(): Promise<CustomVoice[]> {
  let entries: string[];
  try {
    entries = await fsp.readdir(KOKORO_VOICES_DIR);
  } catch {
    return [];
  }

  const ptFiles = entries.filter(e => e.endsWith(".pt"));
  const voices: CustomVoice[] = [];

  for (const f of ptFiles) {
    const name = f.replace(/\.pt$/, "");
    if (STOCK_VOICES.has(name)) continue;

    const ptPath = path.join(KOKORO_VOICES_DIR, f);
    const metaPath = path.join(KOKORO_VOICES_DIR, `${name}.meta.json`);

    let score: number | null = null;
    let createdAt: string | null = null;

    try {
      const raw = await fsp.readFile(metaPath, "utf-8");
      const meta = JSON.parse(raw);
      score = typeof meta.score === "number" ? meta.score : null;
      createdAt = typeof meta.createdAt === "string" ? meta.createdAt : null;
    } catch {
      // No metadata — try to extract score from filename
      score = scoreFromFilename(f);
    }

    voices.push({ name, score, createdAt, ptPath });
  }

  return voices;
}

/**
 * Delete a custom Kokoro voice (.pt + .meta.json) and restart kokoro-fastapi.
 * Refuses to delete stock voices.
 */
export async function deleteVoice(name: string): Promise<{ ok: boolean; error?: string }> {
  if (STOCK_VOICES.has(name)) {
    return { ok: false, error: `Cannot delete stock voice: ${name}` };
  }

  const ptPath = path.join(KOKORO_VOICES_DIR, `${name}.pt`);
  const metaPath = path.join(KOKORO_VOICES_DIR, `${name}.meta.json`);

  // Check the .pt exists
  try {
    await fsp.access(ptPath);
  } catch {
    return { ok: false, error: `Voice not found: ${name}` };
  }

  // Delete .pt
  try {
    await fsp.unlink(ptPath);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return { ok: false, error: `Failed to delete ${ptPath}: ${msg}` };
  }

  // Delete .meta.json (best-effort)
  try {
    await fsp.unlink(metaPath);
  } catch {
    // Not fatal — meta file may not exist
  }

  // Restart kokoro-fastapi
  try {
    await promisifyExec("pm2 restart kokoro-fastapi");
    console.log(`[kvoicewalk] Deleted voice "${name}", restarted kokoro-fastapi`);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return { ok: false, error: `Voice deleted but pm2 restart failed: ${msg}` };
  }

  return { ok: true };
}
