/**
 * GPU Monitor — Collects GPU, CPU, and memory metrics with a 3-day ring buffer.
 *
 * nvidia-smi provides GPU utilization, VRAM, temperature, and power.
 * os module provides CPU utilization and system RAM.
 * Metrics are collected every 30 seconds and stored in a circular buffer.
 *
 * 8640 entries = 30s intervals x 3 days ≈ 1.7MB memory footprint.
 */

import { execFile } from "child_process";
import * as os from "os";

// ---- Types ----

export interface GpuInfo {
  util: number;        // GPU utilization %
  vramUsed: number;    // MB
  vramTotal: number;   // MB
  temp: number;        // Celsius
  powerDraw: number;   // Watts
  powerLimit: number;  // Watts
  name: string;        // e.g. "NVIDIA GeForce RTX 4090"
}

export interface ProcessResourceInfo {
  name: string;        // e.g. "ollama", "kokoro-fastapi", "qwen3-tts", "voiceserver"
  pid: number;
  cpuPercent: number;  // CPU %
  ramMB: number;       // RSS in MB
  vramMB: number;      // GPU VRAM in MB (0 if not using GPU)
}

export interface DiskInfo {
  mountpoint: string;
  totalGB: number;
  usedGB: number;
  availGB: number;
  usedPercent: number;
}

export interface NetworkInfo {
  interface: string;
  rxBytesPerSec: number;  // bytes/sec received
  txBytesPerSec: number;  // bytes/sec transmitted
  rxMbps: number;          // Mbps received
  txMbps: number;          // Mbps transmitted
}

export interface GpuSnapshot {
  ts: number;          // unix ms
  gpu: GpuInfo | null; // null if nvidia-smi unavailable
  cpu: { util: number };
  mem: { used: number; total: number }; // system RAM in MB
  node: { rss: number; heap: number };  // Node.js process in MB
  sessions: number;
  processes?: ProcessResourceInfo[];    // per-process breakdown
  disk?: DiskInfo;                      // disk usage
  network?: NetworkInfo;                // network bandwidth
}

// ---- Process name mapping ----

const TRACKED_PROCESSES: Record<string, string> = {
  "ollama_llama_se": "ollama",    // ollama_llama_server
  "ollama": "ollama",
  "python3": "python",            // could be kokoro or qwen3
  "python": "python",
  "node": "voiceserver",
  "uvicorn": "python",
};

// Track parent→child relationships for qwen3-tts torch workers
let _qwen3MainPid = 0;

function classifyProcess(comm: string, pid: number, cmdline: string): string | null {
  if (cmdline.includes("kokoro") || cmdline.includes("8880")) return "kokoro-fastapi";
  if (cmdline.includes("qwen3") || cmdline.includes("8881")) return "qwen3-tts";
  // qwen3-tts runs as "python3 -m api.main" without "qwen3" in cmdline
  // Its torch compile workers reference parent PID
  if (cmdline.includes("api.main") && !cmdline.includes("8880")) {
    _qwen3MainPid = pid;
    return "qwen3-tts";
  }
  if (cmdline.includes("compile_worker") && _qwen3MainPid > 0 && cmdline.includes(`parent=${_qwen3MainPid}`)) return "qwen3-tts";
  if (cmdline.includes("ollama")) return "ollama";
  if (cmdline.includes("voiceserverV2") || cmdline.includes("dist/index")) return "voiceserver";
  const base = comm.replace(/\s+/g, "");
  if (TRACKED_PROCESSES[base]) return TRACKED_PROCESSES[base];
  return null;
}

// ---- Per-process resource tracking ----

function exec(cmd: string, args: string[], timeout = 5000): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(cmd, args, { timeout }, (err, stdout) => {
      if (err) reject(err);
      else resolve(stdout);
    });
  });
}

async function queryProcessResources(): Promise<ProcessResourceInfo[]> {
  try {
    // Get per-process GPU VRAM usage — try PID matching first, then fall back to name matching
    const vramByPid = new Map<number, number>();
    const vramByName = new Map<string, number>(); // fallback when PIDs are in different namespace
    let totalGpuVram = 0;
    try {
      const vramOut = await exec("nvidia-smi", [
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
      ]);
      for (const line of vramOut.trim().split("\n")) {
        if (!line.trim()) continue;
        const parts = line.split(",").map((s) => s.trim());
        if (parts.length < 3) continue;
        const pid = parseInt(parts[0]);
        const procName = parts[1].toLowerCase();
        const mem = parseFloat(parts[2]) || 0;
        totalGpuVram += mem;
        if (pid) vramByPid.set(pid, (vramByPid.get(pid) || 0) + mem);
        // Classify by nvidia-smi process name for cross-namespace matching
        if (procName.includes("ollama") || procName.includes("llama")) {
          vramByName.set("ollama", (vramByName.get("ollama") || 0) + mem);
        } else if (procName.includes("python") || procName.includes("uvicorn")) {
          // Python processes — accumulate (we'll distribute later)
          vramByName.set("python-gpu", (vramByName.get("python-gpu") || 0) + mem);
        }
      }
    } catch {
      // nvidia-smi not available or no GPU processes
    }

    // Get CPU/RAM for all processes, with full command line for classification
    // Use PSS (Proportional Set Size) for accurate RAM — falls back to RSS
    const psOut = await exec("ps", ["-eo", "pid,comm,%cpu,rss,args", "--no-headers"]);
    const grouped = new Map<string, ProcessResourceInfo>();
    // Track main process PID per service (lowest PID = parent)
    const mainPids = new Map<string, number>();

    for (const line of psOut.trim().split("\n")) {
      if (!line.trim()) continue;
      const match = line.trim().match(/^(\d+)\s+(\S+)\s+([\d.]+)\s+(\d+)\s+(.*)$/);
      if (!match) continue;
      const pid = parseInt(match[1]);
      const comm = match[2];
      const cpu = parseFloat(match[3]) || 0;
      const rssKB = parseInt(match[4]) || 0;
      const cmdline = match[5];

      // Skip compile workers — they share memory with parent and inflate RSS
      if (cmdline.includes("compile_worker") || cmdline.includes("torch/_inductor")) continue;

      const name = classifyProcess(comm, pid, cmdline);
      if (!name) continue;

      const existing = grouped.get(name);
      const vram = vramByPid.get(pid) || 0;
      if (existing) {
        existing.cpuPercent += cpu;
        // Only use the MAIN process RSS (largest), don't sum children
        const thisMB = Math.round(rssKB / 1024);
        if (thisMB > existing.ramMB) {
          existing.ramMB = thisMB;
          existing.pid = pid;
        }
        existing.vramMB += vram;
      } else {
        grouped.set(name, {
          name,
          pid,
          cpuPercent: Math.round(cpu * 10) / 10,
          ramMB: Math.round(rssKB / 1024),
          vramMB: vram,
        });
      }
    }

    // If PID-based VRAM matching didn't work (cross-namespace), use name-based matching
    const anyVramMatched = Array.from(grouped.values()).some((p) => p.vramMB > 0);
    if (!anyVramMatched && totalGpuVram > 0) {
      // Assign ollama VRAM from name match
      const ollamaProc = grouped.get("ollama");
      if (ollamaProc && vramByName.has("ollama")) {
        ollamaProc.vramMB = vramByName.get("ollama")!;
      }
      // Distribute python GPU VRAM between kokoro and qwen3 based on RAM ratio
      const pythonVram = vramByName.get("python-gpu") || 0;
      if (pythonVram > 0) {
        const kokoro = grouped.get("kokoro-fastapi");
        const qwen3 = grouped.get("qwen3-tts");
        if (kokoro && qwen3) {
          const totalRam = kokoro.ramMB + qwen3.ramMB;
          if (totalRam > 0) {
            kokoro.vramMB = Math.round(pythonVram * (kokoro.ramMB / totalRam));
            qwen3.vramMB = pythonVram - kokoro.vramMB;
          }
        } else if (kokoro) {
          kokoro.vramMB = pythonVram;
        } else if (qwen3) {
          qwen3.vramMB = pythonVram;
        }
      }
      // If no name matching worked either, show total on ollama (primary GPU user)
      const anyNowMatched = Array.from(grouped.values()).some((p) => p.vramMB > 0);
      if (!anyNowMatched && ollamaProc) {
        ollamaProc.vramMB = totalGpuVram;
      }
    }

    // Round aggregated values
    for (const info of grouped.values()) {
      info.cpuPercent = Math.round(info.cpuPercent * 10) / 10;
    }

    return Array.from(grouped.values());
  } catch {
    return [];
  }
}

// ---- Disk usage ----

async function queryDiskUsage(): Promise<DiskInfo | null> {
  try {
    const out = await exec("df", ["-BG", "/"]);
    const lines = out.trim().split("\n");
    if (lines.length < 2) return null;
    // Filesystem 1G-blocks Used Available Use% Mounted on
    const parts = lines[1].trim().split(/\s+/);
    if (parts.length < 6) return null;
    const totalGB = parseFloat(parts[1]) || 0;
    const usedGB = parseFloat(parts[2]) || 0;
    const availGB = parseFloat(parts[3]) || 0;
    const usedPercent = parseFloat(parts[4]) || 0;
    return { mountpoint: parts[5], totalGB, usedGB, availGB, usedPercent };
  } catch {
    return null;
  }
}

// ---- Network bandwidth ----

let _prevNetRx = 0;
let _prevNetTx = 0;
let _prevNetTs = 0;

async function queryNetworkBandwidth(): Promise<NetworkInfo | null> {
  try {
    const { readFile } = await import("fs/promises");
    const content = await readFile("/proc/net/dev", "utf-8");
    const lines = content.trim().split("\n");
    let totalRx = 0;
    let totalTx = 0;
    let iface = "all";

    for (const line of lines.slice(2)) { // skip header lines
      const match = line.trim().match(/^(\S+):\s*(.*)/);
      if (!match) continue;
      const name = match[1];
      if (name === "lo") continue; // skip loopback
      const fields = match[2].trim().split(/\s+/);
      totalRx += parseInt(fields[0]) || 0;
      totalTx += parseInt(fields[8]) || 0;
      if (name !== "lo" && iface === "all") iface = name;
    }

    const now = Date.now();
    const elapsed = _prevNetTs > 0 ? (now - _prevNetTs) / 1000 : 0;
    let rxPerSec = 0;
    let txPerSec = 0;

    if (elapsed > 0 && _prevNetRx > 0) {
      rxPerSec = Math.max(0, (totalRx - _prevNetRx) / elapsed);
      txPerSec = Math.max(0, (totalTx - _prevNetTx) / elapsed);
    }

    _prevNetRx = totalRx;
    _prevNetTx = totalTx;
    _prevNetTs = now;

    return {
      interface: iface,
      rxBytesPerSec: Math.round(rxPerSec),
      txBytesPerSec: Math.round(txPerSec),
      rxMbps: Math.round((rxPerSec * 8) / 1_000_000 * 100) / 100,
      txMbps: Math.round((txPerSec * 8) / 1_000_000 * 100) / 100,
    };
  } catch {
    // /proc/net/dev not available (macOS)
    return null;
  }
}

// ---- nvidia-smi ----

function queryGpu(): Promise<GpuInfo | null> {
  return new Promise((resolve) => {
    execFile(
      "nvidia-smi",
      [
        "--query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu,power.draw,power.limit,name",
        "--format=csv,noheader,nounits",
      ],
      { timeout: 5000 },
      (err, stdout) => {
        if (err) {
          resolve(null);
          return;
        }
        try {
          const parts = stdout.trim().split(",").map((s) => s.trim());
          if (parts.length < 9) { resolve(null); return; }
          resolve({
            util: parseFloat(parts[0]) || 0,
            vramUsed: parseFloat(parts[3]) || 0,
            vramTotal: parseFloat(parts[2]) || 0,
            temp: parseFloat(parts[5]) || 0,
            powerDraw: parseFloat(parts[6]) || 0,
            powerLimit: parseFloat(parts[7]) || 0,
            name: parts[8],
          });
        } catch {
          resolve(null);
        }
      }
    );
  });
}

// ---- CPU utilization ----

let _prevCpuIdle = 0;
let _prevCpuTotal = 0;

function getCpuUtil(): number {
  const cpus = os.cpus();
  let idle = 0;
  let total = 0;
  for (const cpu of cpus) {
    idle += cpu.times.idle;
    total += cpu.times.user + cpu.times.nice + cpu.times.sys + cpu.times.idle + cpu.times.irq;
  }
  const idleDelta = idle - _prevCpuIdle;
  const totalDelta = total - _prevCpuTotal;
  _prevCpuIdle = idle;
  _prevCpuTotal = total;
  if (totalDelta === 0) return 0;
  return Math.round((1 - idleDelta / totalDelta) * 100);
}

// ---- Ring Buffer ----

const BUFFER_SIZE = 8640; // 30s x 3 days

export class MetricsRingBuffer {
  private buf: (GpuSnapshot | null)[];
  private writePtr = 0;
  private count = 0;

  constructor() {
    this.buf = new Array(BUFFER_SIZE).fill(null);
  }

  push(snapshot: GpuSnapshot): void {
    this.buf[this.writePtr] = snapshot;
    this.writePtr = (this.writePtr + 1) % BUFFER_SIZE;
    if (this.count < BUFFER_SIZE) this.count++;
  }

  getLatest(): GpuSnapshot | null {
    if (this.count === 0) return null;
    const idx = (this.writePtr - 1 + BUFFER_SIZE) % BUFFER_SIZE;
    return this.buf[idx];
  }

  /** Get snapshots since `sinceMs` (unix ms), downsampled to at most `maxPoints`. */
  getRange(sinceMs: number, maxPoints = 300): GpuSnapshot[] {
    const all: GpuSnapshot[] = [];
    // Read buffer in chronological order
    const start = this.count < BUFFER_SIZE ? 0 : this.writePtr;
    for (let i = 0; i < this.count; i++) {
      const idx = (start + i) % BUFFER_SIZE;
      const snap = this.buf[idx];
      if (snap && snap.ts >= sinceMs) {
        all.push(snap);
      }
    }
    // Downsample if too many
    if (all.length <= maxPoints) return all;
    const step = all.length / maxPoints;
    const result: GpuSnapshot[] = [];
    for (let i = 0; i < maxPoints; i++) {
      result.push(all[Math.floor(i * step)]);
    }
    // Always include the latest point
    if (result[result.length - 1] !== all[all.length - 1]) {
      result.push(all[all.length - 1]);
    }
    return result;
  }

  getCount(): number {
    return this.count;
  }
}

// ---- Singleton ----

export const metricsBuffer = new MetricsRingBuffer();

let _interval: ReturnType<typeof setInterval> | null = null;
let _sessionCountFn: (() => number) | null = null;

/** Set the function that returns current active session count. */
export function setSessionCountProvider(fn: () => number): void {
  _sessionCountFn = fn;
}

/** Collect a single snapshot. */
export async function collectSnapshot(): Promise<GpuSnapshot> {
  const [gpu, processes, disk, network] = await Promise.all([
    queryGpu(),
    queryProcessResources(),
    queryDiskUsage(),
    queryNetworkBandwidth(),
  ]);
  const memTotal = Math.round(os.totalmem() / 1024 / 1024);
  // Use MemAvailable from /proc/meminfo for accurate "usable" memory
  // os.freemem() returns MemFree which excludes reclaimable cache/buffers
  let memFree = Math.round(os.freemem() / 1024 / 1024); // fallback
  try {
    const fs = require("fs");
    const meminfo = fs.readFileSync("/proc/meminfo", "utf8");
    const match = meminfo.match(/MemAvailable:\s+(\d+)/);
    if (match) memFree = Math.round(parseInt(match[1]) / 1024);
  } catch {}
  const procMem = process.memoryUsage();

  const snap: GpuSnapshot = {
    ts: Date.now(),
    gpu,
    cpu: { util: getCpuUtil() },
    mem: { used: memTotal - memFree, total: memTotal },
    node: {
      rss: Math.round(procMem.rss / 1024 / 1024),
      heap: Math.round(procMem.heapUsed / 1024 / 1024),
    },
    sessions: _sessionCountFn ? _sessionCountFn() : 0,
  };

  if (processes.length > 0) snap.processes = processes;
  if (disk) snap.disk = disk;
  if (network) snap.network = network;

  return snap;
}

/** Start collecting metrics every 30 seconds. */
export function startMetricsCollection(): void {
  if (_interval) return;
  console.log("[gpu-monitor] Starting metrics collection (30s interval, 3-day buffer)");
  // Initialize CPU baseline
  getCpuUtil();
  // Collect immediately, then every 30s
  collectSnapshot().then((snap) => metricsBuffer.push(snap));
  _interval = setInterval(async () => {
    try {
      const snap = await collectSnapshot();
      metricsBuffer.push(snap);
    } catch (err) {
      console.error("[gpu-monitor] Collection error:", err);
    }
  }, 30_000);
}

/** Stop metrics collection. */
export function stopMetricsCollection(): void {
  if (_interval) {
    clearInterval(_interval);
    _interval = null;
    console.log("[gpu-monitor] Metrics collection stopped");
  }
}
