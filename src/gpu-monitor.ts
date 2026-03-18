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

export interface GpuSnapshot {
  ts: number;          // unix ms
  gpu: GpuInfo | null; // null if nvidia-smi unavailable
  cpu: { util: number };
  mem: { used: number; total: number }; // system RAM in MB
  node: { rss: number; heap: number };  // Node.js process in MB
  sessions: number;
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
  const [gpu] = await Promise.all([queryGpu()]);
  const memTotal = Math.round(os.totalmem() / 1024 / 1024);
  const memFree = Math.round(os.freemem() / 1024 / 1024);
  const procMem = process.memoryUsage();

  return {
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
