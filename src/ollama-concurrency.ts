/**
 * Module-level Ollama concurrency tracker.
 * Shared across all CallSession instances to track in-flight Ollama LLM requests.
 * Node.js is single-threaded so no locks are needed.
 */

let activeRequests = 0;

export function incrementOllama(): number {
  return ++activeRequests;
}

export function decrementOllama(): number {
  if (activeRequests > 0) activeRequests--;
  return activeRequests;
}

export function getOllamaActiveRequests(): number {
  return activeRequests;
}

export function getOllamaMaxParallel(): number {
  return parseInt(process.env.OLLAMA_NUM_PARALLEL || "8", 10);
}
