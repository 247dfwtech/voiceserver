import OpenAI from "openai";

interface AnalysisConfig {
  summaryEnabled?: boolean;
  summaryPrompt?: string;
  successEvaluationEnabled?: boolean;
  successEvaluationPrompt?: string;
  successEvaluationRubric?: string;
  provider?: string; // "local" | "deepseek" | "openrouter" | "cerebras" | "groq" | "deepinfra"
  model?: string;
}

interface AnalysisResult {
  summary?: string;
  successEvaluation?: string;
  analysisCost?: number;
  analysisProvider?: string;
  analysisModel?: string;
}

/** Per-1K-token pricing for analysis providers */
const ANALYSIS_PRICING: Record<string, { input: number; output: number }> = {
  "deepseek-chat": { input: 0.00014, output: 0.00028 },
  "deepseek-reasoner": { input: 0.00055, output: 0.0022 },
  "meta-llama/llama-3.1-8b-instruct:free": { input: 0, output: 0 },
  "gpt-4o-mini": { input: 0.00015, output: 0.0006 },
  // Cerebras (free tier)
  "llama-3.3-70b": { input: 0, output: 0 },
  // Groq
  "llama-3.1-8b-instant": { input: 0.00005, output: 0.00008 },
  // DeepInfra
  "meta-llama/Meta-Llama-3.1-8B-Instruct": { input: 0.00003, output: 0.00005 },
};

/**
 * Create an OpenAI-compatible client for the given provider.
 */
function createAnalysisClient(provider: string, model?: string): { client: OpenAI; model: string } | null {
  switch (provider) {
    case "deepseek": {
      const apiKey = process.env.DEEPSEEK_API_KEY;
      if (!apiKey) {
        console.warn("[analysis] DeepSeek API key not configured, falling back to local");
        return null;
      }
      return {
        client: new OpenAI({
          apiKey,
          baseURL: process.env.DEEPSEEK_BASE_URL || "https://api.deepseek.com/v1",
        }),
        model: model || "deepseek-chat",
      };
    }

    case "openrouter": {
      const apiKey = process.env.OPENROUTER_API_KEY;
      if (!apiKey) {
        console.warn("[analysis] OpenRouter API key not configured, falling back to local");
        return null;
      }
      return {
        client: new OpenAI({
          apiKey,
          baseURL: "https://openrouter.ai/api/v1",
          defaultHeaders: {
            "HTTP-Referer": process.env.PUBLIC_URL || "https://bushidopros.com",
            "X-Title": "Bushido Pros Voice AI",
          },
        }),
        model: model || "meta-llama/llama-3.1-8b-instruct:free",
      };
    }

    case "cerebras": {
      const apiKey = process.env.CEREBRAS_API_KEY;
      if (!apiKey) {
        console.warn("[analysis] Cerebras API key not configured, falling back to local");
        return null;
      }
      return {
        client: new OpenAI({
          apiKey,
          baseURL: "https://api.cerebras.ai/v1",
        }),
        model: model || "llama-3.3-70b",
      };
    }

    case "groq": {
      const apiKey = process.env.GROQ_API_KEY;
      if (!apiKey) {
        console.warn("[analysis] Groq API key not configured, falling back to local");
        return null;
      }
      return {
        client: new OpenAI({
          apiKey,
          baseURL: "https://api.groq.com/openai/v1",
        }),
        model: model || "llama-3.1-8b-instant",
      };
    }

    case "deepinfra": {
      const apiKey = process.env.DEEPINFRA_API_KEY;
      if (!apiKey) {
        console.warn("[analysis] DeepInfra API key not configured, falling back to local");
        return null;
      }
      return {
        client: new OpenAI({
          apiKey,
          baseURL: "https://api.deepinfra.com/v1/openai",
        }),
        model: model || "meta-llama/Meta-Llama-3.1-8B-Instruct",
      };
    }

    case "local":
    default:
      return null; // Will use Ollama below
  }
}

/**
 * Create Ollama (local) client.
 */
function createLocalClient(): { client: OpenAI; model: string } | null {
  const ollamaUrl = process.env.OLLAMA_URL || "http://localhost:11434/v1";
  const ollamaModel = process.env.DEFAULT_LLM || process.env.OLLAMA_MODEL || "qwen3.5:9b";
  return {
    client: new OpenAI({ apiKey: "ollama", baseURL: ollamaUrl }),
    model: ollamaModel,
  };
}

/**
 * Run post-call analysis: generate summary and success evaluation using LLM.
 *
 * Provider priority:
 * 1. Use the provider specified in analysisConfig (deepseek, openrouter, or local)
 * 2. If specified provider fails/missing key, fall back to local (Ollama)
 * 3. If Ollama unavailable, fall back to OpenAI if key exists
 */
export async function runPostCallAnalysis(
  transcript: string,
  config: AnalysisConfig,
  apiKey?: string
): Promise<AnalysisResult> {
  if (!transcript || (!config.summaryEnabled && !config.successEvaluationEnabled)) {
    return {};
  }

  const provider = config.provider || "local";
  let clientInfo = createAnalysisClient(provider, config.model);

  // If cloud provider configured, try it first
  if (clientInfo) {
    console.log(`[analysis] Using ${provider} (${clientInfo.model}) for post-call analysis`);
  } else {
    // Fall back to local Ollama
    try {
      const ollamaUrl = (process.env.OLLAMA_URL || "http://localhost:11434/v1").replace(/\/v1\/?$/, "");
      const healthCheck = await fetch(`${ollamaUrl}/api/tags`, {
        signal: AbortSignal.timeout(3000),
      });
      if (healthCheck.ok) {
        clientInfo = createLocalClient();
        console.log(`[analysis] Using local Ollama (${clientInfo!.model}) for post-call analysis`);
      } else {
        throw new Error("Ollama not reachable");
      }
    } catch {
      // Last resort: OpenAI
      const openaiKey = apiKey || process.env.OPENAI_API_KEY;
      if (openaiKey) {
        clientInfo = {
          client: new OpenAI({ apiKey: openaiKey }),
          model: "gpt-4o-mini",
        };
        console.log("[analysis] Ollama unavailable, falling back to OpenAI");
      } else {
        console.warn("[analysis] No LLM available for post-call analysis");
        return {};
      }
    }
  }

  if (!clientInfo) return {};

  const { client, model } = clientInfo;
  const result: AnalysisResult = { analysisProvider: provider, analysisModel: model };
  let totalInputTokens = 0;
  let totalOutputTokens = 0;

  if (config.summaryEnabled) {
    try {
      const prompt =
        config.summaryPrompt ||
        "Summarize the following phone call transcript in 2-3 sentences. Focus on what was discussed and the outcome.";

      const completion = await client.chat.completions.create({
        model,
        messages: [
          { role: "system", content: prompt },
          { role: "user", content: transcript },
        ],
        temperature: 0.3,
        max_tokens: 200,
      });

      result.summary = completion.choices[0]?.message?.content || "";
      totalInputTokens += completion.usage?.prompt_tokens || 0;
      totalOutputTokens += completion.usage?.completion_tokens || 0;
    } catch (err) {
      console.error("[analysis] Summary error:", err);
    }
  }

  if (config.successEvaluationEnabled) {
    try {
      const rubric = config.successEvaluationRubric || "PassFail";
      const prompt =
        config.successEvaluationPrompt ||
        `Evaluate the success of this phone call based on the transcript. Return ONLY "${rubric === "PassFail" ? "Pass" : "1-10"}" or "${rubric === "PassFail" ? "Fail" : "a number"}" with no explanation.`;

      const completion = await client.chat.completions.create({
        model,
        messages: [
          { role: "system", content: prompt },
          { role: "user", content: transcript },
        ],
        temperature: 0,
        max_tokens: 10,
      });

      result.successEvaluation =
        completion.choices[0]?.message?.content?.trim() || "";
      totalInputTokens += completion.usage?.prompt_tokens || 0;
      totalOutputTokens += completion.usage?.completion_tokens || 0;
    } catch (err) {
      console.error("[analysis] Success evaluation error:", err);
    }
  }

  // Calculate cost from token usage
  const pricing = ANALYSIS_PRICING[model] ??
    (parseFloat(process.env.COST_LLM_DEEPSEEK || "0") > 0 && provider === "deepseek"
      ? { input: parseFloat(process.env.COST_LLM_DEEPSEEK || "0"), output: parseFloat(process.env.COST_LLM_DEEPSEEK || "0") }
      : parseFloat(process.env.COST_LLM_OPENROUTER || "0") > 0 && provider === "openrouter"
        ? { input: parseFloat(process.env.COST_LLM_OPENROUTER || "0"), output: parseFloat(process.env.COST_LLM_OPENROUTER || "0") }
        : { input: 0, output: 0 });

  const cost = (totalInputTokens / 1000) * pricing.input + (totalOutputTokens / 1000) * pricing.output;
  result.analysisCost = Math.round(cost * 10000) / 10000;

  if (result.analysisCost > 0) {
    console.log(`[analysis] Cost: $${result.analysisCost.toFixed(4)} (${totalInputTokens}in/${totalOutputTokens}out tokens, ${provider}/${model})`);
  }

  return result;
}
