import OpenAI from "openai";

interface AnalysisConfig {
  summaryEnabled?: boolean;
  summaryPrompt?: string;
  successEvaluationEnabled?: boolean;
  successEvaluationPrompt?: string;
  successEvaluationRubric?: string;
}

interface AnalysisResult {
  summary?: string;
  successEvaluation?: string;
}

/**
 * Run post-call analysis: generate summary and success evaluation using LLM.
 *
 * Uses Ollama (via OpenAI-compatible API) by default for free analysis.
 * Falls back to OpenAI if OPENAI_API_KEY is set and Ollama is not available.
 */
export async function runPostCallAnalysis(
  transcript: string,
  config: AnalysisConfig,
  apiKey?: string
): Promise<AnalysisResult> {
  if (!transcript || (!config.summaryEnabled && !config.successEvaluationEnabled)) {
    return {};
  }

  // Determine which LLM to use for analysis
  const ollamaUrl = process.env.OLLAMA_URL || "http://localhost:11434/v1";
  const ollamaModel = process.env.OLLAMA_MODEL || "llama3.1:8b";
  const openaiKey = apiKey || process.env.OPENAI_API_KEY;

  // Try Ollama first (free), fall back to OpenAI
  let client: OpenAI;
  let model: string;

  try {
    // Check if Ollama is reachable
    const ollamaBaseUrl = ollamaUrl.replace(/\/v1\/?$/, "");
    const healthCheck = await fetch(`${ollamaBaseUrl}/api/tags`, {
      signal: AbortSignal.timeout(3000),
    });
    if (healthCheck.ok) {
      client = new OpenAI({ apiKey: "ollama", baseURL: ollamaUrl });
      model = ollamaModel;
    } else {
      throw new Error("Ollama not reachable");
    }
  } catch {
    if (openaiKey) {
      client = new OpenAI({ apiKey: openaiKey });
      model = "gpt-4o-mini";
    } else {
      console.warn("[analysis] Neither Ollama nor OpenAI available for post-call analysis");
      return {};
    }
  }

  const result: AnalysisResult = {};

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
    } catch (err) {
      console.error("[analysis] Success evaluation error:", err);
    }
  }

  return result;
}
