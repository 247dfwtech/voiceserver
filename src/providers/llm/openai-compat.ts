import OpenAI from "openai";
import type { LLMProvider, LLMConfig, LLMMessage, LLMToolDefinition, LLMToolCall } from "./interface";

export class OpenAICompatLLM implements LLMProvider {
  private client: OpenAI;
  private config: LLMConfig;

  constructor(config: LLMConfig, apiKey: string) {
    this.config = config;
    this.client = new OpenAI({
      apiKey,
      baseURL: config.baseUrl,
    });
  }

  streamCompletion(params: {
    messages: LLMMessage[];
    tools?: LLMToolDefinition[];
    onToken: (token: string) => void;
    onToolCall: (toolCall: LLMToolCall) => void;
    onDone: (fullText: string) => void;
  }): { cancel: () => void } {
    let cancelled = false;
    let abortController: AbortController | null = new AbortController();

    const run = async () => {
      try {
        // Ollama/qwen3 thinking models use ~1000 tokens for chain-of-thought
        // before emitting content, so 300 tokens leaves nothing for the reply.
        const rawMax = this.config.maxTokens ?? 300;
        const isOllama = this.config.provider === "ollama";
        const effectiveMaxTokens = (isOllama && rawMax < 2000) ? 2000 : rawMax;
        if (isOllama && rawMax < 2000) {
          console.warn(`[llm] Ollama thinking model: overriding maxTokens from ${rawMax} to 2000 (chain-of-thought needs ~1000 tokens)`);
        }

        const streamParams: OpenAI.ChatCompletionCreateParamsStreaming = {
          model: this.config.model,
          messages: params.messages as OpenAI.ChatCompletionMessageParam[],
          temperature: this.config.temperature ?? 0.3,
          max_tokens: effectiveMaxTokens,
          stream: true,
        };

        if (params.tools && params.tools.length > 0) {
          streamParams.tools = params.tools as OpenAI.ChatCompletionTool[];
        }

        const stream = await this.client.chat.completions.create(streamParams, {
          signal: abortController!.signal,
        });

        let fullText = "";
        const toolCalls: Map<number, { id: string; name: string; args: string }> = new Map();

        for await (const chunk of stream) {
          if (cancelled) break;
          const delta = chunk.choices[0]?.delta;
          if (!delta) continue;

          if (delta.content) {
            fullText += delta.content;
            params.onToken(delta.content);
          }

          if (delta.tool_calls) {
            for (const tc of delta.tool_calls) {
              const idx = tc.index;
              if (!toolCalls.has(idx)) {
                toolCalls.set(idx, {
                  id: tc.id || "",
                  name: tc.function?.name || "",
                  args: "",
                });
              }
              const existing = toolCalls.get(idx)!;
              if (tc.id) existing.id = tc.id;
              if (tc.function?.name) existing.name = tc.function.name;
              if (tc.function?.arguments) existing.args += tc.function.arguments;
            }
          }
        }

        if (!cancelled) {
          // Emit complete tool calls
          for (const tc of Array.from(toolCalls.values())) {
            params.onToolCall({
              id: tc.id,
              type: "function",
              function: { name: tc.name, arguments: tc.args },
            });
          }
          params.onDone(fullText);
        }
      } catch (err: unknown) {
        if (!cancelled) {
          if (err instanceof Error && err.name !== "AbortError") {
            console.error("[llm] Stream error:", err);
          }
          // Always signal done so the session transitions back to waiting_for_speech
          params.onDone("");
        }
      }
    };

    run();

    return {
      cancel: () => {
        cancelled = true;
        abortController?.abort();
        abortController = null;
      },
    };
  }
}
