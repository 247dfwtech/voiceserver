export interface LLMConfig {
  provider: string;
  model: string;
  temperature?: number;
  maxTokens?: number;
  baseUrl?: string;
}

export interface LLMMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  tool_call_id?: string;
  name?: string;
}

export interface LLMToolDefinition {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters: Record<string, unknown>;
  };
}

export interface LLMToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

export interface LLMProvider {
  /** Stream a chat completion. Calls onToken for each text chunk, onToolCall for tool calls. */
  streamCompletion(params: {
    messages: LLMMessage[];
    tools?: LLMToolDefinition[];
    onToken: (token: string) => void;
    onToolCall: (toolCall: LLMToolCall) => void;
    onDone: (fullText: string) => void;
  }): { cancel: () => void };
}
