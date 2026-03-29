import type { LLMToolCall } from "../providers/llm/interface";

/**
 * SSRF protection: block tool webhook requests to internal/private networks.
 * Only allows HTTPS (or HTTP to known safe hosts for dev).
 */
function isAllowedUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    const hostname = parsed.hostname.toLowerCase();

    // Block non-HTTP protocols
    if (parsed.protocol !== "https:" && parsed.protocol !== "http:") return false;

    // Block localhost/loopback
    if (hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1" || hostname === "0.0.0.0") return false;

    // Block private RFC1918 ranges
    if (/^10\./.test(hostname)) return false;
    if (/^172\.(1[6-9]|2\d|3[01])\./.test(hostname)) return false;
    if (/^192\.168\./.test(hostname)) return false;

    // Block link-local
    if (/^169\.254\./.test(hostname)) return false;

    // Block AWS/cloud metadata endpoints
    if (hostname === "metadata.google.internal") return false;

    return true;
  } catch {
    return false;
  }
}

export interface ToolDefinition {
  id: string;
  name: string;
  type: "function" | "transferCall" | "dtmf" | "endCall";
  description?: string;
  config?: {
    url?: string;
    method?: string;
    headers?: Record<string, string>;
    destination?: string;
  };
  functionDefinition?: {
    name: string;
    description?: string;
    parameters: Record<string, unknown>;
  };
  serverUrl?: string;
}

export interface ToolResult {
  toolCallId: string;
  name: string;
  result: string;
  action?: "transfer" | "endCall" | "dtmf" | null;
  actionData?: { destination?: string };
}

/**
 * Execute a tool call during a live call.
 * Returns the result to feed back into the LLM, plus any call actions (transfer, end call).
 */
export async function executeTool(
  toolCall: LLMToolCall,
  toolDefinitions: ToolDefinition[],
  callContext: {
    callId: string;
    customerNumber: string;
    customerName?: string;
    assistantId: string;
    metadata?: Record<string, unknown>;
    fallbackDestination?: string;
  }
): Promise<ToolResult> {
  const toolDef = toolDefinitions.find(
    (t) =>
      t.functionDefinition?.name === toolCall.function.name ||
      t.name === toolCall.function.name
  );

  if (!toolDef) {
    return {
      toolCallId: toolCall.id,
      name: toolCall.function.name,
      result: `Tool "${toolCall.function.name}" not found`,
    };
  }

  // Handle built-in tool types
  if (toolDef.type === "endCall") {
    return {
      toolCallId: toolCall.id,
      name: toolCall.function.name,
      result: "Call will be ended.",
      action: "endCall",
    };
  }

  if (toolDef.type === "transferCall") {
    const destination = toolDef.config?.destination || callContext.fallbackDestination;
    if (!destination) {
      return {
        toolCallId: toolCall.id,
        name: toolCall.function.name,
        result: "Transfer failed: no destination configured",
      };
    }
    return {
      toolCallId: toolCall.id,
      name: toolCall.function.name,
      result: `Transferring call to ${destination}`,
      action: "transfer",
      actionData: { destination },
    };
  }

  if (toolDef.type === "dtmf") {
    const destination = toolDef.config?.destination;
    return {
      toolCallId: toolCall.id,
      name: toolCall.function.name,
      result: `DTMF transfer to ${destination}`,
      action: "dtmf",
      actionData: { destination },
    };
  }

  // Function type -- call the webhook URL
  const url = toolDef.config?.url || toolDef.serverUrl;
  if (!url) {
    return {
      toolCallId: toolCall.id,
      name: toolCall.function.name,
      result: "No webhook URL configured for this tool",
    };
  }

  // SSRF protection: block requests to internal/private networks
  if (!isAllowedUrl(url)) {
    return {
      toolCallId: toolCall.id,
      name: toolCall.function.name,
      result: "Tool webhook URL blocked: internal/private addresses not allowed",
    };
  }

  try {
    let args: Record<string, unknown> = {};
    try {
      args = JSON.parse(toolCall.function.arguments);
    } catch {
      // args stay empty
    }

    const payload = {
      message: {
        type: "tool-calls",
        toolCallList: [
          {
            id: toolCall.id,
            type: "function",
            function: {
              name: toolCall.function.name,
              arguments: args,
            },
          },
        ],
        call: {
          id: callContext.callId,
          customer: {
            number: callContext.customerNumber,
            name: callContext.customerName,
          },
          assistantId: callContext.assistantId,
          metadata: callContext.metadata,
        },
      },
    };

    const response = await fetch(url, {
      method: toolDef.config?.method || "POST",
      headers: {
        "Content-Type": "application/json",
        ...toolDef.config?.headers,
      },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(30000),
    });

    if (!response.ok) {
      return {
        toolCallId: toolCall.id,
        name: toolCall.function.name,
        result: `Tool webhook returned ${response.status}: ${await response.text()}`,
      };
    }

    const data = await response.json();
    return {
      toolCallId: toolCall.id,
      name: toolCall.function.name,
      result: typeof data === "string" ? data : JSON.stringify(data),
    };
  } catch (err) {
    return {
      toolCallId: toolCall.id,
      name: toolCall.function.name,
      result: `Tool execution error: ${err instanceof Error ? err.message : String(err)}`,
    };
  }
}
