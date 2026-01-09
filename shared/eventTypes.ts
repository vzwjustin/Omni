/**
 * Shared event types for WebSocket communication between dashboard and Omni backend.
 * These types define the structure of real-time events streamed to the dashboard.
 */

export type EventType = 
  | "framework_start"
  | "framework_end"
  | "reasoning_step"
  | "llm_call"
  | "context_gathering"
  | "code_generation"
  | "debugging"
  | "token_usage"
  | "error";

export type EventCategory = 
  | "context"
  | "code_generation"
  | "debugging"
  | "reasoning"
  | "llm"
  | "framework"
  | "system";

export type EventStatus = "pending" | "running" | "completed" | "failed";

/**
 * Base event structure for all Omni system events.
 */
export interface OmniEvent {
  id: string;
  type: EventType;
  category: EventCategory;
  title: string;
  description?: string;
  status: EventStatus;
  timestamp: number; // Unix timestamp in milliseconds
  duration?: number; // Duration in milliseconds
  data: Record<string, unknown>;
  metadata?: {
    tokenUsage?: {
      input: number;
      output: number;
      total: number;
    };
    model?: string;
    framework?: string;
    userId?: string;
    [key: string]: unknown;
  };
  error?: {
    message: string;
    code?: string;
    stack?: string;
  };
}

/**
 * Framework execution event - triggered when a framework starts or completes.
 */
export interface FrameworkEvent extends OmniEvent {
  type: "framework_start" | "framework_end";
  data: {
    frameworkName: string;
    query: string;
    stage?: string;
    progress?: number; // 0-100
  };
}

/**
 * Reasoning step event - shows intermediate reasoning during problem-solving.
 */
export interface ReasoningStepEvent extends OmniEvent {
  type: "reasoning_step";
  data: {
    stepNumber: number;
    reasoning: string;
    confidence?: number; // 0-1
    nextStep?: string;
  };
}

/**
 * LLM call event - tracks API calls to language models.
 */
export interface LLMCallEvent extends OmniEvent {
  type: "llm_call";
  data: {
    model: string;
    prompt?: string;
    response?: string;
    temperature?: number;
    maxTokens?: number;
  };
}

/**
 * Context gathering event - shows when system is collecting information.
 */
export interface ContextGatheringEvent extends OmniEvent {
  type: "context_gathering";
  data: {
    source: string;
    itemsFound?: number;
    relevanceScore?: number;
    content?: string;
  };
}

/**
 * Code generation event - tracks code synthesis and execution.
 */
export interface CodeGenerationEvent extends OmniEvent {
  type: "code_generation";
  data: {
    language: string;
    code: string;
    executionResult?: string;
    executionTime?: number;
  };
}

/**
 * Debugging event - shows debugging operations and findings.
 */
export interface DebuggingEvent extends OmniEvent {
  type: "debugging";
  data: {
    issue: string;
    diagnosis?: string;
    solution?: string;
    severity?: "low" | "medium" | "high";
  };
}

/**
 * Token usage event - aggregated token consumption metrics.
 */
export interface TokenUsageEvent extends OmniEvent {
  type: "token_usage";
  data: {
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
    model: string;
    cost?: number;
  };
}

/**
 * Error event - system errors and exceptions.
 */
export interface ErrorEvent extends OmniEvent {
  type: "error";
  data: {
    errorType: string;
    message: string;
    context?: string;
  };
}

/**
 * Union type of all possible event types.
 */
export type AnyOmniEvent = 
  | FrameworkEvent
  | ReasoningStepEvent
  | LLMCallEvent
  | ContextGatheringEvent
  | CodeGenerationEvent
  | DebuggingEvent
  | TokenUsageEvent
  | ErrorEvent;

/**
 * WebSocket message types for dashboard communication.
 */
export interface WebSocketMessage {
  type: "event" | "status" | "command" | "response";
  payload: unknown;
  timestamp?: number;
}

/**
 * Event stream message from server to client.
 */
export interface EventStreamMessage extends WebSocketMessage {
  type: "event";
  payload: AnyOmniEvent;
}

/**
 * Status message for connection and system health.
 */
export interface StatusMessage extends WebSocketMessage {
  type: "status";
  payload: {
    connected: boolean;
    activeFrameworks: number;
    totalEvents: number;
    uptime: number;
    lastEventTime?: number;
  };
}

/**
 * Command message from client to server (e.g., Claude Code CLI execution).
 */
export interface CommandMessage extends WebSocketMessage {
  type: "command";
  payload: {
    commandType: "execute_code" | "filter_events" | "clear_history" | "export_logs";
    data: Record<string, unknown>;
  };
}

/**
 * Response message from server to client.
 */
export interface ResponseMessage extends WebSocketMessage {
  type: "response";
  payload: {
    success: boolean;
    data?: unknown;
    error?: string;
  };
}
