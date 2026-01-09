/**
 * Mock event emitter for testing the dashboard without a real Omni backend.
 * This generates synthetic events that simulate real Omni system operations.
 * 
 * Usage: Import and call startMockEventStream() to begin emitting test events.
 */

import { nanoid } from "nanoid";
import type { AnyOmniEvent } from "../shared/eventTypes";
import { emitEvent } from "./websocket";

const frameworks = ["iterative_reasoning", "code_generation", "context_gathering"];
const models = ["claude-3-sonnet", "claude-3-opus", "gpt-4-turbo"];
const stages = ["planning", "execution", "verification", "refinement"];

/**
 * Generate a mock framework start event.
 */
function createFrameworkStartEvent(): AnyOmniEvent {
  const framework = frameworks[Math.floor(Math.random() * frameworks.length)];
  return {
    id: nanoid(),
    type: "framework_start",
    category: "framework",
    title: `Starting ${framework}`,
    description: `Initializing ${framework} with new query`,
    status: "running",
    timestamp: Date.now(),
    data: {
      frameworkName: framework,
      query: "Analyze the provided code and suggest optimizations",
      stage: "initialization",
      progress: 0,
    },
    metadata: {
      framework: framework,
    },
  };
}

/**
 * Generate a mock reasoning step event.
 */
function createReasoningStepEvent(): AnyOmniEvent {
  const stepNumber = Math.floor(Math.random() * 10) + 1;
  return {
    id: nanoid(),
    type: "reasoning_step",
    category: "reasoning",
    title: `Reasoning Step ${stepNumber}`,
    description: `Analyzing problem space and identifying key patterns`,
    status: "completed",
    timestamp: Date.now(),
    duration: Math.floor(Math.random() * 5000) + 500,
    data: {
      stepNumber: stepNumber,
      reasoning: "The code has inefficient loop iterations that can be optimized using memoization. Additionally, the API calls are not batched, leading to unnecessary network overhead.",
      confidence: Math.random() * 0.5 + 0.5,
      nextStep: "Generate optimized code implementation",
    },
  };
}

/**
 * Generate a mock LLM call event.
 */
function createLLMCallEvent(): AnyOmniEvent {
  const model = models[Math.floor(Math.random() * models.length)];
  const inputTokens = Math.floor(Math.random() * 2000) + 500;
  const outputTokens = Math.floor(Math.random() * 1500) + 300;

  return {
    id: nanoid(),
    type: "llm_call",
    category: "llm",
    title: `LLM Call to ${model}`,
    description: `Generating response with temperature 0.7`,
    status: "completed",
    timestamp: Date.now(),
    duration: Math.floor(Math.random() * 8000) + 1000,
    data: {
      model: model,
      prompt: "Optimize this code for performance and readability",
      response: "Here's an optimized version that uses memoization and batches API calls for better performance...",
      temperature: 0.7,
      maxTokens: 2000,
    },
    metadata: {
      model: model,
      tokenUsage: {
        input: inputTokens,
        output: outputTokens,
        total: inputTokens + outputTokens,
      },
    },
  };
}

/**
 * Generate a mock context gathering event.
 */
function createContextGatheringEvent(): AnyOmniEvent {
  return {
    id: nanoid(),
    type: "context_gathering",
    category: "context",
    title: "Gathering Context",
    description: "Retrieving relevant documentation and code examples",
    status: "completed",
    timestamp: Date.now(),
    duration: Math.floor(Math.random() * 3000) + 500,
    data: {
      source: "documentation_index",
      itemsFound: Math.floor(Math.random() * 20) + 5,
      relevanceScore: Math.random() * 0.3 + 0.7,
      content: "Found 12 relevant documentation pages and 8 code examples matching the query",
    },
  };
}

/**
 * Generate a mock code generation event.
 */
function createCodeGenerationEvent(): AnyOmniEvent {
  return {
    id: nanoid(),
    type: "code_generation",
    category: "code_generation",
    title: "Generating Optimized Code",
    description: "Creating TypeScript implementation",
    status: "completed",
    timestamp: Date.now(),
    duration: Math.floor(Math.random() * 4000) + 1000,
    data: {
      language: "typescript",
      code: `function optimizeArray(arr: number[]): number[] {
  const memo = new Map<string, number>();
  return arr.map(item => {
    const key = String(item);
    if (!memo.has(key)) {
      memo.set(key, expensiveComputation(item));
    }
    return memo.get(key)!;
  });
}`,
      executionResult: "✓ Code executed successfully",
      executionTime: Math.floor(Math.random() * 500) + 50,
    },
  };
}

/**
 * Generate a mock debugging event.
 */
function createDebuggingEvent(): AnyOmniEvent {
  return {
    id: nanoid(),
    type: "debugging",
    category: "debugging",
    title: "Debugging Issue",
    description: "Analyzing and resolving runtime error",
    status: "completed",
    timestamp: Date.now(),
    duration: Math.floor(Math.random() * 2000) + 500,
    data: {
      issue: "TypeError: Cannot read property 'map' of undefined",
      diagnosis: "The input array is not being properly initialized before the map operation",
      solution: "Add null check and default to empty array if undefined",
      severity: "high",
    },
  };
}

/**
 * Generate a mock token usage event.
 */
function createTokenUsageEvent(): AnyOmniEvent {
  const inputTokens = Math.floor(Math.random() * 3000) + 1000;
  const outputTokens = Math.floor(Math.random() * 2000) + 500;

  return {
    id: nanoid(),
    type: "token_usage",
    category: "system",
    title: "Token Usage Summary",
    description: "Aggregated token consumption for current session",
    status: "completed",
    timestamp: Date.now(),
    data: {
      inputTokens: inputTokens,
      outputTokens: outputTokens,
      totalTokens: inputTokens + outputTokens,
      model: models[Math.floor(Math.random() * models.length)],
      cost: (inputTokens * 0.003 + outputTokens * 0.015) / 1000,
    },
  };
}

/**
 * Generate a mock error event.
 */
function createErrorEvent(): AnyOmniEvent {
  return {
    id: nanoid(),
    type: "error",
    category: "system",
    title: "Framework Error",
    description: "Recoverable error during execution",
    status: "failed",
    timestamp: Date.now(),
    data: {
      errorType: "TimeoutError",
      message: "Request to external API timed out after 30 seconds",
      context: "While fetching documentation from external source",
    },
    error: {
      message: "Request to external API timed out after 30 seconds",
      code: "TIMEOUT_ERROR",
      stack: "at fetchDocumentation (context.ts:45:12)\n  at processContext (gateway.ts:120:8)",
    },
  };
}

/**
 * Generate a mock framework end event.
 */
function createFrameworkEndEvent(): AnyOmniEvent {
  return {
    id: nanoid(),
    type: "framework_end",
    category: "framework",
    title: "Framework Completed",
    description: "Successfully completed framework execution",
    status: "completed",
    timestamp: Date.now(),
    duration: Math.floor(Math.random() * 30000) + 5000,
    data: {
      frameworkName: frameworks[Math.floor(Math.random() * frameworks.length)],
      query: "Analyze and optimize the provided code",
      stage: "completion",
      progress: 100,
    },
  };
}

/**
 * Event generator function that returns a random event type.
 */
function generateRandomEvent(): AnyOmniEvent {
  const eventGenerators = [
    createFrameworkStartEvent,
    createReasoningStepEvent,
    createLLMCallEvent,
    createContextGatheringEvent,
    createCodeGenerationEvent,
    createDebuggingEvent,
    createTokenUsageEvent,
    createErrorEvent,
    createFrameworkEndEvent,
  ];

  const generator = eventGenerators[Math.floor(Math.random() * eventGenerators.length)];
  return generator();
}

/**
 * Start emitting mock events at regular intervals.
 * This is useful for testing the dashboard without a real Omni backend.
 */
export function startMockEventStream(intervalMs: number = 2000) {
  console.log("[Mock Events] Starting mock event stream with interval", intervalMs, "ms");

  const interval = setInterval(() => {
    const event = generateRandomEvent();
    emitEvent(event);
    console.log("[Mock Events] Emitted:", event.type, "-", event.title);
  }, intervalMs);

  return () => {
    clearInterval(interval);
    console.log("[Mock Events] Stopped mock event stream");
  };
}

/**
 * Emit a single mock event immediately.
 */
export function emitMockEvent(type?: string): AnyOmniEvent {
  let event: AnyOmniEvent;

  switch (type) {
    case "framework_start":
      event = createFrameworkStartEvent();
      break;
    case "reasoning_step":
      event = createReasoningStepEvent();
      break;
    case "llm_call":
      event = createLLMCallEvent();
      break;
    case "context_gathering":
      event = createContextGatheringEvent();
      break;
    case "code_generation":
      event = createCodeGenerationEvent();
      break;
    case "debugging":
      event = createDebuggingEvent();
      break;
    case "token_usage":
      event = createTokenUsageEvent();
      break;
    case "error":
      event = createErrorEvent();
      break;
    case "framework_end":
      event = createFrameworkEndEvent();
      break;
    default:
      event = generateRandomEvent();
  }

  emitEvent(event);
  return event;
}
