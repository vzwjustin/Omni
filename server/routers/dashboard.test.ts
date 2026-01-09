import { describe, it, expect, beforeEach, vi } from "vitest";
import { dashboardRouter } from "./dashboard";
import type { TrpcContext } from "../_core/context";

type AuthenticatedUser = NonNullable<TrpcContext["user"]>;

/**
 * Create a mock authenticated context for testing.
 */
function createAuthContext(): TrpcContext {
  const user: AuthenticatedUser = {
    id: 1,
    openId: "test-user",
    email: "test@example.com",
    name: "Test User",
    loginMethod: "manus",
    role: "user",
    createdAt: new Date(),
    updatedAt: new Date(),
    lastSignedIn: new Date(),
  };

  return {
    user,
    req: {
      protocol: "https",
      headers: {},
    } as TrpcContext["req"],
    res: {} as TrpcContext["res"],
  };
}

describe("Dashboard Router", () => {
  let ctx: TrpcContext;
  let caller: ReturnType<typeof dashboardRouter.createCaller>;

  beforeEach(() => {
    ctx = createAuthContext();
    caller = dashboardRouter.createCaller(ctx);
  });

  describe("getStatus", () => {
    it("should return WebSocket status", async () => {
      const status = await caller.getStatus();

      expect(status).toBeDefined();
      expect(status).toHaveProperty("omniConnected");
      expect(status).toHaveProperty("omniBackendEnabled");
      expect(status).toHaveProperty("activeConnections");
      expect(status).toHaveProperty("eventHistorySize");
    });

    it("should have correct types for status metrics", async () => {
      const status = await caller.getStatus();

      expect(typeof status.omniConnected).toBe("boolean");
      expect(typeof status.omniBackendEnabled).toBe("boolean");
      expect(typeof status.activeConnections).toBe("number");
      expect(typeof status.eventHistorySize).toBe("number");
    });
  });

  describe("executeCommand", () => {
    it("should execute a valid command", async () => {
      const result = await caller.executeCommand({ command: "ls" });

      expect(result.success).toBe(true);
      expect(result.output).toBeDefined();
      expect(typeof result.duration).toBe("number");
      expect(result.duration).toBeGreaterThan(0);
    });

    it("should handle npm commands", async () => {
      const result = await caller.executeCommand({
        command: "npm install",
      });

      expect(result.success).toBe(true);
      expect(result.output).toContain("added");
    });

    it("should handle git commands", async () => {
      const result = await caller.executeCommand({
        command: "git status",
      });

      expect(result.success).toBe(true);
      expect(result.output).toContain("branch");
    });

    it("should reject empty commands", async () => {
      try {
        await caller.executeCommand({ command: "" });
        expect.fail("Should have thrown an error");
      } catch (error) {
        expect(error).toBeDefined();
      }
    });

    it("should reject whitespace-only commands", async () => {
      try {
        await caller.executeCommand({ command: "   " });
        expect.fail("Should have thrown an error");
      } catch (error) {
        expect(error).toBeDefined();
      }
    });
  });

  describe("emitTestEvent", () => {
    it("should emit a random event when no type specified", async () => {
      const result = await caller.emitTestEvent({});

      expect(result.success).toBe(true);
      expect(result.event).toBeDefined();
      expect(result.event.id).toBeDefined();
      expect(result.event.type).toBeDefined();
      expect(result.event.timestamp).toBeDefined();
    });

    it("should emit framework_start event", async () => {
      const result = await caller.emitTestEvent({
        type: "framework_start",
      });

      expect(result.success).toBe(true);
      expect(result.event.type).toBe("framework_start");
      expect(result.event.category).toBe("framework");
    });

    it("should emit llm_call event", async () => {
      const result = await caller.emitTestEvent({
        type: "llm_call",
      });

      expect(result.success).toBe(true);
      expect(result.event.type).toBe("llm_call");
      expect(result.event.category).toBe("llm");
    });

    it("should emit reasoning_step event", async () => {
      const result = await caller.emitTestEvent({
        type: "reasoning_step",
      });

      expect(result.success).toBe(true);
      expect(result.event.type).toBe("reasoning_step");
      expect(result.event.category).toBe("reasoning");
    });

    it("should emit error event", async () => {
      const result = await caller.emitTestEvent({
        type: "error",
      });

      expect(result.success).toBe(true);
      expect(result.event.type).toBe("error");
      expect(result.event.status).toBe("failed");
    });

    it("should emit code_generation event", async () => {
      const result = await caller.emitTestEvent({
        type: "code_generation",
      });

      expect(result.success).toBe(true);
      expect(result.event.type).toBe("code_generation");
      expect(result.event.data).toHaveProperty("language");
      expect(result.event.data).toHaveProperty("code");
    });

    it("should emit token_usage event with cost calculation", async () => {
      const result = await caller.emitTestEvent({
        type: "token_usage",
      });

      expect(result.success).toBe(true);
      expect(result.event.type).toBe("token_usage");
      expect(result.event.data).toHaveProperty("inputTokens");
      expect(result.event.data).toHaveProperty("outputTokens");
      expect(result.event.data).toHaveProperty("cost");
    });

    it("should emit context_gathering event", async () => {
      const result = await caller.emitTestEvent({
        type: "context_gathering",
      });

      expect(result.success).toBe(true);
      expect(result.event.type).toBe("context_gathering");
      expect(result.event.category).toBe("context");
    });

    it("should emit debugging event", async () => {
      const result = await caller.emitTestEvent({
        type: "debugging",
      });

      expect(result.success).toBe(true);
      expect(result.event.type).toBe("debugging");
      expect(result.event.category).toBe("debugging");
    });

    it("should emit framework_end event", async () => {
      const result = await caller.emitTestEvent({
        type: "framework_end",
      });

      expect(result.success).toBe(true);
      expect(result.event.type).toBe("framework_end");
      expect(result.event.status).toBe("completed");
    });
  });

  describe("startMockStream", () => {
    it("should start mock event stream with default interval", async () => {
      const result = await caller.startMockStream({});

      expect(result.success).toBe(true);
      expect(result.message).toContain("started");
      expect(result.message).toContain("2000");
    });

    it("should start mock event stream with custom interval", async () => {
      const result = await caller.startMockStream({ intervalMs: 5000 });

      expect(result.success).toBe(true);
      expect(result.message).toContain("5000");
    });

    it("should reject interval less than 100ms", async () => {
      try {
        await caller.startMockStream({ intervalMs: 50 });
        expect.fail("Should have thrown an error");
      } catch (error) {
        expect(error).toBeDefined();
      }
    });

    it("should reject interval greater than 30000ms", async () => {
      try {
        await caller.startMockStream({ intervalMs: 40000 });
        expect.fail("Should have thrown an error");
      } catch (error) {
        expect(error).toBeDefined();
      }
    });
  });

  describe("stopMockStream", () => {
    it("should stop mock event stream", async () => {
      // Start the stream first
      await caller.startMockStream({});

      // Stop the stream
      const result = await caller.stopMockStream();

      expect(result.success).toBe(true);
      expect(result.message).toContain("stopped");
    });

    it("should handle stopping when stream is not running", async () => {
      const result = await caller.stopMockStream();

      expect(result.success).toBe(true);
    });
  });

  describe("getCommandHistory", () => {
    it("should return command history with default limit", async () => {
      const history = await caller.getCommandHistory({});

      expect(Array.isArray(history)).toBe(true);
    });

    it("should return command history with custom limit", async () => {
      const history = await caller.getCommandHistory({ limit: 10 });

      expect(Array.isArray(history)).toBe(true);
      expect(history.length).toBeLessThanOrEqual(10);
    });
  });

  describe("Event data integrity", () => {
    it("should generate events with unique IDs", async () => {
      const event1 = await caller.emitTestEvent({});
      const event2 = await caller.emitTestEvent({});

      expect(event1.event.id).not.toBe(event2.event.id);
    });

    it("should generate events with valid timestamps", async () => {
      const result = await caller.emitTestEvent({});
      const timestamp = result.event.timestamp;

      expect(timestamp).toBeGreaterThan(0);
      expect(timestamp).toBeLessThanOrEqual(Date.now());
    });

    it("should include metadata in LLM call events", async () => {
      const result = await caller.emitTestEvent({
        type: "llm_call",
      });

      expect(result.event.metadata).toBeDefined();
      expect(result.event.metadata?.model).toBeDefined();
      expect(result.event.metadata?.tokenUsage).toBeDefined();
    });

    it("should include error details in error events", async () => {
      const result = await caller.emitTestEvent({
        type: "error",
      });

      expect(result.event.error).toBeDefined();
      expect(result.event.error?.message).toBeDefined();
      expect(result.event.error?.code).toBeDefined();
    });
  });
});
