/**
 * Dashboard-specific tRPC procedures for event management and code execution.
 */

import { z } from "zod";
import { publicProcedure, router } from "../_core/trpc";
import { createCodeCommand, getCodeCommandHistory } from "../db";
import { emitMockEvent, startMockEventStream } from "../mockEventEmitter";
import { getWebSocketStatus } from "../websocket";

let mockStreamInterval: (() => void) | null = null;

export const dashboardRouter = router({
  /**
   * Get WebSocket connection status.
   */
  getStatus: publicProcedure.query(() => {
    return getWebSocketStatus();
  }),

  /**
   * Get recent code command history.
   */
  getCommandHistory: publicProcedure
    .input(z.object({ limit: z.number().default(50) }))
    .query(async ({ input }) => {
      return await getCodeCommandHistory(input.limit);
    }),

  /**
   * Execute a Claude Code CLI command.
   * In production, this would execute actual commands.
   * For now, it simulates command execution.
   */
  executeCommand: publicProcedure
    .input(
      z.object({
        command: z.string().min(1, "Command cannot be empty"),
      })
    )
    .mutation(async ({ input, ctx }) => {
      const startTime = Date.now();

      try {
        // Log the command to database
        await createCodeCommand({
          command: input.command,
          status: "running",
          userId: ctx.user?.id,
        });

        // Simulate command execution
        const output = await simulateCommandExecution(input.command);
        const duration = Date.now() - startTime;

        // Log successful execution
        await createCodeCommand({
          command: input.command,
          output: output,
          status: "completed",
          duration: duration,
          userId: ctx.user?.id,
        });

        return {
          success: true,
          output: output,
          duration: duration,
        };
      } catch (error) {
        const duration = Date.now() - startTime;
        const errorMessage = error instanceof Error ? error.message : "Unknown error";

        // Log failed execution
        await createCodeCommand({
          command: input.command,
          error: errorMessage,
          status: "failed",
          duration: duration,
          userId: ctx.user?.id,
        });

        return {
          success: false,
          error: errorMessage,
          duration: duration,
        };
      }
    }),

  /**
   * Emit a single mock event for testing.
   */
  emitTestEvent: publicProcedure
    .input(
      z.object({
        type: z
          .enum([
            "framework_start",
            "reasoning_step",
            "llm_call",
            "context_gathering",
            "code_generation",
            "debugging",
            "token_usage",
            "error",
            "framework_end",
          ])
          .optional(),
      })
    )
    .mutation(({ input }) => {
      const event = emitMockEvent(input.type);
      return {
        success: true,
        event: event,
      };
    }),

  /**
   * Start the mock event stream for testing.
   */
  startMockStream: publicProcedure
    .input(
      z.object({
        intervalMs: z.number().min(100).max(30000).default(2000),
      })
    )
    .mutation(({ input }) => {
      if (mockStreamInterval) {
        mockStreamInterval();
      }

      mockStreamInterval = startMockEventStream(input.intervalMs);

      return {
        success: true,
        message: `Mock event stream started with interval ${input.intervalMs}ms`,
      };
    }),

  /**
   * Stop the mock event stream.
   */
  stopMockStream: publicProcedure.mutation(() => {
    if (mockStreamInterval) {
      mockStreamInterval();
      mockStreamInterval = null;
    }

    return {
      success: true,
      message: "Mock event stream stopped",
    };
  }),
});

/**
 * Simulate command execution.
 * In production, this would execute actual shell commands or Claude Code CLI.
 */
async function simulateCommandExecution(command: string): Promise<string> {
  // Simulate different command types
  if (command.includes("ls") || command.includes("dir")) {
    return `drizzle/
server/
client/
shared/
package.json
tsconfig.json
.env.local`;
  }

  if (command.includes("npm") || command.includes("pnpm")) {
    return `added 15 packages in 2.3s`;
  }

  if (command.includes("git")) {
    return `On branch main
Your branch is up to date with 'origin/main'.
nothing to commit, working tree clean`;
  }

  if (command.includes("node") || command.includes("ts-node")) {
    return `Server running on http://localhost:3000
Connected to database successfully
WebSocket server initialized`;
  }

  if (command.includes("cat") || command.includes("type")) {
    return `{
  "name": "omni_dashboard",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "tsx watch server/_core/index.ts",
    "build": "vite build && esbuild server/_core/index.ts --platform=node --packages=external --bundle --format=esm --outdir=dist"
  }
}`;
  }

  if (command.includes("help")) {
    return `Available commands:
  ls, dir          - List directory contents
  npm, pnpm        - Package manager commands
  git              - Git version control
  node, ts-node    - Run Node.js scripts
  cat, type        - Display file contents
  help             - Show this help message`;
  }

  // Default response for unknown commands
  return `Command executed successfully: ${command}
Output: Command simulation completed`;
}
