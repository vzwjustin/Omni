import { int, mysqlEnum, mysqlTable, text, timestamp, varchar } from "drizzle-orm/mysql-core";

/**
 * Core user table backing auth flow.
 * Extend this file with additional tables as your product grows.
 * Columns use camelCase to match both database fields and generated types.
 */
export const users = mysqlTable("users", {
  /**
   * Surrogate primary key. Auto-incremented numeric value managed by the database.
   * Use this for relations between tables.
   */
  id: int("id").autoincrement().primaryKey(),
  /** Manus OAuth identifier (openId) returned from the OAuth callback. Unique per user. */
  openId: varchar("openId", { length: 64 }).notNull().unique(),
  name: text("name"),
  email: varchar("email", { length: 320 }),
  loginMethod: varchar("loginMethod", { length: 64 }),
  role: mysqlEnum("role", ["user", "admin"]).default("user").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
  lastSignedIn: timestamp("lastSignedIn").defaultNow().notNull(),
});

export type User = typeof users.$inferSelect;
export type InsertUser = typeof users.$inferInsert;

/**
 * Event log table for tracking Omni system operations.
 * Stores real-time events from framework execution, reasoning, and LLM calls.
 */
export const events = mysqlTable("events", {
  id: int("id").autoincrement().primaryKey(),
  /** Event type: framework, reasoning, llm_call, context, code_generation, debugging */
  type: varchar("type", { length: 64 }).notNull(),
  /** Event category for filtering and organization */
  category: varchar("category", { length: 64 }).notNull(),
  /** Event title/summary */
  title: text("title").notNull(),
  /** Detailed event data as JSON */
  data: text("data").notNull(), // JSON stringified
  /** Metadata including timestamps, duration, token usage */
  metadata: text("metadata"), // JSON stringified
  /** Status: pending, running, completed, failed */
  status: mysqlEnum("status", ["pending", "running", "completed", "failed"]).default("pending").notNull(),
  /** User ID who triggered this event */
  userId: int("userId"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type Event = typeof events.$inferSelect;
export type InsertEvent = typeof events.$inferInsert;

/**
 * Claude Code CLI command execution log.
 * Tracks all commands executed through the dashboard.
 */
export const codeCommands = mysqlTable("codeCommands", {
  id: int("id").autoincrement().primaryKey(),
  /** The command that was executed */
  command: text("command").notNull(),
  /** Command output/result */
  output: text("output"),
  /** Error message if command failed */
  error: text("error"),
  /** Execution status */
  status: mysqlEnum("status", ["pending", "running", "completed", "failed"]).default("pending").notNull(),
  /** Execution duration in milliseconds */
  duration: int("duration"),
  /** User ID who executed the command */
  userId: int("userId"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type CodeCommand = typeof codeCommands.$inferSelect;
export type InsertCodeCommand = typeof codeCommands.$inferInsert;