import { eq, desc } from "drizzle-orm";
import { drizzle } from "drizzle-orm/mysql2";
import { InsertUser, users, events, InsertEvent, codeCommands, InsertCodeCommand } from "../drizzle/schema";
import { ENV } from './_core/env';

let _db: ReturnType<typeof drizzle> | null = null;

// Lazily create the drizzle instance so local tooling can run without a DB.
export async function getDb() {
  if (!_db && process.env.DATABASE_URL) {
    try {
      _db = drizzle(process.env.DATABASE_URL);
    } catch (error) {
      console.warn("[Database] Failed to connect:", error);
      _db = null;
    }
  }
  return _db;
}

export async function upsertUser(user: InsertUser): Promise<void> {
  if (!user.openId) {
    throw new Error("User openId is required for upsert");
  }

  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot upsert user: database not available");
    return;
  }

  try {
    const values: InsertUser = {
      openId: user.openId,
    };
    const updateSet: Record<string, unknown> = {};

    const textFields = ["name", "email", "loginMethod"] as const;
    type TextField = (typeof textFields)[number];

    const assignNullable = (field: TextField) => {
      const value = user[field];
      if (value === undefined) return;
      const normalized = value ?? null;
      values[field] = normalized;
      updateSet[field] = normalized;
    };

    textFields.forEach(assignNullable);

    if (user.lastSignedIn !== undefined) {
      values.lastSignedIn = user.lastSignedIn;
      updateSet.lastSignedIn = user.lastSignedIn;
    }
    if (user.role !== undefined) {
      values.role = user.role;
      updateSet.role = user.role;
    } else if (user.openId === ENV.ownerOpenId) {
      values.role = 'admin';
      updateSet.role = 'admin';
    }

    if (!values.lastSignedIn) {
      values.lastSignedIn = new Date();
    }

    if (Object.keys(updateSet).length === 0) {
      updateSet.lastSignedIn = new Date();
    }

    await db.insert(users).values(values).onDuplicateKeyUpdate({
      set: updateSet,
    });
  } catch (error) {
    console.error("[Database] Failed to upsert user:", error);
    throw error;
  }
}

export async function getUserByOpenId(openId: string) {
  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot get user: database not available");
    return undefined;
  }

  const result = await db.select().from(users).where(eq(users.openId, openId)).limit(1);

  return result.length > 0 ? result[0] : undefined;
}

export async function createEvent(event: InsertEvent) {
  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot create event: database not available");
    return undefined;
  }

  try {
    const result = await db.insert(events).values(event);
    return result;
  } catch (error) {
    console.error("[Database] Failed to create event:", error);
    throw error;
  }
}

export async function getRecentEvents(limit: number = 100) {
  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot get events: database not available");
    return [];
  }

  try {
    const result = await db
      .select()
      .from(events)
      .orderBy((e) => desc(e.createdAt))
      .limit(limit);
    return result;
  } catch (error) {
    console.error("[Database] Failed to get events:", error);
    throw error;
  }
}

export async function createCodeCommand(command: InsertCodeCommand) {
  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot create code command: database not available");
    return undefined;
  }

  try {
    const result = await db.insert(codeCommands).values(command);
    return result;
  } catch (error) {
    console.error("[Database] Failed to create code command:", error);
    throw error;
  }
}

export async function getCodeCommandHistory(limit: number = 50) {
  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot get code commands: database not available");
    return [];
  }

  try {
    const result = await db
      .select()
      .from(codeCommands)
      .orderBy((c) => desc(c.createdAt))
      .limit(limit);
    return result;
  } catch (error) {
    console.error("[Database] Failed to get code commands:", error);
    throw error;
  }
}

// TODO: add more feature queries here as your schema grows.
