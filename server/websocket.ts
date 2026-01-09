/**
 * WebSocket server for real-time event streaming from Omni backend to dashboard.
 * 
 * This module provides:
 * - Event streaming via WebSocket connections
 * - Event buffering and history management
 * - Connection lifecycle management
 * - Safe toggle mechanism for enabling/disabling Omni backend connection
 */

import { Server as HTTPServer } from "http";
import { WebSocketServer, WebSocket } from "ws";
import { EventEmitter } from "events";
import type { AnyOmniEvent, WebSocketMessage, EventStreamMessage, StatusMessage } from "../shared/eventTypes";

/**
 * Configuration for WebSocket server and Omni backend connection.
 * 
 * TOGGLE MECHANISM:
 * To safely enable/disable the Omni backend connection, comment/uncomment the OMNI_BACKEND_ENABLED flag.
 * When disabled, the dashboard will still work but won't receive events from the Omni backend.
 */
const OMNI_BACKEND_ENABLED = true; // Toggle: set to false to disable Omni backend connection
const OMNI_BACKEND_URL = process.env.OMNI_BACKEND_URL || "ws://localhost:8000/events";
const MAX_EVENT_HISTORY = 1000;
const EVENT_BROADCAST_INTERVAL = 100; // ms

/**
 * Event manager for handling real-time event streaming.
 */
class EventManager extends EventEmitter {
  private eventHistory: AnyOmniEvent[] = [];
  private activeConnections: Set<WebSocket> = new Set();
  private omniConnection: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000; // Start with 1 second
  private eventBuffer: AnyOmniEvent[] = [];
  private broadcastTimer: NodeJS.Timeout | null = null;

  constructor() {
    super();
    this.initializeOmniConnection();
  }

  /**
   * Initialize connection to Omni backend.
   * Uses exponential backoff for reconnection attempts.
   */
  private initializeOmniConnection() {
    if (!OMNI_BACKEND_ENABLED) {
      console.log("[WebSocket] Omni backend connection disabled via toggle");
      return;
    }

    try {
      console.log(`[WebSocket] Connecting to Omni backend at ${OMNI_BACKEND_URL}`);
      this.omniConnection = new WebSocket(OMNI_BACKEND_URL);

      this.omniConnection.addEventListener("open", () => {
        console.log("[WebSocket] Connected to Omni backend");
        this.reconnectAttempts = 0;
        this.reconnectDelay = 1000;
        this.broadcastStatus();
      });

      this.omniConnection.addEventListener("message", (event: any) => {
        try {
          const data = JSON.parse(event.data);
          this.handleOmniEvent(data);
        } catch (error) {
          console.error("[WebSocket] Failed to parse Omni event:", error);
        }
      });

      this.omniConnection.addEventListener("error", (error: any) => {
        console.error("[WebSocket] Omni connection error:", error);
      });

      this.omniConnection.addEventListener("close", () => {
        console.log("[WebSocket] Omni connection closed");
        this.omniConnection = null;
        this.attemptReconnect();
      });
    } catch (error) {
      console.error("[WebSocket] Failed to initialize Omni connection:", error);
      this.attemptReconnect();
    }
  }

  /**
   * Attempt to reconnect to Omni backend with exponential backoff.
   */
  private attemptReconnect() {
    if (!OMNI_BACKEND_ENABLED || this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log("[WebSocket] Max reconnection attempts reached or connection disabled");
      return;
    }

    this.reconnectAttempts++;
    const attempt = this.reconnectAttempts;
    const max = this.maxReconnectAttempts;
    const delay = this.reconnectDelay;
    console.log(`[WebSocket] Reconnecting to Omni backend (attempt ${attempt}/${max}) in ${delay}ms`);

    setTimeout(() => {
      this.initializeOmniConnection();
    }, this.reconnectDelay);

    // Exponential backoff: double the delay, max 30 seconds
    this.reconnectDelay = Math.min(this.reconnectDelay * 2, 30000);
  }

  /**
   * Handle events received from Omni backend.
   */
  private handleOmniEvent(eventData: unknown) {
    try {
      const event = eventData as AnyOmniEvent;
      
      // Add to history
      this.eventHistory.unshift(event);
      if (this.eventHistory.length > MAX_EVENT_HISTORY) {
        this.eventHistory.pop();
      }

      // Add to broadcast buffer
      this.eventBuffer.push(event);

      // Emit for any local listeners
      this.emit("event", event);
    } catch (error) {
      console.error("[WebSocket] Failed to process Omni event:", error);
    }
  }

  /**
   * Add a client connection to receive events.
   */
  addConnection(ws: WebSocket) {
    this.activeConnections.add(ws);
    const count = this.activeConnections.size;
    console.log(`[WebSocket] Client connected. Active connections: ${count}`);

    // Send recent event history to new client
    this.eventHistory.slice(0, 50).forEach((event) => {
      const message: EventStreamMessage = {
        type: "event",
        payload: event,
        timestamp: Date.now(),
      };
      ws.send(JSON.stringify(message));
    });

    // Send current status
    this.broadcastStatus();
  }

  /**
   * Remove a client connection.
   */
  removeConnection(ws: WebSocket) {
    this.activeConnections.delete(ws);
    const count = this.activeConnections.size;
    console.log(`[WebSocket] Client disconnected. Active connections: ${count}`);
    this.broadcastStatus();
  }

  /**
   * Broadcast events to all connected clients.
   * Uses batching to reduce network overhead.
   */
  private startBroadcasting() {
    if (this.broadcastTimer) return;

    this.broadcastTimer = setInterval(() => {
      if (this.eventBuffer.length === 0) return;

      const eventsToSend = [...this.eventBuffer];
      this.eventBuffer = [];

      eventsToSend.forEach((event) => {
        const message: EventStreamMessage = {
          type: "event",
          payload: event,
          timestamp: Date.now(),
        };

        this.activeConnections.forEach((ws) => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(message));
          }
        });
      });
    }, EVENT_BROADCAST_INTERVAL);
  }

  /**
   * Stop broadcasting events.
   */
  private stopBroadcasting() {
    if (this.broadcastTimer) {
      clearInterval(this.broadcastTimer);
      this.broadcastTimer = null;
    }
  }

  /**
   * Broadcast status message to all connected clients.
   */
  broadcastStatus() {
    const message: StatusMessage = {
      type: "status",
      payload: {
        connected: this.omniConnection?.readyState === WebSocket.OPEN,
        activeFrameworks: 0, // TODO: calculate from active events
        totalEvents: this.eventHistory.length,
        uptime: process.uptime(),
        lastEventTime: this.eventHistory[0]?.timestamp,
      },
      timestamp: Date.now(),
    };

    this.activeConnections.forEach((ws) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(message));
      }
    });
  }

  /**
   * Emit a synthetic event (for testing or manual triggers).
   */
  emitEvent(event: AnyOmniEvent) {
    this.handleOmniEvent(event);
  }

  /**
   * Get event history.
   */
  getHistory(limit: number = 100): AnyOmniEvent[] {
    return this.eventHistory.slice(0, limit);
  }

  /**
   * Clear event history.
   */
  clearHistory() {
    this.eventHistory = [];
    this.eventBuffer = [];
    this.broadcastStatus();
  }

  /**
   * Get connection status.
   */
  getStatus() {
    return {
      omniConnected: this.omniConnection?.readyState === WebSocket.OPEN,
      omniBackendEnabled: OMNI_BACKEND_ENABLED,
      activeConnections: this.activeConnections.size,
      eventHistorySize: this.eventHistory.length,
      reconnectAttempts: this.reconnectAttempts,
    };
  }
}

// Global event manager instance
let eventManager: EventManager | null = null;

/**
 * Initialize WebSocket server for the dashboard.
 */
export function initializeWebSocketServer(httpServer: HTTPServer) {
  if (eventManager) {
    console.warn("[WebSocket] Server already initialized");
    return eventManager;
  }

  eventManager = new EventManager();

  const wss = new WebSocketServer({ server: httpServer, path: "/api/ws" });

  wss.on("connection", (ws: WebSocket) => {
    console.log("[WebSocket] New client connection");
    eventManager!.addConnection(ws);

    ws.on("message", (data: Buffer) => {
      try {
        const message = JSON.parse(data.toString()) as WebSocketMessage;
        handleClientMessage(ws, message);
      } catch (error) {
        console.error("[WebSocket] Failed to parse client message:", error);
      }
    });

    ws.on("close", () => {
      eventManager!.removeConnection(ws);
    });

    ws.on("error", (error: Error) => {
      console.error("[WebSocket] Client error:", error);
    });
  });

  console.log("[WebSocket] Server initialized on /api/ws");
  return eventManager;
}

/**
 * Handle messages from dashboard clients.
 */
function handleClientMessage(ws: WebSocket, message: WebSocketMessage) {
  if (message.type === "command") {
    // Handle dashboard commands (e.g., filter events, export logs)
    console.log("[WebSocket] Received command:", message);
    // TODO: Implement command handling
  }
}

/**
 * Get the global event manager instance.
 */
export function getEventManager(): EventManager | null {
  return eventManager;
}

/**
 * Emit an event to all connected clients.
 */
export function emitEvent(event: AnyOmniEvent) {
  if (eventManager) {
    eventManager.emitEvent(event);
  }
}

/**
 * Get connection status.
 */
export function getWebSocketStatus() {
  return eventManager?.getStatus() || {
    omniConnected: false,
    omniBackendEnabled: OMNI_BACKEND_ENABLED,
    activeConnections: 0,
    eventHistorySize: 0,
    reconnectAttempts: 0,
  };
}
