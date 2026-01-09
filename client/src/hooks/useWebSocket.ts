import { useEffect, useRef, useState, useCallback } from "react";
import type { AnyOmniEvent, StatusMessage, WebSocketMessage } from "@shared/eventTypes";

export interface WebSocketStatus {
  connected: boolean;
  activeFrameworks: number;
  totalEvents: number;
  uptime: number;
  lastEventTime?: number;
}

interface UseWebSocketOptions {
  url?: string;
  onEvent?: (event: AnyOmniEvent) => void;
  onStatus?: (status: WebSocketStatus) => void;
  onError?: (error: Error) => void;
  autoConnect?: boolean;
}

/**
 * Hook for managing WebSocket connection to the dashboard server.
 * Handles real-time event streaming and connection lifecycle.
 */
export function useWebSocket({
  url = `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/api/ws`,
  onEvent,
  onStatus,
  onError,
  autoConnect = true,
}: UseWebSocketOptions = {}) {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [events, setEvents] = useState<AnyOmniEvent[]>([]);
  const [status, setStatus] = useState<WebSocketStatus>({
    connected: false,
    activeFrameworks: 0,
    totalEvents: 0,
    uptime: 0,
  });
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 10;

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      console.log("[WebSocket Client] Connecting to", url);
      const ws = new WebSocket(url);

      ws.addEventListener("open", () => {
        console.log("[WebSocket Client] Connected");
        setConnected(true);
        reconnectAttemptsRef.current = 0;
      });

      ws.addEventListener("message", (event) => {
        try {
          const message = JSON.parse(event.data) as WebSocketMessage;

          if (message.type === "event") {
            const eventMsg = message as any;
            const omniEvent = eventMsg.payload as AnyOmniEvent;
            setEvents((prev) => [omniEvent, ...prev].slice(0, 500));
            onEvent?.(omniEvent);
          } else if (message.type === "status") {
            const statusMsg = message as StatusMessage;
            setStatus(statusMsg.payload);
            onStatus?.(statusMsg.payload);
          }
        } catch (error) {
          console.error("[WebSocket Client] Failed to parse message:", error);
        }
      });

      ws.addEventListener("error", (error) => {
        console.error("[WebSocket Client] Error:", error);
        const err = new Error("WebSocket error");
        onError?.(err);
      });

      ws.addEventListener("close", () => {
        console.log("[WebSocket Client] Disconnected");
        setConnected(false);
        wsRef.current = null;
        attemptReconnect();
      });

      wsRef.current = ws;
    } catch (error) {
      console.error("[WebSocket Client] Failed to connect:", error);
      const err = error instanceof Error ? error : new Error("Connection failed");
      onError?.(err);
      attemptReconnect();
    }
  }, [url, onEvent, onStatus, onError]);

  const attemptReconnect = useCallback(() => {
    if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
      console.log("[WebSocket Client] Max reconnection attempts reached");
      return;
    }

    reconnectAttemptsRef.current++;
    const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current - 1), 30000);
    const attempt = reconnectAttemptsRef.current;
    console.log(`[WebSocket Client] Reconnecting in ${delay}ms (attempt ${attempt}/${maxReconnectAttempts})`);

    reconnectTimeoutRef.current = setTimeout(() => {
      connect();
    }, delay);
  }, [connect]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnected(false);
  }, []);

  const sendMessage = useCallback((message: WebSocketMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn("[WebSocket Client] Not connected, cannot send message");
    }
  }, []);

  const clearEvents = useCallback(() => {
    setEvents([]);
  }, []);

  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  return {
    connected,
    events,
    status,
    connect,
    disconnect,
    sendMessage,
    clearEvents,
  };
}
