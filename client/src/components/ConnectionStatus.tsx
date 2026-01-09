import { Activity, AlertCircle, CheckCircle2, WifiOff } from "lucide-react";
import { Card } from "@/components/ui/card";
import type { WebSocketStatus } from "@/hooks/useWebSocket";

interface ConnectionStatusProps {
  status: WebSocketStatus;
  connected: boolean;
  omniBackendEnabled: boolean;
}

export function ConnectionStatus({
  status,
  connected,
  omniBackendEnabled,
}: ConnectionStatusProps) {
  const getStatusColor = () => {
    if (!omniBackendEnabled) return "bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300";
    if (connected) return "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400";
    return "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400";
  };

  const getStatusIcon = () => {
    if (!omniBackendEnabled) return <AlertCircle className="w-4 h-4" />;
    if (connected) return <CheckCircle2 className="w-4 h-4" />;
    return <WifiOff className="w-4 h-4" />;
  };

  const getStatusText = () => {
    if (!omniBackendEnabled) return "Backend Disabled";
    if (connected) return "Connected";
    return "Disconnected";
  };

  return (
    <Card className="p-4 border border-slate-200 dark:border-slate-700">
      <div className="space-y-3">
        {/* Connection Status */}
        <div className={`flex items-center gap-2 p-2 rounded ${getStatusColor()}`}>
          {getStatusIcon()}
          <span className="font-semibold text-sm">{getStatusText()}</span>
        </div>

        {/* Status Details */}
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="bg-slate-50 dark:bg-slate-800 p-2 rounded">
            <p className="text-slate-600 dark:text-slate-400">Total Events</p>
            <p className="font-semibold text-slate-900 dark:text-slate-100">
              {status.totalEvents}
            </p>
          </div>

          <div className="bg-slate-50 dark:bg-slate-800 p-2 rounded">
            <p className="text-slate-600 dark:text-slate-400">Active Frameworks</p>
            <p className="font-semibold text-slate-900 dark:text-slate-100">
              {status.activeFrameworks}
            </p>
          </div>

          <div className="bg-slate-50 dark:bg-slate-800 p-2 rounded">
            <p className="text-slate-600 dark:text-slate-400">Uptime</p>
            <p className="font-semibold text-slate-900 dark:text-slate-100">
              {formatUptime(status.uptime)}
            </p>
          </div>

          <div className="bg-slate-50 dark:bg-slate-800 p-2 rounded">
            <p className="text-slate-600 dark:text-slate-400">Last Event</p>
            <p className="font-semibold text-slate-900 dark:text-slate-100">
              {status.lastEventTime
                ? formatTimeSince(status.lastEventTime)
                : "Never"}
            </p>
          </div>
        </div>

        {/* Backend Status Info */}
        {!omniBackendEnabled && (
          <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 p-2 rounded text-xs">
            <p className="text-yellow-800 dark:text-yellow-300">
              <strong>Note:</strong> Omni backend connection is disabled. To enable it, uncomment the toggle in <code className="bg-yellow-100 dark:bg-yellow-900 px-1 rounded">server/websocket.ts</code>.
            </p>
          </div>
        )}
      </div>
    </Card>
  );
}

function formatUptime(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  }
  return `${secs}s`;
}

function formatTimeSince(timestamp: number): string {
  const now = Date.now();
  const diff = now - timestamp;
  const seconds = Math.floor(diff / 1000);

  if (seconds < 60) {
    return `${seconds}s ago`;
  }
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m ago`;
  }
  const hours = Math.floor(minutes / 60);
  return `${hours}h ago`;
}
