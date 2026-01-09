import { useAuth } from "@/_core/hooks/useAuth";
import { useWebSocket } from "@/hooks/useWebSocket";
import { EventStream } from "@/components/EventStream";
import { ConnectionStatus } from "@/components/ConnectionStatus";
import { CodeCLIPanel } from "@/components/CodeCLIPanel";
import { Button } from "@/components/ui/button";
import { Download, LogOut, Play, Square, Zap } from "lucide-react";
import { getLoginUrl } from "@/const";
import { useState, useCallback } from "react";
import { trpc } from "@/lib/trpc";
import { toast } from "sonner";

export default function Home() {
  const { user, loading, isAuthenticated, logout } = useAuth();
  const { connected, events, status, clearEvents } = useWebSocket();
  const [showCodeCLI, setShowCodeCLI] = useState(false);
  const [mockStreamRunning, setMockStreamRunning] = useState(false);

  const emitTestEventMutation = trpc.dashboard.emitTestEvent.useMutation();
  const startMockStreamMutation = trpc.dashboard.startMockStream.useMutation();
  const stopMockStreamMutation = trpc.dashboard.stopMockStream.useMutation();

  const handleExportLogs = useCallback(() => {
    const data = {
      timestamp: new Date().toISOString(),
      events: events,
      status: status,
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `omni-events-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Events exported successfully");
  }, [events, status]);

  const handleEmitTestEvent = async (
    type?: 
      | "framework_start"
      | "reasoning_step"
      | "llm_call"
      | "context_gathering"
      | "code_generation"
      | "debugging"
      | "token_usage"
      | "error"
      | "framework_end"
  ) => {
    try {
      await emitTestEventMutation.mutateAsync({ type });
      toast.success(`Test event emitted: ${type || "random"}`);
    } catch (error) {
      toast.error("Failed to emit test event");
    }
  };

  const handleStartMockStream = async () => {
    try {
      await startMockStreamMutation.mutateAsync({ intervalMs: 2000 });
      setMockStreamRunning(true);
      toast.success("Mock event stream started");
    } catch (error) {
      toast.error("Failed to start mock stream");
    }
  };

  const handleStopMockStream = async () => {
    try {
      await stopMockStreamMutation.mutateAsync();
      setMockStreamRunning(false);
      toast.success("Mock event stream stopped");
    } catch (error) {
      toast.error("Failed to stop mock stream");
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p>Loading dashboard...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-100 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-3xl font-bold mb-4">Omni Real-Time Dashboard</h1>
          <p className="text-slate-400 mb-6">Monitor and control Omni system operations</p>
          <Button
            onClick={() => (window.location.href = getLoginUrl())}
            size="lg"
            className="bg-blue-600 hover:bg-blue-700"
          >
            Sign In
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Header */}
      <div className="border-b border-slate-800 bg-slate-900/50 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Omni Dashboard</h1>
            <p className="text-sm text-slate-400">Real-time system monitoring</p>
          </div>
          <div className="flex items-center gap-2 flex-wrap justify-end">
            <Button
              onClick={handleExportLogs}
              variant="outline"
              size="sm"
              className="gap-1"
            >
              <Download className="w-4 h-4" />
              <span className="hidden sm:inline">Export</span>
            </Button>
            <Button
              onClick={() => setShowCodeCLI(!showCodeCLI)}
              variant="outline"
              size="sm"
            >
              {showCodeCLI ? "Hide" : "Show"} CLI
            </Button>
            <Button
              onClick={logout}
              variant="outline"
              size="sm"
              className="gap-1"
            >
              <LogOut className="w-4 h-4" />
              <span className="hidden sm:inline">Sign Out</span>
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar */}
          <div className="lg:col-span-1 space-y-4">
            <ConnectionStatus
              status={status}
              connected={connected}
              omniBackendEnabled={true}
            />

            {/* Testing Controls */}
            <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
              <h3 className="font-semibold text-sm mb-3 text-slate-100">
                Test Controls
              </h3>
              <div className="space-y-2">
                <Button
                  onClick={() => handleEmitTestEvent()}
                  variant="outline"
                  size="sm"
                  className="w-full justify-start gap-2 text-xs"
                  disabled={emitTestEventMutation.isPending}
                >
                  <Zap className="w-3 h-3" />
                  Emit Random Event
                </Button>

                <div className="grid grid-cols-2 gap-1">
                  <Button
                    onClick={() => handleEmitTestEvent("framework_start")}
                    variant="outline"
                    size="sm"
                    className="text-xs"
                    disabled={emitTestEventMutation.isPending}
                  >
                    Framework
                  </Button>
                  <Button
                    onClick={() => handleEmitTestEvent("llm_call")}
                    variant="outline"
                    size="sm"
                    className="text-xs"
                    disabled={emitTestEventMutation.isPending}
                  >
                    LLM Call
                  </Button>
                  <Button
                    onClick={() => handleEmitTestEvent("reasoning_step")}
                    variant="outline"
                    size="sm"
                    className="text-xs"
                    disabled={emitTestEventMutation.isPending}
                  >
                    Reasoning
                  </Button>
                  <Button
                    onClick={() => handleEmitTestEvent("error")}
                    variant="outline"
                    size="sm"
                    className="text-xs"
                    disabled={emitTestEventMutation.isPending}
                  >
                    Error
                  </Button>
                </div>

                <div className="border-t border-slate-700 pt-2 mt-2">
                  <Button
                    onClick={
                      mockStreamRunning
                        ? handleStopMockStream
                        : handleStartMockStream
                    }
                    variant={mockStreamRunning ? "destructive" : "default"}
                    size="sm"
                    className="w-full gap-2"
                    disabled={
                      startMockStreamMutation.isPending ||
                      stopMockStreamMutation.isPending
                    }
                  >
                    {mockStreamRunning ? (
                      <>
                        <Square className="w-3 h-3" />
                        Stop Stream
                      </>
                    ) : (
                      <>
                        <Play className="w-3 h-3" />
                        Start Stream
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </div>
          </div>

          {/* Main Content Area */}
          <div className="lg:col-span-3 space-y-6">
            {/* Event Stream */}
            <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 h-96">
              <div className="flex items-center justify-between mb-4">
                <h2 className="font-semibold">Event Stream</h2>
                <Button
                  onClick={clearEvents}
                  variant="outline"
                  size="sm"
                  className="text-xs"
                >
                  Clear
                </Button>
              </div>
              <EventStream events={events} isLoading={!connected} />
            </div>

            {/* Code CLI Panel */}
            {showCodeCLI && (
              <div className="bg-slate-900 border border-slate-800 rounded-lg overflow-hidden h-80">
                <CodeCLIPanel isExecuting={false} />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
