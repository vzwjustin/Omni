import { useMemo, useState } from "react";
import { format } from "date-fns";
import { ChevronDown, ChevronUp, AlertCircle, CheckCircle2, Clock, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import type { AnyOmniEvent } from "@shared/eventTypes";

interface EventStreamProps {
  events: AnyOmniEvent[];
  isLoading?: boolean;
}

const EventTypeIcons: Record<string, React.ReactNode> = {
  framework_start: <Zap className="w-4 h-4" />,
  framework_end: <CheckCircle2 className="w-4 h-4" />,
  reasoning_step: <Clock className="w-4 h-4" />,
  llm_call: <Zap className="w-4 h-4" />,
  error: <AlertCircle className="w-4 h-4" />,
};

const EventStatusColors: Record<string, string> = {
  pending: "bg-yellow-500/10 text-yellow-700 border-yellow-200",
  running: "bg-blue-500/10 text-blue-700 border-blue-200",
  completed: "bg-green-500/10 text-green-700 border-green-200",
  failed: "bg-red-500/10 text-red-700 border-red-200",
};

function EventCard({ event }: { event: AnyOmniEvent }) {
  const [expanded, setExpanded] = useState(false);
  const statusColor = EventStatusColors[event.status] || EventStatusColors.pending;

  return (
    <Card className="mb-2 overflow-hidden border border-slate-200 dark:border-slate-700">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full text-left p-3 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors flex items-start gap-3"
      >
        <div className="flex-shrink-0 mt-1">
          {expanded ? (
            <ChevronUp className="w-4 h-4 text-slate-500" />
          ) : (
            <ChevronDown className="w-4 h-4 text-slate-500" />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-slate-500">{EventTypeIcons[event.type] || <Clock className="w-4 h-4" />}</span>
            <h3 className="font-semibold text-sm text-slate-900 dark:text-slate-100 truncate">
              {event.title}
            </h3>
            <span className={`text-xs px-2 py-1 rounded border ${statusColor}`}>
              {event.status}
            </span>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            {format(new Date(event.timestamp), "HH:mm:ss.SSS")} • {event.type}
          </p>
        </div>
      </button>

      {expanded && (
        <div className="border-t border-slate-200 dark:border-slate-700 px-3 py-3 bg-slate-50 dark:bg-slate-900 text-sm">
          {event.description && (
            <div className="mb-3">
              <p className="text-slate-700 dark:text-slate-300">{event.description}</p>
            </div>
          )}

          {Object.keys(event.data).length > 0 && (
            <div className="mb-3">
              <p className="font-semibold text-xs text-slate-600 dark:text-slate-400 mb-2">Data</p>
              <pre className="bg-slate-100 dark:bg-slate-800 p-2 rounded text-xs overflow-x-auto">
                {JSON.stringify(event.data, null, 2)}
              </pre>
            </div>
          )}

          {event.metadata && (
            <div className="mb-3">
              <p className="font-semibold text-xs text-slate-600 dark:text-slate-400 mb-2">Metadata</p>
              <div className="grid grid-cols-2 gap-2 text-xs">
                {event.metadata.tokenUsage && (
                  <div className="bg-slate-100 dark:bg-slate-800 p-2 rounded">
                    <p className="font-semibold text-slate-700 dark:text-slate-300">Tokens</p>
                    <p className="text-slate-600 dark:text-slate-400">
                      In: {event.metadata.tokenUsage.input} | Out: {event.metadata.tokenUsage.output}
                    </p>
                  </div>
                )}
                {event.metadata.model && (
                  <div className="bg-slate-100 dark:bg-slate-800 p-2 rounded">
                    <p className="font-semibold text-slate-700 dark:text-slate-300">Model</p>
                    <p className="text-slate-600 dark:text-slate-400">{event.metadata.model}</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {event.error && (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 p-2 rounded">
              <p className="font-semibold text-xs text-red-700 dark:text-red-400 mb-1">Error</p>
              <p className="text-xs text-red-600 dark:text-red-300">{event.error.message}</p>
              {event.error.stack && (
                <pre className="text-xs mt-2 bg-slate-100 dark:bg-slate-800 p-2 rounded overflow-x-auto">
                  {event.error.stack}
                </pre>
              )}
            </div>
          )}

          {event.duration && (
            <p className="text-xs text-slate-600 dark:text-slate-400">
              Duration: {event.duration}ms
            </p>
          )}
        </div>
      )}
    </Card>
  );
}

export function EventStream({ events, isLoading }: EventStreamProps) {
  const [filter, setFilter] = useState<string | null>(null);

  const filteredEvents = useMemo(() => {
    if (!filter) return events;
    return events.filter((e) => e.type === filter || e.category === filter);
  }, [events, filter]);

  const eventTypes = useMemo(() => {
    const types = new Set(events.map((e) => e.type));
    return Array.from(types).sort();
  }, [events]);

  return (
    <div className="flex flex-col h-full">
      {/* Filter Controls */}
      <div className="flex gap-2 mb-4 pb-4 border-b border-slate-200 dark:border-slate-700 overflow-x-auto">
        <Button
          variant={filter === null ? "default" : "outline"}
          size="sm"
          onClick={() => setFilter(null)}
        >
          All ({events.length})
        </Button>
        {eventTypes.map((type) => {
          const count = events.filter((e) => e.type === type).length;
          return (
            <Button
              key={type}
              variant={filter === type ? "default" : "outline"}
              size="sm"
              onClick={() => setFilter(type)}
            >
              {type} ({count})
            </Button>
          );
        })}
      </div>

      {/* Events List */}
      <div className="flex-1 overflow-y-auto">
        {isLoading && (
          <div className="flex items-center justify-center h-32">
            <div className="text-slate-500 dark:text-slate-400">Loading events...</div>
          </div>
        )}
        {filteredEvents.length === 0 && !isLoading && (
          <div className="flex items-center justify-center h-32">
            <div className="text-slate-500 dark:text-slate-400">No events yet</div>
          </div>
        )}
        {filteredEvents.map((event, idx) => (
          <EventCard key={`${event.id}-${idx}`} event={event} />
        ))}
      </div>
    </div>
  );
}
