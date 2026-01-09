import { useState, useRef, useEffect } from "react";
import { Send, Trash2, Copy, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
interface CodeCommand {
  id?: number;
  command: string;
  output?: string;
  error?: string;
  status: "pending" | "running" | "completed" | "failed";
  duration?: number;
  createdAt?: Date;
}

interface CodeCLIPanelProps {
  history?: CodeCommand[];
  onExecute?: (command: string) => Promise<string>;
  isExecuting?: boolean;
}

export function CodeCLIPanel({
  history = [],
  onExecute,
  isExecuting = false,
}: CodeCLIPanelProps) {
  const [command, setCommand] = useState("");
  const [output, setOutput] = useState<string>("");
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const outputRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [output]);

  const handleExecute = async () => {
    if (!command.trim()) return;

    try {
      const result = await onExecute?.(command);
      setOutput((prev) => `${prev}${prev ? "\n" : ""}$ ${command}\n${result || ""}`);
      setCommand("");
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : "Unknown error";
      setOutput((prev) => `${prev}${prev ? "\n" : ""}$ ${command}\nError: ${errorMsg}`);
    }
  };

  const handleCopyOutput = (index: number) => {
    const item = history[index];
    if (item) {
      navigator.clipboard.writeText(item.output || "");
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 2000);
    }
  };

  const handleClearOutput = () => {
    setOutput("");
  };

  return (
    <Card className="flex flex-col h-full border border-slate-200 dark:border-slate-700">
      <div className="p-4 border-b border-slate-200 dark:border-slate-700">
        <h3 className="font-semibold text-sm text-slate-900 dark:text-slate-100">
          Claude Code CLI
        </h3>
        <p className="text-xs text-slate-600 dark:text-slate-400 mt-1">
          Execute commands and see real-time output
        </p>
      </div>

      {/* Output Display */}
      <div
        ref={outputRef}
        className="flex-1 overflow-y-auto bg-slate-900 text-slate-100 p-4 font-mono text-sm"
      >
        {output ? (
          <pre className="whitespace-pre-wrap break-words">{output}</pre>
        ) : (
          <div className="text-slate-500">Ready for commands...</div>
        )}
      </div>

      {/* Command History */}
      {history.length > 0 && (
        <div className="border-t border-slate-200 dark:border-slate-700 p-3 max-h-24 overflow-y-auto">
          <p className="text-xs font-semibold text-slate-600 dark:text-slate-400 mb-2">
            Recent Commands
          </p>
          <div className="space-y-1">
            {history.slice(-5).map((item, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between gap-2 p-2 bg-slate-50 dark:bg-slate-800 rounded text-xs"
              >
                <code className="text-slate-700 dark:text-slate-300 truncate">
                  {item.command}
                </code>
                <button
                  onClick={() => handleCopyOutput(idx)}
                  className="text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
                  title="Copy output"
                >
                  {copiedIndex === idx ? (
                    <Check className="w-3 h-3" />
                  ) : (
                    <Copy className="w-3 h-3" />
                  )}
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Input Area */}
      <div className="border-t border-slate-200 dark:border-slate-700 p-3 space-y-2">
        <div className="flex gap-2">
          <Input
            value={command}
            onChange={(e) => setCommand(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleExecute();
              }
            }}
            placeholder="Enter command... (Shift+Enter for new line)"
            className="text-sm"
            disabled={isExecuting}
          />
          <Button
            onClick={handleExecute}
            disabled={isExecuting || !command.trim()}
            size="sm"
            className="gap-1"
          >
            <Send className="w-3 h-3" />
            <span className="hidden sm:inline">Execute</span>
          </Button>
          <Button
            onClick={handleClearOutput}
            variant="outline"
            size="sm"
            className="gap-1"
          >
            <Trash2 className="w-3 h-3" />
            <span className="hidden sm:inline">Clear</span>
          </Button>
        </div>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Press Enter to execute, Shift+Enter for new line
        </p>
      </div>
    </Card>
  );
}
