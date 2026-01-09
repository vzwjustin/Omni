# Omni Cortex Dashboard (Experimental)

> **⚠️ EXPERIMENTAL FEATURE**
> This dashboard is experimental and may change significantly. It's designed as an optional monitoring tool that can be easily toggled on/off without affecting the core MCP server functionality.

## Overview

The Omni Cortex Dashboard provides real-time monitoring and insights into your reasoning engine's operations. It runs as a separate FastAPI web server alongside the MCP server, giving you visibility into:

- **System Metrics**: Uptime, session counts, token usage, success rates
- **Active Sessions**: Real-time view of ongoing reasoning operations
- **Framework Usage**: Which frameworks are being used and how often
- **Recent Activity**: Historical view of completed sessions
- **Configuration**: Current settings and feature flags

## Architecture

```
┌─────────────────┐         ┌──────────────────┐
│   MCP Server    │────────▶│ Metrics Manager  │
│   (stdio)       │         │  (thread-safe)   │
│   Port: stdio   │         └──────────────────┘
└─────────────────┘                  │
                                     │
                                     ▼
                            ┌──────────────────┐
                            │ Dashboard Server │
                            │  (FastAPI/HTTP)  │
                            │  Port: 8080      │
                            └──────────────────┘
                                     │
                                     ▼
                            ┌──────────────────┐
                            │  Web Browser     │
                            │  (localhost)     │
                            └──────────────────┘
```

### Key Design Principles

1. **Zero Impact When Disabled**: If `ENABLE_DASHBOARD=false`, the metrics manager has zero performance overhead
2. **Separate Process**: Dashboard runs independently from MCP server
3. **Thread-Safe**: Metrics collection is thread-safe for concurrent operations
4. **Non-Breaking**: Can be toggled on/off without affecting MCP server functionality

## Quick Start

### 1. Enable the Dashboard

Add to your `.env` file:

```bash
# Enable experimental dashboard
ENABLE_DASHBOARD=true

# Optional: Configure dashboard settings
DASHBOARD_HOST=127.0.0.1      # Default: localhost only
DASHBOARD_PORT=8080            # Default: 8080
DASHBOARD_METRICS_RETENTION=1000  # Max sessions to retain
```

### 2. Install Dependencies

The dashboard requires FastAPI and Uvicorn:

```bash
pip install fastapi uvicorn
# Or
pip install -r omni_cortex/requirements.txt
```

### 3. Run the Dashboard

**Option A: Standalone Dashboard**

```bash
python -m omni_cortex.server.dashboard_server
```

Then open http://127.0.0.1:8080 in your browser.

**Option B: Run Both Servers**

Terminal 1 - MCP Server:
```bash
python -m omni_cortex.server.main
```

Terminal 2 - Dashboard:
```bash
python -m omni_cortex.server.dashboard_server
```

## Features

### System Metrics

Track overall system health and performance:

- **Uptime**: How long the system has been running
- **Total Sessions**: Cumulative reasoning sessions
- **Active Sessions**: Currently running operations
- **Total Tokens**: Cumulative token consumption
- **Success Rate**: Percentage of successful completions
- **Average Session Time**: Mean duration per session

### Framework Analytics

Understand which reasoning frameworks are most used:

- **Usage Distribution**: Visual bar charts showing framework popularity
- **Success Rates**: Per-framework success percentages
- **Performance**: Average execution time per framework
- **Token Consumption**: Tokens used per framework

### Real-Time Monitoring

Watch your system in action:

- **Active Sessions**: See ongoing reasoning operations
- **Auto-Refresh**: Dashboard updates every 5 seconds
- **Recent Activity**: Historical view of last 20 sessions
- **Error Tracking**: Failed sessions with error details

### Configuration View

Quick reference for current settings:

- **Lean Mode**: Active/inactive status
- **LLM Provider**: Current provider (google, openrouter, etc.)
- **Routing Model**: Active routing model
- **Feature Flags**: Auto-ingest, MCP sampling, etc.

## API Endpoints

The dashboard exposes REST API endpoints you can use programmatically:

### Health Check
```bash
curl http://localhost:8080/api/health
```

### All Metrics
```bash
curl http://localhost:8080/api/metrics
```

### System Metrics Only
```bash
curl http://localhost:8080/api/metrics/system
```

### Framework Metrics
```bash
curl http://localhost:8080/api/metrics/frameworks
```

### Active Sessions
```bash
curl http://localhost:8080/api/sessions/active
```

### Recent Sessions
```bash
curl http://localhost:8080/api/sessions/recent?limit=20
```

### Available Frameworks
```bash
curl http://localhost:8080/api/frameworks
```

### Current Settings
```bash
curl http://localhost:8080/api/settings
```

### Reset Metrics (Admin)
```bash
curl -X POST http://localhost:8080/api/metrics/reset
```

## Configuration

All dashboard settings are in `omni_cortex/app/core/settings.py`:

```python
# Dashboard (Experimental)
enable_dashboard: bool = Field(default=False, alias="ENABLE_DASHBOARD")
dashboard_host: str = Field(default="127.0.0.1", alias="DASHBOARD_HOST")
dashboard_port: int = Field(default=8080, alias="DASHBOARD_PORT")
dashboard_metrics_retention: int = Field(default=1000, alias="DASHBOARD_METRICS_RETENTION")
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_DASHBOARD` | `false` | Enable/disable dashboard |
| `DASHBOARD_HOST` | `127.0.0.1` | Dashboard bind address |
| `DASHBOARD_PORT` | `8080` | Dashboard HTTP port |
| `DASHBOARD_METRICS_RETENTION` | `1000` | Max sessions to retain in memory |

## Integration with Claude Code

The dashboard is designed to work seamlessly with Claude Code:

1. **Automatic Tracking**: Every `reason` tool invocation is automatically tracked
2. **ClaudeCodeBrief Visibility**: See which frameworks Claude is using
3. **Token Monitoring**: Track token consumption across sessions
4. **Performance Insights**: Identify slow or failing operations

### Example Workflow

1. Use Claude Code in your IDE (VS Code, etc.)
2. Invoke Omni Cortex reasoning tools
3. Open dashboard to see real-time metrics
4. Analyze which frameworks work best for your tasks
5. Optimize based on success rates and performance

## Security Considerations

⚠️ **Important Security Notes**:

- **Localhost Only**: Default binding is `127.0.0.1` (localhost only)
- **No Authentication**: Dashboard has no built-in authentication
- **Development Use**: Intended for local development monitoring
- **Production Warning**: Do NOT expose dashboard to public networks

### Production Deployment

If you need to deploy the dashboard in production:

1. Add authentication (JWT, OAuth, etc.)
2. Use HTTPS/TLS encryption
3. Implement rate limiting
4. Add IP whitelisting
5. Consider using a reverse proxy (nginx, traefik)

## Performance Impact

### When Dashboard is Disabled (`ENABLE_DASHBOARD=false`)

- **Zero overhead**: Metrics manager returns immediately
- **No memory usage**: No metrics data collected
- **No performance impact**: Conditional checks are trivial

### When Dashboard is Enabled (`ENABLE_DASHBOARD=true`)

- **Minimal overhead**: ~0.1-0.5ms per session
- **Memory usage**: ~1KB per session × retention limit
- **Thread-safe**: Uses RLock for concurrent access
- **Bounded memory**: Automatically evicts old sessions

## Troubleshooting

### Dashboard Won't Start

**Error**: `Dashboard is disabled`

**Solution**: Set `ENABLE_DASHBOARD=true` in `.env`

```bash
echo "ENABLE_DASHBOARD=true" >> .env
```

**Error**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**: Install dashboard dependencies

```bash
pip install fastapi uvicorn
```

**Error**: `Address already in use`

**Solution**: Change the port or kill the existing process

```bash
# Change port
export DASHBOARD_PORT=8081

# Or kill existing process
lsof -ti:8080 | xargs kill -9
```

### No Metrics Showing

**Issue**: Dashboard shows "No recent sessions"

**Cause**: MCP server hasn't processed any `reason` tool calls yet

**Solution**: Use Claude Code to invoke reasoning, then refresh dashboard

### Metrics Not Updating

**Issue**: Dashboard shows stale data

**Solution**: Check auto-refresh is enabled, or manually click "Refresh" button

## Future Enhancements

Planned improvements for future versions:

- [ ] WebSocket support for real-time streaming
- [ ] Authentication and authorization
- [ ] Custom dashboard themes
- [ ] Export metrics to CSV/JSON
- [ ] Integration with Prometheus/Grafana
- [ ] Per-user metrics tracking
- [ ] Advanced filtering and search
- [ ] Performance profiling tools
- [ ] Alert system for errors/slowness

## Feedback

This is an experimental feature. We'd love your feedback!

- Found a bug? [Open an issue](https://github.com/vzwjustin/Omni/issues)
- Have a feature request? [Start a discussion](https://github.com/vzwjustin/Omni/discussions)
- Want to contribute? [Submit a PR](https://github.com/vzwjustin/Omni/pulls)

## License

Same as Omni Cortex main project.

---

**Last Updated**: 2026-01-09
**Status**: Experimental
**Branch**: `claude/experimental-dashboard-integration-Ze6C5`
