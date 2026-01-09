"""
Experimental Dashboard Server for Omni Cortex

Optional FastAPI-based web dashboard for monitoring and managing Omni Cortex.
Runs independently from the MCP server on a separate port.

To enable:
    Set ENABLE_DASHBOARD=true in your .env file

To run:
    python -m omni_cortex.server.dashboard_server
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError:
    print("ERROR: FastAPI not installed. Install with: pip install fastapi uvicorn")
    sys.exit(1)

from omni_cortex.app.core.settings import get_settings
from omni_cortex.app.core.metrics_manager import get_metrics_manager
from omni_cortex.app.frameworks.registry import FRAMEWORKS


# Initialize
settings = get_settings()
metrics = get_metrics_manager()

# Create FastAPI app
app = FastAPI(
    title="Omni Cortex Dashboard",
    description="Experimental monitoring dashboard for Omni Cortex MCP Server",
    version="0.1.0-experimental"
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root() -> HTMLResponse:
    """Serve the dashboard HTML."""
    html = get_dashboard_html()
    return HTMLResponse(content=html)


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "omni-cortex-dashboard",
        "dashboard_enabled": settings.enable_dashboard,
        "mcp_server_status": "running"
    }


@app.get("/api/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get all metrics data."""
    if not settings.enable_dashboard:
        raise HTTPException(
            status_code=503,
            detail="Dashboard is disabled. Set ENABLE_DASHBOARD=true to enable."
        )

    return metrics.get_dashboard_data()


@app.get("/api/metrics/system")
async def get_system_metrics() -> Dict[str, Any]:
    """Get system-level metrics."""
    if not settings.enable_dashboard:
        raise HTTPException(status_code=503, detail="Dashboard disabled")

    return metrics.get_system_metrics()


@app.get("/api/metrics/frameworks")
async def get_framework_metrics() -> Dict[str, Any]:
    """Get framework usage metrics."""
    if not settings.enable_dashboard:
        raise HTTPException(status_code=503, detail="Dashboard disabled")

    return {
        "frameworks": metrics.get_framework_metrics()
    }


@app.get("/api/sessions/active")
async def get_active_sessions() -> Dict[str, Any]:
    """Get currently active sessions."""
    if not settings.enable_dashboard:
        raise HTTPException(status_code=503, detail="Dashboard disabled")

    return {
        "sessions": metrics.get_active_sessions()
    }


@app.get("/api/sessions/recent")
async def get_recent_sessions(limit: int = 20) -> Dict[str, Any]:
    """Get recently completed sessions."""
    if not settings.enable_dashboard:
        raise HTTPException(status_code=503, detail="Dashboard disabled")

    return {
        "sessions": metrics.get_recent_sessions(limit=min(limit, 100))
    }


@app.get("/api/frameworks")
async def get_frameworks() -> Dict[str, Any]:
    """Get all available frameworks."""
    return {
        "total": len(FRAMEWORKS),
        "frameworks": [
            {
                "id": fw.id,
                "name": fw.name,
                "category": fw.category,
                "description": fw.description,
                "ideal_for": fw.ideal_for,
                "complexity": fw.complexity
            }
            for fw in FRAMEWORKS.values()
        ]
    }


@app.get("/api/settings")
async def get_settings_info() -> Dict[str, Any]:
    """Get current configuration settings."""
    return {
        "lean_mode": settings.lean_mode,
        "enable_dashboard": settings.enable_dashboard,
        "enable_auto_ingest": settings.enable_auto_ingest,
        "enable_mcp_sampling": settings.enable_mcp_sampling,
        "llm_provider": settings.llm_provider,
        "routing_model": settings.routing_model,
        "max_reasoning_depth": settings.max_reasoning_depth,
        "log_level": settings.log_level
    }


@app.post("/api/metrics/reset")
async def reset_metrics() -> Dict[str, str]:
    """Reset all metrics (admin endpoint)."""
    if not settings.enable_dashboard:
        raise HTTPException(status_code=503, detail="Dashboard disabled")

    metrics.reset()
    return {"status": "success", "message": "Metrics reset successfully"}


# ============================================================================
# Dashboard HTML
# ============================================================================

def get_dashboard_html() -> str:
    """Generate the dashboard HTML."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Omni Cortex Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #0f0f1e;
            color: #e0e0e0;
            line-height: 1.6;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            color: white;
        }

        .header .subtitle {
            font-size: 1rem;
            opacity: 0.9;
            color: white;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .card {
            background: #1a1a2e;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            border: 1px solid #2a2a3e;
        }

        .card h2 {
            font-size: 1.25rem;
            margin-bottom: 1rem;
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 0.5rem;
        }

        .metric {
            display: flex;
            justify-content: space-between;
            padding: 0.75rem 0;
            border-bottom: 1px solid #2a2a3e;
        }

        .metric:last-child {
            border-bottom: none;
        }

        .metric-label {
            color: #a0a0a0;
            font-weight: 500;
        }

        .metric-value {
            color: #ffffff;
            font-weight: 600;
            font-size: 1.1rem;
        }

        .status {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .status.running { background: #4ade80; color: #0f172a; }
        .status.completed { background: #3b82f6; color: white; }
        .status.failed { background: #ef4444; color: white; }
        .status.healthy { background: #10b981; color: white; }

        .table-container {
            overflow-x: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }

        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #2a2a3e;
        }

        th {
            background: #667eea;
            color: white;
            font-weight: 600;
        }

        tr:hover {
            background: #2a2a3e;
        }

        .loading {
            text-align: center;
            padding: 2rem;
            color: #a0a0a0;
        }

        .error {
            background: #ef4444;
            color: white;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }

        .refresh-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            transition: background 0.3s;
        }

        .refresh-btn:hover {
            background: #764ba2;
        }

        .auto-refresh {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            margin-left: 1rem;
            color: #a0a0a0;
        }

        .framework-bar {
            background: #2a2a3e;
            height: 24px;
            border-radius: 4px;
            overflow: hidden;
            margin: 0.5rem 0;
        }

        .framework-bar-fill {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            height: 100%;
            display: flex;
            align-items: center;
            padding: 0 0.5rem;
            color: white;
            font-size: 0.75rem;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Omni Cortex Dashboard</h1>
        <p class="subtitle">Experimental Dashboard - Real-time Monitoring</p>
    </div>

    <div class="container">
        <button class="refresh-btn" onclick="loadDashboard()">↻ Refresh</button>
        <span class="auto-refresh">
            <input type="checkbox" id="autoRefresh" checked>
            <label for="autoRefresh">Auto-refresh (5s)</label>
        </span>

        <div id="error" style="display: none;" class="error"></div>

        <div class="grid">
            <div class="card">
                <h2>System Status</h2>
                <div id="systemMetrics" class="loading">Loading...</div>
            </div>

            <div class="card">
                <h2>Active Sessions</h2>
                <div id="activeSessions" class="loading">Loading...</div>
            </div>

            <div class="card">
                <h2>Configuration</h2>
                <div id="settings" class="loading">Loading...</div>
            </div>
        </div>

        <div class="card">
            <h2>Framework Usage</h2>
            <div id="frameworks" class="loading">Loading...</div>
        </div>

        <div class="card">
            <h2>Recent Sessions</h2>
            <div id="recentSessions" class="loading">Loading...</div>
        </div>
    </div>

    <script>
        let autoRefreshInterval = null;

        async function loadDashboard() {
            try {
                const response = await fetch('/api/metrics');
                const data = await response.json();

                if (!data.enabled) {
                    showError(data.message || 'Dashboard is disabled');
                    return;
                }

                hideError();
                renderSystemMetrics(data.system);
                renderActiveSessions(data.active_sessions);
                renderSettings(data.settings);
                renderFrameworks(data.frameworks);
                renderRecentSessions(data.recent_sessions);
            } catch (error) {
                showError(`Failed to load dashboard: ${error.message}`);
            }
        }

        function renderSystemMetrics(system) {
            const html = `
                <div class="metric">
                    <span class="metric-label">Status</span>
                    <span class="status healthy">Healthy</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Uptime</span>
                    <span class="metric-value">${system.uptime_formatted}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Sessions</span>
                    <span class="metric-value">${system.total_sessions}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Active Sessions</span>
                    <span class="metric-value">${system.active_sessions}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Tokens</span>
                    <span class="metric-value">${system.total_tokens.toLocaleString()}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Success Rate</span>
                    <span class="metric-value">${system.success_rate}%</span>
                </div>
            `;
            document.getElementById('systemMetrics').innerHTML = html;
        }

        function renderActiveSessions(sessions) {
            if (sessions.length === 0) {
                document.getElementById('activeSessions').innerHTML = '<p style="color: #a0a0a0;">No active sessions</p>';
                return;
            }

            const html = sessions.map(s => `
                <div class="metric">
                    <div>
                        <div class="metric-label">${s.framework}</div>
                        <div style="font-size: 0.85rem; color: #a0a0a0; margin-top: 0.25rem;">${s.query}</div>
                    </div>
                    <span class="status running">Running</span>
                </div>
            `).join('');
            document.getElementById('activeSessions').innerHTML = html;
        }

        function renderSettings(settings) {
            const html = `
                <div class="metric">
                    <span class="metric-label">Lean Mode</span>
                    <span class="metric-value">${settings.lean_mode ? '✓ Enabled' : '✗ Disabled'}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">LLM Provider</span>
                    <span class="metric-value">${settings.llm_provider}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Routing Model</span>
                    <span class="metric-value">${settings.routing_model}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Auto Ingest</span>
                    <span class="metric-value">${settings.enable_auto_ingest ? '✓ Enabled' : '✗ Disabled'}</span>
                </div>
            `;
            document.getElementById('settings').innerHTML = html;
        }

        function renderFrameworks(frameworks) {
            if (frameworks.length === 0) {
                document.getElementById('frameworks').innerHTML = '<p style="color: #a0a0a0;">No frameworks used yet</p>';
                return;
            }

            frameworks.sort((a, b) => b.total_invocations - a.total_invocations);
            const maxInvocations = frameworks[0]?.total_invocations || 1;

            const html = frameworks.slice(0, 10).map(fw => {
                const percentage = (fw.total_invocations / maxInvocations) * 100;
                return `
                    <div style="margin-bottom: 1.5rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                            <span style="font-weight: 600;">${fw.name}</span>
                            <span style="color: #a0a0a0;">${fw.total_invocations} uses</span>
                        </div>
                        <div class="framework-bar">
                            <div class="framework-bar-fill" style="width: ${percentage}%;">
                                ${fw.success_rate}% success | ${fw.avg_time_ms}ms avg
                            </div>
                        </div>
                    </div>
                `;
            }).join('');

            document.getElementById('frameworks').innerHTML = html;
        }

        function renderRecentSessions(sessions) {
            if (sessions.length === 0) {
                document.getElementById('recentSessions').innerHTML = '<p style="color: #a0a0a0;">No recent sessions</p>';
                return;
            }

            const html = `
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Framework</th>
                                <th>Query</th>
                                <th>Status</th>
                                <th>Duration</th>
                                <th>Tokens</th>
                                <th>Completed</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${sessions.map(s => `
                                <tr>
                                    <td>${s.framework}</td>
                                    <td>${s.query}</td>
                                    <td><span class="status ${s.status}">${s.status}</span></td>
                                    <td>${s.duration_ms ? Math.round(s.duration_ms) + 'ms' : '-'}</td>
                                    <td>${s.tokens_used.toLocaleString()}</td>
                                    <td>${new Date(s.completed_at).toLocaleTimeString()}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            `;
            document.getElementById('recentSessions').innerHTML = html;
        }

        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
        }

        function hideError() {
            document.getElementById('error').style.display = 'none';
        }

        function toggleAutoRefresh() {
            const checkbox = document.getElementById('autoRefresh');
            if (checkbox.checked) {
                autoRefreshInterval = setInterval(loadDashboard, 5000);
            } else {
                clearInterval(autoRefreshInterval);
            }
        }

        document.getElementById('autoRefresh').addEventListener('change', toggleAutoRefresh);

        // Initial load
        loadDashboard();
        toggleAutoRefresh();
    </script>
</body>
</html>
    """


# ============================================================================
# Main Entry Point
# ============================================================================

def run_dashboard():
    """Run the dashboard server."""
    if not settings.enable_dashboard:
        print("❌ Dashboard is disabled")
        print("   Set ENABLE_DASHBOARD=true in your .env file to enable")
        sys.exit(1)

    print("🚀 Starting Omni Cortex Dashboard...")
    print(f"   Host: {settings.dashboard_host}")
    print(f"   Port: {settings.dashboard_port}")
    print(f"   URL: http://{settings.dashboard_host}:{settings.dashboard_port}")
    print("\n⚠️  EXPERIMENTAL: This dashboard is experimental and may change")
    print("   It runs independently from the MCP server")
    print("\n✨ Dashboard ready! Open the URL above in your browser\n")

    uvicorn.run(
        app,
        host=settings.dashboard_host,
        port=settings.dashboard_port,
        log_level="info"
    )


if __name__ == "__main__":
    run_dashboard()
