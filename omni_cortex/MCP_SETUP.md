# Omni-Cortex MCP Setup Guide

Copy-paste configurations for different IDEs and CLI tools.

## Prerequisites

1. Clone and build Omni-Cortex:
```bash
git clone https://github.com/vzwjustin/Omni.git
cd Omni/omni_cortex
./setup.sh  # Creates .env and builds Docker
```

2. Start the server:
```bash
docker-compose up -d
```

---

## Claude Code (CLI)

Add to `~/.claude/.mcp.json`:

```json
{
  "mcpServers": {
    "omni-cortex": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "--env-file", "/path/to/omni_cortex/.env",
        "-v", "/path/to/omni_cortex/data:/app/data",
        "omni_cortex-omni-cortex:latest"
      ]
    }
  }
}
```

**Replace** `/path/to/omni_cortex` with your actual path.

---

## Cursor

Add to Cursor settings (`Cmd+,` > search "MCP"):

```json
{
  "mcpServers": {
    "omni-cortex": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "--env-file", "/path/to/omni_cortex/.env",
        "-v", "/path/to/omni_cortex/data:/app/data",
        "omni_cortex-omni-cortex:latest"
      ]
    }
  }
}
```

---

## VS Code (Cline Extension)

Add to `.vscode/mcp.json` in your project:

```json
{
  "mcpServers": {
    "omni-cortex": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "--env-file", "${workspaceFolder}/../omni_cortex/.env",
        "-v", "${workspaceFolder}/../omni_cortex/data:/app/data",
        "omni_cortex-omni-cortex:latest"
      ]
    }
  }
}
```

Or add to global Cline settings.

---

## Windsurf

Add to Windsurf MCP configuration:

```json
{
  "mcpServers": {
    "omni-cortex": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "--env-file", "/path/to/omni_cortex/.env",
        "-v", "/path/to/omni_cortex/data:/app/data",
        "omni_cortex-omni-cortex:latest"
      ]
    }
  }
}
```

---

## Generic MCP Client (stdio)

For any MCP client that supports stdio transport:

```bash
docker run --rm -i \
  --env-file /path/to/omni_cortex/.env \
  -v /path/to/omni_cortex/data:/app/data \
  omni_cortex-omni-cortex:latest
```

---

## Without Docker (Local Python)

If you prefer running without Docker:

```json
{
  "mcpServers": {
    "omni-cortex": {
      "command": "python",
      "args": ["-m", "server.main"],
      "cwd": "/path/to/omni_cortex",
      "env": {
        "OPENAI_API_KEY": "your-key",
        "EMBEDDING_PROVIDER": "openai"
      }
    }
  }
}
```

Requires: `pip install -r requirements.txt`

---

## Verify Installation

After configuring, you should see these tools available:

**With API Key (48 tools):**
- 40 `think_*` framework tools
- 7 RAG/search tools
- 1 `reason` (smart router)

**Without API Key (41 tools):**
- 40 `think_*` framework tools
- 1 `reason` (smart router)

---

## Troubleshooting

### "RAG: disabled (no API key)"
- Ensure `.env` file has `OPENAI_API_KEY=sk-...`
- Ensure `.env` file has `EMBEDDING_PROVIDER=openai`
- Restart Docker: `docker-compose down && docker-compose up -d`

### Container not found
- Build first: `docker-compose build`
- Check image: `docker images | grep omni_cortex`

### Permission denied
- Check file paths in MCP config
- Ensure `.env` file exists and is readable
