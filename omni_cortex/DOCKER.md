# Docker Deployment Guide

Omni-Cortex runs as a **stdio-based MCP server** in Docker, designed for use with Claude Desktop or other MCP clients.

## Quick Start

### 1. Configuration

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your API key:

```env
# Recommended: Use OpenRouter (single API key for all models)
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-...

# Or use direct Anthropic/OpenAI
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

### 2. Build the Docker Image

```bash
docker-compose build
```

Or manually:

```bash
docker build -t omni-cortex:latest .
```

### 3. Run the MCP Server

The MCP server communicates via **stdin/stdout**, not HTTP.

#### Option A: Docker Compose (Recommended)

```bash
docker-compose up -d
```

#### Option B: Docker Run

```bash
docker run -i \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  omni-cortex:latest
```

**Important flags:**
- `-i`: Keep stdin open for MCP communication
- `--env-file .env`: Load environment variables
- `-v $(pwd)/data:/app/data`: Persist LangChain memory

## MCP Client Configuration

### Claude Desktop (macOS)

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "omni-cortex": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "--env-file",
        "/path/to/omni-cortex/.env",
        "-v",
        "/path/to/omni-cortex/data:/app/data",
        "omni-cortex:latest"
      ]
    }
  }
}
```

### Claude Desktop (Windows)

Edit `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "omni-cortex": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "--env-file",
        "C:\\path\\to\\omni-cortex\\.env",
        "-v",
        "C:\\path\\to\\omni-cortex\\data:/app/data",
        "omni-cortex:latest"
      ]
    }
  }
}
```

### Other MCP Clients

For custom MCP clients:

```bash
docker run -i --env-file .env -v $(pwd)/data:/app/data omni-cortex:latest
```

Then communicate via stdin/stdout using the MCP protocol.

## Environment Variables

### Required

- `LLM_PROVIDER`: `openrouter`, `anthropic`, or `openai`
- API Key (one of):
  - `OPENROUTER_API_KEY`
  - `ANTHROPIC_API_KEY` + `OPENAI_API_KEY`

### Optional

- `DEEP_REASONING_MODEL`: Model for deep reasoning (default: `anthropic/claude-4.5-sonnet`)
- `FAST_SYNTHESIS_MODEL`: Model for fast synthesis (default: `openai/gpt-5.2`)
- `MAX_TOKENS_PER_REQUEST`: Token limit per request (default: `16000`)
- `MAX_REASONING_DEPTH`: Max recursion depth (default: `10`)

See `.env.example` for all options.

### Minimal “vibe coder” toggles (keep it light)
- `ENABLE_AUTO_INGEST=true` (optional) — auto-refresh RAG at startup; set to `false` to disable.
- `LLM_PROVIDER=openrouter` — one key for both models (simpler setup).
- Leave other settings at defaults unless you need them.

## Persistent Memory

LangChain memory and LangGraph checkpoints are stored in `/app/data` inside the container.

**Persist across restarts:**

```bash
# Create data directory
mkdir -p ./data

# Mount it
docker run -i --env-file .env -v $(pwd)/data:/app/data omni-cortex:latest
```

**Clear memory:**

```bash
rm -rf ./data/*
```

## Development Mode

Mount your code for live development:

```yaml
# docker-compose.yml
volumes:
  - cortex-memory:/app/data
  - .:/app  # Mount code for hot reload
```

Then rebuild when you make changes:

```bash
docker-compose restart
```

## Testing

### Interactive Shell

Debug inside the container:

```bash
docker run -it \
  --entrypoint /bin/bash \
  --env-file .env \
  omni-cortex:latest
```

Then run the server manually:

```bash
python -m server.main
```

## Troubleshooting

### Container exits immediately

MCP servers wait for stdin. The container needs `-i` flag:

```bash
docker run -i omni-cortex:latest
```

### "API key not set" error

Verify environment variables are loaded:

```bash
docker run -i --env-file .env omni-cortex:latest
```

Or check the `.env` file exists and has the correct key.

### Memory not persisting

Ensure the volume is mounted:

```bash
docker run -i \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  omni-cortex:latest
```

Check permissions on the `./data` directory.

### Model not found

Update model names in `.env`:

```env
# Use OpenRouter model IDs
DEEP_REASONING_MODEL=anthropic/claude-4.5-sonnet
FAST_SYNTHESIS_MODEL=openai/gpt-5.2
```

Check OpenRouter or provider documentation for valid model names.

## Production Deployment

### Best Practices

1. **Use Docker Compose** for easier management
2. **Mount persistent volumes** for memory
3. **Set restart policy** to `unless-stopped`
4. **Monitor logs**: `docker-compose logs -f omni-cortex`
5. **Backup data directory** regularly
6. **Use secrets** instead of `.env` files in production
7. **Run as non-root** (already configured)

### Resource Limits

Add resource constraints:

```yaml
# docker-compose.yml
services:
  omni-cortex:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          memory: 2G
```

### Monitoring

Check container health:

```bash
docker ps
docker stats omni-cortex
docker logs omni-cortex --tail 100
```

## Updates

Pull and rebuild:

```bash
git pull
docker-compose build --no-cache
docker-compose up -d
```

## Uninstall

Stop and remove:

```bash
docker-compose down -v
docker rmi omni-cortex:latest
```

Keep data:

```bash
docker-compose down
# data/ directory remains
```
