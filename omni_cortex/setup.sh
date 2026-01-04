#!/bin/bash
# Omni-Cortex Setup Script
# Automates Docker build and MCP configuration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  Omni-Cortex MCP Server Setup"
echo "=============================================="

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check for docker-compose
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file..."

    # Prompt for OpenAI API key
    read -p "Enter your OpenAI API key (for embeddings): " OPENAI_KEY

    if [ -z "$OPENAI_KEY" ]; then
        echo "Warning: No API key provided. RAG features will be disabled."
        OPENAI_KEY=""
        EMBEDDING_PROVIDER="none"
    else
        EMBEDDING_PROVIDER="openai"
    fi

    cat > .env << EOF
# Omni-Cortex Configuration
LLM_PROVIDER=pass-through

# Embeddings (for ChromaDB / RAG)
EMBEDDING_PROVIDER=${EMBEDDING_PROVIDER}
OPENAI_API_KEY=${OPENAI_KEY}

# ChromaDB Storage
CHROMA_PERSIST_DIR=./data/chroma

# Feature Flags
ENABLE_AUTO_INGEST=true
ENABLE_DSPY_OPTIMIZATION=true
ENABLE_PRM_SCORING=true

# Limits
MAX_REASONING_DEPTH=10
MCTS_MAX_ROLLOUTS=50
DEBATE_MAX_ROUNDS=5

# Logging
LOG_LEVEL=INFO
EOF
    echo ".env file created!"
else
    echo ".env file already exists, skipping..."
fi

# Create data directory
mkdir -p data/chroma

# Build Docker image
echo ""
echo "Building Docker image..."
docker-compose build

# Configure MCP for Claude Code
echo ""
echo "Configuring MCP..."

CLAUDE_MCP_DIR="$HOME/.claude"
mkdir -p "$CLAUDE_MCP_DIR"

# Create or update .mcp.json
MCP_CONFIG="$CLAUDE_MCP_DIR/.mcp.json"

if [ -f "$MCP_CONFIG" ]; then
    # Check if omni-cortex already configured
    if grep -q "omni-cortex" "$MCP_CONFIG"; then
        echo "omni-cortex already in MCP config, updating..."
    fi
fi

# Write MCP config
cat > "$MCP_CONFIG" << EOF
{
  "mcpServers": {
    "omni-cortex": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "--env-file", "${SCRIPT_DIR}/.env",
        "-v", "${SCRIPT_DIR}/data:/app/data",
        "omni_cortex-omni-cortex:latest"
      ]
    }
  }
}
EOF

echo "MCP configured at: $MCP_CONFIG"

# Test the setup
echo ""
echo "Testing setup..."
docker run --rm --entrypoint python3 --env-file .env omni_cortex-omni-cortex:latest -c "
from app.graph import FRAMEWORK_NODES
print(f'Frameworks loaded: {len(FRAMEWORK_NODES)}')
print('Setup successful!')
"

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "To start the server:"
echo "  docker-compose up -d"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f"
echo ""
echo "The MCP server will be available to Claude Code automatically."
echo ""
