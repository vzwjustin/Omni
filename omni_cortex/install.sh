#!/bin/bash
# Omni MCP Server Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/vzwjustin/Omni/initial-release/omni_cortex/install.sh | bash

set -e

echo "Installing Omni MCP Server..."

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is required. Install from https://docker.com"
    exit 1
fi

# Pull the image
echo "Pulling Docker image..."
docker pull vzwjustin/omni-cortex:latest

# Configure MCP for Claude Code
CLAUDE_MCP_DIR="$HOME/.claude"
mkdir -p "$CLAUDE_MCP_DIR"
MCP_CONFIG="$CLAUDE_MCP_DIR/.mcp.json"

# Create or merge MCP config
if [ -f "$MCP_CONFIG" ]; then
    # Check if already configured
    if grep -q "omni-cortex" "$MCP_CONFIG"; then
        echo "omni-cortex already configured in $MCP_CONFIG"
    else
        echo "Adding omni-cortex to existing MCP config..."
        # Use temp file for safe editing
        tmp=$(mktemp)
        # Insert omni-cortex config before the closing brace of mcpServers
        python3 -c "
import json
with open('$MCP_CONFIG', 'r') as f:
    config = json.load(f)
if 'mcpServers' not in config:
    config['mcpServers'] = {}
config['mcpServers']['omni-cortex'] = {
    'command': 'docker',
    'args': ['run', '--rm', '-i', 'vzwjustin/omni-cortex:latest']
}
with open('$tmp', 'w') as f:
    json.dump(config, f, indent=2)
" && mv "$tmp" "$MCP_CONFIG"
    fi
else
    # Create new config
    cat > "$MCP_CONFIG" << 'EOF'
{
  "mcpServers": {
    "omni-cortex": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "vzwjustin/omni-cortex:latest"]
    }
  }
}
EOF
fi

echo ""
echo "Done! Omni-Cortex is installed."
echo ""
echo "Restart your IDE to start using 40 thinking frameworks."
echo "Just describe what you want - the router picks the right framework."
echo ""
