# MCP Configuration Examples

This directory contains ready-to-use MCP configuration examples for popular IDEs and tools.

## Quick Setup

1. Choose the configuration file for your IDE/tool
2. Copy the JSON content
3. Update the API keys with your actual keys
4. Paste into your MCP client configuration file

---

## 📋 Configuration Files

### `claude-desktop.json`
**For**: Claude Desktop App  
**Location**: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)  
**Usage**: Basic Docker-based setup for Claude Desktop

### `windsurf-mcp.json`
**For**: Windsurf IDE  
**Location**: Windsurf MCP settings  
**Usage**: Full-featured config with all environment variables and metadata

### `cursor-mcp.json`
**For**: Cursor IDE  
**Location**: Cursor MCP settings  
**Usage**: Docker-based setup optimized for Cursor

### `local-development.json`
**For**: Local development (any MCP client)  
**Usage**: Direct Python execution without Docker, includes all configuration options

---

## 🔑 API Key Setup

Replace these placeholder values with your actual API keys:

```json
"ANTHROPIC_API_KEY": "your-anthropic-api-key-here",
"OPENAI_API_KEY": "your-openai-api-key-here",
"OPENROUTER_API_KEY": "your-openrouter-api-key-here"
```

### Getting API Keys

- **OpenRouter** (Recommended): https://openrouter.ai/keys
  - Provides access to both Claude and GPT models with one key
  
- **Anthropic**: https://console.anthropic.com/
  - Required if using `LLM_PROVIDER=anthropic`
  
- **OpenAI**: https://platform.openai.com/api-keys
  - Required if using `LLM_PROVIDER=openai`

---

## 🐳 Docker vs Local

### Docker Setup (Recommended)

**Prerequisites**: 
- Docker and docker-compose installed
- Omni-Cortex container running (`docker-compose up -d`)

**Pros**:
- Isolated environment
- Consistent across systems
- Easy to manage

**Command format**:
```json
{
  "command": "docker",
  "args": ["exec", "-i", "omni-cortex", "python", "-m", "server.main"]
}
```

### Local Development Setup

**Prerequisites**:
- Python 3.12+
- Dependencies installed (`pip install -r requirements.txt`)
- Virtual environment recommended

**Pros**:
- Faster iteration
- Direct debugging
- No Docker overhead

**Command format**:
```json
{
  "command": "python",
  "args": ["-m", "server.main"],
  "cwd": "/absolute/path/to/omni_cortex"
}
```

**Important**: Update `cwd` to the absolute path of your `omni_cortex` directory.

---

## ⚙️ Configuration Options

### Essential Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | `openrouter`, `anthropic`, or `openai` | `openrouter` |
| `DEEP_REASONING_MODEL` | Model for complex reasoning | `anthropic/claude-4.5-sonnet` |
| `FAST_SYNTHESIS_MODEL` | Model for fast generation | `openai/gpt-5.2` |

### Optional Feature Flags

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_AUTO_INGEST` | Auto-ingest repo into vector store | `true` |
| `ENABLE_DSPY_OPTIMIZATION` | DSPy prompt optimization | `true` |
| `ENABLE_PRM_SCORING` | Process Reward Model scoring | `true` |
| `MAX_TOKENS_PER_REQUEST` | Token limit per request | `16000` |
| `MAX_REASONING_DEPTH` | Max depth for recursive frameworks | `10` |
| `MCTS_MAX_ROLLOUTS` | MCTS search iterations | `50` |
| `DEBATE_MAX_ROUNDS` | Multi-agent debate rounds | `5` |

---

## 🧪 Testing Your Configuration

After configuring, test the connection:

1. **Restart your IDE** to load the new MCP configuration
2. **Check MCP server status** in your IDE's MCP panel
3. **Test with a simple query**:
   ```
   Use omni-cortex to explain what reasoning frameworks are available
   ```

### Troubleshooting

**"Server not responding"**
- Verify Docker container is running: `docker ps | grep omni-cortex`
- Check logs: `docker-compose logs -f omni-cortex`
- Ensure API keys are set correctly

**"API key invalid"**
- Verify keys in `.env` file match your provider
- Check key format (no extra spaces or quotes)
- Ensure provider is correctly set (`openrouter`, `anthropic`, or `openai`)

**"Command not found"**
- For Docker: Ensure Docker is running
- For local: Verify Python path and virtual environment
- Check `cwd` path is absolute and correct

---

## 📚 Framework Usage Examples

Once configured, you can use natural language to access frameworks:

```
"wtf is wrong with this code" → Active Inference (debugging)
"clean up this mess" → Graph of Thoughts (refactoring)
"is this secure?" → Chain of Verification (security)
"quick fix needed" → System1 (fast mode)
"complex migration" → Everything of Thought (deep analysis)
```

See the main README.md for the complete framework list and descriptions.

---

## 🔄 Updating Configuration

To modify your configuration after setup:

1. **Edit the config file** in your IDE's MCP settings location
2. **Update environment variables** as needed
3. **Restart your IDE** to apply changes
4. **Test** to verify the changes took effect

---

## 💡 Best Practices

1. **Use OpenRouter**: Simplifies key management (one key for all models)
2. **Enable all features**: Unless you have specific reasons to disable them
3. **Set reasonable limits**: Default token/depth limits work well for most cases
4. **Keep keys secure**: Never commit API keys to version control
5. **Use Docker in production**: More stable and reproducible

---

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review logs: `docker-compose logs -f` (Docker) or terminal output (local)
3. Verify your API keys are valid and have sufficient credits
4. Ensure your IDE supports MCP protocol

For more information, see the main project README.md.
