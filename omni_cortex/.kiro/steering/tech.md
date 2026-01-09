# Technology Stack & Build System

## Core Technologies

- **Python 3.12+** with async/await patterns
- **MCP (Model Context Protocol)** server implementation
- **LangGraph** for workflow orchestration (62 framework nodes)
- **LangChain** for memory management and RAG integration
- **ChromaDB** vector database for embeddings and search
- **Docker & Docker Compose** for containerized deployment
- **Pydantic** for configuration and data validation

## Key Dependencies

- `mcp[cli]>=1.0.0` - MCP server framework
- `langgraph>=0.2.0` - Workflow orchestration
- `langchain>=0.3.0` - Memory and RAG tools
- `chromadb>=0.5.3` - Vector database
- `structlog>=24.0.0` - Structured logging
- `prometheus-client>=0.20.0` - Metrics collection

## Build & Development Commands

### Initial Setup
```bash
./setup.sh                    # Automated setup (creates .env, builds Docker, configures MCP)
```

### Docker Operations
```bash
docker-compose build          # Build the container image
docker-compose up -d          # Run in background
docker-compose up             # Run in foreground
docker-compose logs -f        # View logs
docker-compose down           # Stop services
```

### Local Development
```bash
pip install -r requirements.txt
python -m server.main         # Run MCP server locally (without Docker)
```

### Testing & Validation
```bash
python scripts/verify_learning_offline.py    # Test learning flow with mock embeddings
python scripts/test_mcp_search.py           # Test ChromaDB search (requires API key)
python -m scripts.debug_search              # Debug Chroma/OpenAI configuration
python scripts/validate_frameworks.py       # Validate framework definitions
```

## Configuration

- Copy `.env.example` to `.env` for local configuration
- Never commit API keys to version control
- Use `EMBEDDING_PROVIDER=none` to disable RAG features
- Stdout reserved for MCP JSON-RPC; all logging goes to stderr