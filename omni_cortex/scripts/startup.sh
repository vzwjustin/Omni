#!/bin/bash
set -e

echo "==================================="
echo "Omni-Cortex Startup"
echo "==================================="

# Ensure data directories exist
mkdir -p /app/data/chroma
mkdir -p /app/data/checkpoints

# Check if ChromaDB needs initial ingestion
CHROMA_DB="/app/data/chroma/chroma.sqlite3"
if [ ! -f "$CHROMA_DB" ] || [ "$(python -c "import sqlite3; conn = sqlite3.connect('$CHROMA_DB'); cur = conn.cursor(); cur.execute('SELECT COUNT(*) FROM embeddings'); print(cur.fetchone()[0]); conn.close()")" = "0" ]; then
    echo "==================================="
    echo "ChromaDB empty - ingesting codebase"
    echo "==================================="
    python -m app.ingest_repo || echo "Warning: Ingestion failed, continuing anyway..."
else
    echo "ChromaDB already populated ($(python -c "import sqlite3; conn = sqlite3.connect('$CHROMA_DB'); cur = conn.cursor(); cur.execute('SELECT COUNT(*) FROM embeddings'); print(cur.fetchone()[0]); conn.close()") embeddings)"
fi

echo "==================================="
echo "Starting MCP Server"
echo "==================================="

# Start the MCP server
exec python -m server.main
