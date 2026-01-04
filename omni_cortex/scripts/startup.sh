#!/bin/bash
set -e

echo "==================================="
echo "Omni-Cortex Startup"
echo "==================================="

# Ensure data directories exist
mkdir -p /app/data/chroma
mkdir -p /app/data/checkpoints

# Function to detect if there's a codebase to index
detect_codebase() {
    # Check for Python files in common locations
    local py_count=$(find /app -name "*.py" -type f 2>/dev/null | wc -l | tr -d ' ')
    
    # Check for framework indicators
    local has_requirements=$([ -f "/app/requirements.txt" ] && echo "yes" || echo "no")
    local has_setup=$([ -f "/app/setup.py" ] || [ -f "/app/pyproject.toml" ] && echo "yes" || echo "no")
    
    # Check for external mounted code (common patterns)
    local has_app_dir=$([ -d "/app/app" ] && echo "yes" || echo "no")
    local has_src_dir=$([ -d "/app/src" ] && echo "yes" || echo "no")
    local has_lib_dir=$([ -d "/app/lib" ] && echo "yes" || echo "no")
    
    echo "  Python files: $py_count"
    echo "  requirements.txt: $has_requirements"
    echo "  Project config: $has_setup"
    echo "  Code directories: app=$has_app_dir, src=$has_src_dir, lib=$has_lib_dir"
    
    # Consider it a codebase if we have Python files
    if [ "$py_count" -gt 10 ]; then
        return 0  # true - codebase detected
    else
        return 1  # false - no significant codebase
    fi
}

# Check if ChromaDB needs ingestion
CHROMA_DB="/app/data/chroma/chroma.sqlite3"
EMBEDDING_COUNT=0

if [ -f "$CHROMA_DB" ]; then
    EMBEDDING_COUNT=$(python -c "import sqlite3; conn = sqlite3.connect('$CHROMA_DB'); cur = conn.cursor(); cur.execute('SELECT COUNT(*) FROM embeddings'); print(cur.fetchone()[0]); conn.close()" 2>/dev/null || echo "0")
fi

echo "==================================="
echo "Checking for codebases to index"
echo "==================================="

if detect_codebase; then
    echo "✓ Codebase detected"
    
    if [ "$EMBEDDING_COUNT" = "0" ]; then
        echo "==================================="
        echo "ChromaDB empty - ingesting codebase"
        echo "==================================="
        python -m app.ingest_repo || echo "Warning: Ingestion failed, continuing anyway..."
    else
        echo "ChromaDB already populated ($EMBEDDING_COUNT embeddings)"
        echo "Tip: To re-index, delete /app/data/chroma and restart"
    fi
else
    echo "✗ No significant codebase detected"
    echo "  (Need 10+ Python files for auto-ingestion)"
    if [ "$EMBEDDING_COUNT" -gt 0 ]; then
        echo "  Using existing $EMBEDDING_COUNT embeddings"
    fi
fi

echo "==================================="
echo "Starting MCP Server"
echo "==================================="

# Start the MCP server
exec python -m server.main
