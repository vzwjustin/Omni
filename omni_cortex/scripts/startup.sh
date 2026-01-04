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
    echo "  Scanning for codebases..."
    
    # Check for different language files
    local py_count=$(find /app -name "*.py" -type f 2>/dev/null | wc -l | tr -d ' ')
    local js_count=$(find /app -name "*.js" -o -name "*.jsx" -type f 2>/dev/null | wc -l | tr -d ' ')
    local ts_count=$(find /app -name "*.ts" -o -name "*.tsx" -type f 2>/dev/null | wc -l | tr -d ' ')
    local go_count=$(find /app -name "*.go" -type f 2>/dev/null | wc -l | tr -d ' ')
    local rust_count=$(find /app -name "*.rs" -type f 2>/dev/null | wc -l | tr -d ' ')
    local java_count=$(find /app -name "*.java" -type f 2>/dev/null | wc -l | tr -d ' ')
    local cpp_count=$(find /app -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" -type f 2>/dev/null | wc -l | tr -d ' ')
    local c_count=$(find /app -name "*.c" -o -name "*.h" -type f 2>/dev/null | wc -l | tr -d ' ')
    local ruby_count=$(find /app -name "*.rb" -type f 2>/dev/null | wc -l | tr -d ' ')
    local php_count=$(find /app -name "*.php" -type f 2>/dev/null | wc -l | tr -d ' ')
    
    # Check for framework/project indicators
    local has_python=$([ -f "/app/requirements.txt" ] || [ -f "/app/setup.py" ] || [ -f "/app/pyproject.toml" ] && echo "✓" || echo "✗")
    local has_node=$([ -f "/app/package.json" ] && echo "✓" || echo "✗")
    local has_go=$([ -f "/app/go.mod" ] && echo "✓" || echo "✗")
    local has_rust=$([ -f "/app/Cargo.toml" ] && echo "✓" || echo "✗")
    local has_java=$([ -f "/app/pom.xml" ] || [ -f "/app/build.gradle" ] && echo "✓" || echo "✗")
    local has_ruby=$([ -f "/app/Gemfile" ] && echo "✓" || echo "✗")
    local has_php=$([ -f "/app/composer.json" ] && echo "✓" || echo "✗")
    
    # Display findings
    echo ""
    echo "  Language Files Detected:"
    [ "$py_count" -gt 0 ] && echo "    Python:     $py_count files $has_python"
    [ "$js_count" -gt 0 ] && echo "    JavaScript: $js_count files"
    [ "$ts_count" -gt 0 ] && echo "    TypeScript: $ts_count files $has_node"
    [ "$go_count" -gt 0 ] && echo "    Go:         $go_count files $has_go"
    [ "$rust_count" -gt 0 ] && echo "    Rust:       $rust_count files $has_rust"
    [ "$java_count" -gt 0 ] && echo "    Java:       $java_count files $has_java"
    [ "$cpp_count" -gt 0 ] && echo "    C++:        $cpp_count files"
    [ "$c_count" -gt 0 ] && echo "    C:          $c_count files"
    [ "$ruby_count" -gt 0 ] && echo "    Ruby:       $ruby_count files $has_ruby"
    [ "$php_count" -gt 0 ] && echo "    PHP:        $php_count files $has_php"
    
    # Calculate total significant files
    local total_files=$((py_count + js_count + ts_count + go_count + rust_count + java_count + cpp_count + c_count + ruby_count + php_count))
    
    echo ""
    echo "  Total code files: $total_files"
    
    # Consider it a codebase if we have 10+ files in any supported language
    if [ "$total_files" -gt 10 ]; then
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
    echo "  (Need 10+ code files for auto-ingestion)"
    echo "  Supported: Python, JavaScript/TypeScript, Go, Rust, Java, C/C++, Ruby, PHP"
    if [ "$EMBEDDING_COUNT" -gt 0 ]; then
        echo "  Using existing $EMBEDDING_COUNT embeddings"
    fi
fi

echo "==================================="
echo "Starting MCP Server"
echo "==================================="

# Start the MCP server
exec python -m server.main
