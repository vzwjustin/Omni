# Omni-Cortex MCP Server
# Multi-stage build with cloud test execution

# =============================================================================
# Stage 1: Base - shared between test and production
# =============================================================================
FROM python:3.13-slim AS base

# Metadata
LABEL maintainer="Omni-Cortex Team"
LABEL description="MCP Server with 62 AI reasoning frameworks - multi-turn orchestration via client sampling. No API keys required."
LABEL version="3.0.0"

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONIOENCODING=utf-8

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# =============================================================================
# Stage 2: Test - runs tests in Build Cloud
# Usage: docker buildx build --target test ...
# =============================================================================
FROM base AS test

# Install test dependencies
RUN pip install --no-cache-dir pytest==8.3.4 pytest-asyncio==0.24.0 pytest-cov==6.0.0

# Set test environment
ENV LLM_PROVIDER=pass-through \
    EMBEDDING_PROVIDER=openai \
    ENABLE_AUTO_INGEST=false \
    LEAN_MODE=false \
    LOG_LEVEL=DEBUG \
    PYTHONPATH=/app

# Run tests - this executes during build!
RUN python -m pytest tests/ -v --tb=short 2>&1 | tee /app/test-results.txt; \
    RESULT=$?; \
    echo "Test exit code: $RESULT" >> /app/test-results.txt; \
    exit $RESULT

# =============================================================================
# Stage 3: Production - final lightweight image
# =============================================================================
FROM base AS production

# Create non-root user for security
RUN groupadd --gid 1000 cortex && \
    useradd --uid 1000 --gid cortex --shell /bin/bash --create-home cortex

# Set ownership
RUN chown -R cortex:cortex /app

# Create data directory for persistent memory
RUN mkdir -p /app/data && chown -R cortex:cortex /app/data

# Copy and setup startup script
COPY scripts/startup.sh /app/scripts/startup.sh
RUN chmod +x /app/scripts/startup.sh && chown cortex:cortex /app/scripts/startup.sh

# Switch to non-root user
USER cortex

# MCP servers use stdio, not HTTP ports
# No EXPOSE needed

# Run startup script which handles ingestion and starts server
ENTRYPOINT ["/app/scripts/startup.sh"]
