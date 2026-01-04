# Omni-Cortex MCP Server
# Pass-through MCP - calling LLM does the reasoning

FROM python:3.12-slim

# Metadata
LABEL maintainer="Omni-Cortex Team"
LABEL description="MCP Server exposing 20 thinking frameworks for vibe coders. No API keys required."
LABEL version="2.0.0"

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

# Create non-root user for security
RUN groupadd --gid 1000 cortex && \
    useradd --uid 1000 --gid cortex --shell /bin/bash --create-home cortex

# Set working directory
WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=cortex:cortex . .

# Create data directory for persistent memory
RUN mkdir -p /app/data && chown -R cortex:cortex /app/data

# Switch to non-root user
USER cortex

# MCP servers use stdio, not HTTP ports
# No EXPOSE needed

# No health check for stdio-based server
# Health is checked via successful stdio communication

# Run the MCP server (stdio mode)
ENTRYPOINT ["python", "-m", "server.main"]
