# Incident Response Runbook

## Overview

This runbook covers common incidents and their resolution for Omni-Cortex MCP server.

## Severity Levels

| Level | Description | Response Time |
|-------|-------------|---------------|
| P0 | Service completely down | < 15 min |
| P1 | Major feature broken | < 1 hour |
| P2 | Degraded performance | < 4 hours |
| P3 | Minor issue | < 24 hours |

## Common Incidents

### 1. MCP Server Not Responding

**Symptoms:**
- Claude Code shows "tool not available"
- No response from server

**Diagnosis:**
```bash
# Check if container is running
docker-compose ps

# Check logs
docker-compose logs --tail=100 omni-cortex

# Check health endpoint (if exposed)
curl http://localhost:8000/health
```

**Resolution:**
```bash
# Restart the service
docker-compose restart omni-cortex

# If persistent, rebuild
docker-compose down
docker-compose up -d --build
```

### 2. Framework Routing Failures

**Symptoms:**
- "Framework not found" errors
- Unexpected framework selection

**Diagnosis:**
```bash
# Check router logs
docker-compose logs omni-cortex | grep -i "router\|routing"

# Verify framework registry
python -c "from app.frameworks.registry import FRAMEWORKS; print(len(FRAMEWORKS))"
```

**Resolution:**
- Verify framework is registered in `app/frameworks/registry.py`
- Check vibes dictionary in `app/core/vibe_dictionary.py`
- Restart server to reload registry

### 3. Memory/ChromaDB Issues

**Symptoms:**
- "No embedding provider" errors
- RAG search returns empty results

**Diagnosis:**
```bash
# Check ChromaDB data
ls -la data/chroma/

# Verify embedding provider
grep EMBEDDING_PROVIDER .env
```

**Resolution:**
```bash
# Clear ChromaDB cache
rm -rf data/chroma/*

# Restart with fresh embeddings
docker-compose down
docker-compose up -d
```

### 4. High Latency

**Symptoms:**
- Slow framework responses
- Timeouts

**Diagnosis:**
```bash
# Check Prometheus metrics
curl http://localhost:9090/metrics | grep omni_cortex

# Check container resources
docker stats omni-cortex
```

**Resolution:**
- Enable LEAN_MODE=true for reduced token overhead
- Check LLM provider rate limits
- Scale resources if needed

## Escalation

If issues persist after following runbook:
1. Check GitHub Issues for known problems
2. Open new issue with logs and reproduction steps
3. Tag @vzwjustin for urgent issues

## Post-Incident

After resolving any P0/P1:
1. Document root cause
2. Update runbook if new failure mode
3. Consider adding monitoring/alerting
