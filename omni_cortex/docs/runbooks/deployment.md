# Deployment Runbook

## Pre-Deployment Checklist

- [ ] All tests pass locally (`pytest tests/`)
- [ ] Linter passes (`ruff check app/ server/`)
- [ ] Docker build succeeds (`docker-compose build`)
- [ ] Environment variables documented in `.env.example`
- [ ] CHANGELOG updated (if applicable)

## Deployment Methods

### 1. Docker Compose (Development/Staging)

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose up -d --build

# Verify deployment
docker-compose logs -f omni-cortex
```

### 2. Docker Hub (Production)

```bash
# Build with tag
docker build -t vzwjustin/omni-cortex:v1.0.0 .

# Push to registry
docker push vzwjustin/omni-cortex:v1.0.0

# Deploy on target
docker pull vzwjustin/omni-cortex:v1.0.0
docker-compose up -d
```

### 3. GitHub Release

1. Create and push tag:
```bash
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

2. GitHub Actions will:
   - Run CI checks
   - Build Docker image
   - Push to GHCR
   - Create GitHub Release with changelog

## Rollback Procedure

### Quick Rollback

```bash
# Stop current version
docker-compose down

# Deploy previous version
docker pull vzwjustin/omni-cortex:v0.9.0
docker-compose up -d
```

### Git Rollback

```bash
# Revert to previous commit
git revert HEAD
git push origin main

# Or reset to specific tag
git checkout v0.9.0
docker-compose up -d --build
```

## Health Checks

After deployment, verify:

1. **Server responds:**
```bash
# Via MCP client - call health tool
# Or check logs for startup message
docker-compose logs omni-cortex | grep "Omni-Cortex MCP"
```

2. **Frameworks loaded:**
```bash
docker-compose exec omni-cortex python -c \
  "from app.frameworks.registry import FRAMEWORKS; print(f'{len(FRAMEWORKS)} frameworks')"
```

3. **Memory persistence:**
```bash
ls -la data/chroma/
ls -la data/checkpoints/
```

## Environment Configuration

| Environment | LEAN_MODE | LLM_PROVIDER | Notes |
|-------------|-----------|--------------|-------|
| Development | true | pass-through | Local testing |
| Staging | false | google | Full features |
| Production | true | google | Optimized |
