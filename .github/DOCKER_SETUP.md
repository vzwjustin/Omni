# Docker Hub Auto-Publish Setup

This repository is configured to automatically build and push Docker images to Docker Hub.

## Required Secrets

You need to add the following secrets to your GitHub repository:

1. Go to your repository → **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret** and add:

### `DOCKER_USERNAME`
- Your Docker Hub username
- Example: `vzwjustin`

### `DOCKER_PASSWORD`
- Your Docker Hub access token (recommended) or password
- **Recommended**: Use a Personal Access Token instead of password
  - Go to [Docker Hub Account Settings](https://hub.docker.com/settings/security)
  - Click **New Access Token**
  - Give it a name (e.g., "GitHub Actions")
  - Copy the token and use it as the secret value

## Triggers

The workflow automatically runs on:

- ✅ **Push to `main` or `master`** → Builds and pushes with `latest` tag
- ✅ **Git tags** (e.g., `v3.0.0`) → Builds and pushes with version tags
- ✅ **Pull requests** → Builds only (does not push)
- ✅ **Manual trigger** → Can be run manually from Actions tab

## Image Tags

When you push, the following tags are created:

| Event | Tags Created |
|-------|-------------|
| Push to `main` | `latest`, `main-<sha>` |
| Tag `v3.0.0` | `3.0.0`, `3.0`, `3`, `latest` |
| Push to branch `feature-x` | `feature-x`, `feature-x-<sha>` |

## Docker Image

- **Repository**: `vzwjustin/omni-cortex`
- **Platforms**: `linux/amd64`, `linux/arm64`
- **Cache**: Enabled via GitHub Actions cache

## Manual Trigger

You can manually trigger a build:

1. Go to **Actions** tab
2. Select **Docker Build and Push**
3. Click **Run workflow**
4. Select branch and click **Run workflow**

## Verifying Build

After pushing to `main`, check:

1. **Actions** tab → See the running workflow
2. When complete, verify at: https://hub.docker.com/r/vzwjustin/omni-cortex/tags

## First-Time Setup Checklist

- [ ] Add `DOCKER_USERNAME` secret
- [ ] Add `DOCKER_PASSWORD` secret
- [ ] Verify Docker Hub repository exists (`vzwjustin/omni-cortex`)
- [ ] Push to `main` or create a tag to test
- [ ] Check Actions tab for successful build
- [ ] Verify image appears on Docker Hub
