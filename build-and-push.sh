#!/bin/bash
# Build and push Omni Cortex Docker image
# Usage: ./build-and-push.sh

set -e

echo "🏗️  Building Docker image..."
cd omni_cortex
docker build -t vzwjustin/omni-cortex:latest .

echo "📤 Pushing to Docker Hub..."
docker push vzwjustin/omni-cortex:latest

echo "✅ Done! Image pushed to vzwjustin/omni-cortex:latest"
