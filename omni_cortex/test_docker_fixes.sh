#!/bin/bash
# Test script to verify Docker critical fixes
# Run this after rebuilding your Docker image

set -e

echo "=========================================="
echo "Docker Critical Fixes - Test Script"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Check if container is running
echo "Test 1: Checking container status..."
if docker-compose ps | grep -q "Up"; then
    echo -e "${GREEN}✓${NC} Container is running"
else
    echo -e "${YELLOW}⚠${NC} Container not running, starting..."
    docker-compose up -d
    sleep 3
fi
echo ""

# Test 2: Check for database initialization
echo "Test 2: Verifying database initialization..."
if docker-compose logs | grep -q "checkpointer_initialized"; then
    echo -e "${GREEN}✓${NC} Checkpointer initialized successfully"
    docker-compose logs | grep "checkpointer_initialized" | tail -1
else
    echo -e "${RED}✗${NC} Checkpointer initialization not found in logs"
fi
echo ""

# Test 3: Restart container and check for clean shutdown
echo "Test 3: Testing clean shutdown on restart..."
echo "   Restarting container..."
docker-compose restart > /dev/null 2>&1
sleep 2

echo "   Checking for shutdown messages..."
if docker-compose logs | grep -q "shutting_down"; then
    echo -e "${GREEN}✓${NC} Clean shutdown detected"
    docker-compose logs | grep "shutting_down" | tail -1
else
    echo -e "${YELLOW}⚠${NC} No shutdown message found (may need to trigger MCP request first)"
fi

if docker-compose logs | grep -q "checkpointer_cleaned_up"; then
    echo -e "${GREEN}✓${NC} Checkpointer cleanup executed"
    docker-compose logs | grep "checkpointer_cleaned_up" | tail -1
else
    echo -e "${YELLOW}⚠${NC} No cleanup message found"
fi
echo ""

# Test 4: Check for database lock errors
echo "Test 4: Checking for database lock errors..."
if docker-compose logs | grep -qi "database.*locked\|database.*in use"; then
    echo -e "${RED}✗${NC} Database lock errors found:"
    docker-compose logs | grep -i "database.*locked\|database.*in use"
else
    echo -e "${GREEN}✓${NC} No database lock errors detected"
fi
echo ""

# Test 5: Check for RuntimeError crashes
echo "Test 5: Checking for RuntimeError crashes..."
if docker-compose logs | grep -q "RuntimeError.*dictionary"; then
    echo -e "${RED}✗${NC} RuntimeError found:"
    docker-compose logs | grep "RuntimeError.*dictionary"
else
    echo -e "${GREEN}✓${NC} No RuntimeError crashes detected"
fi
echo ""

# Test 6: Check for recent errors
echo "Test 6: Checking for recent errors (last 50 lines)..."
ERROR_COUNT=$(docker-compose logs --tail=50 | grep -i "ERROR" | wc -l)
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}⚠${NC} Found $ERROR_COUNT error lines in recent logs"
    echo "   Recent errors:"
    docker-compose logs --tail=50 | grep -i "ERROR" | tail -5
else
    echo -e "${GREEN}✓${NC} No errors in recent logs"
fi
echo ""

# Test 7: Full stop/start cycle
echo "Test 7: Testing full stop/start cycle..."
echo "   Stopping container..."
docker-compose down > /dev/null 2>&1
sleep 1

echo "   Starting container..."
docker-compose up -d > /dev/null 2>&1
sleep 3

echo "   Checking startup..."
if docker-compose logs | grep -q "checkpointer_initialized"; then
    echo -e "${GREEN}✓${NC} Container started successfully after full cycle"
else
    echo -e "${RED}✗${NC} Container startup may have issues"
fi
echo ""

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""
echo "Docker fixes verification complete!"
echo ""
echo "Key indicators to watch:"
echo "  1. 'checkpointer_initialized' on startup"
echo "  2. 'shutting_down' and 'checkpointer_cleaned_up' on shutdown"
echo "  3. No 'database locked' errors"
echo "  4. No 'RuntimeError' crashes"
echo ""
echo "To monitor logs in real-time:"
echo "  docker-compose logs -f"
echo ""
echo "To rebuild with latest changes:"
echo "  docker-compose build && docker-compose up -d"
echo ""
