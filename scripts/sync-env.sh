#!/bin/bash

# =============================================================================
# STREAM - Environment File Sync Script
# =============================================================================
# This script copies the root .env file to all directories that need it
# Run this whenever you update API keys or environment variables
#
# Run this from the project root directory.
#
# Usage:
#   ./scripts/sync-env.sh
# =============================================================================

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the project root (go up from scripts/ directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "🔄 Syncing environment files..."
echo -e "${BLUE}📂 Project root: $PROJECT_ROOT${NC}"
echo ""

# Check if root .env exists
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo -e "${RED}❌ Error: .env file not found at $PROJECT_ROOT/.env${NC}"
    echo "Create it first with: cp .env.example .env"
    exit 1
fi

# Directories that need .env files (relative to project root)
TARGETS=(
    "stream/gateway"
    # Add more directories here as needed
    # "middleware"
    # "backend"
)

# Sync to each target
SUCCESS_COUNT=0
FAILED_COUNT=0

for TARGET in "${TARGETS[@]}"; do
    TARGET_DIR="$PROJECT_ROOT/$TARGET"
    TARGET_FILE="$TARGET_DIR/.env"

    echo -e "${BLUE}→ Checking: $TARGET_DIR${NC}"

    # Check if directory exists - FAIL if it doesn't (don't create it)
    if [ ! -d "$TARGET_DIR" ]; then
        echo -e "${RED}❌ Directory does not exist: $TARGET_DIR${NC}"
        echo -e "${YELLOW}   Please create this directory first or remove it from TARGETS array${NC}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        echo ""
        continue
    fi

    # Copy .env file
    if cp "$PROJECT_ROOT/.env" "$TARGET_FILE"; then
        echo -e "${GREEN}✅ Synced to $TARGET/.env${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo -e "${RED}❌ Failed to copy to $TARGET/.env${NC}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
    echo ""
done

echo ""
if [ $SUCCESS_COUNT -gt 0 ]; then
    echo -e "${GREEN}🎉 Successfully synced $SUCCESS_COUNT file(s)${NC}"
fi

if [ $FAILED_COUNT -gt 0 ]; then
    echo -e "${RED}❌ Failed to sync $FAILED_COUNT file(s)${NC}"
fi

echo ""
echo -e "${YELLOW}📝 Next steps:${NC}"
echo "  1. Restart LiteLLM gateway: docker-compose restart litellm"
echo "  2. Restart middleware: stream-middleware"
echo ""
echo -e "${YELLOW}⚠️  Security reminders:${NC}"
echo "  - Never commit .env files to Git"
echo "  - Run this script after updating API keys"
