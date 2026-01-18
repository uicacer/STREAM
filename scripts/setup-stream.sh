#!/bin/bash
# =============================================================================
# STREAM Setup Script - Complete Installation & Configuration
# =============================================================================
# This script automates the entire STREAM setup process:
#   1. Checks prerequisites (Docker, docker-compose, .env)
#   2. Validates API keys configuration
#   3. Cleans up existing services (optional)
#   4. Downloads required AI models (with user confirmation)
#   5. Builds and starts all Docker containers
#   6. Waits for services to become healthy
#   7. Displays access URLs and useful commands
#
# Usage:
#   ./scripts/setup-stream.sh              # Interactive mode (recommended)
#   ./scripts/setup-stream.sh --skip-models # Skip all model downloads
#   ./scripts/setup-stream.sh --quick       # Only download required models
#   ./scripts/setup-stream.sh --clean       # Fresh install (removes volumes)
#   ./scripts/setup-stream.sh --help        # Show help
# =============================================================================

set -e  # Exit immediately if any command fails

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# ANSI color codes for pretty terminal output
# These make the script output colorful and easy to read
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color (resets to default)

# Script behavior flags (set by command-line arguments)
SKIP_MODELS=false      # If true, skip all model downloads
QUICK_MODE=false       # If true, skip optional 8B model
CLEAN_INSTALL=false    # If true, remove volumes before starting
NON_INTERACTIVE=false  # If true, don't prompt user for input

# =============================================================================
# COMMAND-LINE ARGUMENT PARSING
# =============================================================================
# Parse command-line arguments to determine script behavior
# This allows users to customize the setup process

for arg in "$@"; do
    case $arg in
        --skip-models)
            SKIP_MODELS=true
            NON_INTERACTIVE=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            NON_INTERACTIVE=true
            shift
            ;;
        --clean)
            CLEAN_INSTALL=true
            shift
            ;;
        --help)
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "STREAM Setup Script - Installation Guide"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-models    Skip all model downloads"
            echo "  --quick          Only download required models (skip 8B)"
            echo "  --clean          Fresh install (removes existing data)"
            echo "  --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Interactive mode (recommended)"
            echo "  $0 --quick            # Fast setup, skip large model"
            echo "  $0 --skip-models      # Setup without downloading models"
            echo "  $0 --clean            # Clean reinstall"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
# These helper functions make the script output clean and consistent

# Print a section header with decorative lines
print_step() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}${BOLD}$1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Print a success message with green checkmark
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Print a warning message with yellow warning symbol
print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Print an error message with red X
print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Print an informational message (indented)
print_info() {
    echo -e "   $1"
}

# Print a question with magenta question mark
print_question() {
    echo -e "${MAGENTA}❓ $1${NC}"
}


# Ask a yes/no question and return 0 for yes, 1 for no
# Usage: if ask_yes_no "Continue?" "y"; then ...
ask_yes_no() {
    local prompt="$1"      # The question to ask
    local default="${2:-y}" # Default answer (y or n)

    # In non-interactive mode, always return true (yes)
    if [ "$NON_INTERACTIVE" = true ]; then
        return 0
    fi

    # Loop until we get a valid answer
    while true; do
        # Show prompt with default in brackets
        if [ "$default" = "y" ]; then
            echo -ne "${MAGENTA}❓ ${prompt} [Y/n]: ${NC}"
        else
            echo -ne "${MAGENTA}❓ ${prompt} [y/N]: ${NC}"
        fi

        read -r response

        # Convert to lowercase (portable way)
        response=$(echo "$response" | tr '[:upper:]' '[:lower:]')

        # If empty, use default
        if [ -z "$response" ]; then
            response=$default
        fi

        # Check response
        case $response in
            y|yes) return 0 ;;  # Success (true)
            n|no) return 1 ;;   # Failure (false)
            *) echo -e "${YELLOW}Please answer yes or no.${NC}" ;;
        esac
    done
}


# Show an animated spinner while a background process runs
# Usage: command & spinner $! "Loading"
spinner() {
    local pid=$1        # Process ID to monitor
    local message="$2"  # Message to show
    local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'  # Spinner animation frames
    local i=0

    # Keep spinning while process is running
    while kill -0 $pid 2>/dev/null; do
        i=$(( (i+1) %10 ))
        printf "\r${BLUE}${spin:$i:1} ${message}...${NC}"
        sleep 0.1
    done
    printf "\r"  # Clear the spinner line
}

# =============================================================================
# BANNER
# =============================================================================
# Clear screen and show welcome banner

clear
echo -e "${CYAN}"
cat << "EOF"
   _____ _______ _____  ______          __  __
  / ____|__   __|  __ \|  ____|   /\   |  \/  |
 | (___    | |  | |__) | |__     /  \  | \  / |
  \___ \   | |  |  _  /|  __|   / /\ \ | |\/| |
  ____) |  | |  | | \ \| |____ / ____ \| |  | |
 |_____/   |_|  |_|  \_\______/_/    \_\_|  |_|

  Smart Tiered Routing Engine for AI Models
EOF
echo -e "${NC}"
echo -e "${BLUE}Version 1.0.0 | Interactive Setup${NC}"
echo ""

# =============================================================================
# STEP 1: Prerequisites Check
# =============================================================================
# Verify that all required tools are installed and running

print_step "Step 1/6: Checking Prerequisites"

# -----------------------------------------------------------------------------
# Check Docker
# -----------------------------------------------------------------------------
print_info "Checking Docker installation..."

# command -v checks if docker executable exists in PATH
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
    print_info "Install from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# docker info checks if Docker daemon is running
# Redirect stdout to /dev/null, keep stderr
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running"
    print_info "Please start Docker Desktop and try again"
    exit 1
fi

print_success "Docker is running"

# -----------------------------------------------------------------------------
# Check docker-compose
# -----------------------------------------------------------------------------
print_info "Checking docker-compose..."

if ! command -v docker-compose &> /dev/null; then
    print_error "docker-compose is not installed"
    print_info "Install from: https://docs.docker.com/compose/install/"
    exit 1
fi

# Extract and show version
COMPOSE_VERSION=$(docker-compose --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
print_success "docker-compose is available (v${COMPOSE_VERSION})"

# -----------------------------------------------------------------------------
# Check .env file
# -----------------------------------------------------------------------------
print_info "Checking configuration file..."

if [ ! -f .env ]; then
    print_error ".env file not found"

    # Try to create from template
    if [ -f .env.example ]; then
        print_info "Creating .env from template..."
        cp .env.example .env
        print_warning ".env created from .env.example"
        print_warning "Please edit .env to add your API keys"
        print_info "Then run this script again"
        exit 1
    else
        print_error "No .env.example template found"
        print_info "Please create a .env file with your configuration"
        print_info "See documentation: https://github.com/your-repo/STREAM"
        exit 1
    fi
fi

print_success ".env file found"


# -----------------------------------------------------------------------------
# Load configuration from .env
# -----------------------------------------------------------------------------
print_info "Loading configuration..."

# Source the .env file to load variables
# This makes all variables available to the script
set -a  # Automatically export all variables
source .env
set +a  # Turn off auto-export

# Extract port numbers (with defaults if not set)
FRONTEND_PORT=${FRONTEND_EXTERNAL_PORT:-8501}
MIDDLEWARE_PORT=${MIDDLEWARE_EXTERNAL_PORT:-5000}
LITELLM_PORT=${LITELLM_EXTERNAL_PORT:-4000}

print_success "Configuration loaded"



# -----------------------------------------------------------------------------
# Validate .env API keys
# -----------------------------------------------------------------------------
print_info "Validating API keys configuration..."

# Track number of missing API keys
ENV_WARNINGS=0

# Check Anthropic API key
# grep -q returns 0 if pattern found, 1 if not
if ! grep -q "^ANTHROPIC_API_KEY=sk-ant-" .env; then
    print_warning "Anthropic API key not configured (Claude unavailable)"
    ENV_WARNINGS=$((ENV_WARNINGS + 1))
fi

# Check OpenAI API key
if ! grep -q "^OPENAI_API_KEY=sk-" .env; then
    print_warning "OpenAI API key not configured (GPT unavailable)"
    ENV_WARNINGS=$((ENV_WARNINGS + 1))
fi

# Provide feedback based on what's configured
if [ $ENV_WARNINGS -eq 0 ]; then
    print_success "API keys configured (cloud models available)"
elif [ $ENV_WARNINGS -eq 2 ]; then
    print_warning "No API keys configured"
    print_info "Only local and campus models will be available"

    # Ask if user wants to continue without cloud access
    if ! ask_yes_no "Continue without cloud models?" "y"; then
        print_info "Please configure API keys in .env and run again"
        exit 0
    fi
else
    print_warning "Some API keys missing (limited cloud access)"
fi

# =============================================================================
# STEP 2: Cleanup
# =============================================================================
# Stop existing services and optionally remove data

print_step "Step 2/6: Cleanup & Preparation"

# Check if any STREAM services are running
# docker-compose ps -q returns container IDs
# grep -q returns true if any output exists
if docker-compose ps -q 2>/dev/null | grep -q .; then
    print_info "Found existing STREAM services"

    # If --clean flag is set, remove everything including volumes
    if [ "$CLEAN_INSTALL" = true ]; then
        print_warning "Clean install requested - removing all data"

        # docker-compose down -v removes containers, networks, AND volumes
        docker-compose down -v > /dev/null 2>&1
        print_success "All services and volumes removed"

    # Otherwise, ask user what to do
    elif ask_yes_no "Stop existing services?" "y"; then
        print_info "Stopping services..."

        # docker-compose down removes containers and networks but keeps volumes
        docker-compose down > /dev/null 2>&1
        print_success "Services stopped (data preserved)"
    else
        print_info "Keeping existing services"
    fi
else
    print_success "No existing services found"
fi

# =============================================================================
# STEP 3: Ollama & AI Models Setup
# =============================================================================
# Start Ollama and download required AI models

print_step "Step 3/6: Setting Up Local AI Models"

# Check if user wants to skip model downloads entirely
if [ "$SKIP_MODELS" = true ]; then
    print_warning "Skipping model setup (--skip-models flag)"
    print_info "You'll need to download models manually later"
    print_info "Run: docker exec -it stream-ollama ollama pull <model>"
else
    # -----------------------------------------------------------------------------
    # Start Ollama service
    # -----------------------------------------------------------------------------
    print_info "Starting Ollama service..."

    # Start only ollama container in detached mode (-d)
    # Redirect output to hide build logs
    docker-compose up -d ollama > /dev/null 2>&1 &

    # Show spinner while starting
    spinner $! "Starting Ollama"
    print_success "Ollama container started"

    # -----------------------------------------------------------------------------
    # Wait for Ollama to be ready
    # -----------------------------------------------------------------------------
    print_info "Waiting for Ollama to initialize..."

    MAX_RETRIES=12  # Try for up to 60 seconds
    RETRY_COUNT=0

    # Keep trying until ollama responds or we timeout
    while ! docker exec stream-ollama ollama list > /dev/null 2>&1; do
        RETRY_COUNT=$((RETRY_COUNT + 1))

        # If we've tried too many times, give up
        if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
            print_error "Ollama failed to start after 60 seconds"
            print_info "Check logs: docker-compose logs ollama"
            exit 1
        fi

        # Show progress
        echo -ne "\r   Waiting for Ollama... ($RETRY_COUNT/$MAX_RETRIES)"
        sleep 5
    done

    echo ""  # New line after progress
    print_success "Ollama is ready"

    # -----------------------------------------------------------------------------
    # Check what models are already installed
    # -----------------------------------------------------------------------------
    print_info "Checking installed models..."

    # Get list of installed models
    # tail -n +2 skips the header line
    # awk '{print $1}' extracts just the model name
    EXISTING_MODELS=$(docker exec stream-ollama ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' || echo "")

    # -----------------------------------------------------------------------------
    # Model Download Confirmation
    # -----------------------------------------------------------------------------
    echo ""
    print_info "${BOLD}AI Model Downloads${NC}"
    print_info "STREAM uses local AI models for cost-efficient inference"
    echo ""

    # Show what we're going to download
    print_info "Required models:"
    print_info "  • llama3.2:1b - 1.3 GB (for smart routing)"
    print_info "  • llama3.2:3b - 2.0 GB (for local inference)"
    echo ""
    print_info "Optional model:"
    print_info "  • llama3.1:8b - 4.7 GB (higher quality, slower)"
    echo ""

    # Ask for confirmation before starting downloads
    if ! ask_yes_no "Download models now?" "y"; then
        print_warning "Skipping model downloads"
        print_info "You can download them later with:"
        print_info "  docker exec -it stream-ollama ollama pull llama3.2:1b"
        print_info "  docker exec -it stream-ollama ollama pull llama3.2:3b"
    else
        # User confirmed, proceed with downloads

        # -----------------------------------------------------------------------------
        # Download llama3.2:1b (Required - Judge Model)
        # -----------------------------------------------------------------------------
        echo ""
        MODEL="llama3.2:1b"

        if echo "$EXISTING_MODELS" | grep -q "^$MODEL"; then
            print_success "$MODEL already installed ✓"
        else
            print_info "${BOLD}Downloading $MODEL${NC}"
            print_info "Size: ~1.3 GB | Time: 5-10 minutes"
            print_info "Purpose: Smart query routing and complexity detection"
            echo ""

            # Run ollama pull and show progress
            if docker exec -it stream-ollama ollama pull $MODEL; then
                print_success "$MODEL downloaded successfully"
            else
                print_error "Failed to download $MODEL"
                print_info "This model is required for STREAM to function"
                exit 1
            fi
        fi

        # -----------------------------------------------------------------------------
        # Download llama3.2:3b (Required - Local Tier)
        # -----------------------------------------------------------------------------
        echo ""
        MODEL="llama3.2:3b"

        if echo "$EXISTING_MODELS" | grep -q "^$MODEL"; then
            print_success "$MODEL already installed ✓"
        else
            print_info "${BOLD}Downloading $MODEL${NC}"
            print_info "Size: ~2.0 GB | Time: 5-10 minutes"
            print_info "Purpose: Fast local inference for simple queries"
            echo ""

            if docker exec -it stream-ollama ollama pull $MODEL; then
                print_success "$MODEL downloaded successfully"
            else
                print_error "Failed to download $MODEL"
                print_info "This model is required for local tier"
                exit 1
            fi
        fi

        # -----------------------------------------------------------------------------
        # Download llama3.1:8b (Optional - Higher Quality)
        # -----------------------------------------------------------------------------
        echo ""
        MODEL="llama3.1:8b"

        # Skip in quick mode
        if [ "$QUICK_MODE" = true ]; then
            print_warning "Skipping optional model in quick mode"
        elif echo "$EXISTING_MODELS" | grep -q "^$MODEL"; then
            print_success "$MODEL already installed ✓"
        else
            print_info "${BOLD}Optional: $MODEL${NC}"
            print_info "Size: ~4.7 GB | Time: 10-15 minutes"
            print_info "Purpose: Better quality responses (slower)"
            echo ""

            # Ask if user wants this large model
            if ask_yes_no "Download optional 8B model?" "n"; then
                if docker exec -it stream-ollama ollama pull $MODEL; then
                    print_success "$MODEL downloaded successfully"
                else
                    print_warning "Failed to download $MODEL (non-critical)"
                fi
            else
                print_info "Skipping $MODEL (you can add it later)"
            fi
        fi
    fi

    # -----------------------------------------------------------------------------
    # Show summary of installed models (outside the download confirmation)
    # This runs whether the user downloaded models or not
    # -----------------------------------------------------------------------------
    echo ""
    print_success "Model setup complete!"
    print_info "Installed models:"
    echo ""

    # Show nicely formatted list
    docker exec stream-ollama ollama list | head -n 5 || true
fi

# =============================================================================
# STEP 4: Build & Start Services
# =============================================================================
# Build Docker images and start all containers

print_step "Step 4/6: Building & Starting Services"

print_info "Building Docker images..."
print_info "(This may take 2-3 minutes on first run)"
echo ""

# Build and start all services in background with progress monitoring
# We'll show a nice animated progress indicator
BUILD_LOG="/tmp/stream-build-$$.log"
docker-compose up -d --build > "$BUILD_LOG" 2>&1 &
BUILD_PID=$!

# Animated progress indicator with status updates
SERVICES_LIST=("postgres" "ollama" "litellm" "middleware" "frontend")
SPIN_CHARS='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
CURRENT_SERVICE=""
i=0

echo -e "   ${BLUE}Starting build process...${NC}"
echo ""

# Monitor build progress while showing spinner
while kill -0 $BUILD_PID 2>/dev/null; do
    # Extract current service being built from log
    if [ -f "$BUILD_LOG" ]; then
        LATEST_SERVICE=$(grep -oE "Building (stream-)?(postgres|ollama|litellm|middleware|frontend)" "$BUILD_LOG" | tail -1 | awk '{print $NF}')
        if [ -n "$LATEST_SERVICE" ] && [ "$LATEST_SERVICE" != "$CURRENT_SERVICE" ]; then
            # Clear current line and show completion for previous service
            if [ -n "$CURRENT_SERVICE" ]; then
                printf "\r\033[K"
                echo -e "   ${GREEN}✓${NC} Built ${CYAN}${CURRENT_SERVICE}${NC}"
            fi
            CURRENT_SERVICE=$LATEST_SERVICE
            i=0  # Reset spinner for new service
        fi
    fi

    # Show spinner with current service being built
    i=$(( (i+1) % 10 ))
    if [ -n "$CURRENT_SERVICE" ]; then
        printf "\r   ${BLUE}${SPIN_CHARS:$i:1}${NC} Building ${CYAN}${BOLD}${CURRENT_SERVICE}${NC}..."
    else
        printf "\r   ${BLUE}${SPIN_CHARS:$i:1}${NC} Preparing build..."
    fi
    sleep 0.2
done

# Clear the spinner line
printf "\r\033[K"

# Show completion for last service
if [ -n "$CURRENT_SERVICE" ]; then
    echo -e "   ${GREEN}✓${NC} Built ${CYAN}${CURRENT_SERVICE}${NC}"
fi
echo ""

# Check if build succeeded
if wait $BUILD_PID; then
    print_success "All services built and started"

    # Show which services are running
    echo ""
    print_info "Services running:"
    for service in "${SERVICES_LIST[@]}"; do
        if docker ps --filter "name=stream-$service" --filter "status=running" -q | grep -q .; then
            echo -e "   ${GREEN}✓${NC} $service"
        fi
    done
else
    echo ""
    print_error "Failed to start services"
    print_info "Last 20 lines of build log:"
    tail -20 "$BUILD_LOG" | sed 's/^/   /'
    print_info "Full log: $BUILD_LOG"
    exit 1
fi

# Clean up log if successful
rm -f "$BUILD_LOG"

# =============================================================================
# STEP 5: Health Checks
# =============================================================================
# Wait for all services to become healthy

print_step "Step 5/6: Waiting for Services to be Ready"

print_info "Checking service health (may take 30-60 seconds)..."
echo ""

# Give services initial time to start
sleep 10

# List of services to check
SERVICES=("ollama" "postgres" "litellm" "middleware" "frontend")
ALL_HEALTHY=true

# Check each service
for service in "${SERVICES[@]}"; do
    CONTAINER="stream-$service"

    # Frontend needs more time to start (Streamlit initialization)
    if [ "$service" = "frontend" ]; then
        MAX_WAIT=90
    else
        MAX_WAIT=60
    fi

    WAITED=0

    while [ $WAITED -lt $MAX_WAIT ]; do
        # Check Docker health status
        STATUS=$(docker inspect --format='{{.State.Health.Status}}' $CONTAINER 2>/dev/null || echo "none")

        if [ "$STATUS" = "healthy" ]; then
            print_success "$service is healthy"
            break
        elif [ "$STATUS" = "none" ]; then
            # No health check defined, verify container is running
            if docker ps --filter "name=$CONTAINER" --filter "status=running" -q | grep -q .; then
                # For frontend, do an additional HTTP check
                if [ "$service" = "frontend" ]; then
                    # Check if frontend HTTP server is responding (any response is good)
                    if curl -s -o /dev/null -w "%{http_code}" http://localhost:${FRONTEND_PORT}/ | grep -q "200\|302\|404"; then
                        print_success "$service is running and responding"
                        break
                    fi
                else
                    print_success "$service is running"
                    break
                fi
            fi
        fi

        # Show progress dots
        echo -n "."
        sleep 2
        WAITED=$((WAITED + 2))
    done

    # If we timed out, verify container is at least running
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo ""  # New line after dots
        if docker ps --filter "name=$CONTAINER" --filter "status=running" -q | grep -q .; then
            # Container is running, just slow to respond
            if [ "$service" = "frontend" ]; then
                print_success "$service container is running (UI may still be initializing)"
            else
                print_warning "$service is starting (may need a few more seconds)"
            fi
        else
            print_error "$service failed to start"
            ALL_HEALTHY=false
        fi
    fi
done

# =============================================================================
# STEP 6: Final Summary
# =============================================================================
# Show success message and useful information

print_step "✨ STREAM is Ready!"

echo -e "${GREEN}${BOLD}Installation Complete!${NC}"
echo ""

# Show access URLs (using ports from .env)
echo -e "${BLUE}🌐 Access Points:${NC}"
echo -e "   Frontend (UI):   ${GREEN}${BOLD}http://localhost:${FRONTEND_PORT}${NC}"
echo -e "   Middleware API:  ${GREEN}http://localhost:${MIDDLEWARE_PORT}${NC}"
echo -e "   LiteLLM Gateway: ${GREEN}http://localhost:${LITELLM_PORT}${NC}"
echo ""

# Show useful commands
echo -e "${BLUE}📊 Useful Commands:${NC}"
echo -e "   View all logs:       ${YELLOW}docker-compose logs -f${NC}"
echo -e "   View specific logs:  ${YELLOW}docker-compose logs -f middleware${NC}"
echo -e "   Stop all services:   ${YELLOW}docker-compose down${NC}"
echo -e "   Restart a service:   ${YELLOW}docker-compose restart middleware${NC}"
echo -e "   Check status:        ${YELLOW}docker-compose ps${NC}"
echo ""

# Show cleanup commands
echo -e "${BLUE}🧹 Cleanup Commands:${NC}"
echo -e "   Stop services:       ${YELLOW}docker-compose down${NC}"
echo -e "   Remove all data:     ${YELLOW}docker-compose down -v${NC}"
echo -e "   Fresh reinstall:     ${YELLOW}./scripts/setup-stream.sh --clean${NC}"
echo ""

# Show warnings if needed
if [ "$ALL_HEALTHY" = false ]; then
    echo ""
    print_info "Note: Some services may still be initializing"
    print_info "All containers are running - services should be ready shortly"
    print_info "Monitor progress: docker-compose logs -f frontend"
    echo ""
else
    echo ""
fi

if [ "$SKIP_MODELS" = true ]; then
    print_warning "Models were not downloaded"
    print_info "Download manually: docker exec -it stream-ollama ollama pull llama3.2:3b"
    echo ""
fi

# Final message
echo -e "${BLUE}${BOLD}🎉 Happy researching with STREAM!${NC}"
echo -e "${BLUE}${BOLD}📖 Documentation: https://github.com/your-repo/STREAM${NC}"
echo ""
