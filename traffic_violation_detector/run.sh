#!/bin/bash
# =============================================================================
# Traffic Violation Detection - Run Script
# =============================================================================
# Convenience script for running the pipeline

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Function to print colored messages
print_msg() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_err() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_warn "Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    print_msg "Installing dependencies..."
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Run the pipeline with provided arguments
print_msg "Running Traffic Violation Detection Pipeline..."
python -m traffic_violation_detector "$@"
