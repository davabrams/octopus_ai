#!/bin/bash
# Simple test runner script for octopus_ai project

set -e  # Exit on any error

echo "🐙 Running Octopus AI Test Suite"
echo "================================"

# Change to script directory
cd "$(dirname "$0")"

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.7+"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python"
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
fi

echo "Using Python: $PYTHON_CMD"

# Run the test suite
$PYTHON_CMD run_tests.py "$@"