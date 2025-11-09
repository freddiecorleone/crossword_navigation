#!/bin/bash
# run_simulation.sh - Easy script to run the crossword simulation

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to the project directory
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -f "venv/bin/python" ]; then
    echo "❌ Virtual environment not found. Please run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Run the simulation
echo "🚀 Starting Crossword Simulation..."
echo "📁 Working directory: $(pwd)"
echo "🐍 Using Python: $(./venv/bin/python --version)"
echo ""

./venv/bin/python -m src.ssp_modeling.simulation.simulator "$@"