# Crossword Optimization Simulation

A sophisticated crossword solving simulation using XGBoost probability models and policy optimization.

## Quick Start

### Running the Simulation

You have several options to run the simulation:

#### Option 1: Python Runner Script (Recommended)
```bash
# Clean output (default)
python3 run_simulation.py

# With detailed debug output
python3 run_simulation.py --debug

# Quiet mode (minimal output)
python3 run_simulation.py --quiet

# Show help
python3 run_simulation.py --help
```

#### Option 2: Bash Script
```bash
./run_simulation.sh
```

#### Option 3: Direct Python Module
```bash
./venv/bin/python -m src.ssp_modeling.simulation.simulator
```

### Debug Output Control

The simulation has a built-in debug system that can be toggled:

**Enable detailed output:**
- Set `DEBUG = True` in `src/ssp_modeling/simulation/debug.py`
- Or use `python3 run_simulation.py --debug`
- Or set environment variable: `export CROSSWORD_DEBUG=1`

**Disable detailed output:**
- Set `DEBUG = False` in `src/ssp_modeling/simulation/debug.py`
- Or use `python3 run_simulation.py --quiet`
- Or set environment variable: `export CROSSWORD_DEBUG=0`

### What You'll See

**With Debug Output (--debug):**
- 📊 Grid state visualization
- 🎯 Policy recommendations
- 🎲 Individual attempt details with probabilities
- ✅/❌ Success/failure for each move
- 📋 Updated grid after each change

**Without Debug Output (default):**
- Clean epoch summaries
- Final results only
- Faster execution

## Project Structure

```
src/
├── ssp_modeling/
│   ├── simulation/          # Core simulation engine
│   │   ├── simulator.py     # Main simulation runner
│   │   ├── environment.py   # Crossword environment
│   │   ├── policy.py        # Solving policies
│   │   ├── grid_generator.py # Grid generation
│   │   └── debug.py         # Debug output control
│   ├── models/              # XGBoost probability models
│   └── utils/               # Utilities
├── training_app/            # Data collection tools
└── data/                    # Models and datasets
```

## Troubleshooting

### Common Issues

**"No module named 'ssp_modeling'" error:**
- Use the provided runner scripts instead of running Python directly
- Make sure you're in the project root directory

**"Virtual environment not found" error:**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt  # If you have requirements
```

**Import errors:**
- The simulation automatically configures Python paths
- Use the provided runner scripts for best results

### Getting Help

Run the simulation with help:
```bash
python3 run_simulation.py --help
```

Test the debug system:
```bash
python3 src/ssp_modeling/simulation/debug_demo.py
```