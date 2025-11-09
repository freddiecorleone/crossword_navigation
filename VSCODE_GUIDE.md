# VS Code Configuration Guide

## 🎯 Quick Start - Using the Run Button

VS Code is now configured so you can run your simulation scripts directly using the **Run** button (▶️) without import issues!

### How to Run Scripts:

#### **Option 1: Using Launch Configurations (Recommended)**
1. Open any Python file in your project
2. Click the **Run and Debug** panel (Ctrl+Shift+D / Cmd+Shift+D)
3. Select one of these configurations from the dropdown:
   - **"Run Crossword Simulation"** - Clean output
   - **"Run Simulation (Debug Mode)"** - Detailed output  
   - **"Run Experiments"** - Policy comparison experiments
   - **"Current File"** - Run whatever file you have open
   - **"Current File as Module"** - Run current file as Python module

4. Click the **▶️ Start Debugging** button

#### **Option 2: Direct File Running**
1. Open any Python file (like `simulator.py` or `run_experiments.py`)
2. Click the **▶️ Run Python File** button in the top-right corner
3. The file will run with proper Python paths configured

#### **Option 3: Using Tasks (Terminal)**
1. Press **Cmd+Shift+P** (Mac) or **Ctrl+Shift+P** (Windows/Linux)
2. Type "Tasks: Run Task"
3. Select:
   - **"Run Simulation"** - Clean output
   - **"Run Simulation (Debug)"** - Detailed output
   - **"Run Experiments"** - Policy experiments

## 🔧 What's Been Configured

### Python Environment:
- ✅ Virtual environment auto-detected: `./venv/bin/python`
- ✅ Python path includes `./src` for imports
- ✅ Environment variables loaded from `.env` file

### Launch Configurations:
- ✅ **Run Crossword Simulation**: Main simulation with clean output
- ✅ **Run Simulation (Debug Mode)**: Detailed step-by-step output
- ✅ **Run Experiments**: Policy comparison experiments  
- ✅ **Current File**: Run any Python file you have open
- ✅ **Current File as Module**: Run files as Python modules

### Settings Applied:
- ✅ Correct Python interpreter path
- ✅ Auto-activate virtual environment
- ✅ Python analysis with proper imports
- ✅ Environment variables from `.env` file
- ✅ Clean file exclusions (no `__pycache__`)

## 🚀 Examples

### Running the Main Simulation:
1. Open `src/ssp_modeling/simulation/simulator.py`
2. Click **Run and Debug** panel
3. Select "Run Crossword Simulation"
4. Click ▶️ - You'll see clean epoch summaries

### Running with Debug Output:
1. Same as above but select "Run Simulation (Debug Mode)"
2. You'll see detailed move-by-move output with probabilities

### Running Experiments:
1. Open `src/ssp_modeling/simulation/run_experiments.py`  
2. Select "Run Experiments" configuration
3. Compare different policy performances

### Running Any File:
1. Open any Python file
2. Select "Current File" configuration  
3. File runs with proper import paths

## 🐛 Troubleshooting

**Import errors?**
- Make sure you're using the launch configurations, not just F5
- The configurations set `PYTHONPATH` to include `./src`

**Wrong Python interpreter?**
- Press Cmd+Shift+P → "Python: Select Interpreter"
- Choose `./venv/bin/python`

**Module not found?**
- Use "Current File as Module" instead of "Current File"
- This runs files as `python -m module.name`

## 🎛️ Debug Control

**Toggle debug output:**
- Edit `.env` file: Change `CROSSWORD_DEBUG=0` to `CROSSWORD_DEBUG=1`
- Or use "Run Simulation (Debug Mode)" configuration
- Or edit `src/ssp_modeling/simulation/debug.py`: Set `DEBUG = True/False`

**Environment variables:**
- All settings in `.env` file are automatically loaded
- Override in launch configurations as needed

---

🎉 **You're all set!** Just click the ▶️ Run button and your simulations will work without import issues!