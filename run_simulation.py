#!/usr/bin/env python3
"""
Simple runner script for the crossword simulation.

Usage:
    python run_simulation.py           # Run with default settings
    python run_simulation.py --debug   # Run with debug output enabled
    python run_simulation.py --help    # Show help
"""

import sys
import subprocess
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run crossword simulation')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug output (verbose)')
    parser.add_argument('--quiet', action='store_true',
                       help='Disable debug output (clean)')
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent
    python_path = project_root / "venv" / "bin" / "python"
    
    # Check if virtual environment exists
    if not python_path.exists():
        print("❌ Virtual environment not found!")
        print("🔧 Please run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt")
        sys.exit(1)
    
    # Set debug mode if requested
    if args.debug:
        import os
        os.environ['CROSSWORD_DEBUG'] = '1'
        print("🔍 Debug mode enabled")
    elif args.quiet:
        import os
        os.environ['CROSSWORD_DEBUG'] = '0'
        print("🤫 Quiet mode enabled")
    
    # Run the simulation
    print(f"🚀 Starting Crossword Simulation...")
    print(f"📁 Working directory: {project_root}")
    print(f"🐍 Using Python: {python_path}")
    print("")
    
    try:
        subprocess.run([
            str(python_path), "-m", "src.ssp_modeling.simulation.simulator"
        ], cwd=project_root, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Simulation failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⏹️  Simulation interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()