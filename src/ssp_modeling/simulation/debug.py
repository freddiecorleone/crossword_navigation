# src/ssp_modeling/simulation/debug.py
"""
Simple debug utility for toggling print statements on/off throughout the simulation.

Usage:
    from .debug import debug_print
    
    debug_print("This will only print if DEBUG is True")
    debug_print("Multiple", "arguments", "supported")
    debug_print(f"Formatted strings work: {value}")

To toggle debugging:
    Set DEBUG = True/False in this file, or
    Set environment variable: export CROSSWORD_DEBUG=1
"""

import os
from typing import Any

# Main debug flag - change this to toggle all debug output
DEBUG = True

# You can also control it via environment variable
if 'CROSSWORD_DEBUG' in os.environ:
    DEBUG = bool(int(os.environ.get('CROSSWORD_DEBUG', '0')))


def debug_print(*args: Any, **kwargs: Any) -> None:
    """Print only when DEBUG is True. Uses same signature as print()."""
    if DEBUG:
        print(*args, **kwargs)


def set_debug(enabled: bool) -> None:
    """Programmatically enable/disable debug output."""
    global DEBUG
    DEBUG = enabled


def is_debug_enabled() -> bool:
    """Check if debug output is currently enabled."""
    return DEBUG


# Convenience aliases
dprint = debug_print  # Shorter alias