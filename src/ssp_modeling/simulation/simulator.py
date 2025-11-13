#!/usr/bin/env python3
# toy_sim.py — minimal crossword simulation with proper imports

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Any
import math
import random
import sys
import os
from pathlib import Path
from .debug import debug_print
from .environment import Environment
from .value_diff import HintAwareExpectedCostDifference





# Add src directory to Python path FIRST
current_file = Path(__file__)
src_dir = current_file.parent.parent.parent  # Go up to src directory
sys.path.insert(0, str(src_dir))

# Now we can do all imports
from .types import Grid, EntryState, SimConfig, Topology, EpisodeResult, ProbabilityModel
from .policy_core import Policy
from .policies import OneStepRolloutPolicy, NStepRolloutPolicy
from .grid_generator import generate_grid 
from ssp_modeling.models.xgb_model import XGBProbability, DummyModel
from ssp_modeling.utils.path import get_project_root



def reset():
        from .memo import TTABLE
        TTABLE.clear()   # clear global memo

# Environment pool for better memory management
_environment_pool = []
_pool_max_size = 50  # Keep up to 50 environments in pool

def _get_pooled_environment(rows: int, cols: int, seed: Optional[int] = None) -> Environment:
    """Get environment from pool or create new one."""
    global _environment_pool
    
    # Try to find matching environment in pool
    for i, (env, env_rows, env_cols) in enumerate(_environment_pool):
        if env_rows == rows and env_cols == cols:
            # Remove from pool and reset it
            _environment_pool.pop(i)
            _reset_environment(env, seed)
            return env
    
    # No matching environment found, create new one
    return make_random_env(rows, cols, seed)

def _return_environment_to_pool(env: Environment, rows: int, cols: int):
    """Return environment to pool for reuse."""
    global _environment_pool
    
    if len(_environment_pool) < _pool_max_size:
        _environment_pool.append((env, rows, cols))

def _reset_environment(env: Environment, seed: Optional[int] = None):
    """Reset environment state for reuse."""
    # Reset all entries to unsolved state
    for entry in env.grid.entries.values():
        entry.filled_indices.clear()
        entry.solved = False
        entry.guess_with_current_letters = False
        entry.num_attempts = 0
    
    # Reset config seed if provided
    if seed is not None:
        env.cfg.rng_seed = seed



def make_random_env(rows = 5, cols= 5, seed: Optional[int] = random.randint(1, 100000)):
    # Get the Environment class properly
    
    # Simple, direct usage - no complex imports!
    grid_obj, topo = generate_grid(rows = rows, cols = cols, return_topology=True)
    cfg = SimConfig(rng_seed=seed, hint_cost=1)
    return Environment(grid=grid_obj, topo=topo, cfg=cfg)




def simulate_fill(policy: Policy, rows = 5, cols = 5, seed: Optional[int] = None, model: Optional[ProbabilityModel] = None) -> EpisodeResult:

    if model is None:
        model_path = get_project_root() / "data" / "processed" / "crossword_model.pkl"
        print(f"Loading model from: {model_path}")
        model = XGBProbability(model_path=model_path)
    
    # Use pooled environment for better memory management
    env = _get_pooled_environment(rows, cols, seed)
    
    # Debug: Print grid info
    debug_print(f"🔍 Grid Debug - Seed: {seed}")
    debug_print(f"   Grid size: {rows}x{cols}")
    debug_print(f"   Total entries: {len(env.grid.entries)}")
    debug_print(f"   Entry IDs: {list(env.grid.entries.keys())}")
    for entry_id, entry in env.grid.entries.items():
        debug_print(f"   {entry_id}: Length = {entry.L}")
    
    result = env.run_episode(policy, model)
    reset()
    
    # Return environment to pool for reuse
    _return_environment_to_pool(env, rows, cols)
    
    print(f"🎯 Result: {result.steps} epochs, {result.total_cost} total cost")
    return result 

if __name__ == "__main__":
    # Try different grid sizes for more variation
    import random

    grid_sizes = [(7, 7)]

    for rows, cols in grid_sizes:
        debug_print(f"\n{'='*50}")
        debug_print(f"Testing {rows}x{cols} grid:")
        value = HintAwareExpectedCostDifference()

        result = simulate_fill(NStepRolloutPolicy(value=value, depth=2), rows=rows, cols=cols, seed=random.randint(1, 10000))
        debug_print(f"Result: {result.steps} epochs, {result.total_cost} total cost")
        debug_print('='*50)




    
   