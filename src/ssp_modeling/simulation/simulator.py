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
from .value_diff import IsolatedExpectedCost
from .environment import Environment





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



def make_random_env(rows = 5, cols= 5, seed: Optional[int] = random.randint(1, 100000)):
    # Get the Environment class properly
    
    # Simple, direct usage - no complex imports!
    grid_obj, topo = generate_grid(rows = rows, cols = cols, return_topology=True)
    cfg = SimConfig(rng_seed=seed, hint_cost=1)
    return Environment(grid=grid_obj, topo=topo, cfg=cfg)


def make_toy_env(seed: Optional[int] = None):
    """Create a simple 3-entry crossword environment"""
    # Get the classes we need
    Environment, EntryState, Grid, SimConfig = get_environment_classes()
    
    entries = {
        "A1": EntryState(L=5, filled_indices=set()),
        "D1": EntryState(L=4, filled_indices=set()),  
        "D2": EntryState(L=3, filled_indices=set()),
    }
    crossings = {
        "A1": {"D1": [0], "D2": [3]},
        "D1": {"A1": [0]},
        "D2": {"A1": [0]},
    }
    grid = Grid(entries=entries, crossings=crossings)
    cfg = SimConfig(rng_seed=seed, hint_cost=1)
    
    # Create a simple topology for the toy example
    topo = Topology(
        shape=(3, 5),  # Approximating a 3x5 grid
        layout=[[0]*5 for _ in range(3)],  # All white for simplicity
        entry_cells={},  # Could populate this but not needed for basic sim
        cell_to_entries={},
        numbers={}
    )
    
    return Environment(grid=grid, topo=topo, cfg=cfg)

def simulate_fill(policy: Policy, rows = 5, cols = 5, seed: Optional[int] = None, model: Optional[ProbabilityModel] = None) -> EpisodeResult:

    if model is None:
        model_path = get_project_root() / "data" / "processed" / "crossword_model.pkl"
        print(f"Loading model from: {model_path}")
        model = XGBProbability(model_path=model_path)
    
    env = make_random_env(rows = rows, cols = cols, seed=seed)
    
    # Debug: Print grid info
    debug_print(f"🔍 Grid Debug - Seed: {seed}")
    debug_print(f"   Grid size: {rows}x{cols}")
    debug_print(f"   Total entries: {len(env.grid.entries)}")
    debug_print(f"   Entry IDs: {list(env.grid.entries.keys())}")
    for entry_id, entry in env.grid.entries.items():
        debug_print(f"   {entry_id}: Length = {entry.L}")
    
    result = env.run_episode(policy, model)
    reset()
    print(f"🎯 Result: {result.steps} epochs, {result.total_cost} total cost")
    return result 

if __name__ == "__main__":
    # Try different grid sizes for more variation
    import random

    grid_sizes = [(7, 7)]

    for rows, cols in grid_sizes:
        debug_print(f"\n{'='*50}")
        debug_print(f"Testing {rows}x{cols} grid:")
        value = IsolatedExpectedCost()

        result = simulate_fill(NStepRolloutPolicy(value=value, depth=2), rows=rows, cols=cols, seed=random.randint(1, 10000))
        debug_print(f"Result: {result.steps} epochs, {result.total_cost} total cost")
        debug_print('='*50)




    
   