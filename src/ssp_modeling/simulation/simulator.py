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
from .value import ValueFunction, IsolatedExpectedCost




# Add src directory to Python path FIRST
current_file = Path(__file__)
src_dir = current_file.parent.parent.parent  # Go up to src directory
sys.path.insert(0, str(src_dir))

# Now we can do all imports
from .types import Grid, EntryState, SimConfig, Topology, EpisodeResult, ProbabilityModel
from .policy import Policy
from .policy import GreedyP, OneStepRollout
from .grid_generator import generate_grid 
from ssp_modeling.models.xgb_model import XGBProbability, DummyModel
from ssp_modeling.utils.path import get_project_root

# Import base classes first to avoid circular dependencies
# We'll import everything we need in the right order
# Simple imports - no more complex runtime import functions needed!

def get_environment_classes():
    """Import environment classes safely"""
    try:
        from ssp_modeling.simulation import environment
        return environment.Environment, environment.EntryState, environment.Grid, environment.SimConfig
    except ImportError as e:
        print(f"Import error: {e}")
        # Fallback to creating simple versions
        return create_simple_classes()

def get_policy_classes():
    """Import policy classes safely"""  
    try:
        from ssp_modeling.simulation import policy
        return policy.GreedyP, policy.OneStepRollout
    except ImportError:
        return create_simple_policies()

def get_model_classes():
    """Import model classes safely"""
    try:
        from ssp_modeling.models.xgb_model import XGBProbability, DummyModel
        return XGBProbability, DummyModel
    except ImportError:
        return create_simple_models()

def get_value_function():
    """Import ValueFunction safely"""
    try:
        from ssp_modeling.simulation.value import ValueFunction
        return ValueFunction
    except ImportError:
        return create_simple_value_function()

def create_simple_classes():
    """Create simple versions of the classes if imports fail"""
    
    @dataclass 
    class EntryState:
        L: int
        filled_indices: set[int] = field(default_factory=set)
        guess_with_current_letters: bool = False
        solved: bool = False
        
        @property
        def k(self) -> int:
            return len(self.filled_indices)
    
    @dataclass
    class Grid:
        entries: Dict[str, EntryState]
        crossings: Dict[str, Dict[str, List[int]]]
    
    @dataclass
    class SimConfig:
        hint_cost: int = 1
        eps_floor: float = 1e-4  
        max_steps: int = 10000
        rng_seed: Optional[int] = None
    
    class Environment:
        def __init__(self, grid: Grid, cfg: SimConfig):
            self.grid = grid
            self.cfg = cfg
            self.rng = random.Random(cfg.rng_seed)
        
        def run_episode(self, policy, model):
            # Simple episode runner
            steps = 0
            for step in range(self.cfg.max_steps):
                if self.all_solved():
                    break
                actions = policy.plan_epoch(self.grid, model, self.cfg)
                if not actions:
                    break
                # Process actions...
                steps += 1
            
            return type('Result', (), {
                'terminated': self.all_solved(),
                'steps': steps, 
                'total_cost': steps,
                'solved_count': sum(1 for e in self.grid.entries.values() if e.solved),
                'hints_used': 0,
                'log': []
            })()
        
        def all_solved(self) -> bool:
            return all(e.solved for e in self.grid.entries.values())
    
    return Environment, EntryState, Grid, SimConfig

def create_simple_policies():
    """Create simple policy classes"""
    class SimplePolicy:
        def plan_epoch(self, grid, model, config=None):
            unsolved = [eid for eid, e in grid.entries.items() if not e.solved]
            return unsolved[:1] if unsolved else []
        
        def wants_hint(self, grid, model):
            return False
            
        def name(self):
            return "Simple"
    
    return SimplePolicy, SimplePolicy

def create_simple_models():
    """Create simple model classes"""
    class DummyModel:
        def __init__(self, bias: float = 0.0):
            self.bias = bias
        
        def predict_success_probability_simple(self, answer_length: int, first_letter_revealed: bool, num_letters_remaining: int) -> float:
            base_prob = 0.3 + self.bias * 0.1
            if first_letter_revealed:
                base_prob += 0.2
            if num_letters_remaining < answer_length / 2:
                base_prob += 0.3
            return min(max(base_prob, 0.0), 1.0)
    
    return None, DummyModel

def make_random_env(rows = 5, cols= 5, seed: Optional[int] = random.randint(1, 100000)):
    # Get the Environment class properly
    Environment, EntryState, Grid, SimConfig = get_environment_classes()
    
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
    
    print(f"🎯 Result: {result.steps} epochs, {result.total_cost} total cost")
    return result 

if __name__ == "__main__":
    # Try different grid sizes for more variation
    import random
    
    grid_sizes = [(4, 4)]
    
    for rows, cols in grid_sizes:
        debug_print(f"\n{'='*50}")
        debug_print(f"Testing {rows}x{cols} grid:")
        value = ValueFunction()

        result = simulate_fill(GreedyP(), rows=rows, cols=cols, seed=random.randint(1, 10000))
        debug_print(f"Result: {result.steps} epochs, {result.total_cost} total cost")
        debug_print('='*50)


    
   