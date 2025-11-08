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

# Simple, direct imports!
from .types import Grid, EntryState, SimConfig, Topology
from .grid_generator import generate_grid 

# Add src directory to Python path
current_file = Path(__file__)
src_dir = current_file.parent.parent.parent  # Go up to src directory
sys.path.insert(0, str(src_dir))

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
                actions = policy.plan_epoch(self.grid, model)
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
        def plan_epoch(self, grid, model):
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

def create_simple_value_function():
    """Create simple value function"""
    class ValueFunction:
        def __call__(self, *args):
            return 0.0
    return ValueFunction

def make_random_env(seed: Optional[int] = None):
    # Get the Environment class properly
    Environment, EntryState, Grid, SimConfig = get_environment_classes()
    
    # Simple, direct usage - no complex imports!
    grid_obj, topo = generate_grid(rows = 5, cols = 5, return_topology=True)
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

def main():
    print("=== Starting Crossword Simulation ===")
    
    # Get classes
    XGBProbability, DummyModel = get_model_classes()
    GreedyP, OneStepRollout = get_policy_classes()
    ValueFunction = get_value_function()
    
    # Choose model
    use_xgb = True  # flip to True and set model_path if you want to use your XGBoost
    print(f"Using XGBoost model: {use_xgb}")
    
    if use_xgb and XGBProbability:
        model_path = get_project_root() / "data" / "processed" / "crossword_model.pkl"
        print(f"Loading model from: {model_path}")
        model = XGBProbability(model_path=model_path)
        print("✓ XGBoost model loaded successfully")
    else:
        model = DummyModel(bias=-1.0)
        print("✓ Using dummy model")

    # Choose policy: GreedyP or OneStepRollout
    policy_name = "rollout"  # "greedy" or "rollout"
    print(f"Using policy: {policy_name}")
    
    if policy_name == "greedy" and GreedyP:
        policy = GreedyP(min_p_for_attempt=0.0)
        print("✓ Greedy policy initialized")
    elif OneStepRollout and ValueFunction:
        policy = OneStepRollout(ValueFunction(), hint_threshold=0.0)
        print("✓ One-step rollout policy initialized")
    else:
        # Fallback to simple policy
        GreedyP, _ = get_policy_classes()
        policy = GreedyP()
        print("✓ Simple policy initialized")

    print("\n=== Setting up environment ===")
    env = make_random_env() # Random seed for different results each time
    print("✓ Environment created")
    
    print("\n" + "="*60)
    print("🎮 STARTING SIMULATION")
    print("="*60)
    
    res = env.run_episode(policy, model)
    
    print(f"\n" + "="*60)
    print("🏁 SIMULATION COMPLETE")
    print("="*60)
    print(f"📊 FINAL RESULTS:")
    print(f"   Policy: {policy.name() if hasattr(policy,'name') else policy.__class__.__name__}")
    print(f"   Success: {res.terminated} | Epochs: {res.steps} | Total cost: {res.total_cost}")
    print(f"   Entries solved: {res.solved_count}/{len(env.grid.entries)} | Hints used: {res.hints_used}")
    
    if res.steps <= 10:
        print(f"\n📋 EPOCH SUMMARY:")
        for t, ep in enumerate(res.log, 1):
            result = f"✅ Solved {ep.solved_entry}" if ep.solved_entry else f"💡 Used hint"
            print(f"   Epoch {t:2d}: {result} (cost: {ep.cost})")
    else:
        print(f"\n📋 FIRST 10 EPOCHS:")
        for t, ep in enumerate(res.log[:10], 1):
            result = f"✅ Solved {ep.solved_entry}" if ep.solved_entry else f"💡 Used hint"
            print(f"   Epoch {t:2d}: {result} (cost: {ep.cost})")

if __name__ == "__main__":
    main()

