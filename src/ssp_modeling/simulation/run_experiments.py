#!/usr/bin/env python3
"""
Run crossword simulation experiments comparing different policies.
"""

import sys
from pathlib import Path

# Add src directory to Python path FIRST
current_file = Path(__file__)
src_dir = current_file.parent.parent.parent
sys.path.insert(0, str(src_dir))

# Now import everything
from .simulator import simulate_fill
from .policy_core import Policy
from .policies import OneStepRolloutPolicy, NStepRolloutPolicy
from .value_diff import DifferenceValue, IsolatedExpectedCost
from ..models.xgb_model import XGBProbability, DummyModel
from ..utils.path import get_project_root
from .types import SimConfig
from .environment import Environment



def run_experiments(policy: Policy, rows: int, cols: int, num_trials: int, model=None) -> float:
    """Run multiple trials and return average cost."""
    results = []
    
    for i in range(num_trials):
        result = simulate_fill(policy, rows=rows, cols=cols, model=model)
        results.append(result)
    
    # Calculate average total cost
    total_costs = [r.total_cost for r in results if hasattr(r, 'total_cost')]
    if not total_costs:
        # Fallback: if results are just numbers
        total_costs = [r for r in results if isinstance(r, (int, float))]
    
    if total_costs:
        return sum(total_costs) / len(total_costs)
    else:
        print("Warning: No valid cost results found")
        return 0.0


def main():
    """Run experiments comparing policies."""
    print("🧪 Running Crossword Policy Experiments")
    print("=" * 50)
    
    # Load model ONCE for all experiments
    print("🔄 Loading XGB model...")
    try:
        model_path = get_project_root() / "data" / "processed" / "crossword_model.pkl"
        model = XGBProbability(str(model_path))
        print(f"✅ Successfully loaded XGB model from: {model_path}")
    except Exception as e:
        print(f"⚠️ Failed to load XGB model: {e}")
        print("🔄 Using DummyModel instead")
        model = DummyModel()
    
    # Create value function
    value_fn = IsolatedExpectedCost()
    
    # Test different policies
    policies = [
        ("Two step rollout", NStepRolloutPolicy(value_fn, depth=2)),
        ("OneStep rollout", OneStepRolloutPolicy(value_fn))
    ]

    grid_sizes = [(5, 5)]
    num_trials = 25
    avg_costs = []
    for rows, cols in grid_sizes:
        print(f"\n📐 Grid Size: {rows}x{cols} ({num_trials} trials each)")
        print("-" * 30)
        
        for policy_name, policy in policies:
            try:
                avg_cost = run_experiments(policy, rows=rows, cols=cols, num_trials=num_trials, model=model)
                print(f"  {policy_name}: Average cost = {avg_cost:.2f}")
                avg_costs.append([rows, cols, policy_name, avg_cost])
            except Exception as e:
                print(f"  {policy_name}: Error - {e}")
    print(avg_costs)
if __name__ == "__main__":
    main()
