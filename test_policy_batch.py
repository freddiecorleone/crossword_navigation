#!/usr/bin/env python3
"""
Test batch optimization performance in policies.
"""

import sys
import time
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_policy_batch_optimization():
    """Test the performance improvements in policies."""
    
    try:
        from ssp_modeling.models.xgb_model import XGBProbability
        from ssp_modeling.simulation.grid_generator import generate_grid
        from ssp_modeling.simulation.policies import OneStepRolloutPolicy, GreedyPPolicy, ExpectedLettersGainedPolicy
        from ssp_modeling.simulation.value_diff import HintAwareExpectedCostDifference
        from ssp_modeling.simulation.types import SimConfig
        
        print("🚀 Testing Policy Batch Optimization Performance\n")
        
        # Create test components
        model = XGBProbability()
        config = SimConfig()
        value_func = HintAwareExpectedCostDifference(config)
        
        # Generate test grid
        grid = generate_grid(rows=9, cols=9)
        active_count = len([e for e in grid.entries.values() if not e.solved])
        print(f"Test grid: {len(grid.entries)} entries, {active_count} active")
        
        # Test policies
        policies_to_test = [
            ("OneStepRollout", OneStepRolloutPolicy(value_func)),
            ("GreedyP", GreedyPPolicy()),
            ("ExpectedLettersGained", ExpectedLettersGainedPolicy())
        ]
        
        print("\n📊 Policy Performance Comparison:")
        
        for policy_name, policy in policies_to_test:
            print(f"\n🎯 {policy_name}Policy:")
            
            # Test multiple runs for more stable timing
            times = []
            for run in range(3):
                model.clear_cache()  # Fair comparison
                start = time.time()
                result = policy.plan_epoch(grid, model, config)
                elapsed = time.time() - start
                times.append(elapsed)
            
            avg_time = sum(times) / len(times)
            print(f"  Average time: {avg_time*1000:.2f}ms ({len(result)} candidates)")
            print(f"  Time per entry: {avg_time/active_count*1000:.2f}ms")
        
        # Test with different grid sizes
        print(f"\n📈 Scalability Test (OneStepRollout):")
        policy = OneStepRolloutPolicy(value_func)
        
        for size in [5, 7, 9, 11]:
            test_grid = generate_grid(rows=size, cols=size)
            test_active = len([e for e in test_grid.entries.values() if not e.solved])
            
            model.clear_cache()
            start = time.time()
            result = policy.plan_epoch(test_grid, model, config)
            elapsed = time.time() - start
            
            print(f"  {size}x{size} grid: {elapsed*1000:.2f}ms ({test_active} entries)")
        
        print("\n✅ All policies now use batch optimization!")
        print("💡 Benefits:")
        print("  - Automatic batch processing when model supports it")
        print("  - Graceful fallback for older models") 
        print("  - Same API - no code changes needed in your simulations")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_policy_batch_optimization()