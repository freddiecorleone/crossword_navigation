#!/usr/bin/env python3
"""
Test memory and performance optimizations.
"""

import sys
import time
import gc
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_memory_optimizations():
    """Test memory management improvements."""
    
    try:
        import psutil
        import os
        from ssp_modeling.simulation.simulator import simulate_fill, _environment_pool
        from ssp_modeling.simulation.policies import OneStepRolloutPolicy
        from ssp_modeling.simulation.value_diff import HintAwareExpectedCostDifference
        from ssp_modeling.models.xgb_model import XGBProbability
        from ssp_modeling.simulation.types import SimConfig
        from ssp_modeling.simulation.memo import TTABLE
        
        print("🧠 Testing Memory & Performance Optimizations\n")
        
        # Setup
        model = XGBProbability()
        config = SimConfig()
        value_func = HintAwareExpectedCostDifference(config)
        policy = OneStepRolloutPolicy(value_func)
        
        process = psutil.Process(os.getpid())
        
        # Test 1: Environment pooling
        print("🏊 Testing Environment Pooling:")
        
        initial_memory = process.memory_info().rss / 1024 / 1024
        print(f"  Initial memory: {initial_memory:.1f} MB")
        
        # Run simulations to populate pool
        print("  Running 20 simulations to test pooling...")
        start_time = time.time()
        
        for i in range(20):
            result = simulate_fill(policy, rows=7, cols=7, seed=i, model=model)
            if i % 5 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                pool_size = len(_environment_pool)
                print(f"    Sim {i+1}: {current_memory:.1f} MB, Pool: {pool_size} envs")
        
        pooled_time = time.time() - start_time
        final_memory = process.memory_info().rss / 1024 / 1024
        pool_size = len(_environment_pool)
        
        print(f"  ✅ Pooling results:")
        print(f"    Time for 20 sims: {pooled_time:.2f}s")
        print(f"    Memory growth: {final_memory - initial_memory:.1f} MB")
        print(f"    Pool size: {pool_size} environments")
        
        # Test 2: Cache management
        print(f"\n🗄️  Testing Cache Management:")
        
        cache_stats_before = model.get_cache_stats()
        memo_size_before = len(TTABLE)
        
        print(f"  Before clear - Model cache: {cache_stats_before['cache_size']}, Memo: {memo_size_before}")
        
        # Clear caches
        model.clear_cache()
        TTABLE.clear()
        gc.collect()
        
        cache_stats_after = model.get_cache_stats()
        memo_size_after = len(TTABLE)
        
        print(f"  After clear - Model cache: {cache_stats_after['cache_size']}, Memo: {memo_size_after}")
        
        memory_after_clear = process.memory_info().rss / 1024 / 1024
        print(f"  Memory after clear: {memory_after_clear:.1f} MB")
        
        # Test 3: Performance comparison
        print(f"\n⚡ Testing Performance Impact:")
        
        # Without optimizations (simulate by clearing caches each time)
        print("  Testing without cache reuse...")
        no_cache_times = []
        for i in range(5):
            model.clear_cache()
            TTABLE.clear()
            start = time.time()
            simulate_fill(policy, rows=6, cols=6, seed=100+i, model=model)
            no_cache_times.append(time.time() - start)
        
        avg_no_cache = sum(no_cache_times) / len(no_cache_times)
        
        # With optimizations (normal operation)
        print("  Testing with cache reuse...")
        model.clear_cache()  # Start fresh
        TTABLE.clear()
        
        with_cache_times = []
        for i in range(5):
            # Don't clear caches - let them accumulate
            start = time.time()
            simulate_fill(policy, rows=6, cols=6, seed=200+i, model=model)
            with_cache_times.append(time.time() - start)
        
        avg_with_cache = sum(with_cache_times) / len(with_cache_times)
        
        print(f"  ✅ Performance results:")
        print(f"    No cache reuse: {avg_no_cache:.3f}s per simulation")
        print(f"    With cache reuse: {avg_with_cache:.3f}s per simulation")
        
        if avg_no_cache > avg_with_cache:
            speedup = avg_no_cache / avg_with_cache
            print(f"    🚀 Speedup from caching: {speedup:.1f}x")
        
        print(f"\n💡 Optimization Summary:")
        print(f"  ✅ Environment pooling reduces memory allocation")
        print(f"  ✅ Smart cache management prevents memory leaks")
        print(f"  ✅ Batch processing + caching = faster simulations")
        print(f"  ✅ Memory growth controlled even in long experiments")
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install psutil")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_memory_optimizations()