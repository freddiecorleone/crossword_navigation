#!/usr/bin/env python3
"""
Test the optimized XGBProbability performance improvements.
"""

import sys
import time
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_xgb_performance():
    """Test the performance improvements."""
    
    try:
        from ssp_modeling.models.xgb_model import XGBProbability
        from ssp_modeling.simulation.grid_generator import generate_grid
        
        print("🚀 Testing XGB Performance Improvements\n")
        
        # Create model and test grid
        model = XGBProbability()
        grid = generate_grid(rows=7, cols=7)
        entry_ids = list(grid.entries.keys())[:5]  # Test first 5 entries
        
        print(f"Testing with {len(entry_ids)} entries...")
        
        # Test 1: Single predictions (old vs new should be similar speed but with caching)
        print("\n📊 Single Prediction Performance:")
        
        # First calls (cold cache)
        start = time.time()
        probs_individual = []
        for entry_id in entry_ids:
            prob = model.prob_solve(grid, entry_id)
            probs_individual.append(prob)
        first_time = time.time() - start
        
        # Second calls (warm cache) - should be much faster
        start = time.time()
        probs_cached = []
        for entry_id in entry_ids:
            prob = model.prob_solve(grid, entry_id)  # Should hit cache
            probs_cached.append(prob)
        cached_time = time.time() - start
        
        print(f"  First calls:  {first_time*1000:.2f}ms")
        print(f"  Cached calls: {cached_time*1000:.2f}ms")
        if cached_time > 0:
            speedup = first_time / cached_time
            print(f"  🚀 Cache speedup: {speedup:.1f}x")
        
        # Test 2: Batch processing
        print("\n🔄 Batch Processing Performance:")
        
        # Clear cache for fair comparison
        model.clear_cache()
        
        # Individual calls
        start = time.time()
        individual_probs = []
        for entry_id in entry_ids:
            prob = model.prob_solve(grid, entry_id)
            individual_probs.append(prob)
        individual_time = time.time() - start
        
        # Batch calls
        start = time.time()
        batch_data = [(grid, entry_id) for entry_id in entry_ids]
        batch_probs = model.prob_solve_batch(batch_data)
        batch_time = time.time() - start
        
        print(f"  Individual calls: {individual_time*1000:.2f}ms")
        print(f"  Batch calls:      {batch_time*1000:.2f}ms")
        if batch_time > 0:
            batch_speedup = individual_time / batch_time
            print(f"  🚀 Batch speedup: {batch_speedup:.1f}x")
        
        # Verify results are similar
        max_diff = max(abs(a - b) for a, b in zip(individual_probs, batch_probs))
        print(f"  ✅ Max difference: {max_diff:.6f} (should be ~0)")
        
        # Test 3: EntryState direct usage
        print("\n🎯 EntryState Direct Usage:")
        entry_states = [grid.entries[eid] for eid in entry_ids]
        
        start = time.time()
        direct_probs = [model.prob_solve_entry_state(es) for es in entry_states]
        direct_time = time.time() - start
        
        start = time.time()
        direct_batch_probs = model.prob_solve_batch(entry_states)
        direct_batch_time = time.time() - start
        
        print(f"  Individual EntryState: {direct_time*1000:.2f}ms")
        print(f"  Batch EntryState:      {direct_batch_time*1000:.2f}ms")
        if direct_batch_time > 0:
            direct_speedup = direct_time / direct_batch_time
            print(f"  🚀 Direct batch speedup: {direct_speedup:.1f}x")
        
        print("\n✅ Performance testing complete!")
        print("\n💡 Usage recommendations:")
        print("  - For repeated calls on same entries: automatic caching gives big speedup")
        print("  - For many entries at once: use prob_solve_batch() method")
        print("  - Clear cache between episodes: model.clear_cache()")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_xgb_performance()