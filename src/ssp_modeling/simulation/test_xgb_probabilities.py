# src/ssp_modeling/simulation/test_xgb_probabilities.py
"""
Test script for exploring XGB model probability predictions.
This script tests various EntryState configurations to understand model behavior.
"""

from .types import Grid, EntryState, SimConfig, Topology, EpisodeResult, ProbabilityModel
from .value import ValueFunction, IsolatedExpectedCost
from ..models.xgb_model import XGBProbability, DummyModel
from ..utils.path import get_project_root

def test_xgb_model_probabilities():
    """Test XGB model with various EntryState configurations."""
    
    print("🧪 XGB Model Probability Testing")
    print("=" * 50)
    
    # Load the model
    model_path = get_project_root() / "data" / "processed" / "crossword_model.pkl"
    
    try:
        xgb_model = XGBProbability(str(model_path))
        print(f"✅ Loaded XGB model from: {model_path}")
    except Exception as e:
        print(f"❌ Failed to load XGB model: {e}")
        print("🔄 Using DummyModel instead")
        xgb_model = DummyModel()
    
    # Create comparison with DummyModel
    dummy_model = DummyModel()
    
    print("\n📊 Probability Comparisons:")
    print("Format: [XGB_Prob | Dummy_Prob] - Description")
    print("-" * 60)
    
    # Test cases
    test_cases = [
        # Basic cases
        (EntryState(L=5, filled_indices=set()), "5-letter word, no filled letters"),
        (EntryState(L=3, filled_indices=set()), "3-letter word, no filled letters"),
        (EntryState(L=15, filled_indices=set()), "15-letter word, no filled letters"),
        
        # 4-letter word progression (0, 1, 2, 3 letters revealed)
        (EntryState(L=4, filled_indices=set()), "4-letter word, 0 letters revealed"),
        (EntryState(L=4, filled_indices={0}), "4-letter word, 1 letter revealed"),
        (EntryState(L=4, filled_indices={0, 2}), "4-letter word, 2 letters revealed"),
        (EntryState(L=4, filled_indices={0, 1, 3}), "4-letter word, 3 letters revealed"),
        
        # 3-letter word progression (0, 1, 2 letters revealed)
        (EntryState(L=3, filled_indices=set()), "3-letter word, 0 letters revealed"),
        (EntryState(L=3, filled_indices={1}), "3-letter word, 1 letter revealed"),
        (EntryState(L=3, filled_indices={0, 2}), "3-letter word, 2 letters revealed"),
        
        # First letter revealed
        (EntryState(L=5, filled_indices={0}), "5-letter word, first letter revealed"),
        (EntryState(L=5, filled_indices={4}), "5-letter word, last letter revealed"),
        (EntryState(L=5, filled_indices={0, 4}), "5-letter word, first & last letters"),
        
        # Multiple letters
        (EntryState(L=5, filled_indices={0, 1, 2}), "5-letter word, first 3 consecutive"),
        (EntryState(L=5, filled_indices={0, 2, 4}), "5-letter word, alternating pattern"),
        (EntryState(L=5, filled_indices={1, 3}), "5-letter word, middle positions"),
        
        # Nearly complete
        (EntryState(L=5, filled_indices={0, 1, 2, 3}), "5-letter word, 4/5 letters filled"),
        (EntryState(L=5, filled_indices={0, 1, 3, 4}), "5-letter word, missing middle letter"),
        
        # Different lengths with varying fill
        (EntryState(L=7, filled_indices={0, 6}), "7-letter word, first & last"),
        (EntryState(L=10, filled_indices={0, 2, 4, 6, 8}), "10-letter word, every other position"),
        (EntryState(L=4, filled_indices={1, 2}), "4-letter word, middle two letters"),
    ]
    
    for i, (entry_state, description) in enumerate(test_cases, 1):
        try:
            xgb_prob = xgb_model.prob_solve_entry_state(entry_state)
            dummy_prob = dummy_model.prob_solve_entry_state(entry_state)
            
            print(f"{i:2d}. [{xgb_prob:.4f} | {dummy_prob:.4f}] - {description}")
            
        except Exception as e:
            print(f"{i:2d}. [ERROR | ERROR] - {description} (Error: {e})")
    
    print("\n🎯 Key Insights to Look For:")
    print("- How does probability change with word length?")
    print("- Do first/last letters provide more value?") 
    print("- How much does consecutive vs scattered letters matter?")
    print("- At what point does probability become very high?")
    
    return xgb_model, dummy_model

def test_grid_context():
    """Test model with full grid context vs EntryState only."""
    
    print("\n🔗 Grid Context vs EntryState Testing")
    print("=" * 50)
    
    # Create a simple grid
    entries = {
        "1A": EntryState(L=5, filled_indices={0, 2}),
        "1D": EntryState(L=4, filled_indices={0}),  
        "3D": EntryState(L=3, filled_indices=set()),
    }
    crossings = {
        "1A": {"1D": [0], "3D": [4]},
        "1D": {"1A": [0]},
        "3D": {"1A": [0]},
    }
    grid = Grid(entries=entries, crossings=crossings)
    
    model_path = get_project_root() / "data" / "processed" / "crossword_model.pkl"
    
    try:
        model = XGBProbability(str(model_path))
    except:
        model = DummyModel()
    
    print("Comparing Grid context vs EntryState-only predictions:")
    print("Format: [Grid_Context | EntryState_Only] - Entry")
    print("-" * 55)
    
    for entry_id, entry_state in entries.items():
        try:
            grid_prob = model.prob_solve(grid, entry_id)
            entry_prob = model.prob_solve_entry_state(entry_state)
            
            print(f"[{grid_prob:.4f} | {entry_prob:.4f}] - {entry_id} (L={entry_state.L}, filled={len(entry_state.filled_indices)})")
            
        except Exception as e:
            print(f"[ERROR | ERROR] - {entry_id} (Error: {e})")

def test_probability_progression():
    """Test how probability changes as we add more letters to a 5-letter word."""
    
    print("\n📈 Probability Progression Testing")
    print("=" * 50)
    
    model_path = get_project_root() / "data" / "processed" / "crossword_model.pkl"
    
    try:
        model = XGBProbability(str(model_path))
    except:
        model = DummyModel()
        print("Using DummyModel for progression test")
    
    print("5-letter word probability as we add letters:")
    print("Letters Filled | Probability | Pattern")
    print("-" * 40)
    
    # Test different fill patterns
    fill_patterns = [
        (set(), "Empty"),
        ({0}, "First only"),
        ({4}, "Last only"),
        ({0, 4}, "First + Last"),
        ({0, 1}, "First two"),
        ({2}, "Middle only"),
        ({1, 3}, "Symmetric middle"),
        ({0, 2, 4}, "Alternating"),
        ({0, 1, 2}, "First three consecutive"),
        ({0, 1, 2, 3}, "Four consecutive"),
        ({0, 1, 3, 4}, "Missing middle"),
        ({0, 1, 2, 3, 4}, "Complete"),
    ]
    
    for filled_indices, pattern in fill_patterns:
        try:
            entry_state = EntryState(L=5, filled_indices=filled_indices)
            prob = model.prob_solve_entry_state(entry_state)
            
            filled_count = len(filled_indices)
            print(f"     {filled_count}/5      |   {prob:.4f}   | {pattern}")
            
        except Exception as e:
            print(f"     {len(filled_indices)}/5      |   ERROR   | {pattern} ({e})")

if __name__ == "__main__":
    # Run all tests
    print("🚀 Starting XGB Model Testing Suite")
    print("=" * 60)
    
    # Test 1: Basic probability comparisons
    xgb_model, dummy_model = test_xgb_model_probabilities()
    
    # Test 2: Grid context vs EntryState-only
    test_grid_context()
    
    # Test 3: Probability progression
    test_probability_progression()
    
    print("\n✅ Testing Complete!")
    print("=" * 60)