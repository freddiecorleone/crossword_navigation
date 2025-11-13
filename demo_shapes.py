#!/usr/bin/env python3
"""
Demo the interesting grid shapes working properly.
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def demo_interesting_shapes():
    """Demonstrate the new interesting grid shapes."""
    
    try:
        from ssp_modeling.simulation.grid_generator import generate_grid
        
        print("🎨 Crossword Grid Shape Variations\n")
        
        # Test different shapes that work well
        shapes_to_demo = [
            ("random", "Classic Random"),
            ("cross", "Cross-Influenced"),
            ("diamond", "Diamond-Influenced"), 
            ("rings", "Ring-Influenced"),
            ("spiral", "Spiral-Influenced"),
            ("maze", "Maze-Influenced"),
        ]
        
        for shape_style, description in shapes_to_demo:
            print(f"🔹 {description}:")
            
            try:
                grid, topo = generate_grid(
                    rows=11, 
                    cols=11,
                    shape_style=shape_style,
                    return_topology=True,
                    symmetric=True,
                    target_black_frac=0.18,
                    max_restarts=300
                )
                
                # Show full layout for smaller grid
                print("   Layout:")
                for r, row in enumerate(topo.layout):
                    print("   ", end="")
                    for c, cell in enumerate(row):
                        print("█ " if cell == 1 else "· ", end="")
                    print()
                
                print(f"   📊 {len(grid.entries)} entries")
                
                # Show distribution of entry lengths
                lengths = {}
                for entry in grid.entries.values():
                    lengths[entry.L] = lengths.get(entry.L, 0) + 1
                
                length_parts = []
                for length in sorted(lengths.keys()):
                    count = lengths[length]
                    length_parts.append(f"{count}×{length}")
                
                print(f"   📏 Entry lengths: {', '.join(length_parts)}")
                print()
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                print()
        
        # Test auto-selection on larger grid
        print("🎯 Auto Shape Selection (15×15):")
        
        for attempt in range(3):
            try:
                grid, topo = generate_grid(
                    rows=15,
                    cols=15,
                    shape_style="auto", 
                    return_topology=True,
                    symmetric=True,
                    max_restarts=400
                )
                
                print(f"   ✅ Attempt {attempt + 1}: Success! {len(grid.entries)} entries")
                
                # Show corner preview
                print("   Corner preview:")
                for r in range(4):
                    print("   ", end="")
                    for c in range(8):
                        cell = topo.layout[r][c]
                        print("█ " if cell == 1 else "· ", end="")
                    print(" ...")
                print("   ...")
                print()
                break
                
            except Exception as e:
                print(f"   ❌ Attempt {attempt + 1}: {str(e)[:60]}...")
                if attempt == 2:
                    print("   (Large grids can be challenging with shaped patterns)")
                    print()
        
        print("✅ Shape variations ready!")
        print("\n💡 Usage examples:")
        print("   # Classic random (always works)")
        print("   grid = generate_grid(rows=15, cols=15)")
        print()
        print("   # Specific interesting shape")
        print("   grid = generate_grid(rows=13, cols=13, shape_style='cross')")
        print()
        print("   # Auto-select interesting shape for large grids")
        print("   grid = generate_grid(rows=15, cols=15, shape_style='auto')")
        print()
        print("🎯 The shapes work as subtle influences on black square placement,")
        print("   creating more visually interesting patterns while maintaining")
        print("   valid crossword constraints!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_interesting_shapes()