#!/usr/bin/env python3
"""
Test the new interesting grid shapes.
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_interesting_shapes():
    """Test different grid shapes."""
    
    try:
        from ssp_modeling.simulation.grid_generator import generate_grid
        
        print("🎨 Testing Interesting Grid Shapes\n")
        
        # Test different shapes
        shapes_to_test = [
            ("random", "Random (classic)"),
            ("cross", "Cross Pattern"),
            ("diamond", "Diamond Shape"),
            ("rings", "Concentric Rings"),
            ("spiral", "Spiral Pattern"),
            ("maze", "Maze Corridors"),
            ("auto", "Auto-selected")
        ]
        
        grid_size = 11  # Good size to see patterns
        
        for shape_style, description in shapes_to_test:
            print(f"🔹 {description} ({shape_style}):")
            
            try:
                grid, topo = generate_grid(
                    rows=grid_size, 
                    cols=grid_size,
                    shape_style=shape_style,
                    return_topology=True,
                    symmetric=True,  # Makes patterns more obvious
                    target_black_frac=0.18
                )
                
                # Print the layout
                print("   Layout:")
                for r, row in enumerate(topo.layout):
                    print("   ", end="")
                    for c, cell in enumerate(row):
                        if cell == 1:  # Black square
                            print("█ ", end="")
                        else:  # White square
                            print("· ", end="")
                    print()
                
                print(f"   📊 Stats: {len(grid.entries)} entries")
                
                # Count entries by length
                lengths = {}
                for entry in grid.entries.values():
                    lengths[entry.L] = lengths.get(entry.L, 0) + 1
                
                length_str = ", ".join(f"{count}×{length}" for length, count in sorted(lengths.items()))
                print(f"   📏 Lengths: {length_str}")
                print()
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                print()
        
        # Test larger grids with auto-selection
        print("🎯 Testing Larger Grids with Auto Shape Selection:")
        
        for size in [13, 15, 17]:
            print(f"\n📐 {size}×{size} grid:")
            
            try:
                grid, topo = generate_grid(
                    rows=size,
                    cols=size, 
                    shape_style="auto",
                    return_topology=True,
                    symmetric=True,
                    target_black_frac=0.16
                )
                
                print(f"   ✅ Generated successfully!")
                print(f"   📊 {len(grid.entries)} entries")
                
                # Show a mini preview (just corners and center)
                print("   Preview (corners + center):")
                preview_size = 5
                for r in range(preview_size):
                    print("   ", end="")
                    for c in range(preview_size):
                        cell = topo.layout[r][c]
                        print("█ " if cell == 1 else "· ", end="")
                    print(" ... ", end="")
                    for c in range(size - preview_size, size):
                        cell = topo.layout[r][c]
                        print("█ " if cell == 1 else "· ", end="")
                    print()
                
                print("   ...")
                
                for r in range(size - preview_size, size):
                    print("   ", end="")
                    for c in range(preview_size):
                        cell = topo.layout[r][c]
                        print("█ " if cell == 1 else "· ", end="")
                    print(" ... ", end="")
                    for c in range(size - preview_size, size):
                        cell = topo.layout[r][c]
                        print("█ " if cell == 1 else "· ", end="")
                    print()
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        print("\n✅ Shape testing complete!")
        print("\n💡 Usage in your code:")
        print("   grid = generate_grid(rows=15, cols=15, shape_style='cross')")
        print("   grid = generate_grid(rows=17, cols=17, shape_style='auto')  # Auto-picks interesting shapes")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_interesting_shapes()