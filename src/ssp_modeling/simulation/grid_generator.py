# src/ssp_modeling/simulation/grid_generator.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set, Any, Union
import random
from dataclasses import dataclass

# Simple, direct imports - no circular dependency issues!
from .types import Grid, EntryState, EntryId, Cell, Topology




# ------------------------- Generator implementation -------------------------

def generate_grid(
    rows: int = 15,
    cols: int = 15,
    min_entry_len: int = 3,
    symmetric: bool = False,
    target_black_frac: float = 0.16,
    max_restarts: int = 200,
    rng: Optional[random.Random] = None,
    return_topology: bool = False,
) -> Union[Grid, Tuple[Grid, Topology]]:
    """
    Generate a random crossword layout and return your Grid object (and optionally Topology).

    Returns:
      - Grid if return_topology=False
      - (Grid, Topology) if return_topology=True

    Constraints enforced:
      * All across/down entries have length >= min_entry_len (default 3)
      * White cells form a single connected component
      * Optional 180° rotational symmetry
      * Sizes up to 21x21

    Parameters:
      rows, cols           : grid dimensions
      min_entry_len        : minimum allowed entry length (>=3 recommended)
      symmetric            : if True, enforce 180° rotational symmetry in the layout
      target_black_frac    : approximate fraction of black squares (typ. 0.14–0.20)
      max_restarts         : how many random attempts before giving up
      rng                  : optional random.Random instance to make generation reproducible
      return_topology      : if True, also return a Topology object for visualization/numbering

    Prereqs (provided elsewhere in this module/snippet):
      - _random_layout(rows, cols, min_entry_len, symmetric, target_black_frac, rng) -> layout|None
      - _is_single_component(layout) -> bool
      - _build_entries_and_crossings(layout, min_entry_len) -> (entries, crossings, Topology)
      - Data classes: EntryState, Grid, Topology
    """
    assert 1 <= rows <= 21 and 1 <= cols <= 21, "Size must be within 1..21"
    assert min_entry_len >= 3, "Realistic American-style min length is 3+"
    
    rng = rng or random.Random()

    for _ in range(max_restarts):
        layout = _random_layout(
            R=rows,
            C=cols,
            Lmin=min_entry_len,
            symmetric=symmetric,
            target_black_frac=target_black_frac,
            rng=rng,
        )
        if layout is None:
            continue
        if not _is_single_component(layout):
            continue

        entries, crossings, topo = _build_entries_and_crossings(layout, min_entry_len)
        if not entries:
            # Extremely unlikely if _random_layout enforced min lengths, but safeguard anyway.
            continue

        grid_obj = Grid(entries=entries, crossings=crossings)
        return (grid_obj, topo) if return_topology else grid_obj

    raise RuntimeError(
        "Failed to generate a valid grid. Try more restarts or relax constraints: "
        "increase max_restarts, lower min_entry_len (>=3), or adjust target_black_frac (e.g., 0.14–0.20)."
    )


def _random_layout(
    R: int,
    C: int,
    Lmin: int,
    symmetric: bool,
    target_black_frac: float,
    rng: random.Random,
) -> Optional[List[List[int]]]:
    """
    Create a black/white layout as a 2D list of ints: 1 = black, 0 = white.
    Ensures no across or down segments shorter than Lmin.
    """
    # Start all white
    g = [[0 for _ in range(C)] for _ in range(R)]

    # Plan: place ~target number of black squares at random locations (in symmetric pairs if needed),
    # but only accept placements that keep every affected row/col segment either 0 or >= Lmin.
    target_blacks = round(target_black_frac * R * C)

    # Candidate cells in random order
    cells = [(r, c) for r in range(R) for c in range(C)]
    rng.shuffle(cells)

    def mirror(rc: Cell) -> Cell:
        return (R - 1 - rc[0], C - 1 - rc[1])

    placed = 0
    for (r, c) in cells:
        if g[r][c] == 1:
            continue

        to_place = [(r, c)]
        if symmetric:
            mr, mc = mirror((r, c))
            if (mr, mc) != (r, c):
                to_place.append((mr, mc))

        # Tentatively place blacks
        ok = True
        for rr, cc in to_place:
            if g[rr][cc] == 1:
                continue
            g[rr][cc] = 1

            # Check local constraints on affected row and column
            if not (_row_ok(g, rr, Lmin) and _col_ok(g, cc, Lmin)):
                ok = False

        # Global quick spot check (optional but helps): ensure we didn't create any too-short segments
        if ok:
            # This is heavier, but still linear: check just the affected row/col fully.
            pass

        if not ok:
            # Revert
            for rr, cc in to_place:
                g[rr][cc] = 0
            continue

        # Accept
        placed += sum(1 for rr, cc in to_place if g[rr][cc] == 1)

        if placed >= target_blacks:
            break

    # Final pass: if the grid contains any 1- or 2-length segments, reject
    if not _all_rows_ok(g, Lmin) or not _all_cols_ok(g, Lmin):
        return None

    return g


def _row_ok(g: List[List[int]], r: int, Lmin: int) -> bool:
    """Check that row r has no white run of length 1..Lmin-1."""
    run = 0
    for c in range(len(g[0])):
        if g[r][c] == 0:
            run += 1
        else:
            if 0 < run < Lmin:
                return False
            run = 0
    if 0 < run < Lmin:
        return False
    return True


def _col_ok(g: List[List[int]], c: int, Lmin: int) -> bool:
    """Check that column c has no white run of length 1..Lmin-1."""
    run = 0
    for r in range(len(g)):
        if g[r][c] == 0:
            run += 1
        else:
            if 0 < run < Lmin:
                return False
            run = 0
    if 0 < run < Lmin:
        return False
    return True


def _all_rows_ok(g: List[List[int]], Lmin: int) -> bool:
    return all(_row_ok(g, r, Lmin) for r in range(len(g)))


def _all_cols_ok(g: List[List[int]], Lmin: int) -> bool:
    return all(_col_ok(g, c, Lmin) for c in range(len(g[0])))


def _is_single_component(g: List[List[int]]) -> bool:
    """Check white-cell connectivity."""
    R, C = len(g), len(g[0])
    # Find a starting white cell
    start = None
    for r in range(R):
        for c in range(C):
            if g[r][c] == 0:
                start = (r, c)
                break
        if start:
            break
    if not start:
        return False  # all black (reject)

    # BFS
    q = [start]
    seen = {start}
    while q:
        r, c = q.pop()
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            rr, cc = r+dr, c+dc
            if 0 <= rr < R and 0 <= cc < C and g[rr][cc] == 0 and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))

    # Count total white
    total_white = sum(1 for r in range(R) for c in range(C) if g[r][c] == 0)
    return len(seen) == total_white



# Topology class is now imported from types.py


# --- swap your previous _build_entries_and_crossings with this version ---
def _build_entries_and_crossings(
    g: List[List[int]],
    Lmin: int
) -> Tuple[Dict[EntryId, EntryState], Dict[EntryId, Dict[EntryId, List[int]]], Topology]:
    """
    Scan the layout to create Across and Down entries + crossings + topology.
    """
    R, C = len(g), len(g[0])
    entries: Dict[EntryId, EntryState] = {}
    crossings: Dict[EntryId, Dict[EntryId, List[int]]] = {}
    entry_cells: Dict[EntryId, List[Cell]] = {}
    cell_to_entries: Dict[Cell, List[Tuple[EntryId, int]]] = {}

    def add_entry(eid: EntryId, length: int, cells: List[Cell]):
        entries[eid] = EntryState(L=length)
        crossings[eid] = {}
        entry_cells[eid] = cells
        for k, cell in enumerate(cells):
            cell_to_entries.setdefault(cell, []).append((eid, k))

    # Across
    a_counter = 1
    for r in range(R):
        c = 0
        while c < C:
            while c < C and g[r][c] == 1:
                c += 1
            start = c
            path = []
            while c < C and g[r][c] == 0:
                path.append((r, c))
                c += 1
            if len(path) >= Lmin:
                eid = f"A{a_counter}"; a_counter += 1
                add_entry(eid, len(path), path)

    # Down
    d_counter = 1
    for c in range(C):
        r = 0
        while r < R:
            while r < R and g[r][c] == 1:
                r += 1
            start = r
            path = []
            while r < R and g[r][c] == 0:
                path.append((r, c))
                r += 1
            if len(path) >= Lmin:
                eid = f"D{d_counter}"; d_counter += 1
                add_entry(eid, len(path), path)

    # Crossings
    for cell, pairs in cell_to_entries.items():
        if len(pairs) < 2:
            continue
        # For American-style, typically exactly two entries meet (1 across, 1 down).
        for (e1, k1) in pairs:
            for (e2, k2) in pairs:
                if e1 == e2:
                    continue
                crossings.setdefault(e1, {}).setdefault(e2, []).append(k1)

    # Numbering (classic: a cell gets a number if it starts an across and/or a down)
    numbers: Dict[Cell, int] = {}
    clue_num = 1
    for r in range(R):
        for c in range(C):
            if g[r][c] == 1:
                continue
            starts_across = (c == 0 or g[r][c-1] == 1) and (c+1 < C and g[r][c+1] == 0)
            starts_down   = (r == 0 or g[r-1][c] == 1) and (r+1 < R and g[r+1][c] == 0)
            # also ensure the run length is >= Lmin
            if starts_across:
                # lookahead across length
                length = 1
                cc = c+1
                while cc < C and g[r][cc] == 0:
                    length += 1; cc += 1
                starts_across = length >= Lmin
            if starts_down:
                length = 1
                rr = r+1
                while rr < R and g[rr][c] == 0:
                    length += 1; rr += 1
                starts_down = length >= Lmin

            if starts_across or starts_down:
                numbers[(r, c)] = clue_num
                clue_num += 1

    topo = Topology(shape=(R, C), layout=g, entry_cells=entry_cells,
                    cell_to_entries=cell_to_entries, numbers=numbers)
    return entries, crossings, topo


# --- light wrapper to optionally return topology from the generator ---
# --- light wrapper to optionally return topology from the generator ---


# ----------------------------- Convenience utils -----------------------------

def random_size(max_side: int = 21, themed_bias: bool = True, rng: Optional[random.Random] = None) -> Tuple[int, int]:
    """
    Pick a 'realistic' size. Defaults to 15x15; sometimes returns 21x21 (Sundays) or mid-sizes.
    """
    rng = rng or random.Random()
    if themed_bias:
        pick = rng.random()
        if pick < 0.70:
            return 15, 15
        elif pick < 0.82:
            return 21, 21
        else:
            s = rng.choice([13, 16, 17, 18, 19])
            return s, s
    else:
        r = rng.randint(5, max_side)
        c = rng.randint(5, max_side)
        return r, c


def pretty_print_layout(g: List[List[int]]) -> None:
    """Quick ASCII preview: '■' for black, '·' for white."""
    for row in g:
        print("".join("■" if x else "·" for x in row))


def demo():
    rng = random.Random(0)
    grid, topo = generate_grid(15, 15, symmetric=True, return_topology=True, rng=rng)
    

    # If you want to see the layout itself, re-run the layout piece separately,
    # or adapt generate_grid to return the layout alongside Grid.

if __name__ == "__main__":
    demo()