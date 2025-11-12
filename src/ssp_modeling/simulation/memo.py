# memo.py (or near your rollout code)

from typing import Tuple, Dict, Optional
EntryId = str

# Transposition table: (state_key, depth) -> (M_value, argmin_x)
TTABLE: Dict[Tuple[Tuple, int], Tuple[float, Optional[EntryId]]] = {}

def state_key_tuple(grid) -> Tuple:
    """
    Deterministic, hashable snapshot of the grid.
    (solved, locked, sorted filled_indices) per entry, sorted by id.
    """
    items = []
    for eid in sorted(grid.entries.keys()):
        e = grid.entries[eid]
        items.append((
            eid,
            bool(e.solved),
            bool(e.guess_with_current_letters),
            tuple(sorted(e.filled_indices)),
        ))
    return tuple(items)