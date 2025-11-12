# src/simulation/policy_core.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .types import Grid, EntryId, ProbabilityModel, SimConfig, SuccessSnapshot

# ---------- utilities ----------
def active_entries(grid: Grid) -> List[EntryId]:
    """Unsolved and not locked entries."""
    return [i for i,e in grid.entries.items()
            if (not e.solved) and (not e.guess_with_current_letters)]

def expected_immediate_cost(p: float, cfg: SimConfig) -> float:
    """c(x) = p*c_correct + (1-p)*c_incorrect for a single-guess epoch."""
    return p*cfg.solved_correct_cost + (1.0-p)*cfg.solved_incorrect_cost


def get_impacted_crossings(grid: Grid, x: EntryId) -> List[EntryId]:
    """Return list of unsolved crossings impacted by a successful guess on x."""
    #get unfilled letter indices
    indices = set([i for i in range(0, grid.entries[x].L) if i not in grid.entries[x].filled_indices])

    #get names of crossings 
    crossings = list(grid.crossings[x].keys())
    impactedcrossings = []

    for crossing in crossings:
        if set(grid.crossings[x][crossing]).intersection(indices):
            impactedcrossings.append(crossing)

    return impactedcrossings



from dataclasses import dataclass
from typing import Dict, Tuple, FrozenSet, List



def apply_success_inplace(grid: Grid, x: EntryId) -> SuccessSnapshot:
    """
    Mark x solved and propagate to neighbors.
    NOTE: grid.crossings[i][j] stores indices in *i*.
          To add indices to j, we must read grid.crossings[j][i].
    """
    e_x = grid.entries[x]
    snap = SuccessSnapshot(
        x=x,
        x_prev_solved=e_x.solved,
        x_prev_locked=e_x.guess_with_current_letters,
        neighbor_states={}
    )

    # Mark x solved; (locking flag is irrelevant once solved but we restore it on rollback)
    e_x.solved = True
    e_x.guess_with_current_letters = False

    # For each neighbor j of x, propagate *neighbor-side* indices: crossings[j][x]
    for j in grid.crossings.get(x, {}):
        e_j = grid.entries[j]
        if e_j.solved:
            continue

        # Record neighbor's prior state exactly once
        if j not in snap.neighbor_states:
            snap.neighbor_states[j] = (
                e_j.guess_with_current_letters,
                frozenset(e_j.filled_indices),
            )

        before = len(e_j.filled_indices)

        # neighbor-side indices are stored in crossings[j][x]
        idxs_in_j: List[int] = grid.crossings.get(j, {}).get(x, [])
        # If the reciprocal edge is missing, nothing to add (data issue). You can assert if you prefer.
        for idx_j in idxs_in_j:
            e_j.filled_indices.add(idx_j)

        # Unlock j if it gained new info
        if len(e_j.filled_indices) > before:
            e_j.guess_with_current_letters = False

    return snap

def rollback_success(grid: Grid, snap: SuccessSnapshot) -> None:
    # Restore neighbors
    for j, (prev_locked, prev_filled) in snap.neighbor_states.items():
        e_j = grid.entries[j]
        e_j.guess_with_current_letters = prev_locked
        e_j.filled_indices = set(prev_filled)
    # Restore x
    e_x = grid.entries[snap.x]
    e_x.solved = snap.x_prev_solved
    e_x.guess_with_current_letters = snap.x_prev_locked


           
# ---------- value interfaces ----------
class DifferenceValue(ABC):
    """
    Base interface that supplies only *differences*.
    Depth-0: ΔV(x) = hatV(s_x) - hatV(s)
    Depth-(d-1): ΔJ_{d-1}(x|s) = J_{d-1}(s_x) - J_{d-1}(s)
    """
    @abstractmethod
    def delta_depth0(self, grid: Grid, model: ProbabilityModel, x: EntryId) -> float:
        """Return ΔV(x) using your surrogate hatV."""
        ...


# ---------- policy interface ----------
class Policy(ABC):
    @abstractmethod
    def plan_epoch(self, grid: Grid, model: ProbabilityModel, cfg: SimConfig) -> List[EntryId]:
        """Return an ordered list of entries to attempt this epoch (single guess)."""
        ...



