# src/ssp_modeling/simulation/types.py
"""
Shared data types and classes for crossword simulation.
This module contains common classes used across multiple simulation modules
to avoid circular import issues.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Protocol, Any, FrozenSet

# Type aliases
EntryId = str
Cell = Tuple[int, int]



@dataclass
class SuccessSnapshot:
    x: EntryId
    x_prev_solved: bool
    x_prev_locked: bool
    # For each neighbor: (prev_locked, prev_filled_indices)
    neighbor_states: Dict[EntryId, Tuple[bool, FrozenSet[int]]]

@dataclass
class EntryState:
    """Represents the solver's current knowledge about one crossword entry."""
    L: int                                      # total length
    filled_indices: set[int] = field(default_factory=set)
    guess_with_current_letters: bool = False
    solved: bool = False
    num_attempts: int = 0

    @property
    def k(self) -> int:
        return len(self.filled_indices)
    
    @property
    def u(self) -> int:
        return self.L - len(self.filled_indices)

@dataclass
class Grid:
    """
    Crossword as a graph of entries and crossings.
    crossings[i][j] = list of indices IN ENTRY i that are shared with entry j.
    (American-style is usually 0/1 per pair; list lets us generalize.)
    """
    entries: Dict[EntryId, EntryState]
    crossings: Dict[EntryId, Dict[EntryId, List[int]]]

@dataclass
class SimConfig:
    hint_cost: int = 0
    eps_floor: float = 1e-4
    max_steps: int = 10000
    rng_seed: Optional[int] = None
    solved_correct_cost :int = 1
    solved_incorrect_cost :int = 1

@dataclass
class Topology:
    """Static geometry needed for visualization and numbering."""
    shape: Tuple[int, int]                                  # (rows, cols)
    layout: List[List[int]]                                 # 1=black, 0=white
    entry_cells: Dict[EntryId, List[Cell]]                  # ordered cells for each entry
    cell_to_entries: Dict[Cell, List[Tuple[EntryId, int]]]  # (entry_id, index_in_entry)
    numbers: Dict[Cell, int]                                # clue numbers at starting cells

class ProbabilityModel(Protocol):
    """Your XGBoost or any model exposing p(entry|state)."""
    def prob_solve(self, grid: Grid, entry_id: EntryId) -> float: ...

@dataclass
class EpochOutcome:
    cost: int
    solved_entry: Optional[EntryId]
    used_hint: bool
    attempts_order: List[EntryId]
    attempts_taken: int
    info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EpisodeResult:
    total_cost: int
    steps: int
    solved_count: int
    hints_used: int
    log: List["EpochOutcome"]
    terminated: bool