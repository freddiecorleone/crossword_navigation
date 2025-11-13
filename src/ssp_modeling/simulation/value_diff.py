# src/simulation/value_diff.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from .environment import Grid, EntryId, ProbabilityModel, SimConfig
from .types import EntryState
from .policy_core import DifferenceValue, apply_success_inplace, rollback_success
import copy
import random



# value_hint_expected_cost.py
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
# assumes you have: Grid, EntryId, EntryState, ProbabilityModel, SimConfig
# and apply_success_inplace / rollback_success for success transitions.

def mask_from_entry(e: "EntryState") -> int:
    m = 0
    for i in e.filled_indices:
        m |= (1 << i)
    return m

def unrevealed_positions(L: int, mask: int) -> List[int]:
    return [i for i in range(L) if not (mask >> i) & 1]

@dataclass
class EntrySnapshot:
    locked: bool
    filled_mask: int
    solved: bool

def snapshot_entry(e: "EntryState") -> EntrySnapshot:
    return EntrySnapshot(
        locked=e.guess_with_current_letters,
        filled_mask=mask_from_entry(e),
        solved=e.solved,
    )

def restore_entry(e: "EntryState", snap: EntrySnapshot) -> None:
    # reconstruct set from mask
    new_set = set()
    m = snap.filled_mask
    idx = 0
    while m:
        if m & 1: new_set.add(idx)
        m >>= 1; idx += 1
    e.filled_indices = new_set
    e.guess_with_current_letters = snap.locked
    e.solved = snap.solved

class HintAwareExpectedCostDifference(DifferenceValue):
    """
    ΔV(x) for hint-aware surrogate:
        hatV(s) = sum_{active entries j} E_j(mask_j)
    where E_j(mask) follows:
        E(mask) = p(mask)*c_s + (1-p(mask))*(c_f + h + Avg_{pos in unrevealed} E(mask ∪ {pos}))
    We compute ΔV(x) locally: x + neighbors(x) only.
    """

    def __init__(self, cfg: "SimConfig", use_mc: bool = False, mc_samples: int = 16, rng_seed: int = 0):
        self.cfg = cfg
        self.use_mc = use_mc
        self.mc_samples = mc_samples
        import random
        self.rng = random.Random(rng_seed)
        # memo: (entry_id, mask_int) -> expected cost E
        self.memo: Dict[Tuple[str, int], float] = {}

    # --- probability helper (uses current grid features) ---
    def _p_for_mask(self, grid: "Grid", model: "ProbabilityModel", entry_id: str, mask: int, eps: float) -> float:
        e = grid.entries[entry_id]
        # snapshot then temporarily set filled_indices to match 'mask'
        snap = snapshot_entry(e)
        try:
            # rebuild set from mask quickly
            new_set = set()
            m = mask; idx = 0
            while m:
                if (m & 1): new_set.add(idx)
                m >>= 1; idx += 1
            e.filled_indices = new_set
            e.solved = (len(e.filled_indices) == e.L)
            e.guess_with_current_letters = False  # after a hint we allow a new guess
            p = model.prob_solve(grid, entry_id)
            p = max(eps, min(1.0 - eps, p))
        finally:
            restore_entry(e, snap)
        return p

    # --- single-entry expected cost E(entry_id, mask) with memoization ---
    def E_entry(self, grid: "Grid", model: "ProbabilityModel", entry_id: str, mask: int) -> float:
        key = (entry_id, mask)
        if key in self.memo:
            return self.memo[key]

        e = grid.entries[entry_id]
        L = e.L
        k = L - bin(mask).count("1")
        if k == 0:
            self.memo[key] = 0.0
            return 0.0

        p = self._p_for_mask(grid, model, entry_id, mask, self.cfg.eps_floor)
        cs, cf, h = self.cfg.solved_correct_cost, self.cfg.solved_incorrect_cost, self.cfg.hint_cost

        if not self.use_mc:
            # exact averaging over all unrevealed positions
            U = unrevealed_positions(L, mask)
            # average E over adding each pos
            acc = 0.0
            for pos in U:
                next_mask = mask | (1 << pos)
                acc += self.E_entry(grid, model, entry_id, next_mask)
            avg_next = acc / len(U)
        else:
            # Monte Carlo over hint placements (if you insist)
            U = unrevealed_positions(L, mask)
            if not U:
                avg_next = 0.0
            else:
                acc = 0.0
                for _ in range(min(self.mc_samples, len(U))):
                    pos = self.rng.choice(U)
                    next_mask = mask | (1 << pos)
                    acc += self.E_entry(grid, model, entry_id, next_mask)
                avg_next = acc / min(self.mc_samples, max(1, len(U)))

        val = p*cs + (1.0 - p)*(cf + h + avg_next)
        self.memo[key] = val
        return val

    # --- ΔV(x): local difference using E_entry ---
    def delta_depth0(self, grid: "Grid", model: "ProbabilityModel", x: EntryId) -> float:
        ex = grid.entries[x]
        if ex.solved or ex.guess_with_current_letters:
            return 0.0

        # affected ids: x and neighbors
        neighbors = list(grid.crossings.get(x, {}).keys())

        affected = neighbors

        # baseline E for affected
        base: Dict[str, float] = {}
        for j in affected:
            ej = grid.entries[j]
            if ej.solved:
                base[j] = 0.0
            else:
                base[j] = self.E_entry(grid, model, j, mask_from_entry(ej))

        # apply hypothetical success on x (mark solved, propagate crossings to neighbors)
        snap = apply_success_inplace(grid, x)
        try:
            after: Dict[str, float] = {}
            for j in affected:
                ej = grid.entries[j]
                if ej.solved:
                    after[j] = 0.0
                else:
                    after[j] = self.E_entry(grid, model, j, mask_from_entry(ej))
        finally:
            rollback_success(grid, snap)

        # ΔV = Σ (after - base) over affected
        return sum(after[j] - base[j] for j in affected)





