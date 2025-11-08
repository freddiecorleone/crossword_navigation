# src/simulation/value.py
from __future__ import annotations
from typing import Dict, List
from .types import Grid, EntryId, ProbabilityModel


class ValueFunction:
    """V(S) = sum w_j/p_j with w_j = unrevealed letters; one-step score with a simple bump."""
    def weight(self, grid: Grid, i: EntryId) -> float:
        return float(grid.entries[i].u)

    def V(self, grid: Grid, model: ProbabilityModel) -> float:
        total = 0.0
        for i, e in grid.entries.items():
            if e.solved: continue
            if e.guess_with_current_letters: continue
            p = max(1e-8, min(1-1e-8, model.prob_solve(grid, i)))
            total += self.weight(grid, i) / p
        return total

    def _simulate_success_then(self, grid: Grid, entry_id: EntryId, model: ProbabilityModel) -> float:
        """
        Hypothetically solve entry_id: mark solved and reveal crossings (indices)
        for neighbors; compute V; then rollback.
        """
        # snapshot
        e = grid.entries[entry_id]
        snap_solved = e.solved
        neighbor_snaps: Dict[EntryId, set[int]] = {}
        neighbor_guess_snaps: Dict[EntryId, bool] = {}

        # apply hypothetical success
        e.solved = True
        for nbr, idxs in grid.crossings.get(entry_id, {}).items():
            nbr_e = grid.entries[nbr]
            if nbr_e.solved: continue
            neighbor_snaps[nbr] = set(nbr_e.filled_indices)
            neighbor_guess_snaps[nbr] = nbr_e.guess_with_current_letters
            before = len(nbr_e.filled_indices)
            nbr_e.filled_indices.update(idxs)
            if len(nbr_e.filled_indices) > before:
                nbr_e.guess_with_current_letters = False

        # compute V without the solved entry and any guessed-stuck entries
        total = 0.0
        for i2, e2 in grid.entries.items():
            if i2 == entry_id or e2.solved: continue
            if e2.guess_with_current_letters: continue
            p2 = max(1e-8, min(1-1e-8, model.prob_solve(grid, i2)))
            total += self.weight(grid, i2) / p2

        # rollback
        e.solved = snap_solved
        for nbr in neighbor_snaps:
            grid.entries[nbr].filled_indices = neighbor_snaps[nbr]
            grid.entries[nbr].guess_with_current_letters = neighbor_guess_snaps[nbr]
        return total

    def one_step_score(self, grid: Grid, model: ProbabilityModel, i: EntryId) -> float:
        V_fail = self.V(grid, model)
        p_i = max(1e-8, min(1-1e-8, model.prob_solve(grid, i)))
        V_succ = self._simulate_success_then(grid, i, model)
        return 1.0 + p_i * V_succ + (1.0 - p_i) * V_fail
