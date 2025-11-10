# src/simulation/value.py
from __future__ import annotations
from typing import Dict, List
from .types import Grid, EntryId, ProbabilityModel, EntryState

import numpy as np
import copy
import random


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


class IsolatedExpectedCost(ValueFunction):
    """V(S) = sum w_j*(1 - p_j)/p_j with w_j = unrevealed letters; ignores crossings."""
    def V(self, grid: Grid, model: ProbabilityModel) -> float:
        total = 0.0
        for i, e in grid.entries.items():
            if e.solved: continue
            if e.guess_with_current_letters: continue
            p = max(1e-8, min(1-1e-8, model.prob_solve(grid, i)))
            total += self.weight(grid, i) * (1.0 - p) / p
        return total


    def one_step_score(self, grid: Grid, model: ProbabilityModel, i: EntryId) -> float:
        p = max(1e-8, min(1-1e-8, model.prob_solve(grid, i)))
        #get all the neighbors that have unfilled crossing squares with i:
        unfilled_neighbors = []
        for neighbor, indices in grid.crossings.get(i, {}).items():
            
            #get crossing index for neighbor:
            neighbor_crosspoints = grid.crossings[neighbor][i]
            
            #get filled indices for neighbor
            neighbor_filled_indices = grid.entries[neighbor].filled_indices
            
            if any(crosspoint not in neighbor_filled_indices for crosspoint in neighbor_crosspoints):
                unfilled_neighbors.append(neighbor)

        V_delta = sum(self.isolated_E_score(grid, model, i) for i in unfilled_neighbors)

        return p * V_delta
    


    def isolated_E_score(self, grid: Grid, model: ProbabilityModel, k: EntryId) -> float:
        y = grid.entries[k]
        

        # safety: return expected change in cost of 0 if already solved
        if y.solved:
            return 0.0

        #return expected change in cost of 1 if only one letter is unfilled
        if y.L -1 == len(y.filled_indices):
            return 1.0
        
        probs = []
        #calculate probabilities of solving after adding k letters
        for i in range(0, y.L - len(y.filled_indices)):
            i_copy = randomly_add_letters(y, i)
            
            p_i = model.prob_solve_entry_state(i_copy)
        
            probs.append(p_i)
        
        C = 0
        k = len(y.filled_indices)

        for j in range (1, len(probs)):
            if j == 1:
                prod = 1
            else:
                prod = np.prod([1 - probs[m] for m in range (1, j)])

            C += j*prod*probs[j]

        return -(1- probs[0]*C)

            




def randomly_add_letters(entry_state: EntryState, k: int) -> EntryState:
    """
    Randomly adds k letters to an EntryState by adding k random indices
    to the filled_indices set. Returns a new EntryState (does not modify original).
    
    Args:
        entry_state: The EntryState to add letters to
        k: Number of letters to add
        
    Returns:
        New EntryState with k additional random indices filled
        
    Raises:
        ValueError: If k is negative or would exceed available unfilled positions
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    
    # Get available unfilled indices
    all_indices = set(range(entry_state.L))
    available_indices = all_indices - entry_state.filled_indices
    
    if k > len(available_indices):
        raise ValueError(f"Cannot add {k} letters: only {len(available_indices)} positions available")
    
    # Create a copy of the entry state
    new_entry = copy.deepcopy(entry_state)
    
    # Randomly sample k indices to fill
    if k > 0:
        indices_to_add = random.sample(list(available_indices), k)
        new_entry.filled_indices.update(indices_to_add)
        # Reset guess flag since we're adding new information
        if indices_to_add:
            new_entry.guess_with_current_letters = False
    
    return new_entry

