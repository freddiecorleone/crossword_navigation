# src/simulation/value_diff.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from .environment import Grid, EntryId, ProbabilityModel, SimConfig
from .types import EntryState
from .policy_core import active_entries, DifferenceValue, expected_immediate_cost, get_impacted_crossings
import copy
import random


class IsolatedExpectedCost(DifferenceValue):
    """
   
    """
    def delta_depth0(self, grid: Grid, model: ProbabilityModel, x: EntryId, config: SimConfig) -> float:
        # mark x as solved and compute isolated expected cost
        p = max(1e-8, min(1-1e-8, model.prob_solve(grid, x)))
        #get all the neighbors that have unfilled crossing squares with i:
        unfilled_neighbors = []
        for neighbor, indices in grid.crossings.get(x, {}).items():
            
            #get crossing index for neighbor:
            neighbor_crosspoints = grid.crossings[neighbor][x]
            
            #get filled indices for neighbor
            neighbor_filled_indices = grid.entries[neighbor].filled_indices
            
            if any(crosspoint not in neighbor_filled_indices for crosspoint in neighbor_crosspoints):
                unfilled_neighbors.append((neighbor, neighbor_crosspoints))

       
        changes_in_expected_costs = []
        for neighbor, crosspoints in unfilled_neighbors:

            filling_sequence = create_sequence_of_entries(
                grid,
                neighbor,
                crosspoints
            )
            
            probabilities = [model.prob_solve(filling_sequence[i]) for i in range(len(filling_sequence))]
            probabilities[-1] = 1
            
            expected_costs = []

            e_final = 0
          
            expected_costs.append(e_final)
        

            for i in range(2, len(probabilities) + 1):
                ind = len(probabilities) - i
                e_next = (expected_costs[-1] + config.solved_incorrect_cost + config.hint_cost) * (1 - probabilities[ind]) + probabilities[ind] * (config.solved_correct_cost)
                expected_costs.append(e_next)

            
            expected_costs.reverse()
            added_letters = len(crosspoints)
            change_in_expected_cost = expected_costs[added_letters] - expected_costs[0]
            changes_in_expected_costs.append(change_in_expected_cost)
        
        total_change = sum(changes_in_expected_costs)

        return total_change

        



        
        return 




  
    
def create_sequence_of_entries(
    grid: Grid,
    entry_id: EntryId,
    indices_to_fill: List[int],
    seed: Optional[int] = None
) -> List[EntryState]:
    """
    Creates a sequence of EntryStates starting from the original entry state,
    progressively adding indices from indices_to_fill, then randomly filling
    remaining positions until the entry is complete.
    
    Args:
        grid: The grid containing the entry
        entry_id: The ID of the entry to create sequence for
        indices_to_fill: List of specific indices to fill first (in order)
        seed: Optional random seed for reproducible results
        
    Returns:
        List of EntryState objects showing progression from original to fully filled
        
    Raises:
        KeyError: If entry_id not found in grid
        ValueError: If indices_to_fill contains invalid indices or duplicates
    """
    # Error checking
    if entry_id not in grid.entries:
        raise KeyError(f"Entry ID '{entry_id}' not found in grid")
    
    original_entry = grid.entries[entry_id]
    
    # Validate indices_to_fill
    if indices_to_fill:
        invalid_indices = [idx for idx in indices_to_fill if idx < 0 or idx >= original_entry.L]
        if invalid_indices:
            raise ValueError(f"Invalid indices {invalid_indices} for entry of length {original_entry.L}")
        
        if len(set(indices_to_fill)) != len(indices_to_fill):
            raise ValueError("indices_to_fill contains duplicates")
        
        already_filled = [idx for idx in indices_to_fill if idx in original_entry.filled_indices]
        if already_filled:
            raise ValueError(f"Indices {already_filled} are already filled in the original entry")
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    entries = []
    
    # Phase 1: Add states with progressively more indices from indices_to_fill
    current_filled = set(original_entry.filled_indices)  # Start with original filled indices
    
    for i in range(len(indices_to_fill) + 1):  # +1 to include the original state
        entry_copy = copy.deepcopy(original_entry)
        entry_copy.filled_indices = current_filled | set(indices_to_fill[:i])
        entry_copy.num_attempts = original_entry.num_attempts + i  # Increment attempts for each letter added
        entry_copy.guess_with_current_letters = False  # Reset guess flag when adding letters
        
        # Check if entry is now solved
        if len(entry_copy.filled_indices) == entry_copy.L:
            entry_copy.solved = True
        
        entries.append(entry_copy)
    
    # Phase 2: Fill remaining indices randomly until complete
    if entries:
        last_entry = entries[-1]
        
        # Get remaining unfilled indices (more efficient than recreating lists)
        all_indices = set(range(original_entry.L))
        remaining_indices = list(all_indices - last_entry.filled_indices)
        
        # Shuffle remaining indices for random order
        random.shuffle(remaining_indices)
        
        # Add states filling one more random index at a time
        current_entry = last_entry
        for idx in remaining_indices:
            entry_copy = copy.deepcopy(current_entry)
            entry_copy.filled_indices.add(idx)
            entry_copy.num_attempts += 1  # Increment attempts for each additional letter
            entry_copy.guess_with_current_letters = False
            
            # Check if entry is now solved
            if len(entry_copy.filled_indices) == entry_copy.L:
                entry_copy.solved = True
            
            entries.append(entry_copy)
            current_entry = entry_copy
    
    return entries
    
    



def get_successive_probabilities(
    grid: Grid,
    model: ProbabilityModel,
    entry_id: EntryId,
    max_depth: int
) -> List[float]:
    
    
    probabilities = []

    entry_state = grid.entries[entry_id]

    entry_state_copy = copy.deepcopy(entry_state)
    
    return probabilities






