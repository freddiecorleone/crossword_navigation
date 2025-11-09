# src/models/feature_maker.py
from __future__ import annotations
from typing import Dict, Optional, Union, overload
from ..simulation.environment import EntryState, Grid, EntryId

def make_features(grid: Grid, entry_id: EntryId, entry_state: Optional[EntryState] = None) -> Dict[str, float]:
    """
    Compute XGBoost feature vector from Grid and EntryId.
    If entry_state is provided, use it directly. Otherwise, extract from grid.
    """
    if entry_state is not None:
        e = entry_state
    else:
        e: EntryState = grid.entries[entry_id]
        
    filled = sorted(e.filled_indices)
    L = e.L

    # Basic features
    answer_length = L
    first_letter_revealed = int(0 in filled)
    last_letter_revealed = int((L - 1) in filled)
    num_letters_remaining = L - len(filled)

    # Count consecutive sequences
    consecutive_sequences = 0
    if filled:
        consecutive_sequences = 1
        for i in range(1, len(filled)):
            if filled[i] != filled[i - 1] + 1:
                consecutive_sequences += 1

    # Spread - normalized by answer length like in training
    position_spread = 0
    if len(filled) > 1 and L > 1:
        position_spread = (max(filled) - min(filled)) / (L - 1)

    return {
        "answer_length": float(answer_length),
        "first_letter_revealed": float(first_letter_revealed),
        "last_letter_revealed": float(last_letter_revealed),
        "num_letters_remaining": float(num_letters_remaining),
        "consecutive_sequences": float(consecutive_sequences),
        "position_spread": float(position_spread),
    }


def make_features_from_entry_state(entry_state: EntryState) -> Dict[str, float]:
    """Compute XGBoost feature vector directly from EntryState."""
    filled = sorted(entry_state.filled_indices)
    L = entry_state.L

    # Basic features
    answer_length = L
    first_letter_revealed = int(0 in filled)
    last_letter_revealed = int((L - 1) in filled)
    num_letters_remaining = L - len(filled)

    # Count consecutive sequences
    consecutive_sequences = 0
    if filled:
        consecutive_sequences = 1
        for i in range(1, len(filled)):
            if filled[i] != filled[i - 1] + 1:
                consecutive_sequences += 1

    # Spread - normalized by answer length like in training
    position_spread = 0
    if len(filled) > 1 and L > 1:
        position_spread = (max(filled) - min(filled)) / (L - 1)

    return {
        "answer_length": float(answer_length),
        "first_letter_revealed": float(first_letter_revealed),
        "last_letter_revealed": float(last_letter_revealed),
        "num_letters_remaining": float(num_letters_remaining),
        "consecutive_sequences": float(consecutive_sequences),
        "position_spread": float(position_spread),
    }
