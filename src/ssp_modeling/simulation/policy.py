# ---------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------
from typing import Protocol, List, Dict, TYPE_CHECKING

# Clean imports from our centralized types module
from .types import Grid, EntryId, ProbabilityModel
from .value import ValueFunction

class Policy(Protocol):
    def plan_epoch(self, grid: Grid, model: ProbabilityModel) -> List[EntryId]: ...
    def wants_hint(self, grid: Grid, model: ProbabilityModel) -> bool:
        return False
    def name(self) -> str: ...

class GreedyP:
    """Ranks entries by descending solve probability, skipping those guessed already."""
    def __init__(self, min_p_for_attempt: float = 0.0):
        self._min = min_p_for_attempt

    def plan_epoch(self, grid: Grid, model: ProbabilityModel) -> List[EntryId]:
        candidates = [
            i for i, e in grid.entries.items()
            if not e.solved and not e.guess_with_current_letters
        ]
    
        candidates.sort(key=lambda i: model.prob_solve(grid, i), reverse=True)
        if self._min > 0:
            candidates = [i for i in candidates if model.prob_solve(grid, i) >= self._min]
        return candidates

    def wants_hint(self, grid: Grid, model: ProbabilityModel) -> bool:
        if self._min <= 0: return False
        ps = [
            model.prob_solve(grid, i)
            for i, e in grid.entries.items()
            if not e.solved and not e.guess_with_current_letters
        ]
        return (not ps) or (max(ps) < self._min)

    def name(self) -> str:
        return "GreedyP"


class OneStepRollout:
    def __init__(self, value: ValueFunction, hint_threshold: float = 0.0):
        self.value = value
        self.thresh = hint_threshold

    def plan_epoch(self, grid: Grid, model: ProbabilityModel) -> List[EntryId]:
        
        uns = [
            i for i, e in grid.entries.items()
            if not e.solved and not e.guess_with_current_letters
        ]
        

        scored = [(i, self.value.one_step_score(grid, model, i)) for i in uns]
        
        scored.sort(key=lambda t: t[1])
        
        return [i for i, _ in scored]

    def wants_hint(self, grid: Grid, model: ProbabilityModel) -> bool:
        if self.thresh <= 0: return False
        ps = [
            model.prob_solve(grid, i)
            for i, e in grid.entries.items()
            if not e.solved and not e.guess_with_current_letters
        ]
        return (len(ps) > 0) and (max(ps) < self.thresh)

    def name(self) -> str:
        return "OneStepRollout"