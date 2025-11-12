# ---------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------
# src/simulation/policies.py
from __future__ import annotations
from typing import List
from .types import Grid, EntryId, ProbabilityModel, SimConfig, SuccessSnapshot
from .policy_core import Policy, DifferenceValue, active_entries, expected_immediate_cost, apply_success_inplace, rollback_success
from .memo import state_key_tuple, TTABLE

class OneStepRolloutPolicy(Policy):
    def __init__(self, value: DifferenceValue, return_scores: bool = False):
        self.value = value
        self.return_scores = return_scores

    def plan_epoch(self, grid: Grid, model: ProbabilityModel, cfg: SimConfig) -> List[EntryId] | List[dict]:
        cand = active_entries(grid)
        if not cand:
            return []
        # Compute per-x scores
        rows = []
        for x in cand:
            p = max(cfg.eps_floor, min(1.0 - cfg.eps_floor, model.prob_solve(grid, x)))
            c_exp = expected_immediate_cost(p, cfg)
            dV = self.value.delta_depth0(grid, model, x, cfg)
            rows.append({"name": x, "score": c_exp + p*dV, "p": p, "c_exp": c_exp, "delta": dV})
        rows.sort(key=lambda r: r["score"])
        return rows if self.return_scores else [r["name"] for r in rows]


class NStepRolloutPolicy(Policy):
    def __init__(self, value: DifferenceValue, depth: int, return_scores: bool = False):
        assert depth >= 1
        self.value = value
        self.depth = depth
        self.return_scores = return_scores

    def plan_epoch(self, grid: Grid, model: ProbabilityModel, cfg: SimConfig) -> List[EntryId] | List[dict]:
        cand = active_entries(grid)
        if not cand:
            return []

        # Precompute M_{d-1}(s) once
        if self.depth == 1:
            # delegate to one-step
            rows = OneStepRolloutPolicy(self.value, return_scores=True).plan_epoch(grid, model, cfg)
            return rows if self.return_scores else [r["name"] for r in rows]

        M_s, _ = compute_M(grid, model, cfg, self.value, self.depth - 1)

        rows = []
        for x in cand:
            p = max(cfg.eps_floor, min(1.0 - cfg.eps_floor, model.prob_solve(grid, x)))
            c_exp = expected_immediate_cost(p, cfg)
            dV = self.value.delta_depth0(grid, model, x, cfg)

            snap = apply_success_inplace(grid, x)
            try:
                M_sx, _ = compute_M(grid, model, cfg, self.value, self.depth - 1)
            finally:
                rollback_success(grid, snap)

            score = c_exp + p * (dV + M_sx - M_s)
            rows.append({"name": x, "score": score, "p": p, "c_exp": c_exp, "delta": dV,
                         "M_sx": M_sx, "M_s": M_s})

        rows.sort(key=lambda r: r["score"])
        return rows if self.return_scores else [r["name"] for r in rows]
        
        
    

def compute_M(grid: Grid, model: ProbabilityModel, cfg: SimConfig,
              value: DifferenceValue, depth: int) -> tuple[float, EntryId | None]:
    """
    Return (M_d(s), argmin_x) for current state s and rollout depth d.
    For d==1 use the one-step fast ΔV; for d>1 use the ΔJ_{d-1} recursion:
      C_d(x|s) = c(x) + p(x) * [ΔJ_{d-1}(x|s)]
               = c(x) + p(x) * [ΔV(x) + M_{d-1}(s_x) - M_{d-1}(s)]
    NOTE: This function does not mutate the final state.
    """

    key = (state_key_tuple(grid), depth)
    hit = TTABLE.get(key)
    if hit is not None:

        print("Used memoization")
        return hit  # (M_value, argmin_x)

    cand = [i for i,e in grid.entries.items() if not e.solved and not e.guess_with_current_letters]
    if not cand:
        TTABLE[key] = (0.0, None)
        return (0.0, None)




    cand = [i for i, e in grid.entries.items() if not e.solved and not e.guess_with_current_letters]
    if not cand:
        return (0.0, None)

    # Depth-1: pure one-step (ΔV only)
    if depth == 1:
        best = float("inf"); best_x = None
        for x in cand:
            p = max(cfg.eps_floor, min(1.0 - cfg.eps_floor, model.prob_solve(grid, x)))
            c_exp = p*cfg.solved_correct_cost + (1.0-p)*cfg.solved_incorrect_cost
            dV = value.delta_depth0(grid, model, x, cfg)  # ΔV(x) = hatV(s_x)-hatV(s)
            score = c_exp + p * dV
            if score < best:
                best, best_x = score, x

            TTABLE[key] = (best, best_x)
        return (best, best_x)

    # Depth>1
    # 1) Compute M_{d-1}(s) once
    M_s, _ = compute_M(grid, model, cfg, value, depth-1)

    best = float("inf"); best_x = None
    for x in cand:
        p = max(cfg.eps_floor, min(1.0 - cfg.eps_floor, model.prob_solve(grid, x)))
        c_exp = p*cfg.solved_correct_cost + (1.0-p)*cfg.solved_incorrect_cost
        dV = value.delta_depth0(grid, model, x, cfg)

        # 2) Hypothetically solve x → s_x, compute M_{d-1}(s_x), rollback
        snap = apply_success_inplace(grid, x)   # in-place, fast
        try:
            M_sx, _ = compute_M(grid, model, cfg, value, depth-1)
        finally:
            rollback_success(grid, snap)

        score = c_exp + p * (dV + M_sx - M_s)   # <-- correct two-step / n-step formula
        if score < best:
            best, best_x = score, x
        TTABLE[key] = (best, best_x)
    return (best, best_x)     










# src/simulation/policies.py  (add)
class GreedyPPolicy(Policy):
    """Order by descending p(x), return full order."""
    def plan_epoch(self, grid: Grid, model: ProbabilityModel, cfg: SimConfig) -> List[EntryId]:
        cand = active_entries(grid)
        cand.sort(key=lambda x: model.prob_solve(grid, x), reverse=True)
        return cand



class expectedLettersGainedPolicy(Policy):
    """Order by expected letters gained, return full order."""
    def plan_epoch(self, grid: Grid, model: ProbabilityModel, cfg: SimConfig) -> List[EntryId]:
        cand = active_entries(grid)
            
        cand.sort(key=lambda x: self.expected_letters(model, grid, x), reverse=True)
        return cand
    
    def expected_letters(self, model: ProbabilityModel, grid: Grid, entry_id: EntryId) -> float:
        entry = grid.entries[entry_id]
        p = model.prob_solve(grid, entry_id)
        return p * (entry.L - len(entry.filled_indices))