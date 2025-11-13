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
        
        # Use batch solving if available
        if hasattr(model, 'prob_solve_batch'):
            batch_data = [(grid, x) for x in cand]
            probs = model.prob_solve_batch(batch_data)
        else:
            # Fallback to individual calls
            probs = [model.prob_solve(grid, x) for x in cand]
        
        # Compute per-x scores with batch probabilities
        rows = []
        for i, x in enumerate(cand):
            p = max(cfg.eps_floor, min(1.0 - cfg.eps_floor, probs[i]))
            c_exp = expected_immediate_cost(p, cfg)
            dV = self.value.delta_depth0(grid, model, x)
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

        # Get probabilities for all candidates at once
        if hasattr(model, 'prob_solve_batch'):
            batch_data = [(grid, x) for x in cand]
            probs = model.prob_solve_batch(batch_data)
        else:
            # Fallback to individual calls
            probs = [model.prob_solve(grid, x) for x in cand]

        rows = []
        for i, x in enumerate(cand):
            p = max(cfg.eps_floor, min(1.0 - cfg.eps_floor, probs[i]))
            c_exp = expected_immediate_cost(p, cfg)
            dV = self.value.delta_depth0(grid, model, x)

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
        
        # Use batch solving if available
        if hasattr(model, 'prob_solve_batch'):
            batch_data = [(grid, x) for x in cand]
            probs = model.prob_solve_batch(batch_data)
        else:
            # Fallback to individual calls
            probs = [model.prob_solve(grid, x) for x in cand]
        
        for i, x in enumerate(cand):
            p = max(cfg.eps_floor, min(1.0 - cfg.eps_floor, probs[i]))
            c_exp = p*cfg.solved_correct_cost + (1.0-p)*cfg.solved_incorrect_cost
            dV = value.delta_depth0(grid, model, x)  # ΔV(x) = hatV(s_x)-hatV(s)
            score = c_exp + p * dV
            if score < best:
                best, best_x = score, x

        TTABLE[key] = (best, best_x)
        return (best, best_x)

    # Depth>1
    # 1) Compute M_{d-1}(s) once
    M_s, _ = compute_M(grid, model, cfg, value, depth-1)

    # 2) Get probabilities for all candidates at once
    if hasattr(model, 'prob_solve_batch'):
        batch_data = [(grid, x) for x in cand]
        probs = model.prob_solve_batch(batch_data)
    else:
        # Fallback to individual calls
        probs = [model.prob_solve(grid, x) for x in cand]

    best = float("inf"); best_x = None
    for i, x in enumerate(cand):
        p = max(cfg.eps_floor, min(1.0 - cfg.eps_floor, probs[i]))
        c_exp = p*cfg.solved_correct_cost + (1.0-p)*cfg.solved_incorrect_cost
        dV = value.delta_depth0(grid, model, x)

        # 3) Hypothetically solve x → s_x, compute M_{d-1}(s_x), rollback
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
        if not cand:
            return []
        
        # Use batch solving for efficiency
        if hasattr(model, 'prob_solve_batch'):
            batch_data = [(grid, x) for x in cand]
            probs = model.prob_solve_batch(batch_data)
        elif hasattr(model, 'prob_solve_many'):
            probs = model.prob_solve_many(grid, cand)
        else:
            # Fallback to individual calls
            probs = [model.prob_solve(grid, x) for x in cand]
        
        pairs = list(zip(cand, probs))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in pairs]
            

        
        return cand



class ExpectedLettersGainedPolicy(Policy):
    """Order by expected letters gained, return full order."""
    def plan_epoch(self, grid: Grid, model: ProbabilityModel, cfg: SimConfig) -> List[EntryId]:
        cand = active_entries(grid)
        if not cand:
            return []
        
        # Use batch solving for efficiency
        if hasattr(model, 'prob_solve_batch'):
            batch_data = [(grid, x) for x in cand]
            probs = model.prob_solve_batch(batch_data)
        else:
            # Fallback to individual calls
            probs = [model.prob_solve(grid, x) for x in cand]
        
        # Calculate expected letters gained for each entry
        expected_gains = []
        for i, x in enumerate(cand):
            entry = grid.entries[x]
            expected_gain = probs[i] * (entry.L - len(entry.filled_indices))
            expected_gains.append((x, expected_gain))
        
        # Sort by expected gain (descending)
        expected_gains.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in expected_gains]
    


class RandomPolicy(Policy):
    """Return a random ordering of active entries."""
    def __init__(self, rng_seed: int | None = None):
        import random
        self.rng = random.Random(rng_seed)

    def plan_epoch(self, grid: Grid, model: ProbabilityModel, cfg: SimConfig) -> List[EntryId]:
        cand = active_entries(grid)
        self.rng.shuffle(cand)
        return cand