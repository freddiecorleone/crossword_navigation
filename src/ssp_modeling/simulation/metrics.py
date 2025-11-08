# src/simulation/metrics.py
from __future__ import annotations
from typing import List, Dict, Any
import statistics as stats
from .environment import EpisodeResult

def summarize(results: List[EpisodeResult]) -> Dict[str, Any]:
    totals = [r.total_cost for r in results if r.terminated]
    hints  = [r.hints_used for r in results if r.terminated]
    solved = [r.solved_count for r in results]
    return {
        "episodes": len(results),
        "completed": sum(int(r.terminated) for r in results),
        "mean_steps": stats.mean(totals) if totals else None,
        "median_steps": stats.median(totals) if totals else None,
        "p90_steps": (sorted(totals)[int(0.9*len(totals))-1] if totals else None),
        "mean_hints": stats.mean(hints) if hints else None,
        "mean_solved": stats.mean(solved) if solved else None,
    }