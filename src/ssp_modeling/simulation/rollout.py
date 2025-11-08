# src/simulation/rollout.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
from .environment import Environment, Grid, ProbabilityModel, Policy, EpisodeResult

@dataclass
class RunConfig:
    episodes: int = 100
    seeds: List[int] = None  # optional per-episode seeds

class RolloutRunner:
    def __init__(self, env_maker, policy: Policy, model: ProbabilityModel):
        """
        env_maker: callable(seed) -> Environment (fresh grid/state each episode)
        """
        self.env_maker = env_maker
        self.policy = policy
        self.model = model

    def run(self, cfg: RunConfig) -> List[EpisodeResult]:
        results: List[EpisodeResult] = []
        for ep in range(cfg.episodes):
            seed = (cfg.seeds[ep] if cfg.seeds else None)
            env = self.env_maker(seed)
            res = env.run_episode(self.policy, self.model)
            results.append(res)
        return results