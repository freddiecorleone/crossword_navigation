from .value_diff import HintAwareExpectedCostDifference
from .types import Grid, EntryState, SimConfig, Topology, EpisodeResult, ProbabilityModel
from .policies import OneStepRolloutPolicy, NStepRolloutPolicy, ExpectedLettersGainedPolicy
from.policy_core import apply_success_inplace
from ..models.xgb_model import XGBProbability
from .grid_generator import generate_grid
from .environment import render_ascii


entries = {
        "A1": EntryState(L=4, filled_indices=set()),
        "D1": EntryState(L=4, filled_indices=set()),  
        "D2": EntryState(L=3, filled_indices=set()),
    }
crossings = {
        "A1": {"D1": [0], "D2": [3]},
        "D1": {"A1": [0]},
        "D2": {"A1": [0]},
    }

simConfig = SimConfig(hint_cost=0, solved_correct_cost=1, solved_incorrect_cost=1)


grid, topo = generate_grid(rows=5, cols=5, return_topology=True)

apply_success_inplace(grid, "A1")
render_ascii(grid=grid,topo=topo)

value = HintAwareExpectedCostDifference(simConfig)
model = XGBProbability()



policy = OneStepRolloutPolicy(value=value)

policy2 = ExpectedLettersGainedPolicy()

for id in grid.entries.keys():
    print(f"Entry {id}: {value.delta_depth0(grid=grid, model=model, x=id)}")
    print(f"Entry {id} (prob_solve): {model.prob_solve(grid, id)}")
    print("Length:", grid.entries[id].L)
print(policy.plan_epoch(grid, model, simConfig))

print(policy2.plan_epoch(grid, model, simConfig))

        