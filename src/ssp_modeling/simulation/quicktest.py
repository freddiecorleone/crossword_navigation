from .value_diff import create_sequence_of_entries, IsolatedExpectedCost
from .types import Grid, EntryState, SimConfig, Topology, EpisodeResult, ProbabilityModel
from .policies import OneStepRolloutPolicy, NStepRolloutPolicy
from ..models.xgb_model import XGBProbability


entries = {
        "A1": EntryState(L=5, filled_indices=set()),
        "D1": EntryState(L=4, filled_indices=set()),  
        "D2": EntryState(L=3, filled_indices=set()),
    }
crossings = {
        "A1": {"D1": [0], "D2": [3]},
        "D1": {"A1": [0]},
        "D2": {"A1": [0]},
    }

simConfig = SimConfig(hint_cost=0, solved_correct_cost=1, solved_incorrect_cost=1)
grid = Grid(entries=entries, crossings=crossings)
targets = ["D2", "D1", "A1"]


IsolatedCostModel = IsolatedExpectedCost()
model = XGBProbability()
policy = NStepRolloutPolicy(value=IsolatedCostModel, depth=2)

recommended_order = policy.plan_epoch(grid, model, simConfig)
print("Recommended order:", recommended_order)




        