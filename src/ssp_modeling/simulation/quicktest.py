from .types import Grid, EntryState, SimConfig, Topology, EpisodeResult, ProbabilityModel
from .value import ValueFunction, IsolatedExpectedCost
from ..models.xgb_model import XGBProbability, DummyModel
from ..utils.path import get_project_root
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
grid = Grid(entries=entries, crossings=crossings)
targets = ["D2", "D1", "A1"]


function = IsolatedExpectedCost()

model_path = get_project_root() / "data" / "processed" / "crossword_model.pkl"

for target in targets:
    print("Score for", target, ":", function.one_step_score(grid, XGBProbability(model_path), target))




        