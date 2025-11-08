from typing import Dict, List, Optional, Protocol, Any
from ..simulation.environment import EntryState, Grid, EntryId, ProbabilityModel
import math
from ..models.feature_maker import make_features


class DummyModel(ProbabilityModel):
    """
    A quick prior so you can run without XGBoost.
    Logistic over a linear combo of your features with sensible signs:
      - more remaining letters => lower p
      - revealed first/last => slightly higher p
      - more consecutive sequences => higher p
      - larger spread => slightly higher p (you’ve seen letters far apart)
    """
    def __init__(self, bias: float = -1.2):
        self.bias = bias

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0/(1.0+math.exp(-x))

    def prob_solve(self, grid: Grid, entry_id: EntryId) -> float:
        f = make_features(grid, entry_id)
        # Weights are arbitrary but sane for a prior; tune if desired.
        x = (
            self.bias
            + 0.35 * f["first_letter_revealed"]
            + 0.35 * f["last_letter_revealed"]
            + 0.25 * f["consecutive_sequences"]
            + 0.08 * f["position_spread"]
            - 0.28 * f["num_letters_remaining"]
        )
        p = self._sigmoid(x)
        # clamp a bit
        p = max(1e-6, min(1-1e-6, p))
        return p

class XGBProbability(ProbabilityModel):
    """
    Use trained XGBoost model with proper scaler (loaded from .pkl file).
    Requires xgboost, numpy, pandas, and scikit-learn to be installed.
    """
    def __init__(self, model_path: str):
        try:
            import pickle
            import pandas as pd
        except Exception as e:
            raise RuntimeError("pickle and pandas must be installed to use XGBProbability") from e
        
        # Load the full model with scaler from .pkl file
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.pd = pd

    def prob_solve(self, grid: Grid, entry_id: EntryId) -> float:
        f = make_features(grid, entry_id)
        
        # Create DataFrame with proper feature names and order
        features_df = self.pd.DataFrame([[f[name] for name in self.feature_names]], 
                                       columns=self.feature_names)
        
        # Scale features using trained scaler
        features_scaled = self.scaler.transform(features_df)
        
        # Get probability prediction for class 1 (success)
        p = float(self.model.predict_proba(features_scaled)[0, 1])
        return max(1e-6, min(1-1e-6, p))
    


