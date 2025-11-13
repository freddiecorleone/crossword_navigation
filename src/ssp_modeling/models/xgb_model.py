from typing import Dict, List, Optional, Protocol, Any, Union, overload
from ..simulation.environment import EntryState, Grid, EntryId, ProbabilityModel
import math
import numpy as np
from ..models.feature_maker import make_features
import pandas as pd


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

    def prob_solve(self, grid: Grid, entry_id: EntryId, entry_state: Optional[EntryState] = None) -> float:
        """
        Calculate solve probability from Grid and EntryId.
        If entry_state is provided, use it directly. Otherwise, extract from grid.
        """
        f = make_features(grid, entry_id, entry_state)
            
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

    def prob_solve_entry_state(self, entry_state: EntryState) -> float:
        """Calculate solve probability directly from EntryState."""
        from ..models.feature_maker import make_features_from_entry_state
        
        f = make_features_from_entry_state(entry_state)
            
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
    def __init__(self, model_path: Optional[str] = None):
        import pickle
        from pathlib import Path
        import numpy as np
        import pandas as pd
        
        # Store pandas and numpy for later use
        self.pd = pd
        self.np = np

        if model_path is None:
            from ..utils.path import get_project_root
            model_path = get_project_root() / "data" / "processed" / "crossword_model.pkl"

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        self.sklearn_model = data["model"]          # XGBClassifier or similar
        self.booster = self.sklearn_model.get_booster()
        self.scaler = data["scaler"]                # StandardScaler (assumed)
        self.feature_names = data["feature_names"]  # list[str] fixed order

        # Pull scaler params for inline scaling (much faster than scaler.transform)
        self._mean = getattr(self.scaler, "mean_", None)
        self._scale = getattr(self.scaler, "scale_", None)
        self._with_mean = self._mean is not None and getattr(self.scaler, "with_mean", True)
        self._with_std  = self._scale is not None and getattr(self.scaler, "with_std", True)

        # Convert to numpy arrays for faster access
        if self._mean is not None:
            self._mean = np.asarray(self._mean, dtype=np.float32)
        if self._scale is not None:
            self._scale = np.asarray(self._scale, dtype=np.float32)

        # Cache for repeated calls (per episode)
        self._pcache: dict[tuple[str, int], float] = {}
        
        # Pre-build feature name to index mapping for batch processing
        self._feature_idx = {name: i for i, name in enumerate(self.feature_names)}
    def prob_solve(self, grid_or_entry: Union[Grid, EntryState], entry_id: Optional[EntryId] = None, entry_state: Optional[EntryState] = None) -> float:
        """
        Fast calculate solve probability. Can be called in two ways:
        1. prob_solve(grid, entry_id, entry_state=None) - traditional way
        2. prob_solve(entry_state) - direct EntryState usage
        """
        # Check if first argument is an EntryState (new usage)
        if hasattr(grid_or_entry, 'L') and hasattr(grid_or_entry, 'filled_indices') and not hasattr(grid_or_entry, 'entries'):
            return self._prob_solve_fast_entry_state(grid_or_entry)
        
        # Traditional usage with Grid and EntryId
        grid = grid_or_entry
        entry = entry_state if entry_state else grid.entries[entry_id]
        
        # Check cache first (using entry mask as key)
        cache_key = (entry_id, self._mask_from_entry(entry))
        if cache_key in self._pcache:
            return self._pcache[cache_key]
        
        # Fast feature extraction and prediction
        f = make_features(grid, entry_id, entry_state)
        p = self._predict_single_fast(f)
        
        # Cache result
        self._pcache[cache_key] = p
        return p

    def prob_solve_entry_state(self, entry_state: EntryState) -> float:
        """Calculate solve probability directly from EntryState."""
        return self._prob_solve_fast_entry_state(entry_state)
    
    def _prob_solve_fast_entry_state(self, entry_state: EntryState) -> float:
        """Fast solve probability calculation for EntryState."""
        from ..models.feature_maker import make_features_from_entry_state
        
        f = make_features_from_entry_state(entry_state)
        return self._predict_single_fast(f)
    
    def _predict_single_fast(self, features_dict: dict) -> float:
        """Fast single prediction avoiding pandas overhead."""
        # Convert features dict to numpy array in correct order
        features_array = np.array([features_dict[name] for name in self.feature_names], dtype=np.float32)
        
        # Fast inline scaling (avoid scaler.transform overhead)
        if self._with_mean and self._mean is not None:
            features_array -= self._mean
        if self._with_std and self._scale is not None:
            features_array /= self._scale
        
        # Reshape for prediction (booster expects 2D)
        features_2d = features_array.reshape(1, -1)
        
        # Fast prediction using booster directly
        probs = self.booster.inplace_predict(features_2d, predict_type="probability")
        p = float(probs[0] if len(probs.shape) == 1 else probs[0, 1])
        
        # Clamp probability
        return max(1e-6, min(1-1e-6, p))


    def prob_solve_many(self, grid: Grid, entry_ids: list[EntryId], entry_states: Optional[list[EntryState]] = None) -> np.ndarray:
        """
        Batch predict probs for multiple entries in current grid.
        entry_states (optional) lets you pass temp masks (e.g., for E(mask)).
        """
        import numpy as np
        # caching by (entry_id, mask)
        keys = []
        to_build = []
        out = np.empty(len(entry_ids), dtype=np.float32)

        for idx, eid in enumerate(entry_ids):
            e = entry_states[idx] if entry_states else grid.entries[eid]
            mask = self._mask_from_entry(e)
            key = (eid, mask)
            keys.append(key)
            if key in self._pcache:
                out[idx] = self._pcache[key]
            else:
                to_build.append(idx)

        if to_build:
            feats = self._make_features_many(grid, entry_ids, entry_states, to_build)  # 2D np.array
            feats_scaled = self._scale_batch(feats)
            # Fast predict
            probs = self.booster.inplace_predict(feats_scaled, predict_type="probability")
            # booster returns shape (n,) for binary
            probs = np.asarray(probs, dtype=np.float32)
            # clamp
            probs = np.clip(probs, 1e-6, 1 - 1e-6)
            # scatter back and fill cache
            for j, idx in enumerate(to_build):
                out[idx] = probs[j]
                self._pcache[keys[idx]] = float(probs[j])

        return out



 # ---- helpers ----
    @staticmethod
    def _mask_from_entry(e: EntryState) -> int:
        m = 0
        for i in e.filled_indices:
            m |= (1 << i)
        return m

    def _scale_batch(self, X: np.ndarray) -> np.ndarray:
        import numpy as np
        if self._with_mean:
            X = X - self._mean
        if self._with_std:
            X = X / self._scale
        return X

    def _make_features_many(self, grid: Grid, entry_ids: list[EntryId], entry_states: Optional[list[EntryState]], idxs: list[int]) -> np.ndarray:
        """
        Build a 2D array of features for rows in 'idxs' only, in self.feature_names order.
        Optimized to avoid pandas overhead.
        """
        n_feat = len(self.feature_names)
        X = np.empty((len(idxs), n_feat), dtype=np.float32)

        for row_j, idx in enumerate(idxs):
            eid = entry_ids[idx]
            e = entry_states[idx] if entry_states else grid.entries[eid]
            f = make_features(grid, eid, e)  # returns a dict of {feat_name: value}
            # Write features in correct order using pre-built index mapping
            for name, j in self._feature_idx.items():
                X[row_j, j] = f[name]
        return X
    
    def prob_solve_batch(self, entries_data: List[Union[tuple, EntryState]]) -> np.ndarray:
        """
        Simple batch solver for multiple entries.
        
        Args:
            entries_data: List of either:
                - (grid, entry_id) tuples for traditional usage
                - EntryState objects for direct usage
                
        Returns:
            np.ndarray of probabilities
        """
        n_entries = len(entries_data)
        if n_entries == 0:
            return np.array([])
        
        # Pre-allocate result array
        results = np.empty(n_entries, dtype=np.float32)
        features_matrix = np.empty((n_entries, len(self.feature_names)), dtype=np.float32)
        
        # Extract features for all entries
        for i, entry_data in enumerate(entries_data):
            if isinstance(entry_data, tuple):
                # Traditional (grid, entry_id) format
                grid, entry_id = entry_data
                f = make_features(grid, entry_id)
            else:
                # Direct EntryState format
                from ..models.feature_maker import make_features_from_entry_state
                f = make_features_from_entry_state(entry_data)
            
            # Fill feature matrix row
            for name, j in self._feature_idx.items():
                features_matrix[i, j] = f[name]
        
        # Batch scaling
        if self._with_mean and self._mean is not None:
            features_matrix -= self._mean
        if self._with_std and self._scale is not None:
            features_matrix /= self._scale
        
        # Batch prediction
        probs = self.booster.inplace_predict(features_matrix, predict_type="probability")
        
        # Handle different output formats and clamp
        if len(probs.shape) == 1:
            results = np.clip(probs, 1e-6, 1-1e-6)
        else:
            results = np.clip(probs[:, 1], 1e-6, 1-1e-6)
        
        return results
    
    def clear_cache(self):
        """Clear the probability cache (call between episodes)."""
        self._pcache.clear()