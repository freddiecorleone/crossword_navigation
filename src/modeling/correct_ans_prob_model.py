"""
XGBoost Classifier for Crossword Answer Prediction

This module creates a machine learning model that predicts the likelihood of 
correctly guessing a crossword answer based on:
1. Answer length
2. Whether the first letter has been revealed
3. Proportion of letters already revealed

Author: Your Name
Date: November 2025
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


class CrosswordAnswerPredictor:
    """
    XGBoost-based classifier for predicting crossword answer success probability.
    """
    
    def __init__(self, model_save_path: Optional[str] = None):
        """
        Initialize the CrosswordAnswerPredictor.
        
        Args:
            model_save_path: Path to save/load the trained model
        """
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        # Enhanced feature set
        self.feature_names = [
            'answer_length', 'first_letter_revealed', 
            'last_letter_revealed',  'num_letters_remaining', 'consecutive_sequences',
            'position_spread'
        ]
        # Updated to save in data/processed folder
        if model_save_path is None:
            project_root = Path(__file__).parent.parent.parent
            model_save_path = str(project_root / "data" / "processed" / "crossword_model.pkl")
        self.model_save_path = model_save_path
        
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract enhanced features from the crossword session data.
        
        Args:
            df: DataFrame with crossword session data
            
        Returns:
            DataFrame with extracted features
        """
        features_data = []
        
        for _, row in df.iterrows():
            try:
                # Parse the filled_indices JSON
                filled_indices = json.loads(row['filled_indices']) if pd.notna(row['filled_indices']) else []
                answer_length = row['answer_len']
                
                # Basic features
                first_letter_revealed = 0 in filled_indices
                last_letter_revealed = (answer_length - 1) in filled_indices if answer_length > 0 else False
                #proportion_revealed = len(filled_indices) / answer_length if answer_length > 0 else 0
                
                # Advanced pattern features
                num_letters_remaining = answer_length - len(filled_indices)
                
                # Consecutive letters feature - how many consecutive sequences?
                consecutive_sequences = 0
                if filled_indices:
                    sorted_indices = sorted(filled_indices)
                    consecutive_sequences = 1
                    for i in range(1, len(sorted_indices)):
                        if sorted_indices[i] != sorted_indices[i-1] + 1:
                            consecutive_sequences += 1
                
                # Position spread - are letters spread out or clustered?
                position_spread = 0
                if len(filled_indices) > 1 and answer_length > 1:
                    min_pos = min(filled_indices)
                    max_pos = max(filled_indices)
                    position_spread = (max_pos - min_pos) / (answer_length - 1)
                
                # Length category (easier to learn patterns)
                length_category = 'short' if answer_length <= 4 else 'medium' if answer_length <= 7 else 'long'
                
               
                
                features_data.append({
                    # Core features (original)
                    'answer_length': answer_length,
                    'first_letter_revealed': int(first_letter_revealed),
                    #'proportion_revealed': proportion_revealed,
                    
                    # Enhanced pattern features
                    'last_letter_revealed': int(last_letter_revealed),
                    #'num_letters_revealed': num_letters_revealed,
                    'num_letters_remaining': num_letters_remaining,
                    'consecutive_sequences': consecutive_sequences,
                    'position_spread': position_spread,
        
                    
                    # Categorical features (will need encoding)
                    'length_category': length_category,
                    
                    # Target and metadata
                    'user_correct': row['user_correct'],
                    'session_id': row.get('session_id', ''),
                    'source_id': row.get('source_id', ''),
                    'clue': row.get('clue', ''),
                    'answer_raw': row.get('answer_raw', ''),
                    'revealed': row.get('revealed', 0),
                    'voided': row.get('voided', 0),
                    'timed_out': row.get('timed_out', 0)
                })
                
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Error processing row: {e}")
                continue
                
        return pd.DataFrame(features_data)
    
    def load_and_preprocess_data(self, data_path: str) -> pd.DataFrame:
        """
        Load crossword session data and extract features.
        
        Args:
            data_path: Path to the CSV file with session data
            
        Returns:
            DataFrame with features and targets
        """
        print(f"Loading data from: {data_path}")
        
        # Load the CSV data using robust parsing for commas in clue text
        import csv
        records = []
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
        
        df = pd.DataFrame(records)
        
        # Convert numeric columns
        numeric_cols = ['answer_len', 'clue_len', 'user_correct', 'revealed', 'num_remaining_correct']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"Loaded {len(df)} records")
        
        # Filter out voided and revealed entries for training
        clean_df = df[
            (df.get('voided', 0) == 0) &  # Not voided
            (df.get('revealed', 0) == 0)  # Not revealed (user didn't peek)
        ].copy()
        
        print(f"After filtering: {len(clean_df)} records")
        
        # Extract features
        features_df = self.extract_features(clean_df)
        
        print(f"Extracted features from {len(features_df)} records")
        return features_df
    
    def train_model(self, features_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Train the XGBoost model.
        
        Args:
            features_df: DataFrame with features and target
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training results and metrics
        """
        print("Training XGBoost model...")
        
        # Prepare features and target
        X = features_df[self.feature_names]
        y = features_df['user_correct']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost model with monotone constraints and regularization
        self.model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=4,  # Reduced to prevent overfitting
            learning_rate=0.05,  # Reduced learning rate for better generalization
            random_state=random_state,
            eval_metric='logloss',
            # Enhanced monotone constraints for 9 features:
            # 0: answer_length -> 0 (let model decide - could go either way)
            # 1: first_letter_revealed -> 1 (first letter helps)
            # 2: last_letter_revealed -> 1 (last letter helps)
            # 3: num_letters_remaining -> -1 fewer letters remaining helps
            # 4: consecutive_sequences -> 0 (let data decide)
            # 5: position_spread -> 0 (let data decide - could go either way)
            monotone_constraints='(0,1,1,-1,0,0)',
            # Additional regularization to prevent overfitting
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            subsample=0.8,  # Use 80% of samples for each tree
            colsample_bytree=0.8  # Use 80% of features for each tree
        )
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = (y_pred == y_test).mean()
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        results = {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_)),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"AUC Score: {auc_score:.3f}")
        
        return results
    
    def predict_success_probability(self, answer_length: int, first_letter_revealed: bool, 
                                  last_letter_revealed: bool = False,
                                  num_letters_remaining: int = None, consecutive_sequences: int = 1,
                                  position_spread: float = 0.0) -> float:
        """
        Predict the probability of correctly guessing the crossword answer.
        
        Args:
            answer_length: Length of the answer
            first_letter_revealed: Whether the first letter is revealed
            last_letter_revealed: Whether the last letter is revealed
            num_letters_remaining: Number of letters remaining (auto-calculated if None)
            consecutive_sequences: Number of consecutive letter sequences
            position_spread: How spread out the revealed letters are (0.0 to 1.0)
            
        Returns:
            Probability of success (0.0 to 1.0)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Auto-calculate derived features if not provided
        if num_letters_remaining is None:
            num_letters_remaining = answer_length  # Default to all letters remaining
        
        
        # Prepare input features as DataFrame with proper column names
        import pandas as pd
        features_df = pd.DataFrame([[
            answer_length,
            int(first_letter_revealed),
            int(last_letter_revealed),
            num_letters_remaining,
            consecutive_sequences,
            position_spread,
        ]], columns=self.feature_names)
        
        # Scale features (now with proper feature names)
        features_scaled = self.scaler.transform(features_df)
        
        # Get probability prediction
        prob = self.model.predict_proba(features_scaled)[0, 1]
        
        return prob
    
    def predict_success_probability_simple(self, answer_length: int, first_letter_revealed: bool,
                                         num_letters_remaining: int) -> float:
        """
        Simple prediction method for backward compatibility.
        Uses default values for enhanced features.
        """
        return self.predict_success_probability(
            answer_length=answer_length,
            first_letter_revealed=first_letter_revealed,
            num_letters_remaining=num_letters_remaining
        )
    
    def save_model(self, path: Optional[str] = None) -> None:
        """Save the trained model and scaler to disk in data/processed folder."""
        save_path = path or self.model_save_path
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {save_path}")
        
        # Also save XGBoost model as JSON automatically
        self.save_xgb_model_json()
    
    def load_model(self, path: Optional[str] = None) -> None:
        """Load a trained model from disk."""
        load_path = path or self.model_save_path
        
        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from: {load_path}")
    
    def save_xgb_model_json(self, json_path: Optional[str] = None) -> None:
        """Save only the XGBoost model in JSON format."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before saving")
        
        # Get the project root and construct path to data/processed/
        project_root = Path(__file__).parent.parent.parent
        default_path = project_root / "data" / "processed" / "xgb_model.json"
        save_path = json_path or str(default_path)
        
        # Save the XGBoost model in JSON format
        self.model.save_model(save_path)
        print(f"XGBoost model saved to JSON: {save_path}")
    
    def plot_feature_importance(self, results: Dict[str, Any]) -> None:
        """Plot feature importance from training results."""
        importance = results['feature_importance']
        
        plt.figure(figsize=(10, 6))
        features = list(importance.keys())
        values = list(importance.values())
        
        plt.barh(features, values)
        plt.title('Feature Importance in Crossword Success Prediction')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
    
    def analyze_data(self, features_df: pd.DataFrame) -> None:
        """Perform exploratory data analysis on the features."""
        print("\n=== Data Analysis ===")
        print(f"Total samples: {len(features_df)}")
        print(f"Success rate: {features_df['user_correct'].mean():.3f}")
        
        print("\nFeature statistics:")
        print(features_df[self.feature_names + ['user_correct']].describe())
        
        print("\nSuccess rate by first letter revealed:")
        print(features_df.groupby('first_letter_revealed')['user_correct'].agg(['count', 'mean']))
        
        print("\nSuccess rate by answer length:")
        print(features_df.groupby('answer_length')['user_correct'].agg(['count', 'mean']).head(10))


def main():
    """Main function to demonstrate the crossword answer predictor."""
    
    # Suppress sklearn warnings for cleaner output
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    
    # Get the project root and data path
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "crossword_session_LOG.csv"
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please run the crossword collector app to generate training data first.")
        return
    
    # Initialize the predictor
    predictor = CrosswordAnswerPredictor()
    
    try:
        # Load and preprocess data
        features_df = predictor.load_and_preprocess_data(str(data_path))
        
        if len(features_df) < 10:
            print("Not enough data to train the model. Please collect more crossword attempts.")
            return
        
        # Analyze the data
        predictor.analyze_data(features_df)
        
        # Train the model
        results = predictor.train_model(features_df)
        
        # Display results
        print("\n=== Model Performance ===")
        print(results['classification_report'])
        
        # Save the model
        predictor.save_model()
        
        # Example predictions to demonstrate monotone constraints
        print("\n=== Monotone Constraint Validation ===")
        
        # Test 1: Probability should decrease with word length (all letters remaining)
        print("1. Effect of word length (all letters remaining):")
        for length in [3, 4, 5, 6, 7, 8, 10]:
            prob = predictor.predict_success_probability_simple(length, False, length)
            print(f"   Length {length:2d}: {prob:.1%}")
        
        # Test 2: Probability should increase with fewer letters remaining (5-letter word, no first letter)
        print("\n2. Effect of letters remaining (5-letter word, no first letter):")
        for remaining in [5, 4, 3, 2, 1]:
            prob = predictor.predict_success_probability_simple(5, False, remaining)
            print(f"   Remaining {remaining}: {prob:.1%}")
        
        # Test 3: Combined effects
        print("\n=== Combined Effect Examples ===")
        examples = [
            (3, False, 3),   # 3-letter word, all letters remaining
            (3, False, 2),   # 3-letter word, 2 letters remaining
            (3, True, 2),    # 3-letter word, first letter revealed, 2 remaining
            (5, False, 5),   # 5-letter word, all letters remaining
            (5, False, 3),   # 5-letter word, 3 letters remaining
            (5, True, 3),    # 5-letter word, first revealed, 3 remaining
            (8, False, 8),   # 8-letter word, all letters remaining
            (8, False, 6),   # 8-letter word, 6 letters remaining
            (8, True, 6),    # 8-letter word, first revealed, 6 remaining
        ]

        # Test 4: Length effect for fixed letters remaining
        
        for length, first_revealed, remaining in examples:
            prob = predictor.predict_success_probability_simple(length, first_revealed, remaining)
            print(f"Length: {length}, First revealed: {str(first_revealed):5s}, Remaining: {remaining} "
                  f"→ Success probability: {prob:.1%}")
        
        # Plot feature importance
        try:
            predictor.plot_feature_importance(results)
        except ImportError:
            print("Matplotlib not available for plotting. Install with: pip install matplotlib seaborn")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
