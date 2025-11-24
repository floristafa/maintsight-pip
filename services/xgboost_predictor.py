"""XGBoost prediction service for maintenance risk analysis using multiwindow_v2 model."""

import json
import logging
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib

from utils.logger import Logger
from models import RiskPrediction, RiskCategory, CommitData

logger = logging.getLogger(__name__)


class XGBoostPredictor:
    """Standalone risk predictor that uses multiwindow_v2.pkl model and metadata."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize predictor.
        
        Args:
            model_path: Path to model file. If None, uses bundled v2 model.
        """
        self.model = None
        self.calibration_stats = None
        self.feature_engineer = None
        self.logger = Logger('XGBoostPredictor')
        
        if model_path:
            self.load_model(model_path)
        
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load XGBoost model from pickle file.
        
        Args:
            model_path: Path to model file. If None, uses bundled v2 model.
        """
        if model_path is None:
            # Use bundled multiwindow_v2 model
            model_path = str(Path(__file__).parent.parent / 'models' / 'xgboost_degradation_model_multiwindow_v2.pkl')
            
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        self.logger.info(f"Loading model from {model_path}", '🤖')
        
        if joblib is None:
            raise RuntimeError("joblib is required to load the model. Please install it: pip install joblib")
            
        try:
            self.model = joblib.load(model_path_obj)
            self.logger.success("Model loaded successfully", '✅')
        except Exception as e:
            raise RuntimeError(f"Failed to load XGBoost model: {e}")
            
        # Load training prediction statistics for calibration
        metadata_path = model_path_obj.parent / (model_path_obj.stem + "_metadata.json")
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.calibration_stats = json.load(f)
            self.logger.info(f"Loaded calibration stats from {metadata_path}", '📊')
        else:
            self.logger.warn(f"Calibration metadata not found: {metadata_path}. Using default values.", '⚠️')
            self.calibration_stats = None
            
        # Initialize feature engineer
        from services.feature_engineer import FeatureEngineer
        self.feature_engineer = FeatureEngineer()
        
    def predict(self, commit_data_list: List[CommitData]) -> List[RiskPrediction]:
        """Predict maintenance risk for commit data.
        
        Args:
            commit_data_list: List of CommitData objects with commit features
            
        Returns:
            List of RiskPrediction objects
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self.feature_engineer is None:
            raise RuntimeError("Feature engineer not initialized. Call load_model() first.")
            
        self.logger.info(f"Generating features for {len(commit_data_list)} file records...", '🔧')
        commit_data_with_features = self.feature_engineer.transform(commit_data_list)
        
        # Convert CommitData objects to feature vectors for ML model
        feature_vectors = []
        for commit_data in commit_data_with_features:
            feature_vectors.append(commit_data.to_feature_vector())
        
        X = np.array(feature_vectors)
        feature_names = CommitData.feature_names()
        
        # Check if model expects specific features
        try:
            expected_features = self.model.get_booster().feature_names
            if expected_features:
                self.logger.info(f"Model expects {len(expected_features)} features", '📊')
                self.logger.info(f"We have {len(feature_names)} features", '📊')
                
                # Map our features to expected features
                feature_mapping = {}
                for i, feat_name in enumerate(feature_names):
                    if feat_name in expected_features:
                        feature_mapping[expected_features.index(feat_name)] = i
                
                # Create reordered feature matrix
                X_reordered = np.zeros((X.shape[0], len(expected_features)))
                for expected_idx, our_idx in feature_mapping.items():
                    X_reordered[:, expected_idx] = X[:, our_idx]
                
                # Fill missing features with 0
                missing_features = set(expected_features) - set(feature_names)
                if missing_features:
                    self.logger.warn(f"Missing features: {missing_features}", '⚠️')
                
                X = X_reordered
                self.logger.info(f"Using {len(expected_features)} features for prediction", '📊')
        except Exception as e:
            # Fallback if we can't get feature names from model
            self.logger.warn(f"Could not get model feature names: {e}")
            if X.size == 0:
                raise RuntimeError("No valid features found for prediction")
        
        self.logger.info("Running inference...", '🔮')
        
        # Make predictions
        if hasattr(self.model, 'predict_proba'):
            # For classifiers
            raw_predictions = self.model.predict_proba(X)[:, 1]  # Probability of positive class
        else:
            # For regressors
            raw_predictions = self.model.predict(X)
        
        # Apply calibration if available
        if self.calibration_stats:
            degradation_scores = self._calibrate_predictions(raw_predictions)
        else:
            degradation_scores = raw_predictions
        
        # Print prediction statistics
        self.logger.info(f"Raw predictions - Mean: {np.mean(raw_predictions):.3f}, Range: [{np.min(raw_predictions):.3f}, {np.max(raw_predictions):.3f}]", '📊')
        if self.calibration_stats:
            self.logger.info(f"Calibrated predictions - Mean: {np.mean(degradation_scores):.3f}, Range: [{np.min(degradation_scores):.3f}, {np.max(degradation_scores):.3f}]", '📊')
            self.logger.info(f"Std dev: {np.std(degradation_scores):.3f}", '📊')
        
        self.logger.success("Predictions complete", '✅')
        
        # Convert to RiskPrediction objects
        predictions = []
        for i, commit_data in enumerate(commit_data_with_features):
            degradation_score = degradation_scores[i]
            raw_prediction = raw_predictions[i]
            risk_category = RiskCategory.from_score(degradation_score)
            
            predictions.append(RiskPrediction(
                module=commit_data.module,
                risk_category=risk_category,
                degradation_score=degradation_score,
                raw_prediction=raw_prediction
            ))
        
        return predictions
        
    def _predict_single(self, features: List[float]) -> float:
        """Make prediction for a single feature vector.
        
        Args:
            features: Feature vector
            
        Returns:
            Raw prediction score
        """
        # Get base score
        score = 0.5  # default
        
        # Try to extract base score from model
        if (self.model.get('model_data', {})
                .get('learner', {})
                .get('learner_model_param', {})
                .get('base_score')):
            base_score_str = (self.model['model_data']['learner']
                            ['learner_model_param']['base_score'])
            # Parse base score (might be in format like "[-1.201454E-2]")
            if '[' in base_score_str and ']' in base_score_str:
                score_val = base_score_str.strip('[]')
                score = float(score_val)
                
        # Get trees
        trees = (self.model.get('model_data', {})
                .get('learner', {})
                .get('gradient_booster', {})
                .get('model', {})
                .get('trees', []))
                
        if not trees:
            return score
            
        # Predict with each tree
        for tree in trees:
            score += self._predict_tree(tree, features)
            
        # Apply sigmoid transformation
        return 1.0 / (1.0 + math.exp(-score))
        
    def _predict_tree(self, tree: Dict[str, Any], features: List[float]) -> float:
        """Make prediction using a single tree.
        
        Args:
            tree: Tree structure from model
            features: Feature vector
            
        Returns:
            Tree prediction value
        """
        node_id = 0  # Start at root
        
        while True:
            # Check if leaf node
            if tree['left_children'][node_id] == -1:
                return tree['base_weights'][node_id]
                
            # Get feature value and threshold
            feature_idx = tree['split_indices'][node_id]
            feature_value = features[feature_idx]
            threshold = tree['split_conditions'][node_id]
            
            # Navigate to child node
            if feature_value < threshold:
                node_id = tree['left_children'][node_id]
            else:
                node_id = tree['right_children'][node_id]
                
    def _calibrate_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Calibrate predictions to match training data distribution using saved metadata.
        """
        if self.calibration_stats is not None:
            TRAIN_MEAN = self.calibration_stats.get("mean", 0)
            TRAIN_STD = self.calibration_stats.get("std", 1) 
            TRAIN_MIN = self.calibration_stats.get("min", 0)
            TRAIN_MAX = self.calibration_stats.get("max", 1)
        else:
            # Default values if no metadata
            TRAIN_MEAN = 0
            TRAIN_STD = 1
            TRAIN_MIN = 0
            TRAIN_MAX = 1
        
        # Calculate raw prediction statistics
        raw_mean = predictions.mean()
        raw_std = predictions.std()
        
        # Method 1: Z-score normalization then scale to training distribution
        # This preserves relative differences while matching the target distribution
        if raw_std > 0:
            # Standardize (z-score)
            z_scores = (predictions - raw_mean) / raw_std
            # Scale to training distribution
            calibrated = z_scores * TRAIN_STD + TRAIN_MEAN
            # Clip to training data range (with small buffer for unseen cases)
            calibrated = np.clip(calibrated, TRAIN_MIN - 0.1, TRAIN_MAX + 0.1)
        else:
            # All predictions the same - just shift to training mean
            calibrated = np.full_like(predictions, TRAIN_MEAN)
        
        self.logger.info(f"   📊 Calibration: Shifted mean from {raw_mean:.3f} to {calibrated.mean():.3f}", '📊')
        return calibrated