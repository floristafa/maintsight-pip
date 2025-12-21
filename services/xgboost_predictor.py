"""XGBoost prediction service - Clean interface to enhanced logic."""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import xgboost as xgb

from models import RiskPrediction, RiskCategory
from services.post_processor import apply_targeted_post_processing

logger = logging.getLogger(__name__)


class XGBoostPredictor:
    """Clean XGBoost predictor using enhanced logic from standalone script."""
    
    def __init__(self, model_path: Optional[str] = None, metadata_path: Optional[str] = None):
        """Initialize predictor."""
        self.model = None
        self.metadata = None
        self.model_path = model_path
        self.metadata_path = metadata_path
        
    def load_model(self) -> tuple[Any, Dict[str, Any]]:
        """Load XGBoost model and metadata."""
        if self.model_path is None:
            self.model_path = str(Path(__file__).parent.parent / 'models' / 'xgboost_model.pkl')
        if self.metadata_path is None:
            self.metadata_path = str(Path(__file__).parent.parent / 'models' / 'xgboost_model_metadata.json')
            
        logger.info(f"Loading model from {self.model_path}")
        
        self.model = xgb.Booster()
        self.model.load_model(self.model_path)
        
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        logger.info(f"✓ Loaded model v{self.metadata['version']}")
        logger.info(f"   Features: {len(self.metadata['feature_list'])}")
        
        return self.model, self.metadata
        
    def predict_risk(self, features_df: pd.DataFrame, post_processing_mode: str = 'moderate') -> pd.DataFrame:
        """Run predictions WITH post-processing."""
        if self.model is None:
            self.load_model()
            
        logger.info("=" * 80)
        logger.info("RUNNING PREDICTIONS WITH POST-PROCESSING")
        logger.info("=" * 80)
        
        X = features_df.fillna(0).values
        dmatrix = xgb.DMatrix(X)
        
        logger.info(f"Predicting risk for {len(features_df)} files...")
        assert self.model is not None, "Model should be loaded by this point"
        predictions = self.model.predict(dmatrix)
        
        logger.info(f"✨ Applying v3 targeted post-processing (mode={post_processing_mode})...")
        features_for_pp = features_df.reset_index()
        features_for_pp.columns = ['file'] + list(features_df.columns)
        processed_predictions = apply_targeted_post_processing(
        predictions=predictions,
        features_df=features_for_pp,
        mode=post_processing_mode
    )
        final_predictions = processed_predictions
        logger.info("✓ Post-processing applied")

            
        min_score = final_predictions.min()
        max_score = final_predictions.max()
        
        if max_score > min_score:
            normalized_scores = 100 * (final_predictions - min_score) / (max_score - min_score)
        else:
            normalized_scores = final_predictions * 0
        
        results_df = pd.DataFrame({
            'file': features_df.index,
            'risk_score': normalized_scores,
            'raw_prediction': predictions,
            'adjusted_score': final_predictions
        })
        
        results_df = results_df.sort_values('risk_score', ascending=False)
        results_df['rank'] = range(1, len(results_df) + 1)

        logger.info(f"✓ Predictions complete")
        logger.info(f"   Raw score range: {predictions.min():.4f} - {predictions.max():.4f}")
        logger.info(f"   Adjusted score range: {min_score:.4f} - {max_score:.4f}")
        logger.info(f"   Normalized score range: 0.00 - 100.00")

        return results_df
    
    def predict_as_objects(self, features_df: pd.DataFrame, post_processing_mode: str = 'moderate') -> List[RiskPrediction]:
        """Run predictions and return as RiskPrediction objects."""
        results_df = self.predict_risk(features_df, post_processing_mode)
        
        predictions = []
        for _, row in results_df.iterrows():
            # Use 0-100 normalized score directly (no conversion to 0-1)
            normalized_score = float(row['risk_score'])  # This is already 0-100
            risk_category = RiskCategory.from_score(normalized_score)  # Use 0-100 scale
            
            predictions.append(RiskPrediction(
                module=str(row['file']),
                risk_category=risk_category,
                normalized_score=normalized_score,  # Store 0-100 score directly
                raw_prediction=float(row['raw_prediction'])
            ))
        
        return predictions