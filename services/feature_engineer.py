"""Enhanced feature engineering service."""

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List
from services.git_feature_extractor import GitFeatureExtractor
from services.temporal_data_generator import TemporalDataGenerator

logger = logging.getLogger(__name__)
class FeatureEngineer:
    """Simplified interface to feature extraction."""
    
    @staticmethod
    def extract_features(repo_path: str, branch: str, feature_list: List[str]) -> pd.DataFrame:
        """Extract features from repository using standalone script logic."""
        logger.info("=" * 60)
        logger.info(f"EXTRACTING FEATURES FOR: {repo_path}")
        logger.info("=" * 60)

        extractor = GitFeatureExtractor(repo_path, branch=branch)

        time_T = datetime.now()
        feature_start = time_T - timedelta(days=150)
        feature_end = time_T

        logger.info(f"Feature window: [{feature_start.date()} to {feature_end.date()}]")

        # Get files at current time
        temporal_gen = TemporalDataGenerator()
        files = temporal_gen.get_files_at_time(repo_path, branch, time_T)
        logger.info(f"Found {len(files)} source files")

        if not files:
            raise ValueError("No source files found in repository!")

        logger.info("Extracting features...")
        features_df = extractor.extract_features_at_time(
            repo_path, branch, files, feature_start, feature_end
        )

        logger.info(f"✓ Extracted features for {len(features_df)} files")

        expected_features = feature_list
        missing = set(expected_features) - set(features_df.columns)
        if missing:
            logger.warning(f"⚠️  Missing features (filling with 0): {missing}")
            for feat in missing:
                features_df[feat] = 0

        features_df = features_df[expected_features]

        return features_df