"""Post-processing service for prediction adjustments."""

import logging
import numpy as np
import pandas as pd
from typing import Tuple

logger = logging.getLogger(__name__)

class TargetedPostProcessor:
    """Post-processor that applies selective adjustments to XGBoost predictions."""

    def __init__(
        self,
        stabilization_percentile: float = 95,
        recent_activity_percentile: float = 90,
        adjustment_magnitude: float = 0.5
    ):
        self.stabilization_percentile = stabilization_percentile
        self.recent_activity_percentile = recent_activity_percentile
        self.adjustment_magnitude = adjustment_magnitude

        logger.info(f"✨ Initialized TargetedPostProcessor v3:")
        logger.info(f"   Stabilization threshold: {stabilization_percentile}th percentile")
        logger.info(f"   Recent activity threshold: {recent_activity_percentile}th percentile")
        logger.info(f"   Adjustment magnitude: ±{adjustment_magnitude}")

    def process(
        self,
        predictions: np.ndarray,
        features_df: pd.DataFrame
    ) -> np.ndarray:
        """Apply post-processing adjustments to predictions."""
        logger.info("\n🔧 Applying v3 targeted post-processing...")

        adjusted_predictions = predictions.copy()

        stabilization_mask, stab_score = self._identify_stabilized_files(features_df)
        activity_mask, activity_score = self._identify_recently_active_files(features_df)
        bug_spike_mask, bug_score = self._identify_bug_spikes(features_df)

        num_stabilized = stabilization_mask.sum()
        num_active = activity_mask.sum()
        num_bug_spikes = bug_spike_mask.sum()

        if num_stabilized > 0:
            adjusted_predictions[stabilization_mask] += self.adjustment_magnitude
            logger.info(f"   ✓ Adjusted {num_stabilized} stabilized files ({100*num_stabilized/len(predictions):.1f}%) - made SAFER")

        if num_active > 0:
            adjusted_predictions[activity_mask] -= self.adjustment_magnitude
            logger.info(f"   ✓ Adjusted {num_active} recently active files ({100*num_active/len(predictions):.1f}%) - made RISKIER")

        if num_bug_spikes > 0:
            adjusted_predictions[bug_spike_mask] -= self.adjustment_magnitude * 1.5
            logger.info(f"   ✓ Adjusted {num_bug_spikes} bug spike files ({100*num_bug_spikes/len(predictions):.1f}%) - made MUCH RISKIER")

        total_adjusted = len(set(np.where(stabilization_mask)[0]) |
                           set(np.where(activity_mask)[0]) |
                           set(np.where(bug_spike_mask)[0]))

        logger.info(f"\n   📊 Summary:")
        logger.info(f"      Total files adjusted: {total_adjusted}/{len(predictions)} ({100*total_adjusted/len(predictions):.1f}%)")
        logger.info(f"      Score range before: [{predictions.min():.4f}, {predictions.max():.4f}]")
        logger.info(f"      Score range after: [{adjusted_predictions.min():.4f}, {adjusted_predictions.max():.4f}]")

        return adjusted_predictions

    def _identify_stabilized_files(self, features_df: pd.DataFrame) -> tuple:
        """Identify files that are clearly stabilized."""
        commits_total = features_df.get('commits', pd.Series(np.zeros(len(features_df)))).values
        commits_ratio = features_df.get('commits_ratio_30_90', pd.Series(np.ones(len(features_df)))).values

        stabilization_score = commits_total * (1 - commits_ratio)

        if len(stabilization_score) > 0 and stabilization_score.max() > 0:
            threshold = np.percentile(stabilization_score, self.stabilization_percentile)
            mask = stabilization_score > threshold
        else:
            mask = np.zeros(len(features_df), dtype=bool)
            stabilization_score = np.zeros(len(features_df))

        return mask, stabilization_score

    def _identify_recently_active_files(self, features_df: pd.DataFrame) -> tuple:
        """Identify files with strong recent activity."""
        commits_ratio = features_df.get('commits_ratio_30_90', pd.Series(np.zeros(len(features_df)))).values
        activity_intensity = features_df.get('activity_intensity', pd.Series(np.zeros(len(features_df)))).values

        activity_score = commits_ratio * (1 + activity_intensity)

        if len(activity_score) > 0 and activity_score.max() > 0:
            threshold = np.percentile(activity_score, self.recent_activity_percentile)
            mask = activity_score > threshold
        else:
            mask = np.zeros(len(features_df), dtype=bool)
            activity_score = np.zeros(len(features_df))

        return mask, activity_score

    def _identify_bug_spikes(self, features_df: pd.DataFrame) -> tuple:
        """Identify files with clear bug spikes."""
        bug_ratio = features_df.get('bug_ratio_30_90', pd.Series(np.zeros(len(features_df)))).values
        recent_bug_intensity = features_df.get('recent_bug_intensity', pd.Series(np.zeros(len(features_df)))).values

        bug_spike_score = bug_ratio * (1 + recent_bug_intensity)

        if len(bug_spike_score) > 0 and bug_spike_score.max() > 0:
            threshold = np.percentile(bug_spike_score, 90)
            mask = (bug_spike_score > threshold) & (bug_ratio > 1.2)
        else:
            mask = np.zeros(len(features_df), dtype=bool)
            bug_spike_score = np.zeros(len(features_df))

        return mask, bug_spike_score


def apply_targeted_post_processing(
    predictions: np.ndarray,
    features_df: pd.DataFrame,
    mode: str = 'moderate'
) -> np.ndarray:
    """Convenience function to apply v3 targeted post-processing."""
    if mode == 'conservative':
        processor = TargetedPostProcessor(
            stabilization_percentile=98,
            recent_activity_percentile=95,
            adjustment_magnitude=0.3
        )
    elif mode == 'aggressive':
        processor = TargetedPostProcessor(
            stabilization_percentile=85,
            recent_activity_percentile=85,
            adjustment_magnitude=1.0
        )
    else:
        processor = TargetedPostProcessor()

    return processor.process(predictions, features_df)