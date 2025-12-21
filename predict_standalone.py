#!/usr/bin/env python3
"""
Standalone Risk Prediction Script - All Dependencies Included

This is a self-contained version with all dependencies embedded.
No need for external files from the dags/ directory.

Usage:
    python predict_standalone.py <repo_path> --branch <branch>

Example:
    python predict_standalone.py /path/to/repo --branch main

Requirements:
    pip install xgboost pandas numpy gitpython
"""

import sys
import argparse
import logging
from pathlib import Path
import json
import xgboost as xgb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import os
from git import Repo

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# EMBEDDED: GitFeatureExtractorTryAgain
# ============================================================================

class GitFeatureExtractorTryAgain:
    """
    Enhanced feature extractor with temporal ratios and context awareness.
    """

    def __init__(self, repo_path: str, branch: str = "main"):
        self.repo_path = repo_path
        self.repo = Repo(repo_path)
        self.branch = branch
        logger.info(f"Initialized GitFeatureExtractorTryAgain for {repo_path}")

    def extract_features_at_time(
        self,
        repo_path: str,
        branch: str,
        file_paths: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Extract enhanced features for specific files at a historical time point."""
        logger.info(f"   Extracting tryAgain features for {len(file_paths)} files at time {end_date.date()}")

        repo = Repo(repo_path) if repo_path != self.repo_path else self.repo
        commits = list(
            repo.iter_commits(
                branch,
                since=start_date.strftime('%Y-%m-%d'),
                until=end_date.strftime('%Y-%m-%d')
            )
        )

        if not commits:
            logger.warning(f"   No commits found in window [{start_date.date()} to {end_date.date()}]")
            return pd.DataFrame(index=file_paths)

        file_stats = self._aggregate_by_file_with_temporal(commits, start_date, end_date)

        records = []
        for file_path in file_paths:
            if file_path in file_stats:
                features = self._calculate_enhanced_features(
                    file_stats[file_path],
                    start_date,
                    end_date,
                    file_path,
                    repo_path
                )
                features['module'] = file_path
                records.append(features)
            else:
                features = self._get_empty_features(file_path, repo_path)
                features['module'] = file_path
                records.append(features)

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.set_index('module')

        return df

    def _get_empty_features(self, file_path: str, repo_path: str) -> Dict:
        """Return dict of features with zero values (for files with no activity)."""
        context_features = self._extract_context_features(file_path, repo_path)

        return {
            # Activity Features (15)
            'commits': 0.0,
            'commits_last_30d': 0.0,
            'commits_last_60d': 0.0,
            'commits_ratio_30_90': 0.0,
            'commits_ratio_60_90': 0.0,
            'churn': 0.0,
            'churn_last_30d': 0.0,
            'churn_ratio_30_90': 0.0,
            'commits_per_month': 0.0,
            'churn_per_commit': 0.0,
            'num_authors': 0.0,
            'author_ratio_30_90': 0.0,
            'acceleration_score': 0.0,
            'activity_intensity': 0.0,
            'recency_score': 0.0,

            # Bug/Fix Features (10)
            'bug_commits': 0.0,
            'bug_ratio': 0.0,
            'bug_commits_last_30d': 0.0,
            'bug_ratio_30_90': 0.0,
            'revert_commits': 0.0,
            'revert_ratio': 0.0,
            'emergency_commits': 0.0,
            'bug_churn_ratio': 0.0,
            'recent_bug_intensity': 0.0,
            'bug_fix_frequency': 0.0,

            # Complexity Features (8)
            'lines_of_code': 0.0,
            'cyclomatic_complexity': 0.0,
            'num_functions': 0.0,
            'num_classes': 0.0,
            'max_function_complexity': 0.0,
            'complexity_change_rate': 0.0,
            'code_ownership': 1.0,
            'coupling_score': 0.0,

            # Historical Patterns (10)
            'file_age_days': 0.0,
            'days_since_last_commit_normalized': 1.0,
            'maturity_score': 0.0,
            'stability_score': 1.0,
            'revert_rate': 0.0,
            'author_turnover': 0.0,
            'hotspot_score': 0.0,
            'regression_indicator': 0.0,
            'maintenance_burden_score': 0.0,
            'technical_debt_score': 0.0,

            # Context Features (5)
            **context_features
        }

    def _aggregate_by_file_with_temporal(
        self,
        commits: List,
        window_start: datetime,
        window_end: datetime
    ) -> Dict:
        """Aggregate commit information by file with temporal tracking."""
        file_stats = {}
        window_days = (window_end - window_start).days

        for commit in commits:
            commit_date = commit.committed_datetime
            if commit_date.tzinfo is not None:
                commit_date = commit_date.astimezone().replace(tzinfo=None)

            days_before_end = (window_end - commit_date).days

            author = commit.author.email
            message = commit.message.lower()

            is_bug_fix = any(kw in message for kw in ['fix', 'bug', 'patch', 'hotfix'])
            is_feature = any(kw in message for kw in ['feat', 'feature', 'add', 'implement'])
            is_refactor = any(kw in message for kw in ['refactor', 'clean', 'improve'])
            is_revert = 'revert' in message or 'rollback' in message
            is_emergency = any(kw in message for kw in ['hotfix', 'critical', 'urgent', 'emergency'])

            if commit.parents:
                parent = commit.parents[0]
                try:
                    diffs = parent.diff(commit, create_patch=True)
                except:
                    continue

                for diff in diffs:
                    filepath = diff.b_path or diff.a_path
                    if not filepath:
                        continue

                    if filepath not in file_stats:
                        file_stats[filepath] = {
                            'lines_added': 0,
                            'lines_deleted': 0,
                            'commits': 0,
                            'authors': set(),
                            'bug_commits': 0,
                            'feature_commits': 0,
                            'refactor_commits': 0,
                            'revert_commits': 0,
                            'emergency_commits': 0,
                            'first_commit': commit_date,
                            'last_commit': commit_date,
                            'commit_dates': [],
                            'commit_churns': [],
                            'commit_authors': [],
                            'commits_last_30d': 0,
                            'commits_last_60d': 0,
                            'churn_last_30d': 0,
                            'churn_last_60d': 0,
                            'bug_commits_last_30d': 0,
                            'bug_commits_last_60d': 0,
                            'authors_last_30d': set(),
                            'authors_last_60d': set(),
                        }

                    stats = file_stats[filepath]

                    commit_churn = 0
                    if diff.diff:
                        try:
                            diff_text = diff.diff.decode('utf-8', errors='ignore')
                            lines_added = sum(1 for line in diff_text.split('\n')
                                            if line.startswith('+') and not line.startswith('+++'))
                            lines_deleted = sum(1 for line in diff_text.split('\n')
                                              if line.startswith('-') and not line.startswith('---'))
                            stats['lines_added'] += lines_added
                            stats['lines_deleted'] += lines_deleted
                            commit_churn = lines_added + lines_deleted
                        except:
                            pass

                    stats['commits'] += 1
                    stats['authors'].add(author)
                    stats['commit_dates'].append(commit_date)
                    stats['commit_churns'].append(commit_churn)
                    stats['commit_authors'].append(author)

                    if is_bug_fix:
                        stats['bug_commits'] += 1
                    if is_feature:
                        stats['feature_commits'] += 1
                    if is_refactor:
                        stats['refactor_commits'] += 1
                    if is_revert:
                        stats['revert_commits'] += 1
                    if is_emergency:
                        stats['emergency_commits'] += 1

                    stats['first_commit'] = min(stats['first_commit'], commit_date)
                    stats['last_commit'] = max(stats['last_commit'], commit_date)

                    if days_before_end <= 30:
                        stats['commits_last_30d'] += 1
                        stats['churn_last_30d'] += commit_churn
                        stats['authors_last_30d'].add(author)
                        if is_bug_fix:
                            stats['bug_commits_last_30d'] += 1

                    if days_before_end <= 60:
                        stats['commits_last_60d'] += 1
                        stats['churn_last_60d'] += commit_churn
                        stats['authors_last_60d'].add(author)
                        if is_bug_fix:
                            stats['bug_commits_last_60d'] += 1

        return file_stats

    def _calculate_enhanced_features(
        self,
        stats: Dict,
        start_date: datetime,
        end_date: datetime,
        file_path: str,
        repo_path: str
    ) -> Dict:
        """Calculate enhanced feature set with temporal ratios and context."""
        window_days = (end_date - start_date).days
        num_commits = stats['commits']
        num_authors = len(stats['authors'])
        total_churn = stats['lines_added'] + stats['lines_deleted']

        features = {}

        # Activity Features (15)
        features['commits'] = num_commits
        features['commits_last_30d'] = stats['commits_last_30d']
        features['commits_last_60d'] = stats['commits_last_60d']
        features['commits_ratio_30_90'] = stats['commits_last_30d'] / max(num_commits, 1)
        features['commits_ratio_60_90'] = stats['commits_last_60d'] / max(num_commits, 1)
        features['churn'] = total_churn
        features['churn_last_30d'] = stats['churn_last_30d']
        features['churn_ratio_30_90'] = stats['churn_last_30d'] / max(total_churn, 1)
        features['commits_per_month'] = (num_commits / window_days) * 30
        features['churn_per_commit'] = total_churn / max(num_commits, 1)
        features['num_authors'] = num_authors
        features['author_ratio_30_90'] = len(stats['authors_last_30d']) / max(num_authors, 1)
        features['acceleration_score'] = self._calculate_acceleration(stats['commit_dates'], end_date)
        features['activity_intensity'] = features['commits_ratio_30_90'] * features['churn_ratio_30_90']
        days_since_last = (end_date - stats['last_commit']).days
        features['recency_score'] = max(0.0, 1.0 - (days_since_last / window_days))

        # Bug/Fix Features (10)
        features['bug_commits'] = stats['bug_commits']
        features['bug_ratio'] = stats['bug_commits'] / max(num_commits, 1)
        features['bug_commits_last_30d'] = stats['bug_commits_last_30d']
        features['bug_ratio_30_90'] = stats['bug_commits_last_30d'] / max(stats['bug_commits'], 1)
        features['revert_commits'] = stats['revert_commits']
        features['revert_ratio'] = stats['revert_commits'] / max(num_commits, 1)
        features['emergency_commits'] = stats['emergency_commits']
        features['bug_churn_ratio'] = features['bug_ratio']
        features['recent_bug_intensity'] = features['bug_ratio_30_90'] * features['activity_intensity']
        features['bug_fix_frequency'] = (stats['bug_commits'] / window_days) * 30

        # Complexity Features (8)
        features['lines_of_code'] = 0.0
        features['cyclomatic_complexity'] = 0.0
        features['num_functions'] = 0.0
        features['num_classes'] = 0.0
        features['max_function_complexity'] = 0.0
        features['complexity_change_rate'] = features['churn_per_commit'] * features['commits_per_month']
        features['code_ownership'] = 1.0 / max(num_authors, 1)
        features['coupling_score'] = num_authors * num_commits / max(window_days, 1)

        # Historical Patterns (10)
        file_age_days = (end_date - stats['first_commit']).days
        features['file_age_days'] = file_age_days
        features['days_since_last_commit_normalized'] = days_since_last / window_days
        features['maturity_score'] = min(num_commits / 20.0, 1.0)
        features['stability_score'] = 1.0 - features['commits_ratio_30_90']
        features['revert_rate'] = stats['revert_commits'] / max(num_commits, 1)
        features['author_turnover'] = num_authors / max(num_commits, 1)
        features['hotspot_score'] = features['commits_per_month'] * features['bug_ratio']
        features['regression_indicator'] = (
            features['revert_ratio'] + (stats['emergency_commits'] / max(num_commits, 1))
        ) / 2.0
        features['maintenance_burden_score'] = (
            0.3 * features['bug_ratio'] +
            0.3 * features['revert_ratio'] +
            0.2 * features['activity_intensity'] +
            0.2 * (1.0 - features['stability_score'])
        )
        refactor_ratio = stats['refactor_commits'] / max(num_commits, 1)
        features['technical_debt_score'] = 0.5 * features['bug_ratio'] + 0.5 * refactor_ratio

        # Context Features (5)
        context_features = self._extract_context_features(file_path, repo_path)
        features.update(context_features)

        return features

    def _calculate_acceleration(self, commit_dates: List[datetime], end_date: datetime) -> float:
        """Calculate acceleration score (is activity increasing?)."""
        if len(commit_dates) < 4:
            return 0.5

        mid = len(commit_dates) // 2
        first_half = commit_dates[:mid]
        second_half = commit_dates[mid:]

        first_half_days = (max(first_half) - min(first_half)).days or 1
        second_half_days = (max(second_half) - min(second_half)).days or 1

        first_rate = len(first_half) / first_half_days
        second_rate = len(second_half) / second_half_days

        accel_ratio = second_rate / max(first_rate, 0.001)
        return min(accel_ratio / 2.0, 1.0)

    def _extract_context_features(self, file_path: str, repo_path: str) -> Dict:
        """Extract context features about the file."""
        file_depth = file_path.count(os.sep)

        is_test = 1.0 if any(pattern in file_path.lower() for pattern in [
            'test', 'spec', '__test__', '.test.', '.spec.'
        ]) else 0.0

        is_config = 1.0 if any(pattern in file_path.lower() for pattern in [
            'config', 'settings', '.config', 'setup'
        ]) else 0.0

        module_activity_score = max(0.0, 1.0 - (file_depth / 10.0))

        ext = Path(file_path).suffix.lower()
        file_type_risk_map = {
            '.ts': 0.6, '.tsx': 0.6,
            '.js': 0.7, '.jsx': 0.7,
            '.py': 0.5,
            '.java': 0.4,
            '.go': 0.4, '.rs': 0.3,
        }
        file_type_risk = file_type_risk_map.get(ext, 0.5)

        return {
            'file_depth': file_depth,
            'is_test': is_test,
            'is_config': is_config,
            'module_activity_score': module_activity_score,
            'file_type_risk': file_type_risk,
        }


# ============================================================================
# EMBEDDED: TemporalDataGenerator
# ============================================================================

class TemporalDataGenerator:
    """Generates training data with proper temporal splits."""

    def __init__(
        self,
        feature_window_days: int = 150,
        label_window_days: int = 30,
        step_days: int = 30
    ):
        self.feature_window_days = feature_window_days
        self.label_window_days = label_window_days
        self.step_days = step_days

    def get_files_at_time(
        self,
        repo_path: str,
        branch: str,
        time_T: datetime
    ) -> List[str]:
        """Get list of source files that exist at time T."""
        repo = Repo(repo_path)

        source_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
            '.cs', '.rb', '.go', '.rs', '.php', '.swift', '.kt', '.scala'
        }

        try:
            commits = list(repo.iter_commits(
                branch,
                until=time_T.strftime('%Y-%m-%d %H:%M:%S'),
                max_count=1
            ))

            if not commits:
                return []

            commit = commits[0]

            files = []
            for item in commit.tree.traverse():
                if item.type == 'blob':
                    file_path = item.path
                    if any(file_path.endswith(ext) for ext in source_extensions):
                        files.append(file_path)

            return files

        except Exception as e:
            logger.warning(f"Error getting files at {time_T.date()}: {e}")
            return []


# ============================================================================
# EMBEDDED: Post-Processing v3
# ============================================================================

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


# ============================================================================
# MAIN PREDICTION LOGIC
# ============================================================================

def load_model(model_path: str, metadata_path: str):
    """Load XGBoost model and metadata."""
    logger.info(f"Loading model from {model_path}")

    model = xgb.Booster()
    model.load_model(model_path)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    logger.info(f"✓ Loaded tryAgain v{metadata['version']}")
    logger.info(f"   Trained on {metadata['num_repos']} repositories")
    logger.info(f"   Features: {len(metadata['feature_list'])}")

    return model, metadata


def extract_features(repo_path: str, branch: str, feature_list: list):
    """Extract features from repository."""
    logger.info("=" * 80)
    logger.info(f"EXTRACTING FEATURES FOR: {repo_path}")
    logger.info("=" * 80)

    extractor = GitFeatureExtractorTryAgain(repo_path, branch=branch)

    time_T = datetime.now()
    feature_start = time_T - timedelta(days=150)
    feature_end = time_T

    logger.info(f"Feature window: [{feature_start.date()} to {feature_end.date()}]")

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


def predict_risk(model, features_df: pd.DataFrame, post_processing_mode: str = 'moderate'):
    """Run predictions WITH post-processing."""
    logger.info("=" * 80)
    logger.info("RUNNING PREDICTIONS WITH POST-PROCESSING")
    logger.info("=" * 80)

    X = features_df.fillna(0).values
    dmatrix = xgb.DMatrix(X)

    logger.info(f"Predicting risk for {len(features_df)} files...")
    predictions = model.predict(dmatrix)

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
        'raw_score': predictions,
        'adjusted_score': final_predictions
    })

    results_df = results_df.sort_values('risk_score', ascending=False)
    results_df['rank'] = range(1, len(results_df) + 1)

    logger.info(f"✓ Predictions complete")
    logger.info(f"   Raw score range: {predictions.min():.4f} - {predictions.max():.4f}")
    logger.info(f"   Adjusted score range: {min_score:.4f} - {max_score:.4f}")
    logger.info(f"   Normalized score range: 0.00 - 100.00")

    return results_df


def generate_html_report(results_df: pd.DataFrame, metadata: dict, repo_name: str, output_path: str):
    """Generate interactive HTML report."""
    logger.info("=" * 80)
    logger.info("GENERATING HTML REPORT")
    logger.info("=" * 80)

    rows = []
    for _, row in results_df.iterrows():
        risk_score = row['risk_score']

        if risk_score >= 90:
            risk_level = 'Critical'
            color = '#dc3545'
        elif risk_score >= 70:
            risk_level = 'High'
            color = '#fd7e14'
        elif risk_score >= 50:
            risk_level = 'Medium'
            color = '#ffc107'
        else:
            risk_level = 'Low'
            color = '#28a745'

        rows.append(f"""
        <tr>
            <td class="rank">#{row['rank']}</td>
            <td class="file-path">
                <span class="file-path">{row['file']}</span>
            </td>
            <td class="risk-level" style="color: {color}; font-weight: bold;">
                {risk_level}
            </td>
            <td class="risk-score">{risk_score:.2f}</td>
            <td class="raw-score">{row['raw_score']:.4f}</td>
        </tr>
        """)

    rows_html = "".join(rows)

    # Calculate counts for each risk category
    critical_count = len(results_df[results_df['risk_score'] >= 90])
    high_count = len(results_df[(results_df['risk_score'] >= 70) & (results_df['risk_score'] < 90)])
    medium_count = len(results_df[(results_df['risk_score'] >= 50) & (results_df['risk_score'] < 70)])
    low_count = len(results_df[results_df['risk_score'] < 50])

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Risk Report: {repo_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; font-weight: 700; }}
        .header p {{ font-size: 1.2em; opacity: 0.9; }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            padding: 30px 40px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }}
        .stat-card {{
            text-align: center;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }}
        .stat-value {{ font-size: 2em; font-weight: bold; }}
        .stat-label {{ font-size: 0.9em; color: #6c757d; margin-top: 5px; }}
        .stat-range {{ font-size: 0.75em; color: #adb5bd; margin-top: 3px; font-weight: normal; }}
        .stat-critical {{ color: #dc3545; }}
        .stat-high {{ color: #fd7e14; }}
        .stat-medium {{ color: #ffc107; }}
        .stat-low {{ color: #28a745; }}
        .stat-total {{ color: #667eea; }}
        .content {{ padding: 40px; }}
        .search-box {{ margin-bottom: 30px; }}
        .search-box input {{
            width: 100%;
            padding: 15px;
            font-size: 16px;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            transition: border-color 0.3s;
        }}
        .search-box input:focus {{ outline: none; border-color: #667eea; }}
        table {{ width: 100%; border-collapse: collapse; background: white; }}
        thead {{ background: #f8f9fa; position: sticky; top: 0; z-index: 10; }}
        th {{ padding: 15px; text-align: left; font-weight: 600; color: #495057; border-bottom: 2px solid #dee2e6; }}
        td {{ padding: 12px 15px; border-bottom: 1px solid #f1f3f5; }}
        tr:hover {{ background: #f8f9fa; }}
        .rank {{ font-weight: bold; color: #6c757d; }}
        .file-path {{ font-family: 'Monaco', 'Courier New', monospace; font-size: 0.9em; color: #212529; }}
        .risk-score {{ text-align: center; font-weight: bold; }}
        .raw-score {{ text-align: center; font-family: 'Monaco', 'Courier New', monospace; font-size: 0.85em; color: #6c757d; }}
        .footer {{ text-align: center; padding: 30px; background: #f8f9fa; color: #6c757d; border-top: 1px solid #dee2e6; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Risk Prediction Report</h1>
            <p>{repo_name}</p>
        </div>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value stat-total">{len(results_df)}</div>
                <div class="stat-label">Total Files</div>
            </div>
            <div class="stat-card">
                <div class="stat-value stat-critical">{critical_count}</div>
                <div class="stat-label">Critical</div>
                <div class="stat-range">90-100</div>
            </div>
            <div class="stat-card">
                <div class="stat-value stat-high">{high_count}</div>
                <div class="stat-label">High</div>
                <div class="stat-range">70-89</div>
            </div>
            <div class="stat-card">
                <div class="stat-value stat-medium">{medium_count}</div>
                <div class="stat-label">Medium</div>
                <div class="stat-range">50-69</div>
            </div>
            <div class="stat-card">
                <div class="stat-value stat-low">{low_count}</div>
                <div class="stat-label">Low</div>
                <div class="stat-range">0-49</div>
            </div>
        </div>
        <div class="content">
            <div class="search-box">
                <input type="text" id="searchInput" placeholder="🔍 Search files..." onkeyup="filterTable()">
            </div>
            <table id="riskTable">
                <thead>
                    <tr>
                        <th style="width: 80px;">Rank</th>
                        <th>File Path</th>
                        <th style="width: 150px;">Risk Level</th>
                        <th style="width: 120px;">Normalized Score</th>
                        <th style="width: 120px;">Raw Score</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        <div class="footer">
            <p style="margin-top: 20px;">
                Generated by <strong>Code Risk Guard v{metadata['version']}</strong>
            </p>
        </div>
    </div>
    <script>
        function filterTable() {{
            const input = document.getElementById('searchInput');
            const filter = input.value.toLowerCase();
            const table = document.getElementById('riskTable');
            const rows = table.getElementsByTagName('tr');
            for (let i = 1; i < rows.length; i++) {{
                const fileCell = rows[i].getElementsByClassName('file-path')[0];
                if (fileCell) {{
                    const txtValue = fileCell.textContent || fileCell.innerText;
                    rows[i].style.display = txtValue.toLowerCase().indexOf(filter) > -1 ? '' : 'none';
                }}
            }}
        }}
    </script>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html_content)

    logger.info(f"✓ Generated HTML report: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Standalone risk prediction with all dependencies embedded'
    )
    parser.add_argument('repo_path', help='Path to repository')
    parser.add_argument('--branch', default='main', help='Branch to analyze (default: main)')
    parser.add_argument('--output', help='Output HTML path')
    parser.add_argument('--mode', choices=['conservative', 'moderate', 'aggressive'],
                       default='moderate',
                       help='Post-processing mode (default: moderate)')
    parser.add_argument('--model', help='Path to model file (default: models/xgboost_model.pkl)')
    parser.add_argument('--metadata', help='Path to metadata file (default: models/xgboost_model_metadata.json)')

    args = parser.parse_args()

    repo_name = Path(args.repo_path).name
    if not args.output:
        args.output = f'risk_report_{repo_name}_standalone.html'

    try:
        # Set model paths
        if args.model:
            model_path = args.model
        else:
             model_path = str(Path(__file__).parent / 'models' / 'xgboost_model.pkl')

        if args.metadata:
            metadata_path = args.metadata
        else:
            metadata_path = str(Path(__file__).parent / 'models' / 'xgboost_model_metadata.json')

        # Load model
        model, metadata = load_model(str(model_path), str(metadata_path))

        # Extract features
        features_df = extract_features(args.repo_path, args.branch, metadata['feature_list'])

        # Run predictions WITH post-processing
        results_df = predict_risk(model, features_df, post_processing_mode=args.mode)

        # Generate HTML report
        html_path = generate_html_report(results_df, metadata, repo_name, args.output)

        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ PREDICTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"   📊 HTML Report: {html_path}")
        logger.info(f"   🔴 Top risk file: {results_df.iloc[0]['file']}")
        logger.info(f"   📈 Risk score: {results_df.iloc[0]['risk_score']:.4f}")
        logger.info(f"   📁 Total files: {len(results_df)}")
        logger.info("")
        logger.info(f"   To open: open {html_path}")

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
