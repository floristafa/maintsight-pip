"""Enhanced git feature extractor service."""

import os
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from git import Repo

logger = logging.getLogger(__name__)


class GitFeatureExtractor:
    """Enhanced feature extractor with temporal ratios and context awareness."""

    def __init__(self, repo_path: str, branch: str = "main"):
        self.repo_path = repo_path
        self.repo = Repo(repo_path)
        self.branch = branch
        logger.info(f"Initialized GitFeatureExtractor for {repo_path}")

    def extract_features_at_time(
        self,
        repo_path: str,
        branch: str,
        file_paths: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Extract enhanced features for specific files at a historical time point."""
        logger.info(f"   Extracting enhanced features for {len(file_paths)} files at time {end_date.date()}")

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
            empty_df = pd.DataFrame(index=file_paths)
            return empty_df

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