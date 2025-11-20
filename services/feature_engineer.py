"""Feature engineering service for pandas DataFrames (matches train.py - 12 engineered features)."""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """Transforms raw commit data into ML features (matches train.py - 12 engineered features)."""
    
    def __init__(self):
        """Initialize feature engineer."""
        pass
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features used by the model (match training exactly).
        
        Args:
            df: DataFrame with base commit features
            
        Returns:
            DataFrame with all base + engineered features
        """
        df = df.copy()
        
        # 1. net_lines - Code growth
        if 'lines_added' in df.columns and 'lines_deleted' in df.columns:
            df['net_lines'] = df['lines_added'] - df['lines_deleted']
        
        # 2. code_stability - Churn relative to additions
        if 'lines_added' in df.columns and 'churn' in df.columns:
            df['code_stability'] = df['churn'] / (df['lines_added'] + 1)
        
        # 3. is_high_churn_commit - Binary flag for large changes
        if 'churn_per_commit' in df.columns:
            df['is_high_churn_commit'] = (df['churn_per_commit'] > 100).astype(int)
        
        # 4. bug_commit_rate - Proportion of bug commits
        if 'bug_commits' in df.columns and 'commits' in df.columns:
            df['bug_commit_rate'] = np.where(df['commits'] > 0, df['bug_commits'] / df['commits'], 0)
        
        # 5. commits_squared - Non-linear commit activity
        if 'commits' in df.columns:
            df['commits_squared'] = df['commits'] ** 2
        
        # 6. author_concentration - Bus factor
        if 'authors' in df.columns:
            df['author_concentration'] = 1.0 / (df['authors'] + 1)
        
        # 7. lines_per_commit - Average code change size
        if 'lines_added' in df.columns and 'commits' in df.columns:
            df['lines_per_commit'] = df['lines_added'] / (df['commits'] + 1)
        
        # 8. churn_rate - Churn velocity
        if 'churn' in df.columns and 'days_active' in df.columns:
            df['churn_rate'] = df['churn'] / (df['days_active'] + 1)
        
        # 9. modification_ratio - Deletion relative to addition
        if 'lines_added' in df.columns and 'lines_deleted' in df.columns:
            df['modification_ratio'] = df['lines_deleted'] / (df['lines_added'] + 1)
        
        # 10. churn_per_author - Code change per developer
        if 'churn' in df.columns and 'authors' in df.columns:
            df['churn_per_author'] = df['churn'] / (df['authors'] + 1)
        
        # 11. deletion_rate - Code removal rate
        if 'lines_deleted' in df.columns and 'lines_added' in df.columns:
            df['deletion_rate'] = df['lines_deleted'] / (df['lines_added'] + df['lines_deleted'] + 1)
        
        # 12. commit_density - Commit frequency
        if 'commits' in df.columns and 'days_active' in df.columns:
            df['commit_density'] = df['commits'] / (df['days_active'] + 1)
        
        return df
        
    def get_selected_features(self):
        """Get the 14 features selected for the multiwindow_v2 model.
        
        Returns:
            List of 14 selected feature names
        """
        return [
            "days_active", "net_lines", "bug_ratio", "commits_per_day", "commits", "commits_squared",
            "code_stability", "modification_ratio", "commit_density", "bug_commit_rate", "bug_commits",
            "lines_per_commit", "lines_deleted", "author_concentration"
        ]