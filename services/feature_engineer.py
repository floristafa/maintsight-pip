"""Feature engineering service for CommitData objects (matches train.py - 12 engineered features)."""

from typing import List
from models import CommitData


class FeatureEngineer:
    """Transforms raw commit data into ML features (matches train.py - 12 engineered features)."""
    
    def __init__(self):
        """Initialize feature engineer."""
        pass
        
    def transform(self, commit_data_list: List[CommitData]) -> List[CommitData]:
        """
        Generate all features used by the model (match training exactly).
        
        Args:
            commit_data_list: List of CommitData objects with base commit features
            
        Returns:
            List of CommitData objects with all base + engineered features
        """
        # Process each CommitData object and add engineered features
        for commit_data in commit_data_list:
            # 1. degradation_days - Days since creation (temporal feature)
            if commit_data.created_at and commit_data.last_modified:
                commit_data.degradation_days = (commit_data.last_modified - commit_data.created_at).days
            
            # 2. net_lines - Code growth
            commit_data.net_lines = commit_data.lines_added - commit_data.lines_deleted
            
            # 3. code_stability - Churn relative to additions
            commit_data.code_stability = commit_data.churn / (commit_data.lines_added + 1)
            
            # 4. is_high_churn_commit - Binary flag for large changes
            commit_data.is_high_churn_commit = 1 if commit_data.churn_per_commit > 100 else 0
            
            # 5. bug_commit_rate - Proportion of bug commits
            commit_data.bug_commit_rate = commit_data.bug_commits / commit_data.commits if commit_data.commits > 0 else 0
            
            # 6. commits_squared - Non-linear commit activity
            commit_data.commits_squared = commit_data.commits ** 2
            
            # 7. author_concentration - Bus factor
            commit_data.author_concentration = 1.0 / (commit_data.authors + 1)
            
            # 8. lines_per_commit - Average code change size
            commit_data.lines_per_commit = commit_data.lines_added / (commit_data.commits + 1)
            
            # 9. churn_rate - Churn velocity
            commit_data.churn_rate = commit_data.churn / (commit_data.days_active + 1)
            
            # 10. modification_ratio - Deletion relative to addition
            commit_data.modification_ratio = commit_data.lines_deleted / (commit_data.lines_added + 1)
            
            # 11. churn_per_author - Code change per developer
            commit_data.churn_per_author = commit_data.churn / (commit_data.authors + 1)
            
            # 12. deletion_rate - Code removal rate
            commit_data.deletion_rate = commit_data.lines_deleted / (commit_data.lines_added + commit_data.lines_deleted + 1)
            
            # 13. commit_density - Commit frequency
            commit_data.commit_density = commit_data.commits / (commit_data.days_active + 1)
        
        return commit_data_list
        
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