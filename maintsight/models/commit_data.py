"""Commit data model for git repository analysis."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class CommitData:
    """Data structure containing commit metrics and engineered features for a file."""
    
    # Core identification
    module: str
    filename: str
    repo_name: str
    
    # Base git features
    commits: int
    authors: int
    author_names: Optional[List[str]] = None
    lines_added: int = 0
    lines_deleted: int = 0
    churn: int = 0
    bug_commits: int = 0
    refactor_commits: int = 0
    feature_commits: int = 0
    
    # Derived base features
    lines_per_author: float = 0.0
    churn_per_commit: float = 0.0
    bug_ratio: float = 0.0
    days_active: int = 1
    commits_per_day: float = 0.0
    
    # Timestamp fields
    created_at: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    
    # Engineered features (added by FeatureEngineer)
    degradation_days: Optional[int] = None
    net_lines: Optional[int] = None
    code_stability: Optional[float] = None
    is_high_churn_commit: Optional[int] = None
    bug_commit_rate: Optional[float] = None
    commits_squared: Optional[int] = None
    author_concentration: Optional[float] = None
    lines_per_commit: Optional[float] = None
    churn_rate: Optional[float] = None
    modification_ratio: Optional[float] = None
    churn_per_author: Optional[float] = None
    deletion_rate: Optional[float] = None
    commit_density: Optional[float] = None
    
    def __post_init__(self):
        """Post-initialization to set computed defaults."""
        if self.author_names is None:
            self.author_names = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_modified is None:
            self.last_modified = datetime.now()
            
        # Ensure churn is computed if not provided
        if self.churn == 0:
            self.churn = self.lines_added + self.lines_deleted
    
    def to_feature_vector(self) -> List[float]:
        """Convert to feature vector for ML model prediction.
        
        Returns:
            List of 26 features in the order expected by the XGBoost model.
        """
        return [
            float(self.commits),
            float(self.authors), 
            float(self.lines_added),
            float(self.lines_deleted),
            float(self.churn),
            float(self.bug_commits),
            float(self.refactor_commits), 
            float(self.feature_commits),
            float(self.lines_per_author),
            float(self.churn_per_commit),
            float(self.bug_ratio),
            float(self.days_active),
            float(self.commits_per_day),
            float(self.degradation_days or 0),
            float(self.net_lines or 0),
            float(self.code_stability or 0),
            float(self.is_high_churn_commit or 0),
            float(self.bug_commit_rate or 0),
            float(self.commits_squared or 0),
            float(self.author_concentration or 0),
            float(self.lines_per_commit or 0),
            float(self.churn_rate or 0),
            float(self.modification_ratio or 0),
            float(self.churn_per_author or 0),
            float(self.deletion_rate or 0),
            float(self.commit_density or 0),
        ]
    
    @classmethod
    def feature_names(cls) -> List[str]:
        """Get feature names in the order expected by the model.
        
        Returns:
            List of 26 feature names matching the order in to_feature_vector().
        """
        return [
            'commits',
            'authors',
            'lines_added', 
            'lines_deleted',
            'churn',
            'bug_commits',
            'refactor_commits',
            'feature_commits',
            'lines_per_author',
            'churn_per_commit',
            'bug_ratio',
            'days_active', 
            'commits_per_day',
            'degradation_days',
            'net_lines',
            'code_stability',
            'is_high_churn_commit',
            'bug_commit_rate',
            'commits_squared',
            'author_concentration',
            'lines_per_commit',
            'churn_rate',
            'modification_ratio',
            'churn_per_author',
            'deletion_rate',
            'commit_density',
        ]