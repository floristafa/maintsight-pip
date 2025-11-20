"""File statistics data model."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Set, Optional


@dataclass
class FileStats:
    """Statistics for a single file collected from git history."""
    
    lines_added: int = 0
    lines_deleted: int = 0
    commits: int = 0
    authors: Set[str] = field(default_factory=set)
    bug_commits: int = 0
    feature_commits: int = 0
    refactor_commits: int = 0
    first_commit: Optional[datetime] = None
    last_commit: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize optional datetime fields."""
        if self.first_commit is None:
            self.first_commit = datetime.now()
        if self.last_commit is None:
            self.last_commit = datetime.now()
            
    @property
    def churn(self) -> int:
        """Total code churn (lines added + deleted)."""
        return self.lines_added + self.lines_deleted
        
    @property
    def num_authors(self) -> int:
        """Number of unique authors."""
        return len(self.authors)
        
    @property
    def days_active(self) -> int:
        """Number of days between first and last commit."""
        if self.first_commit and self.last_commit:
            delta = self.last_commit - self.first_commit
            return max(delta.days, 1)
        return 1