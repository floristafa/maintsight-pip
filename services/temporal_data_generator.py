"""Temporal data generation service for time-based analysis."""

import logging
from datetime import datetime
from typing import List
from git import Repo

logger = logging.getLogger(__name__)


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
                if hasattr(item, 'type') and item.type == 'blob':  # type: ignore
                    file_path = str(item.path)  # type: ignore
                    if any(file_path.endswith(ext) for ext in source_extensions):
                        files.append(file_path)

            return files

        except Exception as e:
            logger.warning(f"Error getting files at {time_T.date()}: {e}")
            return []