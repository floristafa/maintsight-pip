"""Git commit data collection service for multiwindow_v2 model."""

import os
import re
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Set, Optional, Tuple
import subprocess
from git import Repo

from models import CommitData, FileStats
from utils.logger import Logger


class GitCommitCollector:
    """Collects commit data from local git repository (matches degradation model approach)."""
    
    # Source file extensions to analyze  
    SOURCE_EXTENSIONS = {
        '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp',
        '.cs', '.rb', '.go', '.rs', '.php', '.swift', '.kt', '.scala',
        '.r', '.m', '.jsx', '.tsx', '.vue', '.sol'
    }
    
    def __init__(
        self, 
        repo_path: str,
        branch: str = 'main',
        window_size_days: int = 150,
        only_existing_files: bool = True
    ):
        """Initialize git commit collector.
        
        Args:
            repo_path: Path to git repository
            branch: Git branch to analyze
            window_size_days: Time window in days for analysis
            only_existing_files: Only analyze files that currently exist
        """
        self.repo_path = Path(repo_path).resolve()
        self.branch = branch
        self.window_size_days = window_size_days
        self.only_existing_files = only_existing_files
        self.logger = Logger('GitCommitCollector')
        
        # Validate repository
        if not self.repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
            
        if not (self.repo_path / '.git').exists():
            raise ValueError(f"Not a git repository: {repo_path}")
            
        # Verify branch exists
        try:
            result = self._run_git_command(['branch', '-a'])
            if branch not in result:
                raise ValueError(f"Branch '{branch}' not found")
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Failed to verify branch: {e}")
            
        self.logger.info(f"Initialized repository: {self.repo_path}", '📁')
        self.logger.info(f"Using branch: {branch}", '🌿')
        self.logger.info(f"Window size: {window_size_days} days", '📅')
        
    def _run_git_command(self, args: List[str]) -> str:
        """Run a git command and return output.
        
        Args:
            args: Git command arguments
            
        Returns:
            Command output as string
        """
        cmd = ['git'] + args
        result = subprocess.run(
            cmd,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
        
    def _is_source_file(self, filepath: str) -> bool:
        """Check if file is a source code file to analyze.
        
        Args:
            filepath: Path to file
            
        Returns:
            True if file should be analyzed
        """
        ext = Path(filepath).suffix.lower()
        return ext in self.SOURCE_EXTENSIONS
        
    def _parse_rename_info(self, filepath: str) -> Optional[Tuple[str, Optional[str]]]:
        """Parse git rename information from file path.
        
        Args:
            filepath: Raw filepath from git log
            
        Returns:
            Tuple of (current_path, old_path) or None if invalid
        """
        if not filepath or filepath == '/dev/null' or '\0' in filepath:
            return None
            
        filepath = filepath.strip()
        old_path = None
        
        # Handle various git rename patterns
        
        # Pattern 1: "{old_dir => new_dir}/file.ext" (directory rename)
        dir_rename_match = re.match(r'(.*/)?{([^}]+)\s*=>\s*([^}]+)}(.*)$', filepath)
        if dir_rename_match:
            prefix, old_dir, new_dir, suffix = dir_rename_match.groups()
            old_path = (prefix or '') + old_dir.strip() + suffix
            filepath = (prefix or '') + new_dir.strip() + suffix
            
        # Pattern 2: "{old_file => new_file}" (file rename in braces)
        elif filepath.startswith('{') and filepath.endswith('}') and ' => ' in filepath:
            rename_match = re.match(r'^{(.+)\s*=>\s*(.+)}$', filepath)
            if rename_match:
                old_path = rename_match.group(1).strip()
                filepath = rename_match.group(2).strip()
                if filepath == '/dev/null':
                    return None
                    
        # Pattern 3: "old_path => new_path" (simple rename)
        elif ' => ' in filepath:
            parts = filepath.split(' => ')
            if len(parts) == 2:
                old_path = parts[0].strip()
                filepath = parts[1].strip()
                if filepath == '/dev/null':
                    return None
                    
        # Skip invalid paths
        if any(char in filepath for char in ['=>', '{', '}']):
            return None
            
        return filepath, old_path
        
    def fetch_commit_data(self, max_commits: int = 10000) -> List[CommitData]:
        """Fetch commit data using exact git command logic: git log ${branch} -n ${max} --numstat --find-renames --format="%H|%ae|%at|%s" --since="${since}" --no-merges
        
        Args:
            max_commits: Maximum number of commits to analyze
            
        Returns:
            List of CommitData objects with file-level aggregated commit data
        """
        self.logger.info(f"Fetching commits from {self.repo_path} (branch: {self.branch})", '🔄')
        self.logger.info(f"Max commits: {max_commits}", '📊')
        self.logger.info(f"Time window: last {self.window_size_days} days", '📅')
        
        # Calculate since timestamp (Unix timestamp)
        since_date = datetime.now() - timedelta(days=self.window_size_days)
        since_timestamp = str(int(since_date.timestamp()))
        
        # Execute exact git command: git log ${branch} -n ${max} --numstat --find-renames --format="%H|%ae|%at|%s" --since="${since}" --no-merges
        try:
            git_args = [
                'log',
                self.branch,
                f'-n{max_commits}',
                '--numstat',
                '--find-renames',
                '--format=%H|%ae|%at|%s',
                f'--since={since_timestamp}',
                '--no-merges'
            ]
            
            output = self._run_git_command(git_args)
            
            if not output.strip():
                self.logger.warn("No commits found in time window", '⚠️')
                return []
            
            # Parse git output
            file_stats: Dict[str, Dict] = {}
            current_commit = None
            
            for line in output.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Parse commit header: "hash|author_email|timestamp|subject"
                if '|' in line and len(line.split('|')) == 4:
                    hash_val, author_email, timestamp, subject = line.split('|', 3)
                    current_commit = {
                        'hash': hash_val,
                        'author': author_email,
                        'timestamp': int(timestamp),
                        'subject': subject.lower()
                    }
                    continue
                
                # Parse numstat line: "added_lines\tdeleted_lines\tfilepath"
                if current_commit and '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        added_str, deleted_str = parts[0], parts[1]
                        filepath = '\t'.join(parts[2:])  # Handle filenames with tabs
                        
                        # Handle binary files (marked as '-')
                        try:
                            lines_added = int(added_str) if added_str != '-' else 0
                            lines_deleted = int(deleted_str) if deleted_str != '-' else 0
                        except ValueError:
                            lines_added = lines_deleted = 0
                        
                        # Handle renames (git --find-renames shows "oldpath => newpath")
                        rename_info = self._parse_rename_info(filepath)
                        if rename_info is None:
                            continue
                        
                        filepath, old_path = rename_info
                        
                        # Only process source files
                        if not self._is_source_file(filepath):
                            continue
                        
                        # Only analyze existing files if requested
                        if self.only_existing_files:
                            full_path = self.repo_path / filepath
                            if not full_path.exists():
                                continue
                        
                        # Initialize file stats if not exists
                        if filepath not in file_stats:
                            commit_date = datetime.fromtimestamp(current_commit['timestamp'])
                            file_stats[filepath] = {
                                'lines_added': 0,
                                'lines_deleted': 0,
                                'commits': 0,
                                'authors': set(),
                                'bug_commits': 0,
                                'feature_commits': 0,
                                'refactor_commits': 0,
                                'first_commit': commit_date,
                                'last_commit': commit_date
                            }
                        
                        # Update file stats
                        stats = file_stats[filepath]
                        commit_date = datetime.fromtimestamp(current_commit['timestamp'])
                        
                        stats['lines_added'] += lines_added
                        stats['lines_deleted'] += lines_deleted
                        stats['commits'] += 1
                        stats['authors'].add(current_commit['author'])
                        
                        # Classify commit type based on subject
                        subject = current_commit['subject']
                        if any(kw in subject for kw in ['fix', 'bug', 'patch', 'hotfix', 'bugfix']):
                            stats['bug_commits'] += 1
                        if any(kw in subject for kw in ['feat', 'feature', 'add', 'implement']):
                            stats['feature_commits'] += 1
                        if any(kw in subject for kw in ['refactor', 'clean', 'improve']):
                            stats['refactor_commits'] += 1
                        
                        stats['first_commit'] = min(stats['first_commit'], commit_date)
                        stats['last_commit'] = max(stats['last_commit'], commit_date)
            
            if not file_stats:
                self.logger.warn("No source files found in commits", '⚠️')
                return []
            
            # Convert to CommitData objects
            commit_data_objects = []
            
            for filepath, stats in file_stats.items():
                days_active = max((stats['last_commit'] - stats['first_commit']).days, 1)
                num_authors = len(stats['authors'])
                num_commits = stats['commits']
                
                # Calculate derived features
                lines_per_author = stats['lines_added'] / num_authors if num_authors > 0 else 0
                churn_per_commit = (stats['lines_added'] + stats['lines_deleted']) / num_commits if num_commits > 0 else 0
                bug_ratio = stats['bug_commits'] / num_commits if num_commits > 0 else 0
                commits_per_day = num_commits / days_active
                
                commit_data = CommitData(
                    module=filepath,
                    filename=Path(filepath).name,
                    repo_name=self.repo_path.name,
                    commits=num_commits,
                    authors=num_authors,
                    author_names=list(stats['authors']),
                    lines_added=stats['lines_added'],
                    lines_deleted=stats['lines_deleted'],
                    churn=stats['lines_added'] + stats['lines_deleted'],
                    bug_commits=stats['bug_commits'],
                    refactor_commits=stats['refactor_commits'],
                    feature_commits=stats['feature_commits'],
                    lines_per_author=lines_per_author,
                    churn_per_commit=churn_per_commit,
                    bug_ratio=bug_ratio,
                    days_active=days_active,
                    commits_per_day=commits_per_day,
                    created_at=stats['first_commit'],
                    last_modified=stats['last_commit']
                )
                
                commit_data_objects.append(commit_data)
            
            total_commits = sum(stats['commits'] for stats in file_stats.values()) if file_stats else 0
            self.logger.success(f"Fetched data for {len(commit_data_objects)} files from {total_commits} total file commits", '✅')
            return commit_data_objects
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch commit data: {e}")
        
