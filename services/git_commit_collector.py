"""Git commit data collection service for multiwindow_v2 model."""

import os
import re
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Set, Optional, Tuple
import subprocess
import pandas as pd
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
        
    def fetch_commit_data(self, max_commits: int = 10000) -> pd.DataFrame:
        """Fetch commit data and aggregate by file (matches temporal_git_collector approach).
        
        Args:
            max_commits: Maximum number of commits to analyze
            
        Returns:
            DataFrame with file-level aggregated commit data
        """
        self.logger.info(f"Fetching commits from {self.repo_path} (branch: {self.branch})", '🔄')
        self.logger.info(f"Max commits: {max_commits}", '📊')
        self.logger.info(f"Time window: last {self.window_size_days} days", '📅')
        
        # Calculate since date
        since_date = datetime.now() - timedelta(days=self.window_size_days)
        
        # Use GitPython for better compatibility
        try:
            repo = Repo(self.repo_path)
            commits = list(repo.iter_commits(
                self.branch,
                max_count=max_commits,
                since=since_date
            ))
            self.logger.info(f"Found {len(commits)} commits in time window", '📊')
            
            file_stats: Dict[str, Dict] = {}
            
            for commit in commits:
                commit_date = commit.committed_datetime
                if commit_date.tzinfo is not None:
                    commit_date = commit_date.astimezone(timezone.utc).replace(tzinfo=None)
                
                author = str(commit.author.email)
                message = str(commit.message).lower()
                is_bug_fix = any(kw in message for kw in ['fix', 'bug', 'patch', 'hotfix', 'bugfix'])
                is_feature = any(kw in message for kw in ['feat', 'feature', 'add', 'implement'])
                is_refactor = any(kw in message for kw in ['refactor', 'clean', 'improve'])
                
                if commit.parents:
                    parent = commit.parents[0]
                    diffs = parent.diff(commit, create_patch=True)
                    
                    for diff in diffs:
                        filepath = diff.b_path or diff.a_path
                        if not filepath or not self._is_source_file(filepath):
                            continue
                        
                        if self.only_existing_files:
                            full_path = self.repo_path / filepath
                            if not full_path.exists():
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
                                'first_commit': commit_date,
                                'last_commit': commit_date
                            }
                        
                        stats = file_stats[filepath]
                        
                        if diff.diff:
                            if isinstance(diff.diff, bytes):
                                diff_text = diff.diff.decode('utf-8', errors='ignore')
                            else:
                                diff_text = str(diff.diff)
                            lines_added = sum(1 for line in diff_text.split('\n')
                                             if line.startswith('+') and not line.startswith('+++'))
                            lines_deleted = sum(1 for line in diff_text.split('\n')
                                               if line.startswith('-') and not line.startswith('---'))
                            stats['lines_added'] += lines_added
                            stats['lines_deleted'] += lines_deleted
                        
                        stats['commits'] += 1
                        stats['authors'].add(author)
                        if is_bug_fix:
                            stats['bug_commits'] += 1
                        if is_feature:
                            stats['feature_commits'] += 1
                        if is_refactor:
                            stats['refactor_commits'] += 1
                        
                        stats['first_commit'] = min(stats['first_commit'], commit_date)
                        stats['last_commit'] = max(stats['last_commit'], commit_date)
                else:
                    # Handle initial commit (no parents) - treat all files as added
                    for item in commit.tree.traverse():
                        if item.type == 'blob':  # It's a file
                            filepath = item.path
                            if not self._is_source_file(filepath):
                                continue
                            
                            if self.only_existing_files:
                                full_path = self.repo_path / filepath
                                if not full_path.exists():
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
                                    'first_commit': commit_date,
                                    'last_commit': commit_date
                                }
                            
                            stats = file_stats[filepath]
                            # Estimate lines added for initial commit (can't get exact diff)
                            try:
                                lines_added = len(item.data_stream.read().decode('utf-8', errors='ignore').splitlines())
                                stats['lines_added'] += lines_added
                            except:
                                stats['lines_added'] += 10  # Default estimate
                            
                            stats['commits'] += 1
                            stats['authors'].add(author)
                            if is_bug_fix:
                                stats['bug_commits'] += 1
                            if is_feature:
                                stats['feature_commits'] += 1
                            if is_refactor:
                                stats['refactor_commits'] += 1
                            
                            stats['first_commit'] = min(stats['first_commit'], commit_date)
                            stats['last_commit'] = max(stats['last_commit'], commit_date)
            
            if not file_stats:
                self.logger.warn("No source files found in commits", '⚠️')
                return pd.DataFrame()
            
            # Convert to DataFrame records
            records = []
            
            for filepath, stats in file_stats.items():
                days_active = max((stats['last_commit'] - stats['first_commit']).days, 1)
                num_authors = len(stats['authors'])
                num_commits = stats['commits']
                
                # Calculate base features (matching git_commit_client.py exactly)
                # Base features: 13 total
                # commits, authors, lines_added, lines_deleted, churn
                # bug_commits, refactor_commits, feature_commits
                # lines_per_author, churn_per_commit, bug_ratio, days_active, commits_per_day
                records.append({
                    'module': filepath,
                    'commits': num_commits,
                    'authors': num_authors,
                    'author_names': list(stats['authors']),  # Include actual author names
                    'lines_added': stats['lines_added'],
                    'lines_deleted': stats['lines_deleted'],
                    'churn': stats['lines_added'] + stats['lines_deleted'],
                    'bug_commits': stats['bug_commits'],
                    'refactor_commits': stats['refactor_commits'],
                    'feature_commits': stats['feature_commits'],
                    'lines_per_author': stats['lines_added'] / num_authors if num_authors > 0 else 0,
                    'churn_per_commit': (stats['lines_added'] + stats['lines_deleted']) / num_commits if num_commits > 0 else 0,
                    'bug_ratio': stats['bug_commits'] / num_commits if num_commits > 0 else 0,
                    'days_active': days_active,
                    'commits_per_day': num_commits / days_active
                })
            
            df = pd.DataFrame(records)
            self.logger.success(f"Fetched data for {len(df)} files from {len(commits)} commits", '✅')
            return df
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch commit data: {e}")
        
