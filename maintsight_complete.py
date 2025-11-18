#!/usr/bin/env python3
"""
Complete MaintSight CLI with HTML generation matching the TypeScript version.
This includes the exact same HTML report as the original.
"""

import sys
import os
import subprocess
import json
import re
import math
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Set, Optional, List, Any, Dict
from enum import Enum

# Simple logger without rich
class SimpleLogger:
    def __init__(self, name):
        self.name = name
    def info(self, msg, emoji=None):
        print(f"{emoji or 'ℹ️'} {msg}")
    def warn(self, msg, emoji=None):
        print(f"{emoji or '⚠️'} {msg}")
    def error(self, msg, emoji=None):
        print(f"{emoji or '❌'} {msg}")
    def success(self, msg, emoji=None):
        print(f"{emoji or '✅'} {msg}")

# Risk categories
class RiskCategory(Enum):
    SEVERELY_DEGRADED = "severely_degraded"
    DEGRADED = "degraded"
    STABLE = "stable"
    IMPROVED = "improved"
    
    @classmethod
    def from_score(cls, score: float):
        if score < 0.0:
            return cls.IMPROVED
        elif score <= 0.1:
            return cls.STABLE
        elif score <= 0.2:
            return cls.DEGRADED
        else:
            return cls.SEVERELY_DEGRADED

# Data models
@dataclass
class FileStats:
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
        if self.first_commit is None:
            self.first_commit = datetime.now()
        if self.last_commit is None:
            self.last_commit = datetime.now()
            
    @property
    def num_authors(self) -> int:
        return len(self.authors)

@dataclass
class CommitData:
    module: str
    filename: str
    repo_name: str
    commits: int
    authors: int
    author_names: Optional[List[str]] = None
    lines_added: int = 0
    lines_deleted: int = 0
    churn: int = 0
    bug_commits: int = 0
    refactor_commits: int = 0
    feature_commits: int = 0
    lines_per_author: float = 0.0
    churn_per_commit: float = 0.0
    bug_ratio: float = 0.0
    days_active: int = 1
    commits_per_day: float = 0.0
    created_at: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    
    def __post_init__(self):
        if self.author_names is None:
            self.author_names = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_modified is None:
            self.last_modified = datetime.now()
        if self.churn == 0:
            self.churn = self.lines_added + self.lines_deleted

@dataclass 
class RiskPrediction:
    module: str
    risk_category: RiskCategory
    degradation_score: float
    raw_prediction: float

# HTML generation classes (ported from TypeScript)
class FileTreeNode:
    def __init__(self, name: str, node_type: str, children: Optional[List['FileTreeNode']] = None,
                 prediction: Optional[Any] = None, path: Optional[str] = None):
        self.name = name
        self.type = node_type  # 'file' or 'folder'
        self.children = children or []
        self.prediction = prediction
        self.path = path

class CommitStats:
    def __init__(self, total_commits: int = 0, total_bug_fixes: int = 0,
                 avg_commits_per_file: float = 0.0, avg_authors_per_file: float = 0.0,
                 author_names: Optional[List[str]] = None):
        self.total_commits = total_commits
        self.total_bug_fixes = total_bug_fixes
        self.avg_commits_per_file = avg_commits_per_file
        self.avg_authors_per_file = avg_authors_per_file
        self.author_names = author_names or []

# Git collector (simplified version)
class GitCommitCollector:
    SOURCE_EXTENSIONS = {
        '.js', '.ts', '.jsx', '.tsx', '.mjs', '.cjs', '.vue', '.svelte',
        '.py', '.pyx', '.pyi', '.pyw',
        '.java', '.kt', '.kts', '.scala', '.groovy', '.gradle',
        '.c', '.cpp', '.cxx', '.cc', '.c++', '.h', '.hpp', '.hxx', '.hh', '.h++',
        '.cs', '.vb', '.fs', '.fsx', '.fsi',
        '.swift', '.m', '.mm', '.dart',
        '.php', '.rb', '.perl', '.pl', '.pm',
        '.go', '.rs', '.zig', '.nim', '.d',
        '.hs', '.lhs', '.elm', '.ml', '.mli', '.clj', '.cljs', '.cljc',
        '.r', '.R', '.jl', '.ipynb',
        '.sql', '.mysql', '.pgsql', '.plsql', '.tsql', '.ddl', '.dml',
        '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',
        '.tf', '.hcl', '.yaml', '.yml', '.toml',
        '.sol', '.cairo', '.move', '.vy',
        '.lua', '.crystal', '.ex', '.exs', '.erl', '.hrl',
    }
    
    def __init__(self, repo_path, branch='main', window_size_days=150):
        self.repo_path = Path(repo_path).resolve()
        self.branch = branch
        self.window_size_days = window_size_days
        self.logger = SimpleLogger('GitCommitCollector')
        
        if not self.repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
        if not (self.repo_path / '.git').exists():
            raise ValueError(f"Not a git repository: {repo_path}")
            
        self.logger.info(f"Initialized repository: {self.repo_path}", '📁')
        self.logger.info(f"Using branch: {branch}", '🌿')
        
    def _run_git_command(self, args):
        cmd = ['git'] + args
        result = subprocess.run(
            cmd,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
        
    def _is_source_file(self, filepath):
        ext = Path(filepath).suffix.lower()
        return ext in self.SOURCE_EXTENSIONS
        
    def fetch_commit_data(self, max_commits=10000):
        self.logger.info(f"Fetching commits (max: {max_commits})", '🔄')
        
        # Calculate since date
        since_date = datetime.now() - timedelta(days=self.window_size_days)
        since_timestamp = int(since_date.timestamp())
        
        # Build git log command
        git_args = [
            'log', self.branch,
            f'-n{max_commits}',
            '--numstat',
            '--find-renames', 
            '--format=%H|%ae|%at|%s',
            f'--since={since_timestamp}',
            '--no-merges'
        ]
        
        try:
            output = self._run_git_command(git_args)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to fetch git log: {e}")
            
        # Parse git log output
        file_stats = {}
        lines = [line.strip() for line in output.split('\n') if line.strip()]
        
        current_author = ''
        current_date = datetime.now()
        is_bug_fix = False
        is_feature = False
        is_refactor = False
        
        for line in lines:
            if '|' in line and not line.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                # Commit header line
                parts = line.split('|', 3)
                if len(parts) >= 4:
                    _, author, timestamp, message = parts
                    current_author = author
                    current_date = datetime.fromtimestamp(int(timestamp))
                    
                    # Classify commit type
                    message_lower = message.lower()
                    is_bug_fix = any(kw in message_lower for kw in ['fix', 'bug', 'patch', 'hotfix', 'bugfix'])
                    is_feature = any(kw in message_lower for kw in ['feat', 'feature', 'add', 'implement'])
                    is_refactor = any(kw in message_lower for kw in ['refactor', 'clean', 'improve'])
                    
            elif re.match(r'^\d+\s+\d+\s+', line):
                # File stat line
                parts = line.split('\t')
                if len(parts) >= 3:
                    added_str, removed_str, filepath = parts[0], parts[1], parts[2]
                    
                    # Parse line changes
                    try:
                        added = int(added_str) if added_str != '-' else 0
                        removed = int(removed_str) if removed_str != '-' else 0
                    except ValueError:
                        continue
                        
                    # Skip non-source files
                    if not self._is_source_file(filepath):
                        continue
                        
                    # Filter existing files
                    full_path = self.repo_path / filepath
                    if not full_path.exists():
                        continue
                        
                    # Update file stats
                    if filepath not in file_stats:
                        file_stats[filepath] = FileStats(
                            first_commit=current_date,
                            last_commit=current_date,
                        )
                        
                    stats = file_stats[filepath]
                    stats.lines_added += added
                    stats.lines_deleted += removed
                    stats.commits += 1
                    stats.authors.add(current_author)
                    
                    if is_bug_fix:
                        stats.bug_commits += 1
                    if is_feature:
                        stats.feature_commits += 1
                    if is_refactor:
                        stats.refactor_commits += 1
                        
                    if current_date < stats.first_commit:
                        stats.first_commit = current_date
                    if current_date > stats.last_commit:
                        stats.last_commit = current_date
                    
        if not file_stats:
            self.logger.warn('No source files found in commits', '⚠️')
            return []
            
        # Convert to CommitData objects
        results = []
        repo_name = self.repo_path.name
        
        for filepath, stats in file_stats.items():
            days_active = max((stats.last_commit - stats.first_commit).days, 1)
            churn = stats.lines_added + stats.lines_deleted
            
            commit_data = CommitData(
                module=filepath,
                filename=filepath,
                repo_name=repo_name,
                commits=stats.commits,
                authors=stats.num_authors,
                author_names=list(stats.authors),
                lines_added=stats.lines_added,
                lines_deleted=stats.lines_deleted,
                churn=churn,
                bug_commits=stats.bug_commits,
                refactor_commits=stats.refactor_commits,
                feature_commits=stats.feature_commits,
                lines_per_author=stats.lines_added / max(stats.num_authors, 1),
                churn_per_commit=churn / max(stats.commits, 1),
                bug_ratio=stats.bug_commits / max(stats.commits, 1),
                days_active=days_active,
                commits_per_day=stats.commits / days_active,
                created_at=stats.first_commit,
                last_modified=stats.last_commit,
            )
            
            results.append(commit_data)
            
        self.logger.success(f"Collected data for {len(results)} files", '✅')
        return results

# Simple predictor that creates mock risk predictions
class MockPredictor:
    """Mock predictor for demonstration - creates risk predictions based on commit patterns."""
    
    def predict(self, commit_data: List[CommitData]) -> List[RiskPrediction]:
        """Create mock risk predictions based on commit data patterns."""
        predictions = []
        
        for data in commit_data:
            # Simple heuristic: higher churn and bug commits = higher risk
            # This is a simplified version - the real ML model would be more sophisticated
            
            # Calculate a mock degradation score based on patterns
            churn_factor = min(data.churn / 1000.0, 0.3)  # Normalize churn
            bug_factor = data.bug_ratio * 0.2  # Bug ratio contribution
            author_factor = 1.0 / max(data.authors, 1) * 0.1  # More authors = less risk
            
            # Combine factors with some randomness for demonstration
            base_score = churn_factor + bug_factor + author_factor
            
            # Add some variation based on file type
            if data.module.endswith(('.spec.ts', '.test.ts', '.spec.js', '.test.js')):
                base_score *= 0.5  # Test files typically have higher churn but lower risk
            
            # Normalize to typical range (-0.2 to +0.4)
            degradation_score = (base_score - 0.15) * 2.0
            
            # Clamp to reasonable range
            degradation_score = max(-0.3, min(0.5, degradation_score))
            
            # Determine risk category
            risk_category = RiskCategory.from_score(degradation_score)
            
            prediction = RiskPrediction(
                module=data.module,
                risk_category=risk_category,
                degradation_score=degradation_score,
                raw_prediction=degradation_score
            )
            
            predictions.append(prediction)
            
        return predictions

# HTML Generation Functions (exact port from TypeScript)
def build_file_tree(predictions: List[RiskPrediction]) -> FileTreeNode:
    """Build hierarchical file tree structure from predictions."""
    root = FileTreeNode(name='root', node_type='folder', children=[])
    
    for prediction in predictions:
        path_parts = prediction.module.split('/')
        current_node = root
        
        for index, part in enumerate(path_parts):
            is_file = index == len(path_parts) - 1
            
            # Find existing child
            existing_child = None
            for child in current_node.children:
                if child.name == part:
                    existing_child = child
                    break
                    
            if existing_child:
                current_node = existing_child
            else:
                new_node = FileTreeNode(
                    name=part,
                    node_type='file' if is_file else 'folder',
                    path='/'.join(path_parts[:index + 1]),
                    children=[] if not is_file else None,
                    prediction=prediction if is_file else None
                )
                
                current_node.children.append(new_node)
                current_node = new_node
                
    return root

def calculate_folder_stats(node: FileTreeNode) -> Dict[str, Any]:
    """Calculate average score and category for a folder."""
    total_score = 0.0
    file_count = 0
    
    def traverse(n: FileTreeNode):
        nonlocal total_score, file_count
        
        if not n.children or len(n.children) == 0:
            # This is a file
            if n.prediction:
                total_score += n.prediction.degradation_score
                file_count += 1
        else:
            # This is a folder, traverse children
            for child in n.children:
                traverse(child)
                
    traverse(node)
    
    avg_score = total_score / file_count if file_count > 0 else 0.0
    
    # Determine category based on average score
    if avg_score > 0.2:
        category = 'severely-degraded'
    elif avg_score > 0.1:
        category = 'degraded'
    elif avg_score > 0.0:
        category = 'stable'
    else:
        category = 'improved'
        
    return {
        'avgScore': avg_score,
        'fileCount': file_count,
        'category': category
    }

def generate_tree_html(node: FileTreeNode, depth: int = 0) -> str:
    """Generate HTML for file tree structure."""
    if not node.children or len(node.children) == 0:
        # This is a file
        if node.prediction:
            score = node.prediction.degradation_score
            category = node.prediction.risk_category.value
            category_class = category.replace('_', '-')
            indent_style = f'style="margin-left: {depth * 20}px;"'
            
            return f'''
        <div class="tree-file {category_class}" {indent_style}>
          <div class="file-name">{node.name}</div>
          <div class="file-score">{score:.4f}</div>
          <div class="risk-badge {category_class}">{category.replace('_', ' ')}</div>
        </div>
      '''
        return ''
        
    # This is a folder
    sorted_children = sorted(node.children, key=lambda x: (x.type != 'folder', x.name))
    
    if node.name == 'root':
        # Don't render the root node itself
        children_html = ''.join([generate_tree_html(child, depth) for child in sorted_children])
        return f'<div class="file-tree-container">{children_html}</div>'
        
    children_html = ''.join([generate_tree_html(child, depth + 1) for child in sorted_children])
    
    # Calculate folder statistics
    stats = calculate_folder_stats(node)
    folder_class = stats['category']
    indent_style = f'style="margin-left: {depth * 20}px;"'
    
    return f'''
    <div class="tree-node" data-depth="{depth}">
      <div class="tree-folder {folder_class}" onclick="toggleFolder(this)" {indent_style}>
        <span class="folder-toggle">▶</span>
        <span class="folder-icon">📁</span>
        <span class="folder-name">{node.name}</span>
        <span class="folder-stats">
          <span class="folder-count">{stats['fileCount']} files</span>
          <span class="folder-score">{stats['avgScore']:.3f}</span>
          <span class="risk-badge {folder_class}">{stats['category'].replace('-', ' ')}</span>
        </span>
      </div>
      <div class="collapsible">
        {children_html}
      </div>
    </div>
  '''

def calculate_commit_stats(commit_data: List[CommitData]) -> CommitStats:
    """Calculate commit statistics from commit data."""
    if len(commit_data) == 0:
        return CommitStats()
        
    total_commits = sum(d.commits for d in commit_data)
    total_bug_fixes = sum(d.bug_commits for d in commit_data)
    
    # Extract unique authors from author_names field
    unique_authors = set()
    for d in commit_data:
        if d.author_names:
            for author in d.author_names:
                unique_authors.add(author)
                
    author_names = list(unique_authors)
    avg_authors_per_file = sum(d.authors for d in commit_data) / len(commit_data)
    
    return CommitStats(
        total_commits=total_commits,
        total_bug_fixes=total_bug_fixes,
        avg_commits_per_file=total_commits / len(commit_data),
        avg_authors_per_file=avg_authors_per_file,
        author_names=author_names
    )

def generate_html_report(predictions: List[RiskPrediction], commit_data: List[CommitData], repo_path: str) -> Optional[str]:
    """Generate HTML report and save to .maintsight directory."""
    try:
        # Get repository name
        repo_name = Path(repo_path).name
        
        # Create timestamp for filename  
        timestamp = datetime.now().isoformat().replace(':', '-').replace('.', '-')[:19]
        
        # Create .maintsight directory inside the repo
        maintsight_dir = Path(repo_path) / '.maintsight'
        maintsight_dir.mkdir(exist_ok=True)
        
        # Create HTML filename with repo name and date
        html_filename = f"{repo_name}-{timestamp}.html"
        html_path = maintsight_dir / html_filename
        
        # Generate HTML content
        html_content = format_as_html(predictions, commit_data, repo_path)
        
        # Save HTML file
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"📄 HTML report saved to: {html_path}")
        
        return str(html_path)
        
    except Exception as error:
        print(f"⚠️ Warning: Could not save HTML report: {error}")
        return None

def format_as_html(predictions: List[RiskPrediction], commit_data: List[CommitData], repo_path: str) -> str:
    """Format predictions and commit data as HTML report - EXACT port from TypeScript version."""
    repo_name = Path(repo_path).name
    timestamp = datetime.now().isoformat()
    
    # Calculate statistics
    total_files = len(predictions)
    mean_score = sum(p.degradation_score for p in predictions) / total_files
    std_dev = math.sqrt(
        sum((p.degradation_score - mean_score) ** 2 for p in predictions) / total_files
    )
    
    # Calculate risk distribution
    risk_distribution = {}
    for p in predictions:
        category = p.risk_category.value
        risk_distribution[category] = risk_distribution.get(category, 0) + 1
        
    improved = risk_distribution.get('improved', 0)
    stable = risk_distribution.get('stable', 0)
    degraded = risk_distribution.get('degraded', 0)
    severely_degraded = risk_distribution.get('severely_degraded', 0)
    
    # Calculate commit statistics
    commit_stats = calculate_commit_stats(commit_data)
    
    # Sort predictions by risk score (highest first)
    sorted_predictions = sorted(predictions, key=lambda x: x.degradation_score, reverse=True)
    
    # Build file tree structure
    file_tree = build_file_tree(sorted_predictions)
    
    # Calculate file type statistics
    file_types = {}
    for d in commit_data:
        ext = Path(d.module).suffix.lower() or '.no-ext'
        file_types[ext] = file_types.get(ext, 0) + 1
        
    top_file_types = sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:8]
    
    # Calculate risk by file type
    risk_by_type = {}
    for p in predictions:
        ext = Path(p.module).suffix.lower() or '.no-ext'
        if ext not in risk_by_type:
            risk_by_type[ext] = {'sum': 0, 'count': 0}
        risk_by_type[ext]['sum'] += p.degradation_score
        risk_by_type[ext]['count'] += 1
        
    top_risk_by_type = sorted([
        {
            'ext': ext,
            'avg': data['sum'] / data['count'],
            'count': data['count']
        }
        for ext, data in risk_by_type.items()
    ], key=lambda x: x['avg'], reverse=True)[:8]
    
    # Generate all the HTML sections
    top_files_html = ''.join([
        f'''
                  <div class="top-file-item {p.risk_category.value.replace('_', '-')}">
                    <div class="file-name">{p.module}</div>
                    <div style="display: flex; align-items: center; gap: 10px;">
                      <div class="file-score">{p.degradation_score:.4f}</div>
                      <div class="risk-badge {p.risk_category.value.replace('_', '-')}">{p.risk_category.value.replace('_', ' ')}</div>
                    </div>
                  </div>'''
        for p in sorted_predictions[:30]
    ])
    
    tree_html = generate_tree_html(file_tree)
    
    file_types_html = ''.join([
        f'''
                    <li>
                        <span class="file-type">{ext}</span>
                        <span><strong>{count}</strong> files</span>
                    </li>'''
        for ext, count in top_file_types
    ])
    
    contributors_html = ''
    if commit_stats.author_names:
        contributors_html = ''.join([
            f'''
                    <div class="author-item">
                        <div class="author-avatar">{author[0].upper()}</div>
                        <div class="author-name">{author}</div>
                    </div>'''
            for author in sorted(commit_stats.author_names, key=lambda x: x.lower())
        ])
    
    risk_by_type_html = ''
    if top_risk_by_type:
        for item in top_risk_by_type:
            avg = item['avg']
            if avg >= 0.2:
                risk_class = 'risk-high'
            elif avg >= 0.1:
                risk_class = 'risk-medium'
            elif avg >= 0.0:
                risk_class = 'risk-low'
            else:
                risk_class = 'risk-good'
                
            risk_by_type_html += f'''
                  <li>
                      <span class="file-type">{item['ext']}</span>
                      <span style="display: flex; align-items: center; gap: 10px;">
                          <span class="risk-score {risk_class}">{avg:.3f}</span>
                          <span><strong>{item['count']}</strong> files</span>
                      </span>
                  </li>'''

    # Return the complete HTML (identical to TypeScript version)
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MaintSight - Maintenance Risk Analysis - {repo_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f8f9fa;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}

        .header {{
            background: white;
            padding: 40px 30px;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(22, 104, 220, 0.1);
            margin-bottom: 30px;
            text-align: center;
            border: 1px solid rgba(22, 104, 220, 0.1);
        }}

        .header h1 {{
            color: #1668dc;
            margin-bottom: 8px;
            font-size: 2.2em;
            font-weight: 700;
            letter-spacing: -0.02em;
        }}

        .header p {{
            color: #3c89e8;
            font-size: 1.1em;
            margin-bottom: 20px;
            font-weight: 500;
            opacity: 0.8;
        }}

        .header .meta {{
            color: #3c89e8;
            font-size: 0.9em;
            opacity: 0.7;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid rgba(22, 104, 220, 0.1);
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            grid-template-rows: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 30px;
            max-width: 1000px;
            margin-left: auto;
            margin-right: auto;
        }}

        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}

        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}

        .stat-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}

        .stat-percentage {{
            font-size: 0.8em;
            margin-top: 5px;
        }}

        .improved {{ color: #4CAF50; }}
        .stable {{ color: #1668dc; }}
        .degraded {{ color: #FF9500; }}
        .severely-degraded {{ color: #FF5757; }}

        .section {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}

        .section h2 {{
            color: #1668dc;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
        }}

        .section h2.overview::before {{ content: '📊'; margin-right: 10px; }}
        .section h2.commit-stats::before {{ content: '💻'; margin-right: 10px; }}
        .section h2.file-types::before {{ content: '📁'; margin-right: 10px; }}
        .section h2.top-files::before {{ content: '⚠️'; margin-right: 10px; }}
        .section h2.file-tree::before {{ content: '🌳'; margin-right: 10px; }}

        .two-column {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}

        .stat-list {{
            list-style: none;
            padding: 0;
        }}

        .stat-list li {{
            padding: 8px 0;
            border-bottom: 1px solid #ecf0f1;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .stat-list li:last-child {{
            border-bottom: none;
        }}

        .file-type {{
            font-family: 'Monaco', 'Menlo', monospace;
            background: #f8f9fa;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
        }}

        .risk-score {{
            font-family: 'Monaco', 'Menlo', monospace;
            font-weight: bold;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.85em;
        }}

        .risk-high {{ background: #fdedec; color: #e74c3c; }}
        .risk-medium {{ background: #fef9e7; color: #f39c12; }}
        .risk-low {{ background: #ebf3fd; color: #3498db; }}
        .risk-good {{ background: #eafaf1; color: #27ae60; }}

        .tree-node {{
            margin-bottom: 10px;
        }}

        .tree-folder {{
            font-weight: bold;
            color: #34495e;
            margin-bottom: 10px;
            cursor: pointer;
            user-select: none;
        }}

        .tree-folder:hover {{
            color: #2c3e50;
        }}

        .tree-folder::before {{
            margin-right: 5px;
        }}

        .tree-file {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 6px;
            background: #f8f9fa;
            border-left: 4px solid #ddd;
        }}

        .tree-file.improved {{
            border-left-color: #4CAF50;
            background: #E8F5E8;
        }}

        .tree-file.stable {{
            border-left-color: #1668dc;
            background: #f0f5ff;
        }}

        .tree-file.degraded {{
            border-left-color: #FF9500;
            background: #FFF5E6;
        }}

        .tree-file.severely-degraded {{
            border-left-color: #FF5757;
            background: #FFE8E8;
        }}

        .file-name {{
            flex: 1;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.9em;
        }}

        .file-name::before {{
            content: '📄 ';
            margin-right: 5px;
        }}

        .file-score {{
            font-weight: bold;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.85em;
        }}

        .risk-badge {{
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-left: 10px;
        }}

        .risk-badge.improved {{
            background: #4CAF50;
            color: white;
        }}

        .risk-badge.stable {{
            background: #1668dc;
            color: white;
        }}

        .risk-badge.degraded {{
            background: #FF9500;
            color: white;
        }}

        .risk-badge.severely-degraded {{
            background: #FF5757;
            color: white;
        }}

        .file-tree-container {{
            max-height: 600px;
            overflow-y: auto;
            padding: 10px;
            background: #f0f5ff;
            border-radius: 8px;
            border: 1px solid #65a9f3;
        }}

        .tree-folder {{
            display: flex;
            align-items: center;
            padding: 8px 12px;
            margin: 2px 0;
            border-radius: 6px;
            background: #e6f4ff;
            border-left: 4px solid #1668dc;
            cursor: pointer;
            user-select: none;
            transition: all 0.2s ease;
            font-weight: 500;
        }}

        .tree-folder:hover {{
            background: #f0f5ff;
            border-left-color: #1554ad;
        }}

        .tree-folder.improved {{
            border-left-color: #4CAF50;
            background: #E8F5E8;
        }}

        .tree-folder.stable {{
            border-left-color: #1668dc;
            background: #f0f5ff;
        }}

        .tree-folder.degraded {{
            border-left-color: #FF9500;
            background: #FFF5E6;
        }}

        .tree-folder.severely-degraded {{
            border-left-color: #FF5757;
            background: #FFE8E8;
        }}

        .folder-toggle {{
            margin-right: 8px;
            font-size: 12px;
            transition: transform 0.2s ease;
            color: #1668dc;
        }}

        .folder-toggle.expanded {{
            transform: rotate(90deg);
        }}

        .folder-icon {{
            margin-right: 8px;
            font-size: 14px;
        }}

        .folder-name {{
            flex: 1;
            font-size: 0.9em;
            color: #2D3748;
        }}

        .folder-stats {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.8em;
        }}

        .folder-count {{
            color: #1668dc;
            background: #f0f5ff;
            padding: 2px 6px;
            border-radius: 10px;
            font-weight: 500;
        }}

        .folder-score {{
            font-family: 'Monaco', 'Menlo', monospace;
            font-weight: bold;
            color: #2D3748;
        }}

        .collapsible {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }}

        .collapsible.expanded {{
            max-height: 2000px;
        }}

        .footer {{
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 30px;
            padding: 20px;
        }}

        .top-files-list {{
            max-height: 400px;
            overflow-y: auto;
        }}

        .top-file-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            margin: 5px 0;
            border-radius: 6px;
            background: #f8f9fa;
            border-left: 4px solid #ddd;
        }}

        .top-file-item.severely-degraded {{
            border-left-color: #FF5757;
            background: #FFE8E8;
        }}

        .top-file-item.degraded {{
            border-left-color: #FF9500;
            background: #FFF5E6;
        }}

        .top-file-item.stable {{
            border-left-color: #1668dc;
            background: #f0f5ff;
        }}

        .top-file-item.improved {{
            border-left-color: #4CAF50;
            background: #E8F5E8;
        }}

        .authors-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
            max-height: 280px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #65a9f3;
            border-radius: 8px;
            background: #f0f5ff;
        }}

        .author-item {{
            display: flex;
            align-items: center;
            padding: 10px 15px;
            background: #e6f4ff;
            border-radius: 8px;
            border: 1px solid #65a9f3;
            border-left: 4px solid #1668dc;
            transition: background 0.2s ease;
        }}

        .author-item:hover {{
            background: #e9ecef;
        }}

        .author-avatar {{
            width: 35px;
            height: 35px;
            border-radius: 50%;
            background: #1668dc;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 14px;
            margin-right: 12px;
            flex-shrink: 0;
        }}

        .author-name {{
            font-size: 0.9em;
            color: #2c3e50;
            font-weight: 500;
            word-break: break-word;
        }}

        @media (max-width: 1200px) {{
            .stats-grid {{
                grid-template-columns: repeat(4, 1fr);
                grid-template-rows: repeat(2, 1fr);
            }}
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}

            .stats-grid {{
                grid-template-columns: repeat(2, 1fr);
                grid-template-rows: repeat(4, 1fr);
            }}

            .header {{
                padding: 30px 20px;
            }}

            .header h1 {{
                font-size: 1.8em;
            }}
        }}

            .two-column {{
                grid-template-columns: 1fr;
            }}

            .tree-node {{
                margin-left: 0;
            }}

            .authors-grid {{
                grid-template-columns: 1fr;
                max-height: 400px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 MaintSight - Maintenance Risk Analysis</h1>
            <p>Powered by TechDebtGPT</p>
            <div class="meta">
                <strong>Repository:</strong> {repo_name}<br>
                <strong>Generated:</strong> {datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M:%S')}<br>
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number improved">{improved}</div>
                <div class="stat-label">Improved</div>
                <div class="stat-percentage improved">{(improved / total_files * 100):.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-number stable">{stable}</div>
                <div class="stat-label">Stable</div>
                <div class="stat-percentage stable">{(stable / total_files * 100):.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-number degraded">{degraded}</div>
                <div class="stat-label">Degraded</div>
                <div class="stat-percentage degraded">{(degraded / total_files * 100):.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-number severely-degraded">{severely_degraded}</div>
                <div class="stat-label">Severely Degraded</div>
                <div class="stat-percentage severely-degraded">{(severely_degraded / total_files * 100):.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{total_files}</div>
                <div class="stat-label">Total Files</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{mean_score:.4f}</div>
                <div class="stat-label">Mean Risk Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{std_dev:.4f}</div>
                <div class="stat-label">Standard Deviation</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{commit_stats.total_commits}</div>
                <div class="stat-label">Total Commits</div>
            </div>
        </div>

        <div class="section">
            <h2 class="overview">Analysis Overview</h2>
            <p><strong>Repository Analysis:</strong> This comprehensive report analyzes {total_files} files across {commit_stats.total_commits} commits in the {repo_name} repository to assess maintenance risk and code quality trends.</p>
            <br>
            <p><strong>Risk Categories:</strong></p>
            <ul style="margin-left: 20px; margin-top: 10px;">
                <li><strong class="improved">Improved (< 0.0):</strong> Code quality is improving - excellent maintenance practices</li>
                <li><strong class="stable">Stable (0.0-0.1):</strong> Code quality is stable - minimal degradation detected</li>
                <li><strong class="degraded">Degraded (0.1-0.2):</strong> Moderate degradation - consider refactoring</li>
                <li><strong class="severely-degraded">Severely Degraded (> 0.2):</strong> Critical attention needed - rapid quality decline</li>
            </ul>
        </div>

        <div class="section">
            <h2 class="top-files">Highest Risk Files (Top 30)</h2>
            <div class="top-files-list">
                {top_files_html}
            </div>
        </div>

        <div class="section">
            <h2 class="file-tree">Complete File Analysis Tree</h2>
            {tree_html}
        </div>

        <div class="two-column">
            <div class="section">
                <h2 class="commit-stats">Commit Statistics</h2>
                <ul class="stat-list">
                    <li>
                        <span>Total Commits</span>
                        <span><strong>{commit_stats.total_commits}</strong></span>
                    </li>
                    <li>
                        <span>Total Authors</span>
                        <span><strong>{len(commit_stats.author_names)}</strong></span>
                    </li>
                    <li>
                        <span>Bug Fix Commits</span>
                        <span><strong>{commit_stats.total_bug_fixes}</strong></span>
                    </li>
                    <li>
                        <span>Avg Commits/File</span>
                        <span><strong>{commit_stats.avg_commits_per_file:.1f}</strong></span>
                    </li>
                    <li>
                        <span>Bug Fix Rate</span>
                        <span><strong>{(commit_stats.total_bug_fixes / commit_stats.total_commits * 100 if commit_stats.total_commits > 0 else 0):.1f}%</strong></span>
                    </li>
                </ul>
            </div>

            <div class="section">
                <h2 class="file-types">File Type Distribution</h2>
                <ul class="stat-list">
                    {file_types_html}
                </ul>
            </div>
        </div>

        {f'''
        <div class="section">
            <h2 class="commit-stats">Repository Contributors ({len(commit_stats.author_names)})</h2>
            <div class="authors-grid">
                {contributors_html}
            </div>
        </div>
        ''' if commit_stats.author_names else ''}

        {f'''
        <div class="section">
            <h2 class="file-types">Average Risk by File Type</h2>
            <ul class="stat-list">
                {risk_by_type_html}
            </ul>
        </div>
        ''' if top_risk_by_type else ''}

        <div class="footer">
            Generated by <strong>MaintSight</strong> using XGBoost Machine Learning<br>
            Risk scores based on commit patterns, code churn, and development activity analysis<br>
            <em>Analysis includes both prediction and statistical insights</em>
        </div>
    </div>

    <script>
        // Toggle folder function
        function toggleFolder(element) {{
            const content = element.nextElementSibling;
            const toggle = element.querySelector('.folder-toggle');

            if (content && content.classList.contains('collapsible')) {{
                content.classList.toggle('expanded');
                toggle.classList.toggle('expanded');
            }}
        }}

        // Initialize - all folders collapsed by default (no auto-expand)
        document.addEventListener('DOMContentLoaded', function() {{
            // All folders start collapsed by default
            console.log('File tree initialized - all folders collapsed');
        }});

        // Smooth scrolling for any internal links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
            anchor.addEventListener('click', function (e) {{
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {{
                    target.scrollIntoView({{ behavior: 'smooth' }});
                }}
            }});
        }});

        // Add keyboard navigation
        document.addEventListener('keydown', function(e) {{
            if (e.target.classList.contains('tree-folder')) {{
                if (e.key === 'Enter' || e.key === ' ') {{
                    e.preventDefault();
                    toggleFolder(e.target);
                }}
            }}
        }});
    </script>
</body>
</html>'''

def main():
    """Complete CLI with HTML generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='MaintSight - Complete Git repository analysis with HTML reports')
    parser.add_argument('path', nargs='?', default='.', help='Repository path')
    parser.add_argument('-b', '--branch', default='main', help='Git branch')
    parser.add_argument('-n', '--max-commits', type=int, default=1000, help='Max commits')
    parser.add_argument('-w', '--window-days', type=int, default=150, help='Time window in days')
    parser.add_argument('-f', '--format', choices=['json', 'summary', 'html'], default='html', help='Output format')
    
    args = parser.parse_args()
    
    try:
        # Collect git data
        collector = GitCommitCollector(
            repo_path=args.path,
            branch=args.branch,
            window_size_days=args.window_days
        )
        
        commit_data = collector.fetch_commit_data(args.max_commits)
        
        if not commit_data:
            print("❌ No source files found in git history")
            sys.exit(1)
            
        # Create mock risk predictions (replace with real ML model)
        predictor = MockPredictor()
        predictions = predictor.predict(commit_data)
        
        print(f"🤖 Generated risk predictions for {len(predictions)} files")
        
        if args.format == 'json':
            # Output JSON
            output = []
            for pred in predictions:
                output.append({
                    'module': pred.module,
                    'degradation_score': round(pred.degradation_score, 4),
                    'raw_prediction': round(pred.raw_prediction, 4),
                    'risk_category': pred.risk_category.value
                })
            print(json.dumps(output, indent=2))
            
        elif args.format == 'summary':
            # Summary output
            print(f"\n📊 Repository Analysis Summary")
            print(f"Repository: {commit_data[0].repo_name}")
            print(f"Files analyzed: {len(commit_data)}")
            print(f"Branch: {args.branch}")
            print(f"Time window: {args.window_days} days")
            
            # Sort by risk score
            sorted_preds = sorted(predictions, key=lambda x: x.degradation_score, reverse=True)
            
            print(f"\n🔥 Top 10 Highest Risk Files:")
            print(f"{'File':<50} {'Risk Score':<12} {'Category':<20}")
            print("-" * 82)
            
            for pred in sorted_preds[:10]:
                print(f"{pred.module:<50} {pred.degradation_score:<12.4f} {pred.risk_category.value:<20}")
                
        else:  # HTML format
            # Generate HTML report
            html_path = generate_html_report(predictions, commit_data, args.path)
            
            if html_path:
                print(f"🌐 Complete HTML report generated!")
                
                # Try to open in browser (macOS)
                try:
                    subprocess.run(['open', html_path], check=True)
                    print(f"✅ HTML report opened in browser")
                except:
                    print(f"📂 HTML report saved to: {html_path}")
                    print(f"🌐 Open in browser: file://{html_path}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()