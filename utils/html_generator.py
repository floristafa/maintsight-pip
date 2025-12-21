"""HTML report generator for MaintSight - exact port from TypeScript version."""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import math

from models import RiskPrediction, CommitData


class FileTreeNode:
    """File tree node for hierarchical display."""
    
    def __init__(
        self, 
        name: str, 
        node_type: str, 
        children: Optional[List['FileTreeNode']] = None,
        prediction: Optional[Any] = None,
        path: Optional[str] = None
    ):
        self.name = name
        self.type = node_type  # 'file' or 'folder'
        self.children = children or []
        self.prediction = prediction
        self.path = path


class CommitStats:
    """Commit statistics for the repository."""
    
    def __init__(
        self,
        total_commits: int = 0,
        total_bug_fixes: int = 0,
        avg_commits_per_file: float = 0.0,
        avg_authors_per_file: float = 0.0,
        author_names: Optional[List[str]] = None
    ):
        self.total_commits = total_commits
        self.total_bug_fixes = total_bug_fixes
        self.avg_commits_per_file = avg_commits_per_file
        self.avg_authors_per_file = avg_authors_per_file
        self.author_names = author_names or []


def generate_html_report(
    predictions: List[Any], 
    commit_data: List[Any], 
    repo_path: str
) -> Optional[str]:
    """Generate HTML report and save to .maintsight directory.
    
    Args:
        predictions: List of RiskPrediction objects
        commit_data: List of CommitData objects
        repo_path: Path to git repository
        
    Returns:
        Path to generated HTML file or None if failed
    """
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
            
        print(f"HTML report saved to: {html_path}")
        
        return str(html_path)
        
    except Exception as error:
        print(f"Warning: Could not save HTML report: {error}")
        return None


def build_file_tree(predictions: List[Any]) -> FileTreeNode:
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
                total_score += n.prediction.normalized_score
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
            score = node.prediction.normalized_score
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


def calculate_commit_stats(commit_data: List[Any]) -> CommitStats:
    """Calculate commit statistics from commit data."""
    if len(commit_data) == 0:
        return CommitStats()
        
    total_commits = sum(getattr(d, 'commits', 0) for d in commit_data)
    total_bug_fixes = sum(getattr(d, 'bug_commits', 0) for d in commit_data)
    
    # Extract unique authors from author_names field
    unique_authors = set()
    for d in commit_data:
        author_names = getattr(d, 'author_names', None)
        if author_names and isinstance(author_names, list):
            for author in author_names:
                unique_authors.add(author)
                
    author_names = list(unique_authors)
    avg_authors_per_file = sum(getattr(d, 'authors', 0) for d in commit_data) / len(commit_data)
    
    return CommitStats(
        total_commits=total_commits,
        total_bug_fixes=total_bug_fixes,
        avg_commits_per_file=total_commits / len(commit_data),
        avg_authors_per_file=avg_authors_per_file,
        author_names=author_names
    )


def format_as_html(
    predictions: List[Any], 
    commit_data: List[Any], 
    repo_path: str
) -> str:
    """Format predictions and commit data as HTML report.
    
    Args:
        predictions: List of RiskPrediction objects  
        commit_data: List of CommitData objects
        repo_path: Path to git repository
        
    Returns:
        Complete HTML report as string
    """
    repo_name = Path(repo_path).name
    timestamp = datetime.now().isoformat()
    
    # Calculate statistics
    total_files = len(predictions)
    mean_score = sum(p.normalized_score for p in predictions) / total_files
    std_dev = math.sqrt(
        sum((p.normalized_score - mean_score) ** 2 for p in predictions) / total_files
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
    sorted_predictions = sorted(predictions, key=lambda x: x.normalized_score, reverse=True)
    
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
        risk_by_type[ext]['sum'] += p.normalized_score
        risk_by_type[ext]['count'] += 1
        
    top_risk_by_type = sorted(
        [
            {
                'ext': ext,
                'avg': data['sum'] / data['count'],
                'count': data['count']
            }
            for ext, data in risk_by_type.items()
        ],
        key=lambda x: x['avg'],
        reverse=True
    )[:8]
    
    # Generate top files HTML
    top_files_html = ''
    for p in sorted_predictions[:30]:
        score = p.normalized_score
        category_class = p.risk_category.value.replace('_', '-')
        top_files_html += f'''
                  <div class="top-file-item {category_class}">
                    <div class="file-name">{p.module}</div>
                    <div style="display: flex; align-items: center; gap: 10px;">
                      <div class="file-score">{score:.4f}</div>
                      <div class="risk-badge {category_class}">{p.risk_category.value.replace('_', ' ')}</div>
                    </div>
                  </div>'''
                  
    # Generate file tree HTML
    tree_html = generate_tree_html(file_tree)
    
    # Generate file type distribution HTML
    file_types_html = ''
    for ext, count in top_file_types:
        file_types_html += f'''
                    <li>
                        <span class="file-type">{ext}</span>
                        <span><strong>{count}</strong> files</span>
                    </li>'''
                    
    # Generate contributors HTML
    contributors_html = ''
    if commit_stats.author_names:
        sorted_authors = sorted(commit_stats.author_names, key=lambda x: x.lower())
        for author in sorted_authors:
            contributors_html += f'''
                    <div class="author-item">
                        <div class="author-avatar">{author[0].upper()}</div>
                        <div class="author-name">{author}</div>
                    </div>'''
                    
    # Generate risk by file type HTML
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
    
    # Generate the complete HTML
    html_content = f'''<!DOCTYPE html>
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
    
    return html_content