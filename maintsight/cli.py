#!/usr/bin/env python3
"""
MaintSight CLI - AI-powered maintenance risk predictor for git repositories.
"""

import click
import json
import sys
from pathlib import Path
from typing import Optional

# For now, ignore import errors - we'll fix them later
try:
    from .services import GitCommitCollector, FeatureEngineer, XGBoostPredictor
    from .models import RiskCategory
    from .utils import Logger
except ImportError:
    # Fallback for development
    pass


@click.group()
@click.version_option(version="0.1.0", prog_name="maintsight")
def main():
    """MaintSight - AI-powered maintenance risk predictor for git repositories."""
    pass


@main.command()
@click.argument('path', default='.', type=click.Path(exists=True))
@click.option('-b', '--branch', default='main', help='Git branch to analyze')
@click.option('-n', '--max-commits', default=10000, help='Maximum commits to analyze')
@click.option('-w', '--window-size-days', default=150, help='Time window in days for analysis')
@click.option('-o', '--output', type=click.Path(), help='Output file path')
@click.option('-f', '--format', 
              type=click.Choice(['json', 'csv', 'markdown', 'html']),
              default='json',
              help='Output format')
@click.option('-t', '--threshold', type=float, default=0.0,
              help='Only show files above degradation threshold')
@click.option('-v', '--verbose', is_flag=True, help='Verbose output')
def predict(
    path: str,
    branch: str,
    max_commits: int,
    window_size_days: int,
    output: Optional[str],
    format: str,
    threshold: float,
    verbose: bool
):
    """Run maintenance risk predictions on a git repository."""
    
    logger = Logger('MaintSight')
    
    try:
        # Initialize services
        if verbose:
            logger.info("Loading XGBoost model...", '📁')
            
        predictor = XGBoostPredictor()
        predictor.load_model()
        
        if verbose:
            logger.info(f"Analyzing git history (branch: {branch})...", '🔄')
            
        # Collect git data
        collector = GitCommitCollector(
            repo_path=path,
            branch=branch,
            window_size_days=window_size_days,
            only_existing_files=True
        )
        
        commit_data = collector.fetch_commit_data(max_commits)
        
        if commit_data is None or commit_data.empty:
            logger.error("No source files found in git history")
            sys.exit(1)
            
        if verbose:
            logger.info(f"Running predictions on {len(commit_data)} files...", '🤖')
            
        # Run predictions
        predictions = predictor.predict(commit_data)
        
        # Filter by threshold
        if threshold > 0:
            predictions = [p for p in predictions if p.degradation_score >= threshold]
            
        if verbose:
            logger.success(f"Predictions complete: {len(predictions)} files analyzed", '✅')
            
        # Format output
        if format == 'json':
            output_data = []
            for pred in predictions:
                output_data.append({
                    'module': pred.module,
                    'degradation_score': round(pred.degradation_score, 4),
                    'raw_prediction': round(pred.raw_prediction, 4),
                    'risk_category': pred.risk_category.value
                })
            result = json.dumps(output_data, indent=2)
            
        elif format == 'csv':
            lines = ['module,degradation_score,raw_prediction,risk_category']
            for pred in predictions:
                lines.append(f'"{pred.module}",{pred.degradation_score:.4f},'
                           f'{pred.raw_prediction:.4f},{pred.risk_category.value}')
            result = '\n'.join(lines)
            
        elif format == 'markdown':
            result = _format_markdown_report(predictions, Path(path).name)
            
        elif format == 'html':
            result = _format_html_report(predictions, Path(path).name)
            
        # Output results
        if output:
            with open(output, 'w') as f:
                f.write(result)
            logger.success(f"Results saved to: {output}", '✅')
        else:
            print(result)
            
        # Show summary
        if verbose:
            _show_summary(predictions, logger)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _format_markdown_report(predictions, repo_name: str) -> str:
    """Format predictions as markdown report."""
    from datetime import datetime
    
    # Sort by degradation score
    sorted_preds = sorted(predictions, key=lambda p: p.degradation_score, reverse=True)
    
    # Calculate distribution
    dist = {}
    for pred in predictions:
        category = pred.risk_category.value
        dist[category] = dist.get(category, 0) + 1
        
    total = len(predictions)
    
    report = f"""# MaintSight - Maintenance Risk Analysis Report

**Repository:** {repo_name}
**Date:** {datetime.now().isoformat()}
**Files Analyzed:** {total}

## Risk Distribution

| Risk Level | Count | Percentage |
|------------|-------|------------|
| Severely Degraded | {dist.get('severely_degraded', 0)} | {(dist.get('severely_degraded', 0)/total*100):.1f}% |
| Degraded | {dist.get('degraded', 0)} | {(dist.get('degraded', 0)/total*100):.1f}% |
| Stable | {dist.get('stable', 0)} | {(dist.get('stable', 0)/total*100):.1f}% |
| Improved | {dist.get('improved', 0)} | {(dist.get('improved', 0)/total*100):.1f}% |

## Top 20 High-Risk Files

| File | Degradation Score | Category |
|------|------------------|----------|
"""
    
    for pred in sorted_preds[:20]:
        report += f"| `{pred.module}` | {pred.degradation_score:.4f} | {pred.risk_category.value} |\n"
        
    report += """
## Risk Categories

- **Severely Degraded (> 0.2)**: Critical attention needed - code quality declining rapidly
- **Degraded (0.1-0.2)**: Moderate degradation - consider refactoring
- **Stable (0.0-0.1)**: Code quality stable - minimal degradation  
- **Improved (< 0.0)**: Code quality improving - good maintenance practices

---
*Generated by MaintSight using XGBoost*
"""
    
    return report


def _format_html_report(predictions, repo_name: str) -> str:
    """Format predictions as HTML report."""
    # Basic HTML report - can be enhanced later
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>MaintSight Report - {repo_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .severely_degraded {{ background-color: #ffebee; }}
        .degraded {{ background-color: #fff8e1; }}
        .stable {{ background-color: #e3f2fd; }}
        .improved {{ background-color: #e8f5e8; }}
    </style>
</head>
<body>
    <h1>MaintSight Report - {repo_name}</h1>
    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Files Analyzed: {len(predictions)}</h2>
    
    <table>
        <tr>
            <th>File</th>
            <th>Degradation Score</th>
            <th>Risk Category</th>
        </tr>
"""
    
    # Sort by score
    sorted_preds = sorted(predictions, key=lambda p: p.degradation_score, reverse=True)
    
    for pred in sorted_preds:
        html += f"""        <tr class="{pred.risk_category.value}">
            <td>{pred.module}</td>
            <td>{pred.degradation_score:.4f}</td>
            <td>{pred.risk_category.display_name}</td>
        </tr>
"""
    
    html += """    </table>
</body>
</html>"""
    
    return html


def _show_summary(predictions, logger):
    """Show summary statistics."""
    dist = {}
    for pred in predictions:
        category = pred.risk_category.value  
        dist[category] = dist.get(category, 0) + 1
        
    logger.info("Summary:", '📊')
    print(f"Total files: {len(predictions)}")
    print(f"Severely degraded: {dist.get('severely_degraded', 0)}")
    print(f"Degraded: {dist.get('degraded', 0)}")
    print(f"Stable: {dist.get('stable', 0)}")
    print(f"Improved: {dist.get('improved', 0)}")


if __name__ == '__main__':
    main()