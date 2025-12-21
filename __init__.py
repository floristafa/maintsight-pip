"""
MaintSight - AI-powered maintenance risk predictor for git repositories using XGBoost.

This package provides tools to analyze git repositories and predict maintenance degradation
risk using machine learning. It analyzes commit patterns, code churn, and collaboration
metrics to identify files that may need attention.
"""

__version__ = "0.4.0"
__author__ = "TechDebtGPT Team"
__email__ = "support@techdebtgpt.com"
__license__ = "Apache-2.0"

from models import (
    CommitData,
    RiskPrediction, 
    RiskCategory,
    FileStats,
)

from services import (
    GitCommitCollector,
    FeatureEngineer,
    XGBoostPredictor,
)

from utils.logger import Logger

__all__ = [
    # Models
    "CommitData",
    "RiskPrediction", 
    "RiskCategory",
    "FileStats",
    # Services
    "GitCommitCollector",
    "FeatureEngineer", 
    "XGBoostPredictor",
    # Utils
    "Logger",
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]