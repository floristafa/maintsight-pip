# 🔍 MaintSight (Python)

[![PyPI version](https://img.shields.io/pypi/v/maintsight.svg)](https://pypi.org/project/maintsight/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

> **AI-powered maintenance degradation predictor for git repositories using XGBoost machine learning**

MaintSight analyzes your git repository's commit history and code patterns to predict maintenance degradation at the file level. Using a trained XGBoost model, it identifies code quality trends and helps prioritize refactoring efforts by detecting files that are degrading over time.

## ✨ Features

- 🤖 **XGBoost ML Predictions**: Pre-trained model for maintenance degradation scoring
- 📊 **Git History Analysis**: Analyzes commits, changes, and collaboration patterns
- 📈 **Multiple Output Formats**: JSON, CSV, Markdown, or HTML reports
- 🎯 **Degradation Categorization**: Four-level classification (Improved/Stable/Degraded/Severely Degraded)
- 🔍 **Threshold Filtering**: Focus on degraded files only
- ⚡ **Fast & Efficient**: Analyzes hundreds of files in seconds
- 🛠️ **Easy Integration**: Simple CLI interface and Python API

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI
pip install maintsight

# Or install with development dependencies
pip install maintsight[dev]

# Or install with HTML report support
pip install maintsight[html]
```

### Basic Usage

```bash
# Analyze current directory
maintsight predict

# Analyze specific repository
maintsight predict /path/to/repo

# Show only degraded files
maintsight predict -t 0.1

# Generate markdown report
maintsight predict -f markdown -o report.md

# Generate CSV for Excel
maintsight predict -f csv -o analysis.csv
```

### Python API

```python
from maintsight import GitCommitCollector, XGBoostPredictor, FeatureEngineer

# Initialize components
collector = GitCommitCollector(repo_path="./my-repo", branch="main")
predictor = XGBoostPredictor()
predictor.load_model()

# Collect commit data
commit_data = collector.fetch_commit_data(max_commits=5000)

# Make predictions
predictions = predictor.predict(commit_data)

# Process results
for pred in predictions:
    if pred.is_degraded:
        print(f"{pred.module}: {pred.risk_category.display_name} "
              f"(score: {pred.degradation_score:.3f})")
```

## 📊 Output Formats

### JSON (Default)

```json
[
  {
    "module": "src/legacy/parser.py",
    "degradation_score": 0.3456,
    "raw_prediction": 0.3456,
    "risk_category": "severely_degraded"
  },
  {
    "module": "src/utils/helpers.py",
    "degradation_score": -0.1234,
    "raw_prediction": -0.1234,
    "risk_category": "improved"
  }
]
```

### CSV

```csv
module,degradation_score,raw_prediction,risk_category
"src/legacy/parser.py","0.3456","0.3456","severely_degraded"
"src/utils/helpers.py","-0.1234","-0.1234","improved"
```

## 🎯 Risk Categories

| Score Range | Category             | Description                      | Action                     |
| ----------- | -------------------- | -------------------------------- | -------------------------- |
| < 0.0       | 🟢 Improved          | Code quality improving over time | Continue good practices    |
| 0.0-0.1     | 🔵 Stable            | Code quality stable              | Regular maintenance        |
| 0.1-0.2     | 🟡 Degraded          | Code quality declining           | Schedule for refactoring   |
| > 0.2       | 🔴 Severely Degraded | Rapid quality decline            | Immediate attention needed |

## 📚 Command Reference

### `maintsight predict`

Analyze repository and predict maintenance degradation.

```bash
maintsight predict [PATH] [OPTIONS]
```

**Arguments:**

- `PATH`: Path to git repository (default: current directory)

**Options:**

- `-b, --branch TEXT`: Git branch to analyze (default: main)
- `-n, --max-commits INTEGER`: Maximum commits to analyze (default: 10000)
- `-w, --window-size-days INTEGER`: Time window in days (default: 150)
- `-o, --output PATH`: Output file path
- `-f, --format [json|csv|markdown|html]`: Output format (default: json)
- `-t, --threshold FLOAT`: Degradation threshold filter (default: 0.0)
- `-v, --verbose`: Verbose output
- `--help`: Show help message

## 🧠 Model Information

MaintSight uses an XGBoost model trained on software maintenance degradation patterns. The model analyzes 26 engineered features:

### Base Features (from git history)

- Commit patterns: frequency, authors, timing
- Code churn: lines added/deleted/modified
- Bug indicators: commits mentioning fixes
- Collaboration: number of contributors
- Temporal factors: file age, activity periods

### Engineered Features

- Code stability metrics
- Author concentration (bus factor)
- Commit density and patterns
- Modification ratios
- Quality trend indicators

### Prediction Output

- **degradation_score**: Numerical trend score (-0.5 to +0.5 range)
- **risk_category**: Classification (improved/stable/degraded/severely_degraded)
- **raw_prediction**: Unprocessed model output

## 🔧 Development

### Prerequisites

- Python >= 3.8
- Git
- pip or poetry

### Setup

```bash
# Clone repository
git clone https://github.com/techdebtgpt/maintsight.git
cd maintsight

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
black maintsight/
isort maintsight/
flake8 maintsight/
```

### Project Structure

```
maintsight/
├── maintsight/
│   ├── models/           # Data models and XGBoost model
│   │   ├── commit_data.py
│   │   ├── risk_prediction.py
│   │   ├── risk_category.py
│   │   ├── xgboost_degradation_model_multiwindow_v2.pkl
│   │   └── xgboost_degradation_model_multiwindow_v2_metadata.json
│   ├── services/         # Core services
│   │   ├── git_commit_collector.py
│   │   ├── feature_engineer.py
│   │   └── xgboost_predictor.py
│   ├── utils/           # Utilities
│   │   └── logger.py
│   └── cli.py           # CLI interface
├── tests/               # Test files
└── setup.py             # Package configuration
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=maintsight

# Run specific test file
pytest tests/test_risk_category.py

# Run with verbose output
pytest -v
```

## 📦 Building and Publishing

```bash
# Build package
python -m build

# Upload to PyPI (maintainers only)
python -m twine upload dist/*

# Install locally
pip install -e .
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Quick Start

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies (`pip install -e ".[dev]"`)
4. Write tests for your changes
5. Run tests (`pytest`) and linting (`black maintsight/`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 🐛 Bug Reports

Found a bug? Please [open an issue](https://github.com/techdebtgpt/maintsight/issues/new) with:

- MaintSight version (`maintsight --version`)
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- XGBoost community for the excellent gradient boosting framework
- Git community for robust version control
- All contributors who help improve MaintSight

---

**Made with ❤️ by the TechDebtGPT Team**

[Repository](https://github.com/techdebtgpt/maintsight) | [Documentation](https://github.com/techdebtgpt/maintsight#readme) | [Issues](https://github.com/techdebtgpt/maintsight/issues) | [PyPI](https://pypi.org/project/maintsight/)
