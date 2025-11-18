# 🔍 MaintSight

[![PyPI version](https://img.shields.io/pypi/v/maintsight.svg)](https://pypi.org/project/maintsight/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

> **AI-powered maintenance degradation predictor for git repositories using XGBoost machine learning**

MaintSight analyzes your git repository's commit history and code patterns to predict maintenance degradation at the file level. Using a trained XGBoost model, it identifies code quality trends and helps prioritize refactoring efforts by detecting files that are degrading over time.

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Output Formats](#-output-formats)
- [Degradation Categories](#-degradation-categories)
- [Command Reference](#-command-reference)
- [Model Information](#-model-information)
- [Development](#-development)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- 🤖 **XGBoost ML Predictions**: Pre-trained model for maintenance degradation scoring
- 📊 **Git History Analysis**: Analyzes commits, changes, and collaboration patterns
- 📈 **Multiple Output Formats**: JSON, CSV, Markdown, or interactive HTML reports
- 🎯 **Degradation Categorization**: Four-level classification (Improved/Stable/Degraded/Severely Degraded)
- 🔍 **Threshold Filtering**: Focus on degraded files only
- 🌐 **Interactive HTML Reports**: Rich, interactive analysis with visualizations
- ⚡ **Fast & Efficient**: Analyzes hundreds of files in seconds
- 🛠️ **Easy Integration**: Simple CLI interface and npm package

## 🚀 Quick Start

```bash
# Install from PyPI
pip install maintsight

# Run predictions on current directory (generates interactive HTML report)
python3 maintsight_complete.py

# Show only degraded files with threshold
python3 maintsight_complete.py -f summary

# Generate JSON output
python3 maintsight_complete.py -f json

# Analyze specific repository
python3 maintsight_complete.py /path/to/repo
```

## 📦 Installation

### From PyPI (Coming Soon)

```bash
pip install maintsight
```

### From Source (Current)

```bash
git clone https://github.com/techdebtgpt/maintsight.git
cd maintsight-pip
pip install -r requirements.txt

# Run the complete version
python3 maintsight_complete.py
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## 📖 Usage

### Basic Prediction

```bash
# Analyze current directory (generates HTML report)
python3 maintsight_complete.py

# Analyze specific repository
python3 maintsight_complete.py /path/to/repo

# Generate summary output
python3 maintsight_complete.py -f summary
```

### Advanced Options

```bash
# Analyze specific branch
python3 maintsight_complete.py -b develop

# Limit commit analysis window
python3 maintsight_complete.py -w 90  # Analyze last 90 days

# Limit number of commits
python3 maintsight_complete.py -n 5000

# Generate JSON output
python3 maintsight_complete.py -f json

# All options together
python3 maintsight_complete.py /path/to/repo -b main -w 150 -n 1000 -f html
```

### Python API Usage

```python
from maintsight import GitCommitCollector, MockPredictor
from maintsight.utils.html_generator import generate_html_report

# Collect git data
collector = GitCommitCollector(repo_path="./", branch="main")
commit_data = collector.fetch_commit_data()

# Generate predictions
predictor = MockPredictor()
predictions = predictor.predict(commit_data)

# Generate HTML report
html_path = generate_html_report(predictions, commit_data, "./")
```

## 📊 Output Formats

### JSON (Default)

```json
[
  {
    "module": "src/legacy/parser.ts",
    "degradation_score": 0.3456,
    "raw_prediction": 0.3456,
    "risk_category": "severely_degraded"
  },
  {
    "module": "src/utils/helpers.ts",
    "degradation_score": -0.1234,
    "raw_prediction": -0.1234,
    "risk_category": "improved"
  }
]
```

### CSV

```csv
module,degradation_score,raw_prediction,risk_category
"src/legacy/parser.ts","0.3456","0.3456","severely_degraded"
"src/utils/helpers.ts","-0.1234","-0.1234","improved"
```

### Markdown Report

Generates a comprehensive report with:

- Degradation distribution summary
- Top 20 most degraded files
- Category breakdown with percentages
- Actionable recommendations

### Interactive HTML Report

Always generated automatically in `.maintsight/` folder with:

- Visual degradation trends
- Interactive file explorer
- Detailed metrics per file
- Commit history analysis

## 🎯 Degradation Categories

| Score Range | Category             | Description                      | Action                     |
| ----------- | -------------------- | -------------------------------- | -------------------------- |
| < 0.0       | 🟢 Improved          | Code quality improving over time | Continue good practices    |
| 0.0-0.1     | 🔵 Stable            | Code quality stable              | Regular maintenance        |
| 0.1-0.2     | 🟡 Degraded          | Code quality declining           | Schedule for refactoring   |
| > 0.2       | 🔴 Severely Degraded | Rapid quality decline            | Immediate attention needed |

## 📚 Command Reference

### `maintsight_complete.py`

Analyze repository and predict maintenance degradation.

```bash
python3 maintsight_complete.py [path] [options]
```

**Arguments:**

- `path` - Repository path (default: current directory)

**Options:**

- `-b, --branch BRANCH` - Git branch to analyze (default: "main")
- `-n, --max-commits N` - Maximum commits to analyze (default: 1000)
- `-w, --window-days N` - Time window in days for analysis (default: 150)
- `-f, --format FORMAT` - Output format: json|summary|html (default: "html")
- `-h, --help` - Show help information

### Examples

```bash
# Generate HTML report with default settings
python3 maintsight_complete.py

# Analyze last 90 days on develop branch
python3 maintsight_complete.py -b develop -w 90

# Get JSON output for processing
python3 maintsight_complete.py -f json > results.json

# Show summary for quick overview
python3 maintsight_complete.py -f summary
```

## 🧠 Model Information

MaintSight uses an XGBoost model trained on software maintenance degradation patterns. The model predicts how code quality changes over time by analyzing git commit patterns and code evolution metrics.

### Key Features Analyzed

The model considers multiple dimensions of code evolution:

- **Commit patterns**: Frequency, size, and timing of changes
- **Author collaboration**: Number of contributors and collaboration patterns
- **Code churn**: Lines added, removed, and modified over time
- **Change consistency**: Regularity and predictability of modifications
- **Bug indicators**: Patterns suggesting defects or fixes
- **Temporal factors**: File age and time since last modification

### Prediction Output

- **degradation_score**: Numerical score indicating code quality trend
  - Negative values: Quality improving
  - Positive values: Quality degrading
  - Higher magnitude = stronger trend
- **risk_category**: Classification based on degradation severity
- **raw_prediction**: Unprocessed model output

## 🔧 Development

### Prerequisites

- Python >= 3.8
- Git

### Setup

```bash
# Clone repository
git clone https://github.com/techdebtgpt/maintsight.git
cd maintsight-pip

# Install in development mode
pip install -e ".[dev]"

# Or install requirements directly
pip install -r requirements.txt

# Run the main script
python3 maintsight_complete.py
```

### Project Structure

```
maintsight-pip/
├── maintsight/                    # Python package
│   ├── __init__.py
│   ├── cli.py                     # Click-based CLI
│   ├── models/                    # Data models
│   │   ├── __init__.py
│   │   ├── commit_data.py         # CommitData dataclass
│   │   ├── risk_category.py       # RiskCategory enum
│   │   ├── risk_prediction.py     # RiskPrediction dataclass
│   │   ├── file_stats.py          # FileStats dataclass
│   │   ├── xgboost_model.py       # XGBoost model structures
│   │   ├── xgboost_degradation_model_multiwindow_v2.pkl      # Pre-trained model
│   │   └── xgboost_degradation_model_multiwindow_v2_metadata.json  # Model metadata
│   ├── services/                  # Core services
│   │   ├── __init__.py
│   │   ├── git_commit_collector.py
│   │   ├── feature_engineer.py
│   │   └── xgboost_predictor.py
│   └── utils/                     # Utilities
│       ├── __init__.py
│       ├── logger.py              # Rich-based logger
│       └── html_generator.py      # HTML report generator
├── tests/                         # pytest tests
│   ├── __init__.py
│   └── test_risk_category.py
├── maintsight_complete.py         # Standalone complete script
├── pyproject.toml                 # Modern Python packaging
├── setup.py                       # Legacy setuptools support
├── requirements.txt               # Runtime dependencies
└── requirements-dev.txt           # Development dependencies
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

# Install test dependencies
pip install -e ".[dev]"
```

### Test Coverage Goals

- Services: 80%+
- Utils: 90%+
- CLI: 70%+

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Quick Start

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`npm test`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

- Use Python 3.8+ features
- Follow PEP 8 style guide
- Use black for code formatting
- Use type hints where appropriate
- Write meaningful commit messages
- Add tests for new features
- Update documentation as needed

```bash
# Format code
black maintsight/

# Sort imports
isort maintsight/

# Lint code
flake8 maintsight/

# Type checking
mypy maintsight/
```

## 🐛 Bug Reports

Found a bug? Please [open an issue](https://github.com/techdebtgpt/maintsight/issues/new) with:

- MaintSight version (`python3 maintsight_complete.py --help`)
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

[Repository](https://github.com/techdebtgpt/maintsight) | [Documentation](https://github.com/techdebtgpt/maintsight#readme) | [Issues](https://github.com/techdebtgpt/maintsight/issues)
