# MaintSight User Guide

This guide provides comprehensive instructions for using MaintSight to analyze and predict maintenance degradation in your git repositories.

## 📚 Table of Contents

- [**Installation**](#-installation)
- [**Quick Start**](#-quick-start)
- [**CLI Reference**](#-cli-reference)
  - [predict](#predict)
- [**Output Formats**](#-output-formats)
- [**Degradation Categories**](#-degradation-categories)
- [**Configuration**](#-configuration)
- [**Usage Examples**](#-usage-examples)
- [**Troubleshooting**](#-troubleshooting)

## 📦 Installation

You can install MaintSight globally to use it as a command-line tool.

```bash
# Using pip (recommended)
pip install maintsight

# From source
git clone https://github.com/floristafa/maintsight-pip.git
cd maintsight-pip
pip install -e .
```

## 🚀 Quick Start

### 1. Run a Basic Prediction

The simplest way to get started is by running a prediction on your current directory:

```bash
maintsight predict
```

This will analyze your git repository and output maintenance degradation predictions in JSON format, plus generate an interactive HTML report.

### 2. Filter Degraded Files

Focus on files that are degrading and need attention:

```bash
maintsight predict --threshold 0.1
```

## CLI Reference

### `predict`

Analyzes a git repository and predicts maintenance degradation for each file.

```bash
maintsight predict [path] [options]
```

**Arguments:**

- `[path]`: Path to the git repository to analyze. Defaults to the current directory.

**Options:**

| Flag                           | Description                                            | Default |
| ------------------------------ | ------------------------------------------------------ | ------- |
| `-b, --branch <branch>`        | Git branch to analyze                                  | `main`  |
| `-n, --max-commits <num>`      | Maximum number of commits to analyze                   | `10000` |
| `-w, --window-size-days <num>` | Time window in days for commit analysis                | `150`   |
| `-o, --output <path>`          | Output file path (defaults to stdout)                  | stdout  |
| `-f, --format <format>`        | Output format: `json`, `csv`, `markdown`, or `html`    | `json`  |
| `-t, --threshold <num>`        | Only show files with degradation score above threshold | `0`     |
| `-v, --verbose`                | Enable verbose output with detailed logs               | `false` |

## 📊 Output Formats

### JSON Format

```json
[
  {
    "module": "src/services/user.service.ts",
    "normalized_score": 0.2345,
    "raw_prediction": 0.2345,
    "risk_category": "severely_degraded"
  }
]
```

### CSV Format

```csv
module,normalized_score,raw_prediction,risk_category
src/services/user.service.ts,0.2345,0.2345,severely_degraded
src/controllers/auth.controller.ts,-0.0567,-0.0567,improved
```

### Markdown Format

Generates a comprehensive markdown report with:

- Degradation distribution summary
- Top degraded files
- Risk category breakdown with percentages
- Actionable recommendations

### HTML Format

Interactive HTML report with:

- Visual charts and graphs
- File-by-file analysis
- Commit history integration
- Exportable results

## 🎯 Degradation Categories

MaintSight categorizes files into four degradation levels:

- **🔴 Severely Degraded** (> 0.2): Rapid quality decline, requires immediate attention
- **🟡 Degraded** (0.1-0.2): Code quality declining, should be scheduled for refactoring
- **🔵 Stable** (0.0-0.1): Code quality stable, regular maintenance sufficient
- **🟢 Improved** (< 0.0): Code quality improving over time, continue good practices

## ⚙️ Configuration

### Environment Variables

MaintSight respects the following environment variables:

- `MAINTSIGHT_MODEL_PATH`: Custom path to the XGBoost model file
- `MAINTSIGHT_LOG_LEVEL`: Logging level (ERROR, WARN, INFO, DEBUG)

### Automatic Reports

MaintSight automatically generates interactive HTML reports saved to:

```
<repository>/.maintsight/report.html
```

These reports include:

- Visual degradation trends
- Interactive file explorer
- Detailed metrics per file
- Historical analysis capabilities

## 📋 Usage Examples

### Analyze a Specific Repository

```bash
maintsight predict /path/to/repo
```

### Generate a Markdown Report

```bash
maintsight predict --format markdown --output report.md
```

### Analyze Feature Branch

```bash
maintsight predict --branch feature/new-feature
```

### Quick Analysis (Last 60 Days)

```bash
maintsight predict --window-size-days 60
```

### CI/CD Integration

```bash
# Fail if any file has severe degradation
maintsight predict --format json --threshold 0.2 || exit 1
```

## 🔍 Troubleshooting

### Common Issues

1. **"Not a git repository" error**
   - Ensure you're running MaintSight in a directory with a `.git` folder
   - Initialize git if needed: `git init`

2. **"Branch not found" error**
   - Check available branches: `git branch -a`
   - Use `--branch` to specify the correct branch

3. **No output or empty results**
   - Ensure the repository has commit history
   - Try increasing `--max-commits` for more data
   - Check that there are source code files (not just configs/docs)

4. **Model not found error**
   - Ensure the npm package was installed correctly
   - Check that `models/model.json` exists in the package

### Getting Help

- View command help: `maintsight --help`
- View subcommand help: `maintsight predict --help`
- Report issues: [GitHub Issues](https://github.com/floristafa/maintsight-pip/issues)
