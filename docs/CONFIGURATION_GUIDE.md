# MaintSight Configuration Guide

This guide covers all configuration options available in MaintSight.

## 📚 Table of Contents

- [**Environment Variables**](#-environment-variables)
- [**Output Configuration**](#-output-configuration)
- [**Model Configuration**](#-model-configuration)
- [**Logging Configuration**](#-logging-configuration)
- [**Data Storage**](#-data-storage)

## 🔧 Environment Variables

MaintSight uses environment variables for configuration that should not be hardcoded:

### `MAINTSIGHT_MODEL_PATH`

Specifies a custom path to the XGBoost model file.

```bash
export MAINTSIGHT_MODEL_PATH=/custom/path/to/model.json
```

**Default**: `<package_root>/models/model.json`

### `MAINTSIGHT_LOG_LEVEL`

Controls the verbosity of logging output.

```bash
export MAINTSIGHT_LOG_LEVEL=DEBUG
```

**Valid values**:

- `ERROR`: Only show errors
- `WARN`: Show warnings and errors
- `INFO`: Show informational messages (default)
- `DEBUG`: Show detailed debug information

### `MAINTSIGHT_DATA_DIR`

Directory where HTML reports are saved.

```bash
export MAINTSIGHT_DATA_DIR=/path/to/data
```

**Default**: `<repository>/.maintsight`

## 📁 Output Configuration

### Output Formats

MaintSight supports multiple output formats:

- **JSON**: Machine-readable format, ideal for integration
- **CSV**: Spreadsheet-compatible format, good for analysis
- **Markdown**: Human-readable format, perfect for reports
- **HTML**: Interactive format with visualizations and detailed analysis

### Output Location

By default, MaintSight outputs to stdout. Use `-o` or `--output` to save to a file:

```bash
maintsight predict -o results.json
```

## 🤖 Model Configuration

### Model Features

The XGBoost model uses 16 features extracted from git history:

1. **Total commits**: Number of commits to the file
2. **Unique authors**: Number of different contributors
3. **Lines added/removed**: Code churn metrics
4. **File age**: Days since first commit
5. **Days since last update**: Staleness indicator
6. **Bug fix commits**: Number of commits fixing bugs
7. **Average commit size**: Typical change magnitude
8. **Commit frequency**: Changes per time period
9. **Author concentration**: How concentrated changes are
10. **Recent activity**: Recent change patterns
11. **Collaboration metrics**: Multi-author patterns
12. **Change patterns**: Consistency of modifications
13. **Bug density**: Bug fixes relative to total commits
14. **Refactoring indicators**: Large-scale changes
15. **Code complexity proxy**: Inferred from change patterns
16. **Maintenance burden**: Combined metric

### Degradation Thresholds

The model categorizes files based on degradation score:

```
Improved:           < 0.0
Stable:           0.0 - 0.1
Degraded:         0.1 - 0.2
Severely Degraded:  > 0.2
```

## 📊 Logging Configuration

### Log Format

Logs follow this format:

```
[timestamp] [level] [component] message
```

Example:

```
[2024-01-15T10:30:45.123Z] [INFO] [GitCommitCollector] 🔄 Fetching commits from repository
```

### Verbose Mode

Enable verbose mode for detailed analysis information:

```bash
maintsight predict --verbose
```

This shows:

- Feature extraction details
- Model prediction steps
- Performance metrics
- Detailed error messages

## 💾 Data Storage

### Automatic Report Generation

MaintSight automatically generates interactive HTML reports in your repository:

```
my-project/
└── .maintsight/
    └── report.html
```

### HTML Report Features

Interactive reports contain:

- **Visual charts**: Degradation distribution and trends
- **File explorer**: Drill-down analysis by file
- **Metrics dashboard**: Detailed statistics per file
- **Export options**: Download data in various formats
- **Responsive design**: Works on desktop and mobile

### Accessing Reports

Reports are automatically opened after analysis or can be accessed manually:

```bash
# Report location
open .maintsight/report.html

# Or copy the file:// URL shown in terminal
```

## 🔄 CI/CD Configuration

### GitHub Actions Example

```yaml
name: Maintenance Risk Check
on: [push, pull_request]

jobs:
  risk-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0 # Full history needed

      - uses: actions/setup-node@v3
        with:
          python-version: '3.9'

      - run: pip install maintsight

      - name: Check maintenance risk
        run: |
          maintsight predict --format json --output risk-report.json
          maintsight stats

      - name: Fail if severe degradation
        run: |
          severe=$(jq '[.[] | select(.degradation_score > 0.2)] | length' risk-report.json)
          if [ "$severe" -gt "0" ]; then
            echo "Found $severe severely degraded files!"
            exit 1
          fi
```

### GitLab CI Example

```yaml
maintenance-check:
  image: python:3.9
  script:
    - pip install maintsight
    - maintsight predict --threshold 0.1 --format markdown
  artifacts:
    reports:
      - risk-report.md
```
