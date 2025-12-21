# MaintSight Integration Guide

This guide explains how to integrate MaintSight into your projects, CI/CD pipelines, and development workflows.

## 📚 Table of Contents

- [**CI/CD Integration**](#-cicd-integration)
  - [GitHub Actions](#github-actions)
  - [GitLab CI](#gitlab-ci)
  - [Jenkins](#jenkins)
- [**NPM Scripts**](#-npm-scripts)
- [**Pre-commit Hooks**](#-pre-commit-hooks)
- [**Programmatic Usage**](#-programmatic-usage)
- [**Monitoring & Alerts**](#-monitoring--alerts)

## 🔄 CI/CD Integration

### GitHub Actions

Create a workflow to check maintenance risk on pull requests:

```yaml
name: Maintenance Risk Check

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  risk-check:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0 # Full history needed for analysis

      - uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install MaintSight
        run: pip install maintsight

      - name: Run risk analysis
        run: |
          maintsight predict --format markdown > risk-report.md
          maintsight stats

      - name: Comment PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('risk-report.md', 'utf8');

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: report
            });

      - name: Fail if severe degradation
        run: |
          severe=$(maintsight predict -f json | jq '[.[] | select(.normalized_score > 0.3)] | length')
          if [ "$severe" -gt "0" ]; then
            echo "❌ Found $severe files with severe degradation!"
            exit 1
          fi
```

### GitLab CI

Add to `.gitlab-ci.yml`:

```yaml
maintenance-check:
  stage: test
  image: python:3.9
  before_script:
    - pip install maintsight
  script:
    - maintsight predict --threshold 0.1 --format markdown > risk-report.md
  artifacts:
    reports:
      junit: risk-report.md
  only:
    - merge_requests
    - main
```

### Jenkins

Add to your Jenkinsfile:

```groovy
pipeline {
  agent any

  stages {
    stage('Maintenance Check') {
      steps {
        sh 'pip install maintsight'
        sh 'maintsight predict --format json > risk-report.json'

        script {
          def report = readJSON file: 'risk-report.json'
          def degraded = report.findAll { it.normalized_score > 0.1 }

          if (degraded.size() > 0) {
            echo "⚠️  Found ${degraded.size()} degraded files"
            currentBuild.result = 'UNSTABLE'
          }
        }
      }
    }
  }

  post {
    always {
      archiveArtifacts artifacts: 'risk-report.json'
    }
  }
}
```

## 📦 NPM Scripts

Add MaintSight to your package.json scripts:

```json
{
  "scripts": {
    "maintenance:check": "maintsight predict",
    "maintenance:report": "maintsight predict -f markdown -o maintenance-report.md",
    "maintenance:stats": "maintsight stats",
    "maintenance:degraded": "maintsight predict -t 0.1 -f json",
    "precommit": "maintsight predict -t 0.2 || echo 'Warning: Severely degraded files detected'"
  }
}
```

## 🪝 Pre-commit Hooks

Using husky to check maintenance risk before commits:

```bash
npm install --save-dev husky
npx husky install
```

Create `.husky/pre-commit`:

```bash
#!/usr/bin/env sh
. "$(dirname -- "$0")/_/husky.sh"

# Check for severely degraded files
severe=$(npx maintsight predict -f json -t 0.2 2>/dev/null | jq length)

if [ "$severe" -gt "0" ]; then
  echo "⚠️  Warning: $severe files are severely degraded (>0.2)"
  echo "Run 'maintsight predict -t 0.2' to see details"
  # Uncomment to block commit:
  # exit 1
fi
```

## 💻 Programmatic Usage

Integrate MaintSight into your build tools or scripts:

```javascript
const { GitCommitCollector, XGBoostPredictor } = require('maintsight');
const fs = require('fs').promises;

async function checkMaintenanceDegradation() {
  const predictor = new XGBoostPredictor();
  predictor.loadModel(); // Uses built-in model

  const collector = new GitCommitCollector(process.cwd(), 'main', 150, true);
  const commitData = collector.fetchCommitData(10000);

  const predictions = predictor.predict(commitData);
  const degraded = predictions.filter((p) => p.normalized_score > 0.1);

  if (degraded.length > 0) {
    console.warn(`⚠️  ${degraded.length} degraded files detected`);

    // Save report
    await fs.writeFile('degraded-files.json', JSON.stringify(degraded, null, 2));

    // Optionally fail the build
    if (degraded.some((f) => f.normalized_score > 0.3)) {
      process.exit(1);
    }
  }
}

checkMaintenanceDegradation().catch(console.error);
```

## 📊 Monitoring & Alerts

### Slack Integration

Send alerts for degraded files:

```javascript
const axios = require('axios');

async function sendSlackAlert(degradedFiles) {
  const webhook = process.env.SLACK_WEBHOOK_URL;

  await axios.post(webhook, {
    text: `⚠️ Code Degradation Alert`,
    attachments: [
      {
        color: 'warning',
        fields: degradedFiles.slice(0, 5).map((f) => ({
          title: f.module,
          value: `Degradation: ${f.normalized_score.toFixed(3)}`,
          short: true,
        })),
      },
    ],
  });
}
```

### Dashboard Integration

Track trends over time:

```bash
#!/bin/bash
# Run daily and save results
DATE=$(date +%Y-%m-%d)
maintsight predict -f json > "reports/degradation-${DATE}.json"

# Generate trend data
jq -s '[.[] | {
  date: input_filename | split("/")[-1] | split(".")[0] | split("-")[1:4] | join("-"),
  degraded: [.[] | select(.normalized_score > 0.1)] | length,
  severely_degraded: [.[] | select(.normalized_score > 0.2)] | length,
  total: length
}]' reports/degradation-*.json > trend-data.json
```

## Best Practices

1. **Regular Analysis**: Run MaintSight weekly or on each release
2. **Track Trends**: Monitor degradation scores over time
3. **Set Thresholds**: Define acceptable degradation levels for your project
4. **Prioritize Refactoring**: Focus on files with consistent degradation
5. **Combine with Code Review**: Use degradation scores to guide review efforts
6. **Use Interactive Reports**: Leverage HTML reports for team discussions
