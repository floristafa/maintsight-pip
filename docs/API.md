# MaintSight API Reference

MaintSight provides a Python API for integrating maintenance degradation prediction into your applications.

## Installation

```bash
pip install maintsight
```

## Basic Usage

```python
from services import GitCommitCollector, FeatureEngineer, XGBoostPredictor

def analyze_maintenance(repo_path: str):
    # Load the model
    predictor = XGBoostPredictor()
    predictor.load_model()  # Uses built-in model

    # Collect git data
    collector = GitCommitCollector(repo_path, 'main', 150, True)
    commit_data = collector.fetch_commit_data(10000)

    # Run predictions
    predictions = predictor.predict(commit_data)

    return predictions
```

## API Classes

### GitCommitCollector

Collects and processes git commit history.

```python
class GitCommitCollector:
    def __init__(
        self,
        repo_path: str,
        branch: str = 'main',
        window_size_days: int = 150,
        only_existing_files: bool = True,
    ):
        ...

    def fetch_commit_data(self, max_commits: int = 10000) -> List[CommitData]:
        ...
```

### FeatureEngineer

Transforms commit data into ML features (used internally by XGBoostPredictor).

```typescript
class FeatureEngineer {
  generateFeatures(fileCommitData: FileCommitData[]): Features[];
}
```

### XGBoostPredictor

Loads the model and makes predictions.

```typescript
class XGBoostPredictor {
  loadModel(modelPath?: string): void; // Uses built-in model if no path provided

  predict(fileCommitData: FileCommitData[]): RiskPrediction[];

  predictSingle(features: CommitFeatures): RiskPrediction;
}
```

## Types

```typescript
interface FileCommitData {
  file: string;
  commits: CommitInfo[];
}

interface CommitInfo {
  hash: string;
  author: string;
  date: Date;
  message: string;
  linesAdded: number;
  linesDeleted: number;
}

interface RiskPrediction {
  module: string;
  degradation_score: number;
  raw_prediction: number;
  risk_category: 'improved' | 'stable' | 'degraded' | 'severely_degraded';
}

interface Features {
  module: string;
  total_commits: number;
  unique_authors: number;
  total_lines_added: number;
  total_lines_removed: number;
  file_age_days: number;
  days_since_last_update: number;
  bug_fix_commits: number;
  avg_commit_size: number;
  commit_frequency: number;
  author_concentration: number;
  recent_activity_score: number;
  collaboration_factor: number;
  change_consistency: number;
  bug_density: number;
  refactoring_score: number;
  maintenance_burden: number;
}
```

## Example: Custom Integration

```typescript
import { GitCommitCollector, XGBoostPredictor } from 'maintsight';
import * as fs from 'fs/promises';

async function analyzeAndSaveResults(repoPath: string, outputPath: string) {
  try {
    // Initialize predictor
    const predictor = new XGBoostPredictor();
    predictor.loadModel(); // Uses built-in model

    // Collect commit data
    const collector = new GitCommitCollector(repoPath, 'main', 150, true);
    const commitData = collector.fetchCommitData(10000);

    if (commitData.length === 0) {
      throw new Error('No source files found in repository');
    }

    // Get predictions
    const predictions = predictor.predict(commitData);

    // Filter degraded files
    const degradedFiles = predictions.filter((p) => p.degradation_score > 0.1);

    // Save results
    await fs.writeFile(outputPath, JSON.stringify(degradedFiles, null, 2), 'utf-8');

    console.log(`Found ${degradedFiles.length} degraded files`);

    return degradedFiles;
  } catch (error) {
    console.error('Analysis failed:', error);
    throw error;
  }
}
```

## Error Handling

All API methods throw errors with descriptive messages:

```typescript
try {
  const predictor = new XGBoostPredictor();
  await predictor.loadModel('invalid-path.json');
} catch (error) {
  // Error: Model file not found: invalid-path.json
}
```

## Environment Variables

- `MAINTSIGHT_MODEL_PATH`: Override default model path
- `MAINTSIGHT_LOG_LEVEL`: Set logging verbosity (ERROR, WARN, INFO, DEBUG)
