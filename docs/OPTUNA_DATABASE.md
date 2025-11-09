# Optuna Database Storage

## Implementation

Optuna optimization results are now saved to a SQLite database in the `db/` directory.

## Features

### 1. Persistent Storage
- Database: `db/optuna_{experimento}.db`
- Study name: `lgbm_{experimento}`
- All trials are saved automatically

### 2. Resume Capability
If the script is interrupted or you want to add more trials:
- Automatically detects existing study
- Loads completed trials
- Only runs remaining trials

**Example:**
```
First run (30 trials configured):
  Created new study: lgbm_seg-001
  Running 30 remaining trials (out of 30 total)...

Second run (interrupted at trial 15):
  Loaded existing study: 15 trials already completed
  Running 15 remaining trials (out of 30 total)...

Third run (add more trials, change config to 50):
  Loaded existing study: 30 trials already completed
  Running 20 remaining trials (out of 50 total)...
```

### 3. Database Benefits

**Query trials:**
```python
import optuna
study = optuna.load_study(
    study_name="lgbm_seg-001",
    storage="sqlite:///db/optuna_seg-001.db"
)

# Best trial
print(study.best_trial)
print(study.best_params)
print(study.best_value)

# All trials
df_trials = study.trials_dataframe()
print(df_trials.head())

# Visualize
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
```

**Analyze externally:**
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('db/optuna_seg-001.db')
trials = pd.read_sql("SELECT * FROM trials", conn)
print(trials.head())
```

### 4. Multiple Experiments

Each experiment gets its own database:
```
db/
├── optuna_seg-001.db  # Experiment seg-001
├── optuna_seg-002.db  # Experiment seg-002
└── optuna_seg-003.db  # Experiment seg-003
```

## Configuration

In `src/config.py`:
```python
PARAM = {
    "experimento": "seg-001",  # Used for DB filename
    "hipeparametertuning": {
        "BO_iteraciones": 30,   # Number of trials
    }
}
```

## Output

```
==================================================
RUNNING BAYESIAN OPTIMIZATION
==================================================
Study database: sqlite:///db/optuna_seg-001.db
Created new study: lgbm_seg-001
Running 30 remaining trials (out of 30 total)...
[Progress bar...]

Best gain: $45,678,000
Best parameters:
  num_iterations: 512
  learning_rate: 0.0234
  feature_fraction: 0.7823
  min_data_in_leaf: 234
  num_leaves: 89
```

## Resuming After Interruption

Just run the script again! It will:
1. Detect the existing database
2. Load completed trials
3. Continue from where it left off

No need to start from scratch!

## Advantages

1. **No data loss**: Trials are saved immediately
2. **Resume anytime**: Interrupted? Just restart
3. **Add more trials**: Increase `BO_iteraciones` and rerun
4. **Analysis**: Query database for deeper insights
5. **Reproducibility**: All trial history preserved
6. **Multiple experiments**: Compare different runs

## Technical Details

- **Database**: SQLite (no server needed)
- **Location**: `db/optuna_{experimento}.db`
- **Format**: Standard Optuna schema
- **Size**: ~10-50 KB per trial (very small)

## Cleanup

To start fresh, simply delete the database:
```bash
rm db/optuna_seg-001.db
```

Or delete entire db directory:
```bash
rm -rf db/
```

The script will create a new study on next run.

