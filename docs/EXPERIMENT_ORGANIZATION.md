# Experiment Organization

## Overview

All experiment outputs are now organized by experiment name in separate directories. This prevents different experiments from overwriting each other's files and makes it easy to compare results.

## Directory Structure

When you run an experiment with `"experimento": "seg-001"`, the following structure is created:

```
output/
├── seg-001/                          # Experiment-specific directory
│   ├── BO_log.txt                   # Bayesian Optimization log
│   ├── impo_1.txt                   # Feature importance (best trial)
│   ├── impo_2.txt                   # Feature importance (if better)
│   ├── ...
│   ├── modelo.model                 # Random Forest model (if RF features used)
│   ├── prediccion.txt               # Final predictions
│   ├── predicciones_ensemble.parquet # Individual seed predictions
│   ├── modelitos/                   # Final ensemble models
│   │   ├── mod_123456.txt
│   │   ├── mod_234567.txt
│   │   └── ...
│   ├── validation/                  # Validation set analysis (NEW)
│   │   ├── predicciones_validation.parquet  # Predictions on Optuna test set
│   │   ├── gain_curve_202104.png            # Validation gain curve
│   │   └── gain_stats_202104.csv            # Validation statistics
│   └── analysis/                    # Future set gain analysis
│       ├── gain_curve_202106.png
│       ├── gain_curve_ensemble_202106.png
│       ├── gain_by_cutoff_202106.csv
│       └── gain_ensemble_stats.csv
├── seg-002/                          # Another experiment
│   └── ...
└── kaggle/                           # Shared Kaggle submissions
    ├── KAseg-001_202104_11000.csv
    ├── KAseg-001_202106_11000.csv
    ├── KAseg-002_202104_11000.csv
    └── ...
```

## Database Organization

All experiments share a single Optuna database with separate studies:

```
db/
└── optuna.db            # Shared database for all experiments
    ├── study: lgbm_seg-001
    ├── study: lgbm_seg-002
    └── study: lgbm_seg-003
```

**Benefits:**
- Easy to compare experiments in Optuna Dashboard
- All optimization history in one place
- Simpler to manage and backup

## Validation vs Future Analysis

Starting from the latest version, the workflow generates **two separate gain analyses**:

### 1. **Validation Analysis** (`output/{experimento}/validation/`)

- **Purpose**: Evaluate model performance on the **Optuna test set**
- **Training data**: Same as Optuna (e.g., up to 202102)
- **Test data**: Same as Optuna validation (e.g., 202104)
- **When**: After Bayesian Optimization, before final training
- **Why**: 
  - See how well the best hyperparameters generalize
  - Compare with Optuna's optimization metric
  - Validate that the ensemble performs as expected

**Files generated:**
- `predicciones_validation.parquet`: Individual model predictions
- `gain_curve_202104.png`: Gain curve visualization
- `gain_stats_202104.csv`: Statistics by cutoff

### 2. **Future Analysis** (`output/{experimento}/analysis/`)

- **Purpose**: Evaluate final ensemble on **true future data**
- **Training data**: Extended (e.g., up to 202104)
- **Test data**: Future month (e.g., 202106)
- **When**: After final ensemble training
- **Why**:
  - Final performance estimate for Kaggle submission
  - Determine optimal cutoff for submission
  - Assess ensemble stability

**Files generated:**
- `gain_curve_202106.png`: Individual gain curves
- `gain_curve_ensemble_202106.png`: Ensemble statistics
- `gain_by_cutoff_202106.csv`: Detailed cutoff analysis
- `gain_ensemble_stats.csv`: Summary statistics

### Why Both?

Having both analyses helps you:
1. **Validate hyperparameters**: Validation set confirms Optuna found good params
2. **Detect overfitting**: Compare validation vs future performance
3. **Choose cutoffs wisely**: Use future analysis for final submission decisions

## Benefits

### 1. **No File Conflicts**
Different experiments don't overwrite each other's outputs.

### 2. **Easy Comparison**
Compare results from different experiments side-by-side:
```bash
# Compare gain curves
open output/seg-001/analysis/gain_curve.png
open output/seg-002/analysis/gain_curve.png

# Compare feature importance
diff output/seg-001/impo_1.txt output/seg-002/impo_1.txt
```

### 3. **Clean Restarts**
Start a new experiment from scratch by changing the experiment name:
```python
# In config.py
"experimento": "seg-002"  # New experiment, fresh start
```

### 4. **Reproducibility**
All files for an experiment are in one place, making it easy to archive or share.

## Running Multiple Experiments

### Sequential Experiments
Just change the experiment name in `config.py`:

```python
# First experiment
"experimento": "seg-001"
# Run workflow...

# Second experiment
"experimento": "seg-002"
# Run workflow again...
```

### Continuing an Experiment
If you use the same experiment name, Optuna will continue from where it left off:
- Existing trials are loaded
- New trials are added to the study
- Output files are updated/overwritten

### Fresh Start
To start completely fresh:
1. Change experiment name (creates new study in shared DB), OR
2. Delete the experiment directory:
   ```bash
   rm -rf output/seg-001
   ```
   
Note: Optuna studies remain in the shared database. To delete a specific study:
```bash
# Using Optuna CLI
optuna delete-study --study-name lgbm_seg-001 --storage sqlite:///db/optuna.db
```

Or view/delete via Optuna Dashboard:
```bash
optuna-dashboard sqlite:///db/optuna.db
```

## Shared Resources

Some directories remain shared across experiments:

- **`output/kaggle/`**: Submission files (filename includes experiment name)
- **`logs/`**: Run logs (filename includes timestamp)
- **`data/`**: Input data (shared across all experiments)

## Migration from Old Structure

If you have outputs from before this change (without experiment directories), they will be in the root `output/` directory. 

### Automatic Migration (PowerShell)

To organize existing files under your current experiment name:

```powershell
cd output

# Get experiment name from config (or set manually)
$experimento = "seg-001"  # Change to your experiment name

# Create directory structure
mkdir -p $experimento/analysis

# Move files
if (Test-Path BO_log.txt) { Move-Item BO_log.txt $experimento/ -Force }
if (Test-Path modelo.model) { Move-Item modelo.model $experimento/ -Force }
if (Test-Path prediccion.txt) { Move-Item prediccion.txt $experimento/ -Force }
Move-Item impo_*.txt $experimento/ -Force -ErrorAction SilentlyContinue
Move-Item analysis/* $experimento/analysis/ -Force -ErrorAction SilentlyContinue
if (Test-Path modelitos) { Move-Item modelitos $experimento/ -Force }

# Clean up empty directories
Remove-Item analysis -Force -ErrorAction SilentlyContinue
```

### Manual Migration (Bash/Linux)

```bash
cd output

# Get experiment name
experimento="seg-001"  # Change to your experiment name

# Create directory structure
mkdir -p $experimento/analysis

# Move files
[ -f BO_log.txt ] && mv BO_log.txt $experimento/
[ -f modelo.model ] && mv modelo.model $experimento/
[ -f prediccion.txt ] && mv prediccion.txt $experimento/
mv impo_*.txt $experimento/ 2>/dev/null
mv analysis/* $experimento/analysis/ 2>/dev/null
[ -d modelitos ] && mv modelitos $experimento/

# Clean up
rmdir analysis 2>/dev/null
```

## Implementation Details

All experiment-specific paths are constructed using:
```python
experimento = config["experimento"]
output_path = f"output/{experimento}/..."
```

This ensures consistency across all modules:
- `src/training.py`: BO logs, feature importance, modelitos
- `src/rf_features.py`: Random Forest model
- `src/scoring.py`: Predictions, ensemble predictions
- `src/gain_analysis.py`: Gain curves and statistics

