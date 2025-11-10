# Standalone Scripts Guide

This document explains how to run the preprocessing and feature engineering steps independently.

## Overview

The workflow has been split into modular standalone scripts that can be run separately:

1. **`run_preprocessing.py`** - Runs data preprocessing only
2. **`run_feature_engineering.py`** - Runs feature engineering only
3. **`run.py`** - Runs the complete end-to-end workflow

## Why Use Standalone Scripts?

- **Faster iteration**: Skip preprocessing when experimenting with features
- **Debugging**: Isolate and test specific pipeline stages
- **Resource management**: Run expensive steps once, reuse results
- **Flexibility**: Mix and match different preprocessing/feature versions

## Usage

### Step 1: Preprocessing

Runs data loading, clase_ternaria generation, feature elimination, MICE imputation, and IPC inflation correction.

```bash
python run_preprocessing.py
```

**Input:** `data/competencia_02_target.parquet`  
**Output:** `data/preprocessed_data.parquet`

**What it does:**
1. Loads the raw dataset
2. Generates `clase_ternaria` (CONTINUA, BAJA+1, BAJA+2)
3. Eliminates unnecessary features (placeholder)
4. Fixes data quality issues in 202006 using MICE
5. Corrects inflation drift using IPC indicators

**Execution time:** ~5-10 minutes (depending on MICE iterations)

---

### Step 2: Feature Engineering

Runs intra-month features, historical lags/deltas/trends, and optional RF features.

```bash
python run_feature_engineering.py
```

**Input:** `data/preprocessed_data.parquet`  
**Output:** `data/featured_data.parquet`

**What it does:**
1. Loads preprocessed data
2. Adds intra-month features (kmes, ctrx_quarter_normalizado, mpayroll_sobre_edad)
3. Adds lag features (lag1, lag2)
4. Adds delta features (delta1, delta2)
5. Adds trend features (tend6, avg6, min6, max6, ratios)
6. Optionally adds Random Forest leaf features (commented out by default)

**Execution time:** ~10-20 minutes (depending on trend calculations)

**Note:** RF features are commented out by default. To enable, edit `run_feature_engineering.py` and uncomment:
```python
df = add_rf_features(df, PARAM, campos_buenos)
```

---

### Step 3: Complete Workflow

Runs everything end-to-end: preprocessing, feature engineering, training, optimization, and scoring.

```bash
python run.py
```

**Input:** `data/competencia_02_target.parquet` (or intermediate parquet files)  
**Output:** Multiple files in `output/{experimento}/` directory

**Smart Loading:**
- If `data/final_dataset.parquet` exists → loads it directly (skips all feature engineering)
- Else if `data/featured_data.parquet` exists → loads it and adds RF features
- Else → runs everything from scratch

**What it does:**
1. Loads data (using smart loading above)
2. Prepares training/testing data
3. Runs Bayesian Optimization with Optuna
4. Trains validation ensemble and analyzes gain on Optuna test set
5. Trains final ensemble of models
6. Scores future data
7. Generates Kaggle submissions
8. Creates gain curve analysis for future data

**Execution time:** 
- With `final_dataset.parquet`: ~1-2 hours
- With `featured_data.parquet`: ~1.5-2.5 hours (adds RF features)
- From scratch: ~2-3 hours

---

## Workflow Comparison

| Script | Preprocessing | Feature Eng. | Training | Optimization | Scoring |
|--------|--------------|--------------|----------|--------------|---------|
| `run_preprocessing.py` | ✅ | ❌ | ❌ | ❌ | ❌ |
| `run_feature_engineering.py` | ❌ | ✅ | ❌ | ❌ | ❌ |
| `run.py` | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## Example Workflows

### Scenario 1: First Time Setup
```bash
# Run everything once
python run.py
```

### Scenario 2: Fast Iteration (Using Cached Data)
```bash
# First run: Generate all intermediate files
python run.py  # Creates final_dataset.parquet

# Subsequent runs: Automatically uses cached data
python run.py  # Loads final_dataset.parquet instantly!
```

### Scenario 3: Experimenting with Features
```bash
# Build features once
python run_preprocessing.py
python run_feature_engineering.py

# Try different RF configurations
# Edit config.py, then run.py will load featured_data.parquet and regenerate RF features
python run.py
```

### Scenario 4: Debugging Preprocessing
```bash
# Test preprocessing changes
python run_preprocessing.py

# Verify output
python -c "import polars as pl; df = pl.read_parquet('data/preprocessed_data.parquet'); print(df.shape, df.columns)"
```

---

## Output Files

| File | Created By | Description |
|------|-----------|-------------|
| `data/preprocessed_data.parquet` | `run_preprocessing.py` | Cleaned data with clase_ternaria |
| `data/featured_data.parquet` | `run_feature_engineering.py` | Data with historical features |
| `data/final_dataset.parquet` | `run.py` | Complete dataset with RF features |
| `output/{experimento}/*` | `run.py` | Models, predictions, submissions |

---

## Configuration

Both standalone scripts use the configuration from `src/config.py`:

- **PARAM['semilla_primigenia']**: Random seed
- **PARAM['experimento']**: Experiment name/ID
- **PARAM['FE_hist']**: Feature engineering settings (lags, trends, windows)

Edit `src/config.py` to change these settings.

---

## Tips

1. **Save intermediate results**: The parquet files are your checkpoints
2. **Version control**: Keep different versions of preprocessed/featured data
3. **Disk space**: Parquet files are compressed but can still be large
4. **Memory**: Feature engineering can be memory-intensive with large datasets
5. **Reproducibility**: Use the same seed across runs for consistency

---

## Troubleshooting

**Error: File not found**
- Make sure you run scripts from the project root directory
- Check that input files exist in the `data/` folder

**Error: Memory issues**
- Reduce MICE iterations in `data_quality_fixes()`
- Reduce trend window size in config
- Process fewer columns at once

**Error: Module not found**
- Install dependencies: `pip install -r requirements.txt`
- Make sure you're in the correct Python environment

---

## Next Steps

After running these scripts, you can:
1. Use `featured_data.parquet` for training
2. Continue with the training steps from `run.py`
3. Build custom analysis scripts using the preprocessed data
4. Experiment with different feature combinations

For the complete workflow documentation, see `docs/README.md`.

