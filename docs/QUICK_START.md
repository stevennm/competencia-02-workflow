# Quick Start Guide

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify the data file exists:**
   - Ensure `data/competencia_02_target.parquet` is present
   - This file should contain the competition dataset

## Running the Workflow

### Full Workflow
To run the complete pipeline (preprocessing → feature engineering → optimization → training → scoring):

```bash
python run.py
```

**Expected output files:**
- `BO_log.txt` - Bayesian Optimization log with all trials
- `prediccion.txt` - Full predictions on future data (202108)
- `kaggle/KAseg-001_11000.csv` - Kaggle submission file
- `modelitos/mod_*.txt` - 30 trained LightGBM models
- `impo_*.txt` - Feature importance files for best iterations

### Estimated Runtime
- **Full workflow**: 2-6 hours (depends on hardware)
- **Preprocessing**: 5-10 minutes
- **Feature Engineering**: 10-30 minutes
- **RF Features**: 5-15 minutes
- **Bayesian Optimization**: 1-3 hours (30 trials)
- **Final Training**: 30-60 minutes (30 models)
- **Scoring**: 2-5 minutes

## Configuration

Edit `src/config.py` to customize:

```python
PARAM = {
    "experimento": "seg-001",           # Experiment name
    "semilla_primigenia": 102191,       # Random seed
    
    # For faster testing, reduce these:
    "hipeparametertuning": {
        "BO_iteraciones": 30,           # Try 10 for testing
    },
    "train_final": {
        "ksemillerio": 30,              # Try 5 for testing
    }
}
```

## Testing with Reduced Configuration

For quick testing (10-30 minutes), modify `src/config.py`:

```python
# Reduce Bayesian Optimization iterations
PARAM["hipeparametertuning"]["BO_iteraciones"] = 5

# Reduce final ensemble size
PARAM["train_final"]["ksemillerio"] = 3

# Reduce RF features
PARAM["FE_rf"]["arbolitos"] = 10
```

## Monitoring Progress

The workflow provides detailed console output:

```
==================================================================
  STEP 1: PREPROCESSING
==================================================================
Loading dataset from data/competencia_02_target.parquet...
Dataset loaded: 500000 rows, 150 columns
...
```

**Logs to watch:**
- Console output shows current step and progress bars
- `BO_log.txt` updates after each Optuna trial
- Check `impo_*.txt` files to see which features are important

## Troubleshooting

### Memory Errors
If you encounter memory errors:
1. Close other applications
2. Reduce training data size in `src/config.py`:
   ```python
   PARAM["trainingstrategy"]["undersampling"] = 0.02  # Lower from 0.05
   ```

### Slow Execution
If running too slow:
1. Reduce Bayesian Optimization trials (see above)
2. Reduce final ensemble size
3. Disable trend calculation:
   ```python
   PARAM["FE_hist"]["Tendencias"]["run"] = False
   ```

### Import Errors
Ensure all dependencies are installed:
```bash
pip install --upgrade polars numpy lightgbm optuna tqdm
```

## Understanding the Output

### BO_log.txt
Contains optimization history:
- `iter`: Trial number
- `num_iterations`, `learning_rate`, etc.: Hyperparameters tried
- `metrica`: Gain achieved in this trial
- `metrica_mejor`: Best gain so far

### Kaggle Submission
The file `kaggle/KA{experiment}_{n_envios}.csv` contains:
- `numero_de_cliente`: Customer ID
- `Predicted`: 1 for top 11,000 predictions, 0 otherwise

### Feature Importance
Files `impo_*.txt` show which features contributed most to the best models.

## Next Steps

1. **Run the workflow** with default settings
2. **Analyze results** in `BO_log.txt`
3. **Implement TODO functions** in `src/preprocessing.py`:
   - Feature elimination
   - Data quality fixes
   - Data drifting correction
4. **Re-run** with improved preprocessing
5. **Submit** to Kaggle: `kaggle/KA{experiment}_11000.csv`

## Support

Check the following files for more information:
- `README.md` - Full documentation
- `IMPLEMENTATION_SUMMARY.md` - Implementation details
- `docs/workflow-jueves.ipynb` - Original R notebook (reference)

