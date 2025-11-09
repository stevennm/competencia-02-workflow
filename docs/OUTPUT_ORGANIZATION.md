# Output Folder Organization

All model outputs are now saved to the `output/` directory for better organization.

## Directory Structure

```
output/
├── modelo.model              # Random Forest model for leaf features
├── BO_log.txt               # Bayesian Optimization trial log
├── prediccion.txt           # Full predictions on future data
├── impo_*.txt               # Feature importance files (best iterations)
├── modelitos/               # Final ensemble models
│   ├── mod_123456.txt
│   ├── mod_234567.txt
│   └── ... (30 models)
└── kaggle/                  # Kaggle submissions
    └── KAseg-001_11000.csv
```

## Files Modified

1. **src/rf_features.py** - RF model saved to `output/modelo.model`
2. **src/training.py** - Log, importance files, and models saved to `output/`
3. **src/scoring.py** - Predictions and submissions saved to `output/`
4. **run.py** - Updated final output messages
5. **.gitignore** - Added `output/` to ignore list

## Clean Workflow

The main directory stays clean - only source code and data. All generated files go to `output/` which is gitignored by default.

