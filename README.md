# LightGBM Workflow with Bayesian Optimization

Python implementation of the LightGBM workflow using Polars for data handling and Optuna for Bayesian Optimization.

## Project Structure

```
.
├── run.py                      # Main orchestrator script
├── requirements.txt            # Python dependencies
├── data/
│   └── competencia_02_target.parquet  # Input dataset
├── src/
│   ├── config.py              # Configuration and parameters
│   ├── preprocessing.py       # Data loading and clase_ternaria generation
│   ├── feature_engineering.py # Intra-month and historical features
│   ├── rf_features.py         # Random Forest leaf features
│   ├── training.py            # Bayesian Optimization and model training
│   └── scoring.py             # Scoring and submission generation
├── modelitos/                 # Trained models (generated)
├── kaggle/                    # Submission files (generated)
└── docs/
    └── workflow-jueves.ipynb  # Original R notebook (reference)
```

## Installation

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the complete workflow:

```bash
python run.py
```

This will execute all steps:
1. Load and preprocess data
2. Generate clase_ternaria (CONTINUA, BAJA+1, BAJA+2)
3. Add intra-month features
4. Add historical features (lags, deltas, trends)
5. Generate Random Forest leaf features
6. Run Bayesian Optimization with Optuna
7. Train final ensemble of models
8. Score future data
9. Generate Kaggle submission

## Configuration

Edit `src/config.py` to modify:
- Experiment name and seed
- Training/testing months
- Undersampling rates
- Number of Bayesian Optimization iterations
- LightGBM parameters
- Final ensemble size

## Output Files

- `BO_log.txt` - Log of Bayesian Optimization trials
- `prediccion.txt` - Full predictions on future data
- `kaggle/KA{experiment}_{n_envios}.csv` - Kaggle submission file
- `modelitos/mod_*.txt` - Trained LightGBM models
- `impo_*.txt` - Feature importance for best iterations

## Placeholder Functions

The following functions are marked as TODO and should be implemented based on exploratory data analysis:

1. **Feature Elimination** (`src/preprocessing.py`):
   - Remove features that are not useful for the model
   
2. **Data Quality Fixes** (`src/preprocessing.py`):
   - Repair attributes where all values are zero for certain months
   - Options: leave as is, replace with NA, interpolate, or use MICE
   
3. **Data Drifting Correction** (`src/preprocessing.py`):
   - Correct for inflation effects on monetary values
   - Options: IPC, Dollar rates, UVA adjustment

## Key Features

- **Polars** for fast data handling (no Pandas)
- **LightGBM** for gradient boosting
- **Optuna** for Bayesian Optimization
- **Ensemble of 30 models** for final predictions
- **Custom gain metric** with smoothed plateau (meseta)
- **Random Forest leaf features** for additional signal
- **Historical features**: lags, deltas, and trends over 6-month windows

## Notes

- The workflow is designed for the second competition dataset
- Target is clase_ternaria: CONTINUA, BAJA+1, BAJA+2
- Final submission selects top 11,000 predictions
- Execution time depends on hardware and configuration

