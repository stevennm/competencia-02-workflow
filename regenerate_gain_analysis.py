"""
Script to regenerate gain analysis for semillerio_optuna experiment
"""

import polars as pl
from src.config import PARAM
from src.gain_analysis import create_gain_curve, create_ensemble_gain_curve

# Override experiment name to use semillerio_optuna
PARAM_ANALYSIS = PARAM.copy()
PARAM_ANALYSIS["experimento"] = "semillerio_optuna"

print("="*70)
print("REGENERATING GAIN ANALYSIS FOR semillerio_optuna")
print("="*70)

# Load data
print("\nLoading data...")
df = pl.read_parquet("data/final_dataset.parquet")
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Load predictions
print("\nLoading predictions...")
df_pred = pl.read_csv("output/semillerio_optuna/prediccion.txt", separator="\t")
print(f"Predictions loaded: {df_pred.shape[0]} rows")

# Check ensemble predictions
print("\nChecking ensemble predictions...")
df_ensemble = pl.read_parquet("output/semillerio_optuna/predicciones_ensemble.parquet")
prob_cols = [c for c in df_ensemble.columns if c.startswith("prob_seed_")]
print(f"Found {len(prob_cols)} individual seed predictions")
print(f"Columns: {df_ensemble.columns}")

# Regenerate gain curve
print("\n" + "="*70)
print("STEP 1: Single model gain curve")
print("="*70)
create_gain_curve(df, df_pred, PARAM_ANALYSIS)

# Regenerate ensemble gain curve
print("\n" + "="*70)
print("STEP 2: Ensemble gain curve")
print("="*70)
create_ensemble_gain_curve(df, PARAM_ANALYSIS)

print("\n" + "="*70)
print("GAIN ANALYSIS REGENERATION COMPLETE")
print("="*70)
print("\nCheck the following files:")
print("  - output/semillerio_optuna/analysis/gain_curve.png")
print("  - output/semillerio_optuna/analysis/gain_curve_ensemble.png")
print("  - output/semillerio_optuna/analysis/gain_by_cutoff.csv")
print("  - output/semillerio_optuna/analysis/gain_ensemble_stats.csv")

