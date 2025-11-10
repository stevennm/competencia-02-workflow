"""
Scoring and submission generation module
"""

import polars as pl
import numpy as np
import lightgbm as lgb
from typing import Dict, List
import os
from glob import glob


def score_future_data(df: pl.DataFrame, config: Dict, campos_buenos: List[str]) -> pl.DataFrame:
    """
    Apply trained models to future data
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        campos_buenos: List of feature columns
        
    Returns:
        DataFrame with predictions
    """
    print("\n" + "="*50)
    print("SCORING FUTURE DATA")
    print("="*50)
    
    # Filter to future month
    future_months = config["train_final"]["future"]
    df_future = df.filter(pl.col("foto_mes").is_in(future_months))
    
    print(f"Future data: {df_future.shape[0]} rows")
    
    # Get valid feature columns
    campos_buenos_valid = [col for col in campos_buenos 
                           if col in df_future.columns and 
                           col not in ["clase_ternaria", "numero_de_cliente", "foto_mes"]]
    
    # Prepare feature matrix
    X_future = df_future.select(campos_buenos_valid).to_numpy()
    
    # Load all models and make predictions
    model_files = sorted(glob("output/modelitos/mod_*.txt"))
    
    if len(model_files) == 0:
        raise FileNotFoundError("No models found in output/modelitos/ directory!")
    
    print(f"Loading {len(model_files)} models...")
    
    # Store individual predictions for ensemble analysis
    all_predictions = []
    predictions_sum = np.zeros(X_future.shape[0])
    n_models = 0
    
    for model_file in model_files:
        modelo = lgb.Booster(model_file=model_file)
        predictions = modelo.predict(X_future)
        all_predictions.append(predictions)
        predictions_sum += predictions
        n_models += 1
    
    # Average predictions
    predictions_avg = predictions_sum / n_models
    
    print(f"Predictions complete using {n_models} models")
    
    # Create prediction dataframe
    df_pred = df_future.select(["numero_de_cliente", "foto_mes"]).with_columns([
        pl.Series("prob", predictions_avg)
    ])
    
    # Save predictions
    os.makedirs("output", exist_ok=True)
    print("Saving predictions to output/prediccion.txt...")
    df_pred.write_csv("output/prediccion.txt", separator="\t")
    
    # Save individual predictions for ensemble analysis
    print("Saving individual model predictions...")
    df_individual = df_future.select(["numero_de_cliente", "foto_mes"])
    for i, preds in enumerate(all_predictions):
        df_individual = df_individual.with_columns([
            pl.Series(f"prob_seed_{i+1}", preds)
        ])
    df_individual.write_parquet("output/predicciones_ensemble.parquet")
    print("✓ Individual predictions saved to output/predicciones_ensemble.parquet")
    
    return df_pred


def generate_submission(df_pred: pl.DataFrame, config: Dict, n_envios: int = 11000) -> None:
    """
    Generate Kaggle submission file
    
    Args:
        df_pred: DataFrame with predictions
        config: Configuration dictionary
        n_envios: Number of top predictions to submit (default 11000)
    """
    print("\n" + "="*50)
    print("GENERATING KAGGLE SUBMISSION")
    print("="*50)
    
    # Sort by probability descending
    df_pred = df_pred.sort("prob", descending=True)
    
    # Create Predicted column
    df_pred = df_pred.with_columns([
        pl.lit(0).alias("Predicted")
    ])
    
    # Mark top n_envios as 1
    df_pred = df_pred.with_row_count(name="row_num")
    df_pred = df_pred.with_columns([
        pl.when(pl.col("row_num") < n_envios)
        .then(pl.lit(1))
        .otherwise(pl.col("Predicted"))
        .alias("Predicted")
    ])
    
    # Create kaggle directory
    os.makedirs("output/kaggle", exist_ok=True)
    
    # Generate filename
    experimento = config["experimento"]
    submission_file = f"output/kaggle/KA{experimento}_{n_envios}.csv"
    
    # Save submission
    print(f"Saving submission to {submission_file}...")
    df_submission = df_pred.select(["numero_de_cliente", "Predicted"])
    df_submission.write_csv(submission_file)
    
    # Print statistics
    n_predicted = df_pred.filter(pl.col("Predicted") == 1).shape[0]
    print(f"\nSubmission statistics:")
    print(f"  Total records: {df_pred.shape[0]}")
    print(f"  Predicted as positive: {n_predicted}")
    print(f"  Top probability: {df_pred.select('prob').head(1).item():.6f}")
    print(f"  Probability at cutoff: {df_pred.filter(pl.col('row_num') == n_envios - 1).select('prob').item():.6f}")
    
    print(f"\n✓ Submission saved to {submission_file}")

