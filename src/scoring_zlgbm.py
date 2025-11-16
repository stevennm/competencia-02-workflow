"""
Scoring module for zLightGBM models
"""

import polars as pl
import numpy as np
import lightgbm as lgb
from typing import Dict, List
from pathlib import Path


def score_zlgbm_future_data(df: pl.DataFrame, config: Dict, campos_buenos: List[str]) -> pl.DataFrame:
    """
    Score future data using zLightGBM model(s)
    
    If ksemillerio > 1, loads multiple models and averages their predictions (ensemble)
    
    IMPORTANT: Future data must also have canaries at the beginning
    
    Args:
        df: Input DataFrame (must have canaries)
        config: Configuration dictionary
        campos_buenos: List of feature columns (canaries must be first)
        
    Returns:
        DataFrame with predictions
    """
    print("\n" + "="*70)
    print("SCORING FUTURE DATA (zLightGBM)")
    print("="*70)
    
    # Get configuration
    zlgbm_config = config["zlgbm"]
    n_canaritos = zlgbm_config["qcanaritos"]
    future_months = zlgbm_config["train_final"]["future"]
    ksemillerio = zlgbm_config["train_final"]["ksemillerio"]
    
    # Filter future month
    df_future = df.filter(pl.col("foto_mes").is_in(future_months))
    
    print(f"\nFuture data:")
    print(f"  Months: {future_months}")
    print(f"  Records: {df_future.shape[0]:,}")
    print(f"  Ensemble size: {ksemillerio}")
    
    # Verify canaries exist
    expected_canaritos = [f"canarito_{i+1}" for i in range(n_canaritos)]
    missing_canaritos = [c for c in expected_canaritos if c not in df_future.columns]
    
    if missing_canaritos:
        raise ValueError(
            f"ERROR: Future data is missing canaries!\n"
            f"Missing: {missing_canaritos[:5]}...\n"
            f"Canaries must be added to future data before scoring."
        )
    
    print(f"✓ Canaries verified in future data")
    
    # Get valid feature columns
    campos_buenos_valid = [col for col in campos_buenos 
                          if col in df_future.columns and 
                          col not in ["clase_ternaria", "numero_de_cliente", "foto_mes"]]
    
    X_future = df_future.select(campos_buenos_valid).to_numpy()
    
    print(f"  Features: {X_future.shape[1]:,} (including {n_canaritos} canaries)")
    
    # Setup paths
    experimento = config["experimento"]
    output_dir = Path(f"output/{experimento}")
    
    # Predefined seeds (same as training)
    SEMILLAS = [123479, 123491, 123493, 123499, 123503]
    semillas_to_use = SEMILLAS[:ksemillerio]
    
    print(f"\n{'='*70}")
    print(f"LOADING {ksemillerio} MODEL(S) FOR ENSEMBLE PREDICTION")
    print(f"{'='*70}")
    
    all_predictions = []
    
    for i, semilla in enumerate(semillas_to_use, 1):
        # Determine model filename
        if ksemillerio == 1:
            model_file = output_dir / "zmodelo.txt"
        else:
            model_file = output_dir / f"zmodelo_{semilla}.txt"
        
        print(f"\nModel {i}/{ksemillerio} (seed={semilla}):")
        print(f"  Loading from {model_file}...")
        
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model not found: {model_file}\n"
                f"Train the model first using train_zlgbm_final_model()"
            )
        
        modelo = lgb.Booster(model_file=str(model_file))
        n_trees = modelo.num_trees()
        print(f"  ✓ Loaded: {n_trees} trees")
        
        # Predict
        print(f"  Predicting...")
        predictions = modelo.predict(X_future)
        all_predictions.append(predictions)
        
        print(f"  ✓ Predictions: min={predictions.min():.6f}, max={predictions.max():.6f}, mean={predictions.mean():.6f}")
        
        # Clean up
        del modelo
    
    # Average predictions (ensemble)
    print(f"\n{'='*70}")
    print(f"AVERAGING ENSEMBLE PREDICTIONS")
    print(f"{'='*70}")
    
    all_predictions = np.array(all_predictions)  # Shape: (ksemillerio, n_samples)
    avg_predictions = all_predictions.mean(axis=0)
    
    print(f"\nEnsemble statistics:")
    print(f"  Models: {ksemillerio}")
    print(f"  Final predictions:")
    print(f"    Min: {avg_predictions.min():.6f}")
    print(f"    Max: {avg_predictions.max():.6f}")
    print(f"    Mean: {avg_predictions.mean():.6f}")
    print(f"    Median: {np.median(avg_predictions):.6f}")
    
    if ksemillerio > 1:
        # Show variance across models
        std_predictions = all_predictions.std(axis=0)
        print(f"  Prediction std dev (across models):")
        print(f"    Mean: {std_predictions.mean():.6f}")
        print(f"    Max: {std_predictions.max():.6f}")
    
    predictions = avg_predictions
    
    # Create prediction DataFrame
    df_pred = df_future.select(["numero_de_cliente", "foto_mes"]).with_columns([
        pl.Series("prob", predictions)
    ])
    
    # Save predictions
    output_dir = Path(f"output/{experimento}")
    pred_file = output_dir / "prediccion.txt"
    df_pred.write_csv(pred_file, separator="\t")
    print(f"\n✓ Predictions saved to {pred_file}")
    
    return df_pred


def generate_zlgbm_submission(df_pred: pl.DataFrame, config: Dict, n_envios: int = 11000) -> None:
    """
    Generate Kaggle submission file from zLightGBM predictions
    
    Args:
        df_pred: DataFrame with predictions
        config: Configuration dictionary
        n_envios: Number of top predictions to submit (default 11000 from notebook)
    """
    print("\n" + "="*70)
    print("GENERATING KAGGLE SUBMISSION (zLightGBM)")
    print("="*70)
    
    print(f"\nCutoff: {n_envios} envíos")
    
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
    experimento = config["experimento"]
    kaggle_dir = Path("output/kaggle")
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    submission_file = kaggle_dir / f"KA{experimento}_{n_envios}.csv"
    
    # Save submission
    print(f"\nSaving submission to {submission_file}...")
    df_submission = df_pred.select(["numero_de_cliente", "Predicted"])
    df_submission.write_csv(submission_file)
    
    # Print statistics
    n_predicted = df_pred.filter(pl.col("Predicted") == 1).shape[0]
    prob_at_cutoff = df_pred.filter(pl.col("row_num") == n_envios - 1).select("prob").item()
    
    print(f"\nSubmission statistics:")
    print(f"  Total records: {df_pred.shape[0]:,}")
    print(f"  Predicted as positive: {n_predicted:,}")
    print(f"  Top probability: {df_pred.select('prob').head(1).item():.6f}")
    print(f"  Probability at cutoff ({n_envios}): {prob_at_cutoff:.6f}")
    
    print(f"\n✓ Submission saved to {submission_file}")
    print(f"\n{'='*70}")
    print("SUBMISSION GENERATION COMPLETED")
    print(f"{'='*70}")

