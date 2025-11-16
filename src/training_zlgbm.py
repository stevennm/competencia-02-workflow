"""
Training module for zLightGBM (without Bayesian Optimization)
"""

import polars as pl
import numpy as np
import lightgbm as lgb
import logging
from typing import Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


def calculate_month_weights(months: np.ndarray, strategy: str = "equal") -> np.ndarray:
    """
    Calculate weights for each sample based on its month
    
    Args:
        months: Array of foto_mes values for each sample (format YYYYMM)
        strategy: Weighting strategy
            - "equal": All months have weight 1.0
            - "step": Step decay (last 6 months = 1.0, next 6 = 0.7, rest = 0.4)
            - "linear": Linear decay from newest to oldest (1.0 to 0.3)
            - "exponential": Exponential decay (more weight to recent months)
    
    Returns:
        Array of weights for each sample
    """
    if strategy == "equal":
        return np.ones(len(months))
    
    # Get unique months and sort them
    unique_months = np.unique(months)
    unique_months.sort()
    
    # Calculate real month differences from the most recent month
    most_recent = unique_months[-1]
    
    # Create weight map for each month
    n_months = len(unique_months)
    month_weight_map = {}
    
    if strategy == "step":
        # Step decay: last 6 months = 1.0, next 6 = 0.7, rest = 0.4
        for month in unique_months:
            # Calculate how many months back this is from the most recent
            months_back = calculate_month_difference(most_recent, month)
            
            if months_back < 6:
                month_weight_map[month] = 1.0
            elif months_back < 12:
                month_weight_map[month] = 0.7
            else:
                month_weight_map[month] = 0.4
    
    elif strategy == "linear":
        # Linear decay: newest = 1.0, oldest = 0.3
        oldest = unique_months[0]
        total_months = calculate_month_difference(most_recent, oldest)
        
        for month in unique_months:
            months_back = calculate_month_difference(most_recent, month)
            # Linear interpolation: months_back=0 -> weight=1.0, months_back=total -> weight=0.3
            weight = 1.0 - (0.7 * months_back / total_months)
            month_weight_map[month] = weight
    
    elif strategy == "exponential":
        # Exponential decay: much more weight to recent months
        # Using decay factor of 0.95 per month back
        for month in unique_months:
            months_back = calculate_month_difference(most_recent, month)
            weight = 0.95 ** months_back
            month_weight_map[month] = weight
    
    else:
        raise ValueError(f"Unknown weighting strategy: {strategy}")
    
    # Map weights to each sample
    weights = np.array([month_weight_map[m] for m in months])
    
    return weights


def calculate_month_difference(month1: int, month2: int) -> int:
    """
    Calculate difference in months between two YYYYMM dates
    
    Args:
        month1: More recent month (YYYYMM)
        month2: Older month (YYYYMM)
    
    Returns:
        Number of months difference
    
    Example:
        calculate_month_difference(202106, 202101) -> 5
        calculate_month_difference(202101, 202012) -> 1
        calculate_month_difference(202101, 201912) -> 1
    """
    year1 = month1 // 100
    mon1 = month1 % 100
    year2 = month2 // 100
    mon2 = month2 % 100
    
    return (year1 - year2) * 12 + (mon1 - mon2)


def train_zlgbm_final_model(df: pl.DataFrame, config: Dict, campos_buenos: List[str]) -> None:
    """
    Train final model(s) using zLightGBM
    
    Args:
        df: Input DataFrame (must already have canaries at the beginning)
        config: Configuration dictionary
        campos_buenos: List of feature columns (canaries must be first)
    """
    # Get zLightGBM configuration
    zlgbm_config = config["zlgbm"]
    n_canaritos = zlgbm_config["qcanaritos"]
    training_months = zlgbm_config["train_final"]["training"]
    undersampling = zlgbm_config["train_final"]["undersampling"]
    ksemillerio = zlgbm_config["train_final"]["ksemillerio"]
    
    logger.info(f"Training {ksemillerio} model(s)")
    logger.info(f"Undersampling: {undersampling}")
    logger.info(f"Canaries: {n_canaritos}")
    
    # Verify canaries are at the beginning
    expected_canaritos = [f"canarito_{i+1}" for i in range(n_canaritos)]
    actual_first_cols = campos_buenos[:n_canaritos]
    
    if actual_first_cols != expected_canaritos:
        raise ValueError(
            f"ERROR: Canaries must be the first {n_canaritos} columns!\n"
            f"Expected: {expected_canaritos[:5]}...\n"
            f"Got: {actual_first_cols[:5]}..."
        )
    
    # Prepare binary target
    df = df.with_columns([
        pl.when(pl.col("clase_ternaria").is_in(["BAJA+2", "BAJA+1"]))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("clase01")
    ])
    
    # Filter to training months only (before loop)
    df_training_months = df.filter(pl.col("foto_mes").is_in(training_months))
    
    # Get valid feature columns (same for all models)
    campos_buenos_valid = [col for col in campos_buenos
                          if col in df.columns and 
                          col not in ["clase_ternaria", "clase01", "numero_de_cliente", "foto_mes"]]
    
    # Setup output directory
    experimento = config["experimento"]
    output_dir = Path(f"output/{experimento}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Predefined seeds (same as R notebook)
    SEMILLAS = [123479, 123491, 123493, 123499, 123503]
    
    if ksemillerio > len(SEMILLAS):
        raise ValueError(
            f"ksemillerio ({ksemillerio}) exceeds available seeds ({len(SEMILLAS)})\n"
            f"Available seeds: {SEMILLAS}"
        )
    
    semillas_to_use = SEMILLAS[:ksemillerio]
    logger.info(f"Seeds: {semillas_to_use}")
    
    # Train each model with different seed
    for i, semilla in enumerate(semillas_to_use, 1):
        logger.info(f"Training model {i}/{ksemillerio} (seed {semilla})")
        
        # Set seed for this model
        np.random.seed(semilla)
        
        # Generate random column for undersampling (different for each seed)
        azar = np.random.uniform(0, 1, df_training_months.shape[0])
        df_with_azar = df_training_months.with_columns([pl.Series("azar", azar)])
        
        # Apply undersampling: keep all positives + random sample of negatives
        df_train = df_with_azar.filter(
            (pl.col("clase01") == 1) |  # All positives
            (pl.col("azar") <= undersampling)  # Random sample of negatives
        )
        
        # Extract training data
        X_train = df_train.select(campos_buenos_valid).to_numpy()
        y_train = df_train.select("clase01").to_numpy().ravel()
        months_train = df_train.select("foto_mes").to_numpy().ravel()
        
        logger.info(f"Samples: {X_train.shape[0]:,}, Features: {X_train.shape[1]:,}, Positives: {y_train.sum():,} ({y_train.sum()/len(y_train)*100:.2f}%)")
        
        # Calculate month weights
        month_weight_strategy = zlgbm_config["train_final"].get("month_weights", "equal")
        sample_weights = calculate_month_weights(months_train, month_weight_strategy)
        
        if i == 1:  # Only log once
            logger.info(f"Month weighting: {month_weight_strategy}")
        
        # Save month weights to file (only for first model)
        if i == 1:
            weights_file = output_dir / "month_weights.txt"
            with open(weights_file, 'w') as f:
                f.write(f"Month Weighting Strategy: {month_weight_strategy}\n")
                f.write(f"Experiment: {experimento}\n")
                f.write(f"Ensemble size: {ksemillerio}\n")
                f.write("="*50 + "\n\n")
                
                unique_months_sorted = np.unique(months_train)
                unique_months_sorted.sort()
                
                f.write(f"{'Month':<10} {'Weight':<10} {'Samples':<10}\n")
                f.write("-"*30 + "\n")
                
                for month in unique_months_sorted:
                    weight = sample_weights[months_train == month][0]
                    n_samples = (months_train == month).sum()
                    f.write(f"{month:<10} {weight:<10.4f} {n_samples:<10}\n")
                
                f.write("\n" + "="*50 + "\n")
                f.write("Statistics:\n")
                f.write(f"  Total samples: {len(sample_weights)}\n")
                f.write(f"  Unique months: {len(unique_months_sorted)}\n")
                f.write(f"  Weight range: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]\n")
                f.write(f"  Mean weight: {sample_weights.mean():.4f}\n")
            
            logger.info(f"Month weights saved to {weights_file}")
        
        # Create LightGBM dataset with weights
        dtrain = lgb.Dataset(X_train, label=y_train, weight=sample_weights,
                            free_raw_data=False, feature_name=campos_buenos_valid)
        
        # Get zLightGBM parameters
        lgb_params = zlgbm_config["param"].copy()
        lgb_params["seed"] = semilla  # Use ensemble seed, not base seed
        
        # Train model
        modelo = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=lgb_params["num_iterations"]
        )
        
        # Get model info
        n_trees = modelo.num_trees()
        logger.info(f"Model {i}/{ksemillerio} trained: {n_trees} trees")
        
        # Save model with seed suffix
        if ksemillerio == 1:
            model_file = output_dir / "zmodelo.txt"
        else:
            model_file = output_dir / f"zmodelo_{semilla}.txt"
        
        try:
            modelo.save_model(str(model_file))
            
            # Verify the file was actually created and has content
            if not model_file.exists():
                raise IOError(f"Model file was not created: {model_file}")
            
            file_size = model_file.stat().st_size
            if file_size == 0:
                raise IOError(f"Model file is empty: {model_file}")
            
            logger.info(f"Model saved: {file_size / (1024*1024):.2f} MB")
            
        except Exception as e:
            logger.error(f"Failed to save model {i}: {e}")
            raise RuntimeError(f"Failed to save model {i}: {e}") from e
        
        # Save feature importance for this model
        try:
            importance = modelo.feature_importance(importance_type='gain')
            
            # Create feature importance DataFrame
            importance_df = pl.DataFrame({
                'feature': campos_buenos_valid,
                'importance': importance,
                'is_canary': [name.startswith("canarito_") for name in campos_buenos_valid]
            })
            
            # Sort and add metadata
            importance_df = importance_df.sort('importance', descending=True)
            importance_df = importance_df.with_row_count(name='rank', offset=1)
            
            total_importance = importance_df['importance'].sum()
            importance_df = importance_df.with_columns([
                (pl.col('importance') / total_importance * 100).alias('importance_pct'),
            ])
            importance_df = importance_df.with_columns([
                pl.col('importance_pct').cum_sum().alias('importance_cumsum_pct')
            ])
            
            # Save to file
            if ksemillerio == 1:
                importance_file = output_dir / "feature_importance.txt"
            else:
                importance_file = output_dir / f"feature_importance_{semilla}.txt"
            
            importance_df.write_csv(importance_file, separator="\t")
            logger.info(f"Feature importance saved: {len(importance_df)} features")
            
            # Canary analysis
            canary_df = importance_df.filter(pl.col('is_canary'))
            canary_mean = canary_df['importance'].mean()
            
            if canary_mean == 0:
                logger.info("Canaries have ZERO importance")
            else:
                real_mean = importance_df.filter(~pl.col('is_canary'))['importance'].mean()
                ratio = real_mean / canary_mean if canary_mean > 0 else float('inf')
                logger.warning(f"Canary avg importance: {canary_mean:.1f} (ratio: {ratio:.2f}x)")
            
        except Exception as e:
            logger.warning(f"Could not save feature importance: {e}")
        
        # Clean up to free memory
        del X_train, y_train, months_train, sample_weights, dtrain, modelo
        if 'df_train' in locals():
            del df_train
        if 'df_with_azar' in locals():
            del df_with_azar
    
    # Final summary
    logger.info(f"Training completed: {ksemillerio} model(s) saved to {output_dir}")

