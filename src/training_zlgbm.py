"""
Training module for zLightGBM (without Bayesian Optimization)

zLightGBM uses canary features for automatic overfitting control,
eliminating the need for hyperparameter optimization.
"""

import polars as pl
import numpy as np
import lightgbm as lgb
from typing import Dict, List
from pathlib import Path


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
    Train final model using zLightGBM (no Bayesian Optimization needed)
    
    Key differences from standard LightGBM:
    - Uses canary features for overfitting control
    - Stops automatically when it can't improve
    - Only trains 1 model (no ensemble needed)
    - More conservative undersampling (0.50 vs 0.10)
    
    Args:
        df: Input DataFrame (must already have canaries at the beginning)
        config: Configuration dictionary
        campos_buenos: List of feature columns (canaries must be first)
    """
    print("\n" + "="*70)
    print("TRAINING FINAL MODEL WITH zLightGBM")
    print("="*70)
    
    # Get zLightGBM configuration
    zlgbm_config = config["zlgbm"]
    n_canaritos = zlgbm_config["qcanaritos"]
    training_months = zlgbm_config["train_final"]["training"]
    undersampling = zlgbm_config["train_final"]["undersampling"]
    
    print(f"\nConfiguration:")
    print(f"  Training months: {training_months}")
    print(f"  Undersampling: {undersampling} ({undersampling*100:.0f}%)")
    print(f"  Canaries: {n_canaritos}")
    
    # Verify canaries are at the beginning
    expected_canaritos = [f"canarito_{i+1}" for i in range(n_canaritos)]
    actual_first_cols = campos_buenos[:n_canaritos]
    
    if actual_first_cols != expected_canaritos:
        raise ValueError(
            f"ERROR: Canaries must be the first {n_canaritos} columns!\n"
            f"Expected: {expected_canaritos[:5]}...\n"
            f"Got: {actual_first_cols[:5]}..."
        )
    
    print(f"✓ Canaries verified at the beginning")
    
    # Prepare binary target
    df = df.with_columns([
        pl.when(pl.col("clase_ternaria").is_in(["BAJA+2", "BAJA+1"]))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("clase01")
    ])
    
    # Undersampling with random column
    np.random.seed(config["semilla_primigenia"])
    azar = np.random.uniform(0, 1, df.shape[0])
    df = df.with_columns([pl.Series("azar", azar)])
    
    # Mark training records (undersampling)
    df = df.with_columns([
        pl.when(
            (pl.col("foto_mes").is_in(training_months)) &
            ((pl.col("azar") <= undersampling) | 
             (pl.col("clase_ternaria").is_in(["BAJA+1", "BAJA+2"])))
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("training")
    ])
    
    # Filter training data
    df_train = df.filter(pl.col("training") == 1)
    
    # Get valid feature columns
    campos_buenos_valid = [col for col in campos_buenos
                          if col in df.columns and 
                          col not in ["clase_ternaria", "clase01", "azar", "training",
                                     "numero_de_cliente", "foto_mes"]]
    
    X_train = df_train.select(campos_buenos_valid).to_numpy()
    y_train = df_train.select("clase01").to_numpy().ravel()
    months_train = df_train.select("foto_mes").to_numpy().ravel()
    
    print(f"\nTraining data:")
    print(f"  Samples: {X_train.shape[0]:,}")
    print(f"  Features: {X_train.shape[1]:,} (including {n_canaritos} canaries)")
    print(f"  Positives: {y_train.sum():,} ({y_train.sum()/len(y_train)*100:.2f}%)")
    print(f"  Negatives: {len(y_train) - y_train.sum():,}")
    
    # Calculate month weights
    month_weight_strategy = zlgbm_config["train_final"].get("month_weights", "equal")
    sample_weights = calculate_month_weights(months_train, month_weight_strategy)
    
    print(f"\nMonth weighting strategy: {month_weight_strategy}")
    if month_weight_strategy != "equal":
        unique_months = np.unique(months_train)
        unique_months.sort()
        print(f"  Month weights:")
        for month in unique_months[:3]:  # First 3 months
            weight = sample_weights[months_train == month][0]
            print(f"    {month}: {weight:.3f}")
        print(f"    ...")
        for month in unique_months[-3:]:  # Last 3 months
            weight = sample_weights[months_train == month][0]
            print(f"    {month}: {weight:.3f}")
    
    # Save month weights to file
    experimento = config["experimento"]
    output_dir = Path(f"output/{experimento}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    weights_file = output_dir / "month_weights.txt"
    with open(weights_file, 'w') as f:
        f.write(f"Month Weighting Strategy: {month_weight_strategy}\n")
        f.write(f"Experiment: {experimento}\n")
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
    
    print(f"✓ Month weights saved to {weights_file}")
    
    # Create LightGBM dataset with weights
    dtrain = lgb.Dataset(X_train, label=y_train, weight=sample_weights,
                        free_raw_data=False, feature_name=campos_buenos_valid)
    
    # Get zLightGBM parameters
    lgb_params = zlgbm_config["param"].copy()
    lgb_params["seed"] = config["semilla_primigenia"]
    
    print(f"\nzLightGBM parameters:")
    print(f"  canaritos: {lgb_params['canaritos']}")
    print(f"  gradient_bound: {lgb_params['gradient_bound']}")
    print(f"  learning_rate: {lgb_params['learning_rate']}")
    print(f"  feature_fraction: {lgb_params['feature_fraction']}")
    print(f"  min_data_in_leaf: {lgb_params['min_data_in_leaf']}")
    print(f"  num_iterations (max): {lgb_params['num_iterations']}")
    print(f"  num_leaves (max): {lgb_params['num_leaves']}")
    
    # Train model
    print(f"\n{'='*70}")
    print("TRAINING MODEL (zLightGBM will stop automatically)")
    print(f"{'='*70}\n")
    
    modelo = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=lgb_params["num_iterations"]
    )
    
    # Get model info
    n_trees = modelo.num_trees()
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Trees built: {n_trees} (max was {lgb_params['num_iterations']})")
    print(f"  zLightGBM stopped automatically at {n_trees} trees")
    
    # Save model
    experimento = config["experimento"]
    output_dir = Path(f"output/{experimento}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_file = output_dir / "zmodelo.txt"
    
    print(f"\nSaving model to {model_file}...")
    try:
        modelo.save_model(str(model_file))
        
        # Verify the file was actually created and has content
        if not model_file.exists():
            raise IOError(f"Model file was not created: {model_file}")
        
        file_size = model_file.stat().st_size
        if file_size == 0:
            raise IOError(f"Model file is empty: {model_file}")
        
        print(f"✓ Model saved successfully: {file_size / (1024*1024):.2f} MB")
        
    except Exception as e:
        error_msg = f"CRITICAL ERROR: Failed to save model: {e}"
        print(f"\n❌ {error_msg}")
        raise RuntimeError(error_msg) from e
    
    # Save tree structure for analysis
    try:
        tree_df = modelo.trees_to_dataframe()
        tree_file = output_dir / "tb_arboles.txt"
        tree_df.to_csv(tree_file, sep="\t", index=False)
        print(f"✓ Tree structure saved to {tree_file}")
        
        # Analyze tree structure (if leaf_index column exists)
        if 'leaf_index' in tree_df.columns:
            leaves_per_tree = tree_df.groupby('tree_index')['leaf_index'].max() + 1
            
            print(f"\nTree structure analysis:")
            print(f"  Total trees: {n_trees}")
            print(f"  Leaves per tree:")
            print(f"    Min: {leaves_per_tree.min()}")
            print(f"    Median: {leaves_per_tree.median():.0f}")
            print(f"    Mean: {leaves_per_tree.mean():.1f}")
            print(f"    Max: {leaves_per_tree.max()}")
        else:
            print(f"  ℹ Tree structure saved (leaf_index column not available for detailed analysis)")
        
    except Exception as e:
        print(f"⚠ Could not save tree structure: {e}")
    
    # Feature importance
    print(f"\n{'='*70}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*70}")
    
    try:
        # Get feature importance from model
        importance = modelo.feature_importance(importance_type='gain')
        feature_names = campos_buenos_valid
        
        print(f"\n  Extracted importance for {len(importance)} features")
        print(f"  Total importance (gain): {importance.sum():,.1f}")
        print(f"  Non-zero features: {(importance > 0).sum()}")
        
        # Validate we have data
        if len(importance) == 0 or len(feature_names) == 0:
            raise ValueError("No feature importance data extracted from model")
        
        if len(importance) != len(feature_names):
            raise ValueError(f"Mismatch: {len(importance)} importance values vs {len(feature_names)} feature names")
        
        # Create feature importance DataFrame using Polars
        importance_df = pl.DataFrame({
            'feature': feature_names,
            'importance': importance,
            'is_canary': [name.startswith("canarito_") for name in feature_names]
        })
        
        print(f"  Created DataFrame with {len(importance_df)} rows")
        
        # Sort by importance
        importance_df = importance_df.sort('importance', descending=True)
        
        # Add rank (using with_row_count is the proper way in Polars)
        importance_df = importance_df.with_row_count(name='rank', offset=1)
        
        # Add relative importance (percentage)
        total_importance = importance_df['importance'].sum()
        importance_df = importance_df.with_columns([
            (pl.col('importance') / total_importance * 100).alias('importance_pct'),
        ])
        
        # Add cumulative sum
        importance_df = importance_df.with_columns([
            pl.col('importance_pct').cum_sum().alias('importance_cumsum_pct')
        ])
        
        # Validate DataFrame before saving
        if len(importance_df) == 0:
            raise ValueError("Feature importance DataFrame is empty after processing")
        
        # Save to file
        importance_file = output_dir / "feature_importance.txt"
        importance_df.write_csv(importance_file, separator="\t")
        print(f"\n✓ Feature importance saved to {importance_file} ({len(importance_df)} features)")
        
        # Get top features
        top_features = importance_df.head(20)
        
        print(f"\nTop 20 features by importance:")
        print(f"  {'Rank':<5} {'Feature':<40} {'Importance':>12} {'%':>8} {'Cumsum %':>10}")
        print(f"  {'-'*80}")
        for row in top_features.iter_rows(named=True):
            marker = "🐤" if row['is_canary'] else "  "
            print(f"  {row['rank']:>3d}. {marker} {row['feature']:<38s} "
                  f"{row['importance']:>12,.1f} {row['importance_pct']:>7.2f}% "
                  f"{row['importance_cumsum_pct']:>9.2f}%")
        
        # Analyze canary importance
        canary_df = importance_df.filter(pl.col('is_canary'))
        real_df = importance_df.filter(~pl.col('is_canary'))
        
        print(f"\nCanary analysis:")
        print(f"  Canary features: {len(canary_df)}")
        print(f"  Real features: {len(real_df)}")
        
        canary_mean = canary_df['importance'].mean()
        real_mean = real_df['importance'].mean()
        
        print(f"  Avg canary importance: {canary_mean:,.1f}")
        print(f"  Avg real feature importance: {real_mean:,.1f}")
        
        # Calculate ratio (handle division by zero)
        if canary_mean > 0:
            ratio = real_mean / canary_mean
            print(f"  Ratio (real/canary): {ratio:.2f}x")
        else:
            print(f"  Ratio (real/canary): ∞ (canaries have zero importance)")
        
        # Check how many canaries are in top features
        top_100_canaries = importance_df.head(100).filter(pl.col('is_canary')).shape[0]
        top_200_canaries = importance_df.head(200).filter(pl.col('is_canary')).shape[0]
        
        print(f"  Canaries in top 100: {top_100_canaries}")
        print(f"  Canaries in top 200: {top_200_canaries}")
        
        if real_mean > canary_mean:
            print(f"  ✓ Real features are more important than canaries (good!)")
        else:
            print(f"  ⚠ Canaries have similar importance to real features (check for overfitting)")
        
    except Exception as e:
        error_msg = f"CRITICAL ERROR in feature importance analysis: {e}"
        print(f"\n❌ {error_msg}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(error_msg) from e
    
    # Final validation: ensure all critical files were created
    print(f"\n{'='*70}")
    print("VALIDATION: Checking output files")
    print(f"{'='*70}")
    
    critical_files = {
        "Model": model_file,
        "Feature Importance": importance_file,
        "Month Weights": weights_file
    }
    
    missing_files = []
    for name, filepath in critical_files.items():
        if not filepath.exists():
            missing_files.append(f"{name} ({filepath})")
            print(f"  ❌ {name}: MISSING")
        elif filepath.stat().st_size == 0:
            missing_files.append(f"{name} ({filepath}) - EMPTY")
            print(f"  ❌ {name}: EMPTY FILE")
        else:
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {name}: {size_mb:.2f} MB")
    
    # Check feature importance has actual data (more than just header)
    importance_lines = importance_file.read_text().strip().split('\n')
    if len(importance_lines) <= 1:
        missing_files.append(f"Feature Importance ({importance_file}) - NO DATA (only header)")
        print(f"  ❌ Feature Importance: NO DATA (only header found)")
    else:
        print(f"  ✓ Feature Importance: {len(importance_lines)-1} features")
    
    if missing_files:
        error_msg = f"Training validation FAILED. Missing or invalid files:\n" + "\n".join(f"  - {f}" for f in missing_files)
        print(f"\n❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    print(f"\n✓ All output files validated successfully")
    
    print(f"\n{'='*70}")
    print("zLightGBM TRAINING COMPLETED SUCCESSFULLY")
    print(f"{'='*70}")

