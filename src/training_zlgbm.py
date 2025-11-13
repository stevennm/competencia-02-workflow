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
    
    print(f"\nTraining data:")
    print(f"  Samples: {X_train.shape[0]:,}")
    print(f"  Features: {X_train.shape[1]:,} (including {n_canaritos} canaries)")
    print(f"  Positives: {y_train.sum():,} ({y_train.sum()/len(y_train)*100:.2f}%)")
    print(f"  Negatives: {len(y_train) - y_train.sum():,}")
    
    # Create LightGBM dataset
    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False,
                        feature_name=campos_buenos_valid)
    
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
    modelo.save_model(str(model_file))
    print(f"\n✓ Model saved to {model_file}")
    
    # Save tree structure for analysis
    try:
        tree_df = modelo.trees_to_dataframe()
        tree_file = output_dir / "tb_arboles.txt"
        tree_df.to_csv(tree_file, sep="\t", index=False)
        print(f"✓ Tree structure saved to {tree_file}")
        
        # Analyze tree structure
        leaves_per_tree = tree_df.groupby('tree_index')['leaf_index'].max() + 1
        
        print(f"\nTree structure analysis:")
        print(f"  Total trees: {n_trees}")
        print(f"  Leaves per tree:")
        print(f"    Min: {leaves_per_tree.min()}")
        print(f"    Median: {leaves_per_tree.median():.0f}")
        print(f"    Mean: {leaves_per_tree.mean():.1f}")
        print(f"    Max: {leaves_per_tree.max()}")
        
    except Exception as e:
        print(f"⚠ Could not save tree structure: {e}")
    
    # Feature importance
    try:
        importance = modelo.feature_importance(importance_type='gain')
        feature_names = campos_buenos_valid
        
        # Get top features
        top_indices = np.argsort(importance)[::-1][:20]
        
        print(f"\nTop 20 features by importance:")
        for i, idx in enumerate(top_indices, 1):
            feat_name = feature_names[idx]
            is_canary = feat_name.startswith("canarito_")
            marker = "🐤" if is_canary else "  "
            print(f"  {i:2d}. {marker} {feat_name:40s} {importance[idx]:>12,.1f}")
        
        # Analyze canary importance
        canary_importance = importance[:n_canaritos]
        real_importance = importance[n_canaritos:]
        
        print(f"\nCanary analysis:")
        print(f"  Avg canary importance: {canary_importance.mean():,.1f}")
        print(f"  Avg real feature importance: {real_importance.mean():,.1f}")
        print(f"  Ratio (real/canary): {real_importance.mean() / canary_importance.mean():.2f}x")
        
        if real_importance.mean() > canary_importance.mean():
            print(f"  ✓ Real features are more important than canaries (good!)")
        else:
            print(f"  ⚠ Canaries have similar importance to real features (check for overfitting)")
        
    except Exception as e:
        print(f"⚠ Could not analyze feature importance: {e}")
    
    print(f"\n{'='*70}")
    print("zLightGBM TRAINING COMPLETED SUCCESSFULLY")
    print(f"{'='*70}")

