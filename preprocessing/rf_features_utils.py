"""
Random Forest Feature Engineering module
Creates binary features from Random Forest leaf predictions
"""

import polars as pl
import numpy as np
import lightgbm as lgb
import os
from typing import Dict, List
from tqdm import tqdm


def add_rf_features(df: pl.DataFrame, config: Dict, campos_buenos: List[str]) -> pl.DataFrame:
    """
    Add Random Forest leaf features (MEMORY OPTIMIZED VERSION)
    
    Trains a small LightGBM (configured as Random Forest) and creates
    binary features for each tree-leaf combination
    
    This version processes periods in chunks and uses streaming to reduce memory
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        campos_buenos: List of feature columns to use
        
    Returns:
        DataFrame with RF leaf features added
    """
    print("\n" + "="*50)
    print("ADDING RANDOM FOREST LEAF FEATURES (MEMORY OPTIMIZED)")
    print("="*50)
    
    # Create binary target
    print("Creating binary target...")
    df = df.with_columns([
        pl.when(pl.col("clase_ternaria").is_in(["BAJA+2", "BAJA+1"]))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("clase01")
    ])
    
    # Mark training data
    training_months = config["FE_rf"]["train"]["training"]
    df = df.with_columns([
        pl.when(pl.col("foto_mes").is_in(training_months))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("entrenamiento")
    ])
    
    # Prepare training data
    print("Preparing training data...")
    df_train = df.filter(pl.col("entrenamiento") == 1)
    
    # Get campos_buenos that exist in df
    campos_buenos_valid = [col for col in campos_buenos if col in df.columns]
    
    X_train = df_train.select(campos_buenos_valid).to_numpy()
    y_train = df_train.select("clase01").to_numpy().ravel()
    
    print(f"Training shape: {X_train.shape}")
    print(f"Number of positives: {y_train.sum()}")
    
    # Free memory
    del df_train
    
    # Train Random Forest (LightGBM configured as RF)
    print("Training Random Forest...")
    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    
    rf_params = config["FE_rf"]["lgb_param"].copy()
    
    modelo = lgb.train(
        rf_params,
        dtrain,
        num_boost_round=rf_params["num_iterations"],
        valid_sets=None,
        callbacks=[lgb.log_evaluation(period=0)]  # Suppress output
    )
    
    print("Random Forest trained successfully")
    
    # Free training data
    del X_train, y_train, dtrain
    
    # Save model (experiment-specific path)
    print("Saving model...")
    experimento = config["experimento"]
    os.makedirs(f"output/{experimento}", exist_ok=True)
    modelo.save_model(f"output/{experimento}/modelo.model")
    
    # Get unique periods
    periodos = df.select("foto_mes").unique().sort("foto_mes").to_series().to_list()
    
    # =========================================================================
    # MEMORY OPTIMIZATION: Process and join period by period (streaming)
    # =========================================================================
    print(f"\nProcessing {len(periodos)} periods (streaming mode for memory efficiency)...")
    
    # Get all possible RF features from the model (to ensure consistent columns)
    n_trees = rf_params["num_iterations"]
    max_leaves = rf_params["num_leaves"]
    
    print(f"Model info: {n_trees} trees, up to {max_leaves} leaves per tree")
    
    # Process each period and join immediately (instead of accumulating)
    for i, periodo in enumerate(tqdm(periodos, desc="Processing periods")):
        # Get data for this period
        df_periodo = df.filter(pl.col("foto_mes") == periodo)
        X_periodo = df_periodo.select(campos_buenos_valid).to_numpy()
        
        # Get leaf predictions
        leaf_preds = modelo.predict(X_periodo, pred_leaf=True)
        n_samples, n_trees_actual = leaf_preds.shape
        
        # Create binary features for this period
        period_features = {}
        
        for tree_idx in range(n_trees_actual):
            tree_leaves = leaf_preds[:, tree_idx]
            unique_leaves = np.unique(tree_leaves)
            
            for leaf_id in unique_leaves:
                feature_name = f"rf_{tree_idx:03d}_{leaf_id:03d}"
                period_features[feature_name] = (tree_leaves == leaf_id).astype(np.int8)  # int8 to save memory
        
        # Create small dataframe with these features
        period_df = pl.DataFrame(period_features)
        
        # Add row index for this period
        period_df = period_df.with_columns([
            pl.Series("_idx", np.arange(n_samples))
        ])
        
        # Add index to the portion of df for this period
        df_periodo_indexed = df_periodo.with_columns([
            pl.Series("_idx", np.arange(n_samples))
        ])
        
        # Join RF features for this period only
        df_periodo_with_rf = df_periodo_indexed.join(period_df, on="_idx", how="left")
        
        # Drop the index column
        df_periodo_with_rf = df_periodo_with_rf.drop("_idx")
        
        # Update the main dataframe for this period
        if i == 0:
            # First period: replace
            df_result = df_periodo_with_rf
        else:
            # Subsequent periods: concatenate
            # Use diagonal concat to handle different RF features across periods
            df_result = pl.concat([df_result, df_periodo_with_rf], how="diagonal")
        
        # Free memory
        del df_periodo, X_periodo, leaf_preds, period_features, period_df, df_periodo_indexed, df_periodo_with_rf
    
    # Replace original dataframe
    df = df_result
    del df_result
    
    # Drop temporary columns
    df = df.drop(["clase01", "entrenamiento"])
    
    # Fill NaN values in RF features with 0
    rf_cols = [col for col in df.columns if col.startswith("rf_")]
    if rf_cols:
        print(f"\nFilling null values in {len(rf_cols)} RF features...")
        df = df.with_columns([
            pl.col(col).fill_null(0).cast(pl.Int8) for col in rf_cols  # Cast to int8 to save memory
        ])
    
    print(f"\n✓ RF features added: {len(rf_cols)} features")
    print(f"✓ Total columns: {df.shape[1]}")
    
    return df

