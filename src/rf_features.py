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
    Add Random Forest leaf features
    
    Trains a small LightGBM (configured as Random Forest) and creates
    binary features for each tree-leaf combination
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        campos_buenos: List of feature columns to use
        
    Returns:
        DataFrame with RF leaf features added
    """
    print("\n" + "="*50)
    print("ADDING RANDOM FOREST LEAF FEATURES")
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
    
    # Save model
    print("Saving model...")
    os.makedirs("output", exist_ok=True)
    modelo.save_model("output/modelo.model")
    
    # Get unique periods
    periodos = df.select("foto_mes").unique().sort("foto_mes").to_series().to_list()
    
    # Process each period
    print(f"Processing {len(periodos)} periods...")
    
    # Create a list to store all new leaf features
    all_rf_features = []
    
    for periodo in tqdm(periodos, desc="Processing periods"):
        # Get data for this period
        df_periodo = df.filter(pl.col("foto_mes") == periodo)
        X_periodo = df_periodo.select(campos_buenos_valid).to_numpy()
        
        # Get leaf predictions (which leaf each sample falls into for each tree)
        leaf_preds = modelo.predict(X_periodo, pred_leaf=True)
        
        # leaf_preds shape: (n_samples, n_trees)
        n_samples, n_trees = leaf_preds.shape
        
        # Create binary features for each tree and leaf
        period_features = {}
        
        for tree_idx in range(n_trees):
            tree_leaves = leaf_preds[:, tree_idx]
            unique_leaves = np.unique(tree_leaves)
            
            for leaf_id in unique_leaves:
                feature_name = f"rf_{tree_idx:03d}_{leaf_id:03d}"
                # Binary feature: 1 if sample is in this leaf, 0 otherwise
                period_features[feature_name] = (tree_leaves == leaf_id).astype(np.int32)
        
        # Create a small dataframe with these features
        period_df = pl.DataFrame(period_features)
        
        # Add foto_mes and index for joining
        period_df = period_df.with_columns([
            pl.lit(periodo).alias("foto_mes"),
            pl.Series("_idx", np.arange(n_samples))
        ])
        
        all_rf_features.append(period_df)
    
    # Concatenate all period features
    print("Concatenating RF features...")
    rf_features_df = pl.concat(all_rf_features)
    
    # Add index to original dataframe for joining
    df = df.with_columns([
        pl.int_range(0, pl.count()).over("foto_mes").alias("_idx")
    ])
    
    # Join RF features with original dataframe
    print("Joining RF features with main dataset...")
    df = df.join(rf_features_df, on=["foto_mes", "_idx"], how="left")
    
    # Drop temporary columns
    df = df.drop(["clase01", "entrenamiento", "_idx"])
    
    # Fill NaN values in RF features with 0
    rf_cols = [col for col in df.columns if col.startswith("rf_")]
    if rf_cols:
        df = df.with_columns([
            pl.col(col).fill_null(0) for col in rf_cols
        ])
    
    print(f"\nRF features added: {len(rf_cols)} features")
    print(f"Total columns: {df.shape[1]}")
    
    return df

