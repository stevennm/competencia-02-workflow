"""
Training module with Bayesian Optimization using Optuna
"""

import polars as pl
import numpy as np
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from typing import Dict, List, Tuple
import os
from datetime import datetime


def prepare_training_data(df: pl.DataFrame, config: Dict, campos_buenos: List[str]) -> Tuple:
    """
    Prepare training and testing data with undersampling
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        campos_buenos: List of feature columns
        
    Returns:
        Tuple of (dtrain, dataset_test, test_matrix, campos_buenos_valid)
    """
    print("\n" + "="*50)
    print("PREPARING TRAINING DATA")
    print("="*50)
    
    # Create binary target
    df = df.with_columns([
        pl.when(pl.col("clase_ternaria").is_in(["BAJA+2", "BAJA+1"]))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("clase01")
    ])
    
    # Add random column for undersampling
    np.random.seed(config["semilla_primigenia"])
    azar = np.random.uniform(0, 1, df.shape[0])
    df = df.with_columns([
        pl.Series("azar", azar)
    ])
    
    # Mark training records with undersampling
    training_months = config["trainingstrategy"]["training"]
    undersampling = config["trainingstrategy"]["undersampling"]
    
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
    
    # Get valid feature columns
    campos_buenos_valid = [col for col in campos_buenos 
                           if col in df.columns and col not in ["clase_ternaria", "clase01", "azar", "training"]]
    
    print(f"Training features: {len(campos_buenos_valid)}")
    
    # Prepare training dataset
    df_train = df.filter(pl.col("training") == 1)
    X_train = df_train.select(campos_buenos_valid).to_numpy()
    y_train = df_train.select("clase01").to_numpy().ravel()
    
    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    
    print(f"Training set: {X_train.shape[0]} rows, {X_train.shape[1]} columns")
    print(f"  Positives: {y_train.sum()}, Negatives: {len(y_train) - y_train.sum()}")
    
    # Prepare testing dataset
    testing_months = config["trainingstrategy"]["testing"]
    df_test = df.filter(pl.col("foto_mes").is_in(testing_months))
    
    # Add gan column for gain calculation
    df_test = df_test.with_columns([
        pl.lit(-20000.0).alias("gan")
    ])
    df_test = df_test.with_columns([
        pl.when(pl.col("clase_ternaria") == "BAJA+2")
        .then(pl.lit(780000.0))
        .otherwise(pl.col("gan"))
        .alias("gan")
    ])
    
    test_matrix = df_test.select(campos_buenos_valid).to_numpy()
    
    print(f"Testing set: {test_matrix.shape[0]} rows")
    print(f"  BAJA+2: {df_test.filter(pl.col('clase_ternaria') == 'BAJA+2').shape[0]}")
    
    n_train = X_train.shape[0]
    return dtrain, df_test, test_matrix, campos_buenos_valid, n_train


def calculate_gain_with_meseta(predictions: np.ndarray, gan_values: np.ndarray, 
                                window_size: int = 2001) -> float:
    """
    Calculate gain with smoothed plateau (meseta)
    
    Args:
        predictions: Array of probabilities
        gan_values: Array of gain values
        window_size: Window size for smoothing (default 2001)
        
    Returns:
        Maximum smoothed gain
    """
    # Sort by probability descending
    sorted_idx = np.argsort(-predictions)
    sorted_gan = gan_values[sorted_idx]
    
    # Calculate cumulative gain
    gan_acum = np.cumsum(sorted_gan)
    
    # Calculate smoothed gain (meseta) using rolling mean
    # Pad with NaN for proper alignment
    half_window = window_size // 2
    padded_gan = np.concatenate([
        np.full(half_window, np.nan),
        gan_acum,
        np.full(half_window, np.nan)
    ])
    
    # Calculate rolling mean
    gan_meseta = np.full(len(gan_acum), np.nan)
    for i in range(len(gan_acum)):
        window = padded_gan[i:i + window_size]
        valid_values = window[~np.isnan(window)]
        if len(valid_values) > 0:
            gan_meseta[i] = np.mean(valid_values)
    
    # Return max gain
    max_gain = np.nanmax(gan_meseta)
    return max_gain


def create_objective_function(dtrain: lgb.Dataset, df_test: pl.DataFrame, 
                              test_matrix: np.ndarray, config: Dict,
                              n_train: int, log_file: str = "output/BO_log.txt"):
    """
    Create Optuna objective function
    
    Args:
        dtrain: Training dataset
        df_test: Testing DataFrame with gan column
        test_matrix: Testing feature matrix
        config: Configuration dictionary
        n_train: Number of training samples
        log_file: Path to log file
        
    Returns:
        Objective function for Optuna
    """
    gan_values = df_test.select("gan").to_numpy().ravel()
    fixed_params = config["lgbm"]["param_fijos"].copy()
    fixed_params["seed"] = config["semilla_primigenia"]
    
    # Initialize log file
    os.makedirs("output", exist_ok=True)
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("fecha\titer\tnum_iterations\tlearning_rate\tfeature_fraction\t"
                   "min_data_in_leaf\tnum_leaves\tmetrica\tmetrica_mejor\n")
    
    # Track best result
    best_gain = {"value": -np.inf, "iter": 0}
    iteration = {"count": 0}
    
    def objective(trial: optuna.Trial) -> float:
        iteration["count"] += 1
        
        # Suggest hyperparameters
        num_iterations = trial.suggest_int("num_iterations", 1, 2048, log=True)
        learning_rate = trial.suggest_float("learning_rate", 1/256, 1/2, log=True)
        feature_fraction = trial.suggest_float("feature_fraction", 0.05, 1.0)
        min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 1, 8192, log=True)
        num_leaves = trial.suggest_int("num_leaves", 2, 1024, log=True)
        
        # Check constraint: min_data_in_leaf * num_leaves <= n_training
        if min_data_in_leaf * num_leaves > n_train:
            return -np.inf
        
        # Combine parameters
        params = fixed_params.copy()
        params.update({
            "num_iterations": num_iterations,
            "learning_rate": learning_rate,
            "feature_fraction": feature_fraction,
            "min_data_in_leaf": min_data_in_leaf,
            "num_leaves": num_leaves
        })
        
        # Train model
        modelo = lgb.train(
            params,
            dtrain,
            num_boost_round=num_iterations,
            valid_sets=None,
            callbacks=[lgb.log_evaluation(period=0)]
        )
        
        # Predict on test
        predictions = modelo.predict(test_matrix)
        
        # Calculate gain
        gain = calculate_gain_with_meseta(predictions, gan_values)
        
        # Update best
        if gain > best_gain["value"]:
            best_gain["value"] = gain
            best_gain["iter"] = iteration["count"]
            
            # Save feature importance
            importance_df = pl.DataFrame({
                "Feature": modelo.feature_name(),
                "Gain": modelo.feature_importance(importance_type="gain")
            })
            importance_df = importance_df.sort("Gain", descending=True)
            importance_df.write_csv(f"output/impo_{iteration['count']}.txt", separator="\t")
        
        # Log to file
        timestamp = datetime.now().strftime("%Y%m%d.%H%M%S")
        with open(log_file, "a") as f:
            f.write(f"{timestamp}\t{iteration['count']}\t{num_iterations}\t"
                   f"{learning_rate}\t{feature_fraction}\t{min_data_in_leaf}\t"
                   f"{num_leaves}\t{gain}\t{best_gain['value']}\n")
        
        return gain
    
    return objective


def run_bayesian_optimization(dtrain: lgb.Dataset, df_test: pl.DataFrame,
                              test_matrix: np.ndarray, config: Dict, n_train: int) -> Dict:
    """
    Run Bayesian Optimization with Optuna
    
    Args:
        dtrain: Training dataset
        df_test: Testing DataFrame
        test_matrix: Testing feature matrix
        config: Configuration dictionary
        n_train: Number of training samples
        
    Returns:
        Dictionary with best parameters
    """
    print("\n" + "="*50)
    print("RUNNING BAYESIAN OPTIMIZATION")
    print("="*50)
    
    n_trials = config["hipeparametertuning"]["BO_iteraciones"]
    experimento = config["experimento"]
    
    # Create db directory
    os.makedirs("db", exist_ok=True)
    
    # Create database URL
    db_url = f"sqlite:///db/optuna_{experimento}.db"
    print(f"Study database: {db_url}")
    
    # Create objective function
    objective = create_objective_function(dtrain, df_test, test_matrix, config, n_train)
    
    # Create or load study (load_if_exists handles both cases)
    study_name = f"lgbm_{experimento}"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=db_url,
        direction="maximize",
        sampler=TPESampler(seed=config["semilla_primigenia"]),
        load_if_exists=True  # Loads if exists, creates if not
    )
    
    # Check if study already had trials
    trials_completed = len(study.trials)
    if trials_completed > 0:
        print(f"Loaded existing study: {trials_completed} trials already completed")
    else:
        print(f"Created new study: {study_name}")
    
    # Calculate remaining trials
    trials_remaining = max(0, n_trials - trials_completed)
    
    if trials_remaining == 0:
        print(f"All {n_trials} trials already completed!")
    else:
        print(f"Running {trials_remaining} remaining trials (out of {n_trials} total)...")
        # Run optimization
        study.optimize(objective, n_trials=trials_remaining, show_progress_bar=True)
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\nBest gain: {best_value:,.0f}")
    print("\nBest parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    return best_params


def train_final_models(df: pl.DataFrame, config: Dict, 
                      campos_buenos: List[str], best_params: Dict) -> None:
    """
    Train final ensemble of models
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        campos_buenos: List of feature columns
        best_params: Best hyperparameters from optimization
    """
    print("\n" + "="*50)
    print("TRAINING FINAL MODELS")
    print("="*50)
    
    # Prepare final training data
    df = df.with_columns([
        pl.when(pl.col("clase_ternaria").is_in(["BAJA+2", "BAJA+1"]))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("clase01")
    ])
    
    # Filter to training months
    training_months = config["train_final"]["training"]
    undersampling = config["train_final"]["undersampling"]
    
    # Add random column
    np.random.seed(config["semilla_primigenia"])
    azar = np.random.uniform(0, 1, df.shape[0])
    df = df.with_columns([
        pl.Series("azar", azar)
    ])
    
    # Mark training records
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
    
    # Get training data
    df_train_final = df.filter(pl.col("training") == 1)
    
    campos_buenos_valid = [col for col in campos_buenos 
                           if col in df.columns and col not in ["clase_ternaria", "clase01", "azar", "training"]]
    
    X_train_final = df_train_final.select(campos_buenos_valid).to_numpy()
    y_train_final = df_train_final.select("clase01").to_numpy().ravel()
    
    dtrain_final = lgb.Dataset(X_train_final, label=y_train_final, free_raw_data=False)
    
    print(f"Final training set: {X_train_final.shape[0]} rows, {X_train_final.shape[1]} columns")
    
    # Adjust min_data_in_leaf for larger training set
    original_train_size = df.filter(
        pl.col("foto_mes").is_in(config["trainingstrategy"]["training"]) &
        ((pl.col("azar") <= config["trainingstrategy"]["undersampling"]) | 
         (pl.col("clase_ternaria").is_in(["BAJA+1", "BAJA+2"])))
    ).shape[0]
    
    scale_factor = X_train_final.shape[0] / original_train_size
    adjusted_min_data = int(best_params["min_data_in_leaf"] * scale_factor)
    
    print(f"Adjusted min_data_in_leaf: {best_params['min_data_in_leaf']} -> {adjusted_min_data}")
    
    # Prepare parameters
    final_params = config["lgbm"]["param_fijos"].copy()
    final_params.update(best_params)
    final_params["min_data_in_leaf"] = adjusted_min_data
    
    # Generate seeds
    np.random.seed(config["semilla_primigenia"])
    # Generate prime numbers for seeds (simplified - just use random large numbers)
    seeds = np.random.randint(100000, 1000000, size=config["train_final"]["ksemillerio"])
    
    # Create modelitos directory
    os.makedirs("output/modelitos", exist_ok=True)
    
    # Train ensemble
    print(f"Training {len(seeds)} models...")
    for idx, seed in enumerate(seeds):
        model_file = f"output/modelitos/mod_{seed}.txt"
        
        if os.path.exists(model_file):
            print(f"  Model {idx+1}/{len(seeds)} already exists, skipping...")
            continue
        
        print(f"  Training model {idx+1}/{len(seeds)} (seed={seed})...")
        
        params = final_params.copy()
        params["seed"] = int(seed)
        
        modelo = lgb.train(
            params,
            dtrain_final,
            num_boost_round=params["num_iterations"],
            valid_sets=None,
            callbacks=[lgb.log_evaluation(period=0)]
        )
        
        modelo.save_model(model_file)
    
    print("\nFinal models trained and saved!")

