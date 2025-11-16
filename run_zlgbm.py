"""
Main script for zLightGBM workflow (without Bayesian Optimization)

This is a simplified workflow that uses zLightGBM's automatic overfitting control
through canary features, eliminating the need for hyperparameter optimization.

Usage:
    uv run run_zlgbm.py
"""

import sys
import time
import logging
from datetime import datetime
from pathlib import Path
import polars as pl

# Import modules
from src.config import PARAM
from src.training_zlgbm import train_zlgbm_final_model
from src.scoring_zlgbm import score_zlgbm_future_data, generate_zlgbm_submission
from src.bucket_utils import get_bucket_info, sync_output_to_bucket

# Import preprocessing utilities (now in preprocessing package)
# Note: We don't need load_data or add_canaritos since we use preprocessed data
import numpy as np


def add_canaritos(df: pl.DataFrame, n_canaritos: int, seed: int) -> tuple[pl.DataFrame, list[str]]:
    """
    Add canary features (random noise) to detect overfitting
    
    Args:
        df: Input DataFrame
        n_canaritos: Number of canary features to add
        seed: Random seed
        
    Returns:
        Tuple of (DataFrame with canaries, list of canary names)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Adding {n_canaritos} canary features")
    
    np.random.seed(seed)
    n_rows = df.shape[0]
    
    # Generate canary features (random uniform between 0 and 1)
    canaritos_data = {}
    canaritos_names = []
    
    for i in range(n_canaritos):
        canary_name = f"canarito_{i+1}"
        canaritos_names.append(canary_name)
        canaritos_data[canary_name] = np.random.uniform(0, 1, n_rows)
    
    # Create DataFrame with canaries
    df_canaritos = pl.DataFrame(canaritos_data)
    
    # Add canaries to the beginning of the DataFrame
    df = pl.concat([df_canaritos, df], how="horizontal")
    
    logger.info(f"Added {n_canaritos} canaries at the beginning of DataFrame")
    
    return df, canaritos_names


def setup_logging():
    """Setup logging to file and console"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_zlgbm_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    
    return logger


def main():
    """Main execution function for zLightGBM workflow"""
    start_time = time.time()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info(f"ZLIGHTGBM WORKFLOW - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Update experiment name for zLightGBM
    PARAM["experimento"] = f"{PARAM['experimento']}_zlgbm"
    
    # Check if experiment already exists
    experimento = PARAM['experimento']
    experiment_dir = Path(f"output/{experimento}")
    
    if experiment_dir.exists():
        error_msg = f"ERROR: Experiment '{experimento}' already exists at output/{experimento}/"
        logger.error(error_msg)
        logger.error("Change experiment name in config.py or delete the directory")
        raise FileExistsError(error_msg)
    
    # Log experiment configuration
    zlgbm_config = PARAM["zlgbm"]
    
    logger.info("="*70)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("="*70)
    logger.info(f"Experiment: {PARAM['experimento']}")
    logger.info(f"Seed: {PARAM['semilla_primigenia']}")
    logger.info(f"Training months: {zlgbm_config['train_final']['training']}")
    logger.info(f"Future months: {zlgbm_config['train_final']['future']}")
    logger.info(f"Undersampling: {zlgbm_config['train_final']['undersampling']}")
    logger.info(f"Ensemble models: {zlgbm_config['train_final']['ksemillerio']}")
    logger.info(f"Canaries: {zlgbm_config['qcanaritos']}")
    logger.info(f"gradient_bound: {zlgbm_config['param']['gradient_bound']}")
    logger.info(f"learning_rate: {zlgbm_config['param']['learning_rate']}")
    logger.info(f"feature_fraction: {zlgbm_config['param']['feature_fraction']}")
    logger.info(f"min_data_in_leaf: {zlgbm_config['param']['min_data_in_leaf']}")
    logger.info(f"max_bin: {zlgbm_config['param']['max_bin']}")
    logger.info(f"num_iterations: {zlgbm_config['param']['num_iterations']}")
    logger.info(f"num_leaves: {zlgbm_config['param']['num_leaves']}")
    logger.info("="*70)
    
    # Log bucket configuration
    bucket_info = get_bucket_info(PARAM)
    if bucket_info["enabled"]:
        logger.info(f"Bucket: {bucket_info['base_path']}")
    else:
        logger.info("Bucket: Disabled")
    
    try:
        # Load preprocessed data
        logger.info("Loading preprocessed data")
        
        # Assuming you already have preprocessed data with feature engineering
        data_file = "data/final_dataset.parquet"
        
        if not Path(data_file).exists():
            raise FileNotFoundError(
                f"Data file not found: {data_file}\n"
                f"Run the preprocessing pipeline first:\n"
                f"  python preprocessing/run_all.py\n"
                f"\nOr run individual steps:\n"
                f"  1. python preprocessing/01_generate_clase_ternaria.py\n"
                f"  2. python preprocessing/02_preprocessing.py\n"
                f"  3. python preprocessing/03_feature_engineering.py\n"
                f"  4. python preprocessing/04_rf_features.py"
            )
        
        df = pl.read_parquet(data_file)
        logger.info(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
        
        # Replace 100% zero columns with nulls
        zero_cols = []
        for col in df.columns:
            if col not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]:
                if df[col].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
                    if (df[col] == 0).all():
                        zero_cols.append(col)
        
        if zero_cols:
            logger.info(f"Replacing {len(zero_cols)} 100% zero columns with nulls:")
            for col in zero_cols:
                logger.info(f"  - {col}")
            df = df.with_columns([pl.lit(None).alias(col) for col in zero_cols])
        
        # Add canary features
        logger.info("Adding canary features")
        
        n_canaritos = PARAM["zlgbm"]["qcanaritos"]
        df, canaritos_names = add_canaritos(df, n_canaritos, PARAM["semilla_primigenia"])
        
        logger.info(f"New dataset shape: {df.shape}")
        
        # Prepare feature list
        logger.info("Preparing feature list")
        
        # Get base features (excluding identifiers and target)
        campos_buenos_base = [col for col in df.columns 
                             if col not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]
                             and not col.startswith("canarito_")]
        
        # Canaries MUST be first
        campos_buenos = canaritos_names + campos_buenos_base
        
        logger.info(f"Total features: {len(campos_buenos):,}")
        logger.info(f"Canaries: {len(canaritos_names)}")
        logger.info(f"Real features: {len(campos_buenos_base):,}")
        
        # Verify canaries are first
        if campos_buenos[:n_canaritos] != canaritos_names:
            raise ValueError("ERROR: Canaries are not at the beginning of feature list!")
        
        logger.info("Canaries verified at the beginning of feature list")
        
        # Train model
        logger.info("Training zLightGBM model")
        train_zlgbm_final_model(df, PARAM, campos_buenos)
        
        # Score future data
        logger.info("Scoring future data")
        
        df_pred = score_zlgbm_future_data(df, PARAM, campos_buenos)
        logger.info(f"Scored {len(df_pred):,} predictions")
        logger.info(f"Min: {df_pred['prob'].min():.6f}, Max: {df_pred['prob'].max():.6f}, Mean: {df_pred['prob'].mean():.6f}")
        
        # Generate Kaggle submission
        logger.info("Generating Kaggle submission")
        n_envios = 11000
        generate_zlgbm_submission(df_pred, PARAM, n_envios=n_envios)
        logger.info(f"Submission: output/kaggle/KA{PARAM['experimento']}_{n_envios}.csv")
        
        # Sync to bucket
        if PARAM.get("bucket", {}).get("enabled", False):
            logger.info("Syncing to bucket")
            sync_output_to_bucket(PARAM, logger)
        
        # Completion
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        time_str = f"{hours:02d}h {minutes:02d}m {seconds:02d}s"
        
        logger.info(f"WORKFLOW COMPLETED - {time_str}")
        logger.info(f"Experiment: {PARAM['experimento']}")
        logger.info(f"Output: output/{experimento}/")
        
        return 0
        
    except Exception as e:
        import traceback
        logger.error(f"ERROR: {str(e)}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())

