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
from src.preprocessing import load_data, add_canaritos
from src.training_zlgbm import train_zlgbm_final_model
from src.scoring_zlgbm import score_zlgbm_future_data, generate_zlgbm_submission
from src.bucket_utils import get_bucket_info, sync_output_to_bucket


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


def print_section(title: str):
    """Print a formatted section header"""
    separator = "="*70
    print(f"\n{separator}")
    print(f"  {title}")
    print(separator)
    logging.info(separator)
    logging.info(f"  {title}")
    logging.info(separator)


def main():
    """Main execution function for zLightGBM workflow"""
    start_time = time.time()
    
    # Setup logging
    logger = setup_logging()
    
    print_section(f"zLightGBM WORKFLOW - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Update experiment name for zLightGBM
    PARAM["experimento"] = f"{PARAM['experimento']}_zlgbm"
    
    # Check if experiment already exists
    experimento = PARAM['experimento']
    experiment_dir = Path(f"output/{experimento}")
    
    if experiment_dir.exists():
        error_msg = f"ERROR: Experiment '{experimento}' already exists at output/{experimento}/"
        print(f"\n❌ {error_msg}")
        print(f"💡 Change experiment name in config.py or delete the directory")
        logger.error(error_msg)
        raise FileExistsError(error_msg)
    
    # Log experiment configuration
    zlgbm_config = PARAM["zlgbm"]
    
    logger.info("="*70)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("="*70)
    logger.info(f"Experiment: {PARAM['experimento']}")
    logger.info(f"Seed: {PARAM['semilla_primigenia']}")
    logger.info(f"Mode: zLightGBM (no Bayesian Optimization)")
    logger.info("")
    logger.info("Training Strategy:")
    logger.info(f"  Training months: {zlgbm_config['train_final']['training']}")
    logger.info(f"  Future months: {zlgbm_config['train_final']['future']}")
    logger.info(f"  Undersampling: {zlgbm_config['train_final']['undersampling']} ({zlgbm_config['train_final']['undersampling']*100:.0f}%)")
    logger.info(f"  Models in ensemble: {zlgbm_config['train_final']['ksemillerio']}")
    logger.info("")
    logger.info("zLightGBM Parameters:")
    logger.info(f"  Canaries: {zlgbm_config['qcanaritos']}")
    logger.info(f"  gradient_bound: {zlgbm_config['param']['gradient_bound']}")
    logger.info(f"  learning_rate: {zlgbm_config['param']['learning_rate']}")
    logger.info(f"  feature_fraction: {zlgbm_config['param']['feature_fraction']}")
    logger.info(f"  min_data_in_leaf: {zlgbm_config['param']['min_data_in_leaf']}")
    logger.info(f"  max_bin: {zlgbm_config['param']['max_bin']}")
    logger.info(f"  num_iterations (max): {zlgbm_config['param']['num_iterations']}")
    logger.info(f"  num_leaves (max): {zlgbm_config['param']['num_leaves']}")
    logger.info("="*70)
    
    print(f"\n🚀 Mode: zLightGBM")
    print(f"Experiment: {PARAM['experimento']}")
    print(f"Seed: {PARAM['semilla_primigenia']}")
    print(f"\n📊 Configuration:")
    print(f"  Training months: {zlgbm_config['train_final']['training']}")
    print(f"  Future months: {zlgbm_config['train_final']['future']}")
    print(f"  Undersampling: {zlgbm_config['train_final']['undersampling']*100:.0f}%")
    print(f"  Canaries: {zlgbm_config['qcanaritos']}")
    print(f"  gradient_bound: {zlgbm_config['param']['gradient_bound']}")
    print(f"  feature_fraction: {zlgbm_config['param']['feature_fraction']}")
    
    # Log bucket configuration
    bucket_info = get_bucket_info(PARAM)
    if bucket_info["enabled"]:
        logger.info("")
        logger.info("Bucket Configuration:")
        logger.info(f"  Enabled: Yes")
        logger.info(f"  Base path: {bucket_info['base_path']}")
        logger.info(f"  Exp path: {bucket_info['exp_path']}")
        logger.info(f"  Bucket exists: {bucket_info['exists']}")
        print(f"\n💾 Bucket: {bucket_info['base_path']}")
    else:
        logger.info("Bucket: Disabled")
        print(f"\n💾 Bucket: Disabled")
    
    try:
        # =====================================================================
        # STEP 1: LOAD PREPROCESSED DATA
        # =====================================================================
        print_section("STEP 1: LOAD PREPROCESSED DATA")
        
        # Assuming you already have preprocessed data with feature engineering
        data_file = "data/final_dataset.parquet"
        
        if not Path(data_file).exists():
            raise FileNotFoundError(
                f"Data file not found: {data_file}\n"
                f"Run the preprocessing and feature engineering steps first:\n"
                f"  1. python run_preprocessing.py\n"
                f"  2. python run_feature_engineering.py"
            )
        
        df = pl.read_parquet(data_file)
        logger.info(f"Loaded data: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
        logger.info(f"Columns: {len(df.columns)}")
        
        # Log month distribution
        month_counts = df.group_by("foto_mes").agg(pl.count().alias("count")).sort("foto_mes")
        logger.info("Month distribution:")
        for row in month_counts.iter_rows():
            logger.info(f"  {row[0]}: {row[1]:,} records")
        
        print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
        
        # =====================================================================
        # STEP 2: ADD CANARY FEATURES
        # =====================================================================
        print_section("STEP 2: ADD CANARY FEATURES")
        
        n_canaritos = PARAM["zlgbm"]["qcanaritos"]
        df, canaritos_names = add_canaritos(df, n_canaritos, PARAM["semilla_primigenia"])
        
        logger.info(f"Added {n_canaritos} canaries")
        print(f"\n✓ Added {n_canaritos} canary features")
        print(f"✓ New dataset shape: {df.shape}")
        
        # =====================================================================
        # STEP 3: PREPARE FEATURE LIST
        # =====================================================================
        print_section("STEP 3: PREPARE FEATURES")
        
        # Get base features (excluding identifiers and target)
        campos_buenos_base = [col for col in df.columns 
                             if col not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]
                             and not col.startswith("canarito_")]
        
        # Canaries MUST be first
        campos_buenos = canaritos_names + campos_buenos_base
        
        logger.info(f"Total features: {len(campos_buenos):,}")
        logger.info(f"  Canaries: {len(canaritos_names)}")
        logger.info(f"  Real features: {len(campos_buenos_base):,}")
        logger.info(f"  First 5 features: {campos_buenos[:5]}")
        logger.info(f"  Last 5 features: {campos_buenos[-5:]}")
        
        print(f"\nFeature composition:")
        print(f"  Canaries: {len(canaritos_names)}")
        print(f"  Real features: {len(campos_buenos_base):,}")
        print(f"  Total: {len(campos_buenos):,}")
        
        # Verify canaries are first
        if campos_buenos[:n_canaritos] != canaritos_names:
            raise ValueError("ERROR: Canaries are not at the beginning of feature list!")
        
        print(f"✓ Canaries verified at the beginning")
        
        # =====================================================================
        # STEP 4: TRAIN zLightGBM MODEL
        # =====================================================================
        print_section("STEP 4: TRAIN zLightGBM MODEL")
        
        logger.info("Training zLightGBM model...")
        logger.info(f"Training months: {zlgbm_config['train_final']['training']}")
        logger.info(f"Undersampling: {zlgbm_config['train_final']['undersampling']}")
        train_zlgbm_final_model(df, PARAM, campos_buenos)
        logger.info("Model trained successfully")
        
        # =====================================================================
        # STEP 5: SCORE FUTURE DATA
        # =====================================================================
        print_section("STEP 5: SCORE FUTURE DATA")
        
        logger.info("Scoring future data...")
        logger.info(f"Future months: {zlgbm_config['train_final']['future']}")
        df_pred = score_zlgbm_future_data(df, PARAM, campos_buenos)
        logger.info(f"Scored {len(df_pred):,} predictions")
        logger.info(f"Prediction stats:")
        logger.info(f"  Min prob: {df_pred['prob'].min():.6f}")
        logger.info(f"  Max prob: {df_pred['prob'].max():.6f}")
        logger.info(f"  Mean prob: {df_pred['prob'].mean():.6f}")
        logger.info(f"  Median prob: {df_pred['prob'].median():.6f}")
        
        # =====================================================================
        # STEP 6: GENERATE KAGGLE SUBMISSION
        # =====================================================================
        print_section("STEP 6: GENERATE KAGGLE SUBMISSION")
        
        n_envios = 11000
        logger.info("Generating Kaggle submission...")
        logger.info(f"Cutoff: {n_envios:,} envíos")
        generate_zlgbm_submission(df_pred, PARAM, n_envios=n_envios)
        logger.info(f"Submission file: output/kaggle/KA{PARAM['experimento']}_{n_envios}.csv")
        
        # =====================================================================
        # STEP 7: SYNC TO BUCKET
        # =====================================================================
        if PARAM.get("bucket", {}).get("enabled", False):
            print_section("STEP 7: SYNC TO BUCKET")
            logger.info("Syncing outputs to bucket...")
            sync_output_to_bucket(PARAM, logger)
            logger.info("Bucket sync complete")
        
        # =====================================================================
        # COMPLETION
        # =====================================================================
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        print_section("zLightGBM WORKFLOW COMPLETED SUCCESSFULLY")
        
        time_str = f"{hours:02d}h {minutes:02d}m {seconds:02d}s"
        logger.info(f"Total execution time: {time_str}")
        logger.info(f"Experiment: {PARAM['experimento']}")
        
        print(f"\n⏱️  Total execution time: {time_str}")
        print(f"📁 Experiment: {PARAM['experimento']}")
        print(f"\n📊 Output files:")
        print(f"  - output/{experimento}/zmodelo.txt (zLightGBM model)")
        print(f"  - output/{experimento}/tb_arboles.txt (Tree structure)")
        print(f"  - output/{experimento}/prediccion.txt (Predictions)")
        print(f"  - output/kaggle/KA{PARAM['experimento']}_11000.csv (Submission)")
        
        print(f"\n💡 Key advantages of zLightGBM:")
        print(f"  ✓ No Bayesian Optimization needed (faster)")
        print(f"  ✓ Automatic overfitting control (canaries)")
        print(f"  ✓ Single model (no ensemble needed)")
        print(f"  ✓ Self-stopping (optimal tree count)")
        
        logger.info("All output files saved successfully")
        logger.info("="*70)
        logger.info("zLightGBM WORKFLOW COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        
        return 0
        
    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        print(f"\n{'='*70}")
        print(error_msg)
        print(f"{'='*70}")
        
        logging.error(error_msg)
        logging.error("="*70)
        
        import traceback
        traceback.print_exc()
        
        # Log full traceback
        logging.error("Full traceback:")
        logging.error(traceback.format_exc())
        
        return 1


if __name__ == "__main__":
    sys.exit(main())

