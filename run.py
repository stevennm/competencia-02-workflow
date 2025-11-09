"""
Main orchestrator script for the LightGBM workflow
Executes the complete pipeline from data loading to Kaggle submission

For running steps independently, see:
- run_preprocessing.py (preprocessing only)
- run_feature_engineering.py (feature engineering only)
- STANDALONE_SCRIPTS.md (documentation)
"""

import sys
import time
import logging
from datetime import datetime
from pathlib import Path
import polars as pl
import os

# Import all modules
from src.config import PARAM
from src.preprocessing import preprocess_data
from src.feature_engineering import add_intra_month_features, add_historical_features
from src.rf_features import add_rf_features
from src.training import (
    prepare_training_data, 
    run_bayesian_optimization, 
    train_final_models
)
from src.scoring import score_future_data, generate_submission
from src.gain_analysis import create_gain_curve


def setup_logging():
    """Setup logging to file and console"""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{timestamp}.log"
    
    # Configure logging
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
    """Main execution function"""
    start_time = time.time()
    
    # Setup logging
    logger = setup_logging()
    
    print_section(f"LIGHTGBM WORKFLOW - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Log experiment configuration
    logger.info(f"Experiment: {PARAM['experimento']}")
    logger.info(f"Seed: {PARAM['semilla_primigenia']}")
    print(f"\nExperiment: {PARAM['experimento']}")
    print(f"Seed: {PARAM['semilla_primigenia']}")
    
    try:
        # =====================================================================
        # STEP 1: PREPROCESSING
        # =====================================================================
        # print_section("STEP 1: PREPROCESSING")
        # df = preprocess_data("data/competencia_02_target.parquet")
        # df = pl.read_parquet("data/preprocessed_data.parquet")
        
        # =====================================================================
        # STEP 2: FEATURE ENGINEERING - INTRA-MONTH
        # =====================================================================
        # print_section("STEP 2: INTRA-MONTH FEATURE ENGINEERING")
        # df = add_intra_month_features(df)
        
        # =====================================================================
        # STEP 3: FEATURE ENGINEERING - HISTORICAL
        # =====================================================================
        # print_section("STEP 3: HISTORICAL FEATURE ENGINEERING")
        
        # # Define columns that can have lags
        # cols_lagueables = [col for col in df.columns 
        #                   if col not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]]
        
        # print(f"Base features for lags: {len(cols_lagueables)}")
        
        # df = add_historical_features(df, cols_lagueables, PARAM)
        logger.info("Loading featured data from parquet file")
        df = pl.read_parquet("data/featured_data.parquet")
        logger.info(f"Loaded featured data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # =====================================================================
        # STEP 4: RANDOM FOREST FEATURES
        # =====================================================================
        print_section("STEP 4: RANDOM FOREST LEAF FEATURES")
        
        # Update campos_buenos to include all features except identifiers and target
        campos_buenos = [col for col in df.columns 
                        if col not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]]
        
        df = add_rf_features(df, PARAM, campos_buenos)
        
        # =====================================================================
        # STEP 5: PREPARE DATA FOR TRAINING
        # =====================================================================
        print_section("STEP 5: PREPARE TRAINING DATA")
        
        # Update campos_buenos again after RF features
        campos_buenos = [col for col in df.columns 
                        if col not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]]
        
        logger.info(f"Total features: {len(campos_buenos)}")
        print(f"Total features: {len(campos_buenos)}")
        
        logger.info("Preparing training data...")
        dtrain, df_test, test_matrix, campos_buenos_valid, n_train = prepare_training_data(
            df, PARAM, campos_buenos
        )
        logger.info(f"Training samples: {n_train}")
        
        # =====================================================================
        # STEP 6: BAYESIAN OPTIMIZATION
        # =====================================================================
        print_section("STEP 6: BAYESIAN OPTIMIZATION WITH OPTUNA")
        
        logger.info("Starting Bayesian optimization...")
        best_params = run_bayesian_optimization(dtrain, df_test, test_matrix, PARAM, n_train)
        logger.info(f"Best parameters found: {best_params}")
        
        # Store best parameters
        PARAM["train_final"]["param_mejores"] = best_params
        
        # =====================================================================
        # STEP 7: TRAIN FINAL MODELS
        # =====================================================================
        print_section("STEP 7: TRAIN FINAL ENSEMBLE")
        
        logger.info("Training final ensemble models...")
        train_final_models(df, PARAM, campos_buenos, best_params)
        logger.info("Final models trained successfully")
        
        # =====================================================================
        # STEP 8: SCORING
        # =====================================================================
        print_section("STEP 8: SCORE FUTURE DATA")
        
        logger.info("Scoring future data...")
        df_pred = score_future_data(df, PARAM, campos_buenos)
        logger.info(f"Scored {len(df_pred)} predictions")
        
        # =====================================================================
        # STEP 9: GENERATE SUBMISSION
        # =====================================================================
        print_section("STEP 9: GENERATE KAGGLE SUBMISSION")
        
        logger.info("Generating Kaggle submission...")
        generate_submission(df_pred, PARAM, n_envios=11000)
        logger.info(f"Submission file: kaggle/KA{PARAM['experimento']}_11000.csv")
        
        # =====================================================================
        # STEP 10: GAIN CURVE ANALYSIS (if labels available)
        # =====================================================================
        print_section("STEP 10: GAIN CURVE ANALYSIS")
        
        logger.info("Creating gain curve analysis...")
        create_gain_curve(df, df_pred, PARAM)
        logger.info("Gain curve analysis complete")
        
        # =====================================================================
        # COMPLETION
        # =====================================================================
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        print_section("WORKFLOW COMPLETED SUCCESSFULLY")
        
        time_str = f"{hours:02d}h {minutes:02d}m {seconds:02d}s"
        logger.info(f"Total execution time: {time_str}")
        logger.info(f"Experiment: {PARAM['experimento']}")
        
        print(f"\nTotal execution time: {time_str}")
        print(f"Experiment: {PARAM['experimento']}")
        print(f"\nOutput files (in output/ directory):")
        print(f"  - output/BO_log.txt (Bayesian Optimization log)")
        print(f"  - output/prediccion.txt (Full predictions)")
        print(f"  - output/kaggle/KA{PARAM['experimento']}_11000.csv (Submission file)")
        print(f"  - output/modelitos/ (Trained models directory)")
        print(f"  - output/impo_*.txt (Feature importance files)")
        print(f"  - output/modelo.model (Random Forest model)")
        print(f"  - output/analysis/gain_curve.png (Gain curve visualization)")
        print(f"  - output/analysis/gain_by_cutoff.csv (Gain statistics)")
        
        logger.info("All output files saved successfully")
        logger.info("="*70)
        logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
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

