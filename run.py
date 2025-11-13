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
    train_validation_ensemble,
    train_final_models
)
from src.scoring import score_future_data, generate_submission
from src.gain_analysis import create_gain_curve, create_ensemble_gain_curve, create_validation_gain_curve
from src.bucket_utils import get_bucket_info, sync_output_to_bucket


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
    bo_iterations = PARAM["hipeparametertuning"]["BO_iteraciones"]
    
    logger.info("="*70)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("="*70)
    logger.info(f"Experiment: {PARAM['experimento']}")
    logger.info(f"Seed: {PARAM['semilla_primigenia']}")
    logger.info(f"Mode: {'Bayesian Optimization' if bo_iterations > 0 else 'Pre-configured hyperparameters'}")
    logger.info("")
    logger.info("Training Strategy (BO):")
    logger.info(f"  Training months: {PARAM['trainingstrategy']['training']}")
    logger.info(f"  Testing months: {PARAM['trainingstrategy']['testing']}")
    logger.info(f"  Undersampling: {PARAM['trainingstrategy']['undersampling']} ({PARAM['trainingstrategy']['undersampling']*100:.0f}%)")
    logger.info("")
    logger.info("Final Training Strategy:")
    logger.info(f"  Training months: {PARAM['train_final']['training']}")
    logger.info(f"  Future months: {PARAM['train_final']['future']}")
    logger.info(f"  Undersampling: {PARAM['train_final']['undersampling']} ({PARAM['train_final']['undersampling']*100:.0f}%)")
    logger.info(f"  Models in ensemble: {PARAM['train_final']['ksemillerio']}")
    logger.info("")
    logger.info("Hyperparameter Tuning:")
    logger.info(f"  BO iterations: {bo_iterations}")
    if bo_iterations > 0:
        logger.info(f"  ksemillerio (models per trial): {PARAM['hipeparametertuning']['ksemillerio']}")
        logger.info(f"  repe (repetitions): {PARAM['hipeparametertuning']['repe']}")
        logger.info(f"  Total models per trial: {PARAM['hipeparametertuning']['ksemillerio'] * PARAM['hipeparametertuning']['repe']}")
    else:
        logger.info(f"  Using pre-configured params: {PARAM['train_final']['param_mejores']}")
    logger.info("")
    logger.info("Feature Engineering:")
    logger.info(f"  RF features: {PARAM['FE_rf']['arbolitos']} trees")
    logger.info(f"  Historical features: ventana={PARAM['FE_hist']['Tendencias']['ventana']}")
    logger.info("="*70)
    
    print(f"\nExperiment: {PARAM['experimento']}")
    print(f"Seed: {PARAM['semilla_primigenia']}")
    print(f"Mode: {'Bayesian Optimization' if bo_iterations > 0 else 'Pre-configured hyperparameters'}")
    print(f"\n📊 Configuration:")
    print(f"  BO iterations: {bo_iterations}")
    print(f"  Training months (BO): {len(PARAM['trainingstrategy']['training'])} months")
    print(f"  Training months (Final): {len(PARAM['train_final']['training'])} months")
    print(f"  Future months: {PARAM['train_final']['future']}")
    print(f"  Final ensemble: {PARAM['train_final']['ksemillerio']} models")
    
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
        # logger.info("Loading featured data from parquet file")
        # df = pl.read_parquet("data/featured_data.parquet")
        # logger.info(f"Loaded featured data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # =====================================================================
        # STEP 4: RANDOM FOREST FEATURES
        # =====================================================================
        # print_section("STEP 4: RANDOM FOREST LEAF FEATURES")
        
        # # Update campos_buenos to include all features except identifiers and target
        # campos_buenos = [col for col in df.columns 
        #                 if col not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]]
        
        # df = add_rf_features(df, PARAM, campos_buenos)
        
        # print("Saving final dataset")
        # df.write_parquet("data/final_dataset.parquet")
        # print("Final dataset saved")
        
        logger.info("Loading final dataset from parquet file")
        df = pl.read_parquet("data/final_dataset.parquet")
        logger.info(f"Loaded final dataset: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
        logger.info(f"Columns: {len(df.columns)}")
        
        # Log month distribution
        month_counts = df.group_by("foto_mes").agg(pl.count().alias("count")).sort("foto_mes")
        logger.info("Month distribution:")
        for row in month_counts.iter_rows():
            logger.info(f"  {row[0]}: {row[1]:,} records")

        # =====================================================================
        # STEP 5: PREPARE DATA FOR TRAINING
        # =====================================================================
        print_section("STEP 5: PREPARE TRAINING DATA")
        
        # Update campos_buenos again after RF features
        campos_buenos = [col for col in df.columns 
                        if col not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]]
        
        logger.info(f"Total features: {len(campos_buenos):,}")
        logger.info(f"  First 5 features: {campos_buenos[:5]}")
        logger.info(f"  Last 5 features: {campos_buenos[-5:]}")
        print(f"Total features: {len(campos_buenos):,}")
        
        logger.info("Preparing training data...")
        dtrain, df_test, test_matrix, campos_buenos_valid, n_train = prepare_training_data(
            df, PARAM, campos_buenos
        )
        logger.info(f"Training samples: {n_train}")
        
        # =====================================================================
        # STEP 6: BAYESIAN OPTIMIZATION
        # =====================================================================
        bo_iterations = PARAM["hipeparametertuning"]["BO_iteraciones"]
        
        if bo_iterations > 0:
            print_section("STEP 6: BAYESIAN OPTIMIZATION WITH OPTUNA")
            
            logger.info("Starting Bayesian optimization...")
            logger.info(f"BO iterations: {bo_iterations}")
            logger.info(f"Training months: {PARAM['trainingstrategy']['training']}")
            logger.info(f"Testing months: {PARAM['trainingstrategy']['testing']}")
            best_params = run_bayesian_optimization(dtrain, df_test, test_matrix, PARAM, n_train)
            logger.info(f"Best parameters found:")
            for param, value in best_params.items():
                logger.info(f"  {param}: {value}")
            
            # Store best parameters
            PARAM["train_final"]["param_mejores"] = best_params
            
            # =====================================================================
            # STEP 7: VALIDATION ENSEMBLE & GAIN ANALYSIS
            # =====================================================================
            print_section("STEP 7: VALIDATION ENSEMBLE & GAIN ANALYSIS")
            
            logger.info("Training validation ensemble (Optuna split)...")
            validation_pred = train_validation_ensemble(df, PARAM, campos_buenos, best_params)
            logger.info("Validation ensemble trained")
            
            logger.info("Creating validation gain curve...")
            create_validation_gain_curve(validation_pred, PARAM)
            logger.info("Validation gain analysis complete")
        else:
            print_section("STEP 6: SKIPPING BAYESIAN OPTIMIZATION")
            print("Using pre-configured hyperparameters from config.py")
            
            best_params = PARAM["train_final"]["param_mejores"]
            if best_params is None:
                raise ValueError("BO_iteraciones=0 but param_mejores is None. Please set param_mejores in config.py")
            
            logger.info(f"Using hyperparameters: {best_params}")
            print(f"✓ Hyperparameters loaded: {best_params}")
            
            print("\nNote: Skipping validation ensemble analysis (requires Optuna split)")
            logger.info("Skipping validation ensemble (no Optuna split available)")
        
        # =====================================================================
        # STEP 8: TRAIN FINAL MODELS
        # =====================================================================
        print_section("STEP 8: TRAIN FINAL ENSEMBLE")
        
        logger.info("Training final ensemble models...")
        logger.info(f"Training months: {PARAM['train_final']['training']}")
        logger.info(f"Undersampling: {PARAM['train_final']['undersampling']}")
        logger.info(f"Number of models: {PARAM['train_final']['ksemillerio']}")
        logger.info(f"Hyperparameters: {best_params}")
        train_final_models(df, PARAM, campos_buenos, best_params)
        logger.info("Final models trained successfully")
        
        # =====================================================================
        # STEP 9: SCORING
        # =====================================================================
        print_section("STEP 9: SCORE FUTURE DATA")
        
        logger.info("Scoring future data...")
        logger.info(f"Future months: {PARAM['train_final']['future']}")
        df_pred = score_future_data(df, PARAM, campos_buenos)
        logger.info(f"Scored {len(df_pred):,} predictions")
        logger.info(f"Prediction stats:")
        logger.info(f"  Min prob: {df_pred['prob'].min():.6f}")
        logger.info(f"  Max prob: {df_pred['prob'].max():.6f}")
        logger.info(f"  Mean prob: {df_pred['prob'].mean():.6f}")
        logger.info(f"  Median prob: {df_pred['prob'].median():.6f}")
        
        # =====================================================================
        # STEP 10: GENERATE SUBMISSION
        # =====================================================================
        print_section("STEP 10: GENERATE KAGGLE SUBMISSION")
        
        n_envios = 11000
        logger.info("Generating Kaggle submission...")
        logger.info(f"Cutoff: {n_envios:,} envíos")
        generate_submission(df_pred, PARAM, n_envios=n_envios)
        logger.info(f"Submission file: output/kaggle/KA{PARAM['experimento']}_{n_envios}.csv")
        
        # =====================================================================
        # STEP 11: GAIN CURVE ANALYSIS (if labels available)
        # =====================================================================
        print_section("STEP 11: GAIN CURVE ANALYSIS")
        
        logger.info("Creating gain curve analysis...")
        create_gain_curve(df, df_pred, PARAM)
        logger.info("Gain curve analysis complete")
        
        logger.info("Creating ensemble gain curve...")
        create_ensemble_gain_curve(df, PARAM)
        logger.info("Ensemble gain curve complete")
        
        # =====================================================================
        # STEP 12: SYNC TO BUCKET
        # =====================================================================
        if PARAM.get("bucket", {}).get("enabled", False):
            print_section("STEP 12: SYNC TO BUCKET")
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

