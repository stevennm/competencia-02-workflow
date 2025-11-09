"""
Main orchestrator script for the LightGBM workflow
Executes the complete pipeline from data loading to Kaggle submission
"""

import sys
import time
from datetime import datetime
import polars as pl

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


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def main():
    """Main execution function"""
    start_time = time.time()
    print_section(f"LIGHTGBM WORKFLOW - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\nExperiment: {PARAM['experimento']}")
    print(f"Seed: {PARAM['semilla_primigenia']}")
    
    try:
        # =====================================================================
        # STEP 1: PREPROCESSING
        # =====================================================================
        print_section("STEP 1: PREPROCESSING")
        df = preprocess_data("data/competencia_02_target.parquet")
        
        # =====================================================================
        # STEP 2: FEATURE ENGINEERING - INTRA-MONTH
        # =====================================================================
        print_section("STEP 2: INTRA-MONTH FEATURE ENGINEERING")
        df = add_intra_month_features(df)
        
        # =====================================================================
        # STEP 3: FEATURE ENGINEERING - HISTORICAL
        # =====================================================================
        print_section("STEP 3: HISTORICAL FEATURE ENGINEERING")
        
        # Define columns that can have lags
        cols_lagueables = [col for col in df.columns 
                          if col not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]]
        
        print(f"Base features for lags: {len(cols_lagueables)}")
        
        df = add_historical_features(df, cols_lagueables, PARAM)
        
        # =====================================================================
        # STEP 4: RANDOM FOREST FEATURES
        # =====================================================================
        print_section("STEP 4: RANDOM FOREST LEAF FEATURES")
        
        # Update campos_buenos to include all features except identifiers and target
        campos_buenos = [col for col in df.columns 
                        if col not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]]
        
        #df = add_rf_features(df, PARAM, campos_buenos)
        
        # =====================================================================
        # STEP 5: PREPARE DATA FOR TRAINING
        # =====================================================================
        print_section("STEP 5: PREPARE TRAINING DATA")
        
        # Update campos_buenos again after RF features
        campos_buenos = [col for col in df.columns 
                        if col not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]]
        
        print(f"Total features: {len(campos_buenos)}")
        
        dtrain, df_test, test_matrix, campos_buenos_valid, n_train = prepare_training_data(
            df, PARAM, campos_buenos
        )
        
        # =====================================================================
        # STEP 6: BAYESIAN OPTIMIZATION
        # =====================================================================
        print_section("STEP 6: BAYESIAN OPTIMIZATION WITH OPTUNA")
        
        best_params = run_bayesian_optimization(dtrain, df_test, test_matrix, PARAM, n_train)
        
        # Store best parameters
        PARAM["train_final"]["param_mejores"] = best_params
        
        # =====================================================================
        # STEP 7: TRAIN FINAL MODELS
        # =====================================================================
        print_section("STEP 7: TRAIN FINAL ENSEMBLE")
        
        train_final_models(df, PARAM, campos_buenos, best_params)
        
        # =====================================================================
        # STEP 8: SCORING
        # =====================================================================
        print_section("STEP 8: SCORE FUTURE DATA")
        
        df_pred = score_future_data(df, PARAM, campos_buenos)
        
        # =====================================================================
        # STEP 9: GENERATE SUBMISSION
        # =====================================================================
        print_section("STEP 9: GENERATE KAGGLE SUBMISSION")
        
        generate_submission(df_pred, PARAM, n_envios=11000)
        
        # =====================================================================
        # STEP 10: GAIN CURVE ANALYSIS (if labels available)
        # =====================================================================
        print_section("STEP 10: GAIN CURVE ANALYSIS")
        
        create_gain_curve(df, df_pred, PARAM)
        
        # =====================================================================
        # COMPLETION
        # =====================================================================
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        print_section("WORKFLOW COMPLETED SUCCESSFULLY")
        print(f"\nTotal execution time: {hours:02d}h {minutes:02d}m {seconds:02d}s")
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
        
        return 0
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: {str(e)}")
        print(f"{'='*70}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

