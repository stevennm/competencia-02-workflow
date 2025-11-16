"""
Step 5: Random Forest Features (Optional)
Loads advanced_featured_data.parquet and adds RF leaf features
Configuration from workflow-jueves notebook
"""

import sys
import time
from datetime import datetime
import polars as pl

# Import from local RF features module
from rf_features_utils import add_rf_features


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


# =============================================================================
# RF FEATURES CONFIGURATION
# (from workflow-jueves notebook / config_old.py)
# =============================================================================

RF_CONFIG = {
    "experimento": "rf_feature_engineering",  # Temporary experiment name
    
    "FE_rf": {
        "run": True,
        
        "train": {
            "training": [202101, 202102, 202103]  # Last 3 months for training
        },
        
        "lgb_param": {
            # Parameters that can be changed
            "num_iterations": 20,
            "num_leaves": 16,
            "min_data_in_leaf": 100,
            "feature_fraction_bynode": 0.2,
            
            # LightGBM configured as Random Forest
            "boosting": "rf",
            "bagging_fraction": 1.0 - 1.0/2.718281828,  # (1 - 1/e) ≈ 0.632
            "bagging_freq": 1,
            "feature_fraction": 1.0,
            
            # Generic LightGBM parameters
            "max_bin": 31,
            "objective": "binary",
            "first_metric_only": True,
            "boost_from_average": True,
            "feature_pre_filter": False,
            "force_row_wise": True,
            "verbosity": -100,
            "max_depth": -1,
            "min_gain_to_split": 0.0,
            "min_sum_hessian_in_leaf": 0.001,
            "lambda_l1": 0.0,
            "lambda_l2": 0.0,
            "pos_bagging_fraction": 1.0,
            "neg_bagging_fraction": 1.0,
            "seed": 102191
        }
    }
}


def main():
    """Main RF features execution"""
    start_time = time.time()
    print_section(f"RF FEATURES - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Input and output paths
        input_file = "data/advanced_featured_data.parquet"
        output_file = "data/final_dataset.parquet"
        
        print(f"\nInput:  {input_file}")
        print(f"Output: {output_file}")
        
        # Display RF configuration
        print("\nRandom Forest Configuration:")
        rf_config = RF_CONFIG["FE_rf"]
        print(f"  Run: {rf_config['run']}")
        print(f"  Trees: {rf_config['lgb_param']['num_iterations']}")
        print(f"  Leaves per tree: {rf_config['lgb_param']['num_leaves']}")
        print(f"  Training months: {rf_config['train']['training']}")
        print(f"  Feature fraction bynode: {rf_config['lgb_param']['feature_fraction_bynode']}")
        print(f"  Bagging fraction: {rf_config['lgb_param']['bagging_fraction']:.3f}")
        print(f"  Min data in leaf: {rf_config['lgb_param']['min_data_in_leaf']}")
        
        # Load featured data
        print_section("LOADING ADVANCED FEATURED DATA")
        print(f"Loading from {input_file}...")
        df = pl.read_parquet(input_file)
        print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
        
        # Get campos_buenos (all features except metadata)
        campos_buenos = [col for col in df.columns 
                        if col not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]]
        
        print(f"Features available: {len(campos_buenos)}")
        
        # Add RF features (MEMORY OPTIMIZED - processes period by period)
        print_section("STEP 1: GENERATE RF LEAF FEATURES")
        df = add_rf_features(df, RF_CONFIG, campos_buenos)
        
        # Save results
        print_section("SAVING FINAL DATASET")
        
        # Final feature count
        campos_buenos_final = [col for col in df.columns 
                              if col not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]]
        
        rf_features = [col for col in df.columns if col.startswith("rf_")]
        
        print(f"Final shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
        print(f"Total features: {len(campos_buenos_final)}")
        print(f"RF features added: {len(rf_features)}")
        print(f"Saving to: {output_file}")
        
        df.write_parquet(output_file)
        
        print(f"✓ Saved successfully!")
        
        # Completion
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        print_section("RF FEATURES COMPLETED SUCCESSFULLY")
        print(f"\nExecution time: {minutes:02d}m {seconds:02d}s")
        print(f"Output file: {output_file}")
        print(f"\nNext step: Use final_dataset.parquet for training/modeling")
        
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


