"""
Standalone feature engineering script
Loads preprocessed data and adds all feature engineering steps
"""

import sys
import time
from datetime import datetime
import polars as pl

from src.config import PARAM
from src.feature_engineering import add_intra_month_features, add_historical_features
from src.rf_features import add_rf_features


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def main():
    """Main feature engineering execution"""
    start_time = time.time()
    print_section(f"FEATURE ENGINEERING ONLY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Input and output paths
        input_file = "data/preprocessed_data.parquet"
        output_file = "data/featured_data.parquet"
        
        print(f"\nInput:  {input_file}")
        print(f"Output: {output_file}")
        print(f"\nExperiment: {PARAM['experimento']}")
        print(f"Seed: {PARAM['semilla_primigenia']}")
        
        # =====================================================================
        # LOAD PREPROCESSED DATA
        # =====================================================================
        print_section("LOADING PREPROCESSED DATA")
        print(f"Loading from {input_file}...")
        df = pl.read_parquet(input_file)
        print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # =====================================================================
        # FEATURE ENGINEERING - INTRA-MONTH
        # =====================================================================
        print_section("STEP 1: INTRA-MONTH FEATURE ENGINEERING")
        df = add_intra_month_features(df)
        
        # =====================================================================
        # FEATURE ENGINEERING - HISTORICAL
        # =====================================================================
        print_section("STEP 2: HISTORICAL FEATURE ENGINEERING")
        
        # Define columns that can have lags
        cols_lagueables = [col for col in df.columns 
                          if col not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]]
        
        print(f"Base features for lags: {len(cols_lagueables)}")
        
        df = add_historical_features(df, cols_lagueables, PARAM)
        
        # =====================================================================
        # RANDOM FOREST FEATURES (OPTIONAL)
        # =====================================================================
        print_section("STEP 3: RANDOM FOREST LEAF FEATURES")
        
        # Update campos_buenos to include all features except identifiers and target
        campos_buenos = [col for col in df.columns 
                        if col not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]]
        
        print(f"Total features before RF: {len(campos_buenos)}")
        
        # Uncomment the line below to add RF features
        # df = add_rf_features(df, PARAM, campos_buenos)
        print("  [SKIPPED] RF features are commented out in the code")
        print("  To enable: uncomment the add_rf_features line in this script")
        
        # =====================================================================
        # SAVE RESULTS
        # =====================================================================
        print_section("SAVING FEATURED DATA")
        
        # Final feature count
        campos_buenos_final = [col for col in df.columns 
                              if col not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]]
        
        print(f"Final shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Total features: {len(campos_buenos_final)}")
        print(f"Saving to: {output_file}")
        
        df.write_parquet(output_file)
        
        print(f"✓ Saved successfully!")
        
        # =====================================================================
        # COMPLETION
        # =====================================================================
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        print_section("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
        print(f"\nExecution time: {minutes:02d}m {seconds:02d}s")
        print(f"Output file: {output_file}")
        print(f"\nNext step: Use this data for training/modeling")
        
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

