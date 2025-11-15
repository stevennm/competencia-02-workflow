"""
Step 3: Feature Engineering
Loads preprocessed data and adds all feature engineering steps
Configuration from workflow-jueves notebook
"""

import sys
import time
from datetime import datetime
import polars as pl

# Import from local feature engineering module
from feature_engineering_utils import add_intra_month_features, add_historical_features


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# (from workflow-jueves notebook / config_old.py)
# =============================================================================

FE_CONFIG = {
    "FE_hist": {
        "Tendencias": {
            "run": True,
            "ventana": 6,              # 6-month window
            "tendencia": True,          # Linear trend (slope)
            "minimo": True,             # Rolling minimum
            "maximo": True,             # Rolling maximum
            "promedio": True,           # Rolling average
            "ratioavg": True,           # Ratio to average
            "ratiomax": True            # Ratio to maximum
        }
    }
}


def main():
    """Main feature engineering execution"""
    start_time = time.time()
    print_section(f"FEATURE ENGINEERING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Input and output paths
        input_file = "data/preprocessed_data.parquet"
        output_file = "data/featured_data.parquet"
        
        print(f"\nInput:  {input_file}")
        print(f"Output: {output_file}")
        
        # Display FE configuration
        print("\nFeature Engineering Configuration:")
        tend_config = FE_CONFIG["FE_hist"]["Tendencias"]
        print(f"  Ventana: {tend_config['ventana']} months")
        print(f"  Tendencia: {tend_config['tendencia']}")
        print(f"  Minimo: {tend_config['minimo']}")
        print(f"  Maximo: {tend_config['maximo']}")
        print(f"  Promedio: {tend_config['promedio']}")
        print(f"  Ratio Avg: {tend_config['ratioavg']}")
        print(f"  Ratio Max: {tend_config['ratiomax']}")
        
        # Load preprocessed data
        print_section("LOADING PREPROCESSED DATA")
        print(f"Loading from {input_file}...")
        df = pl.read_parquet(input_file)
        print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
        
        # Intra-month feature engineering
        print_section("STEP 1: INTRA-MONTH FEATURE ENGINEERING")
        df = add_intra_month_features(df)
        
        # Historical feature engineering (lags, deltas, trends)
        print_section("STEP 2: HISTORICAL FEATURE ENGINEERING")
        
        # Define columns that can have lags
        cols_lagueables = [col for col in df.columns 
                          if col not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]]
        
        print(f"Base features for lags: {len(cols_lagueables)}")
        
        df = add_historical_features(df, cols_lagueables, FE_CONFIG)
        
        # Save results
        print_section("SAVING FEATURED DATA")
        
        # Final feature count
        campos_buenos_final = [col for col in df.columns 
                              if col not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]]
        
        print(f"Final shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
        print(f"Total features: {len(campos_buenos_final)}")
        print(f"Saving to: {output_file}")
        
        df.write_parquet(output_file)
        
        print(f"✓ Saved successfully!")
        
        # Feature breakdown
        print("\nFeature breakdown:")
        lag_features = len([col for col in df.columns if '_lag' in col.lower()])
        delta_features = len([col for col in df.columns if '_delta' in col.lower()])
        trend_features = len([col for col in df.columns if '_tend' in col.lower()])
        ratio_features = len([col for col in df.columns if '_ratio' in col.lower()])
        maxmin_features = len([col for col in df.columns if '_max' in col.lower() or '_min' in col.lower()])
        
        print(f"  Lag features: {lag_features}")
        print(f"  Delta features: {delta_features}")
        print(f"  Trend features: {trend_features}")
        print(f"  Ratio features: {ratio_features}")
        print(f"  MaxMin features: {maxmin_features}")
        
        # Completion
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

