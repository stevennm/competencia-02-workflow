"""
Step 2: Preprocessing
Loads competencia_02_target.parquet and applies preprocessing steps
"""

import sys
import time
from datetime import datetime
import polars as pl

# Import from local preprocessing module
from preprocessing_utils import (
    data_quality_fixes,
    data_drifting_correction
)


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def eliminate_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Eliminate features that cause data drifting problems
    
    Based on workflow-jueves notebook: "La auténtica salsa mágica"
    These features cause data drifting and should be removed:
    - cprestamos_personales
    - mprestamos_personales
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with problematic features removed
    """
    features_to_drop = ["cprestamos_personales", "mprestamos_personales"]
    
    existing_to_drop = [col for col in features_to_drop if col in df.columns]
    
    if existing_to_drop:
        print(f"\nDropping {len(existing_to_drop)} features causing data drifting:")
        for col in existing_to_drop:
            print(f"  - {col}")
        df = df.drop(existing_to_drop)
        print(f"Columns after drop: {df.shape[1]}")
    else:
        print("\nNo problematic features found to drop")
    
    return df


def eliminate_problematic_months(df: pl.DataFrame) -> pl.DataFrame:
    """
    Eliminate months with data quality issues
    
    These months had severe data quality problems (>80% zeros in many columns)
    that MICE was unable to fix properly. Instead of imputing, we remove them:
    - 201905
    - 201910
    - 202006
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with problematic months removed
    """
    problematic_months = [201905, 201910, 202006]
    
    # Check which months exist
    available_months = df.select("foto_mes").unique().to_series().to_list()
    months_to_remove = [m for m in problematic_months if m in available_months]
    
    if not months_to_remove:
        print("\nNo problematic months found to remove")
        return df
    
    initial_rows = df.shape[0]
    
    print(f"\nRemoving {len(months_to_remove)} problematic months:")
    for month in months_to_remove:
        month_rows = df.filter(pl.col("foto_mes") == month).shape[0]
        print(f"  - {month}: {month_rows:,} rows")
    
    # Filter out problematic months
    df = df.filter(~pl.col("foto_mes").is_in(problematic_months))
    
    final_rows = df.shape[0]
    removed_rows = initial_rows - final_rows
    
    print(f"\nRows before: {initial_rows:,}")
    print(f"Rows after:  {final_rows:,}")
    print(f"Removed:     {removed_rows:,} rows ({100*removed_rows/initial_rows:.2f}%)")
    
    return df


def main():
    """Main preprocessing execution"""
    start_time = time.time()
    print_section(f"PREPROCESSING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Input and output paths
        input_file = "data/competencia_02_target.parquet"
        output_file = "data/preprocessed_data.parquet"
        
        print(f"\nInput:  {input_file}")
        print(f"Output: {output_file}")
        
        # Load data
        print_section("STEP 1: LOAD DATA")
        print(f"Loading from {input_file}...")
        df = pl.read_parquet(input_file)
        print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
        
        # Eliminate problematic features
        print_section("STEP 2: ELIMINATE FEATURES (Data Drifting)")
        df = eliminate_features(df)
        
        # Eliminate problematic months
        print_section("STEP 3: ELIMINATE PROBLEMATIC MONTHS")
        df = eliminate_problematic_months(df)
        
        # Data quality fixes (MICE imputation) - DISABLED
        print_section("STEP 4: DATA QUALITY FIXES (MICE) - SKIPPED")
        print("[INFO] MICE imputation is DISABLED")
        print("       Problematic months (201905, 201910, 202006) have been removed instead")
        # df = data_quality_fixes(df)  # COMMENTED OUT
        
        # Data drifting correction (IPC adjustment)
        print_section("STEP 5: DATA DRIFTING CORRECTION (IPC)")
        df = data_drifting_correction(df)
        
        # Save results
        print_section("SAVING PREPROCESSED DATA")
        print(f"Final shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
        print(f"Saving to: {output_file}")
        
        df.write_parquet(output_file)
        
        print(f"[OK] Saved successfully!")
        
        # Completion
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        print_section("PREPROCESSING COMPLETED SUCCESSFULLY")
        print(f"\nExecution time: {minutes:02d}m {seconds:02d}s")
        print(f"Output file: {output_file}")
        print(f"\nNext step: Run '03_feature_engineering.py' to add features")
        
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

