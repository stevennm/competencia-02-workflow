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
        
        # Data quality fixes (MICE imputation)
        print_section("STEP 3: DATA QUALITY FIXES (MICE)")
        df = data_quality_fixes(df)
        
        # Data drifting correction (IPC adjustment)
        print_section("STEP 4: DATA DRIFTING CORRECTION (IPC)")
        df = data_drifting_correction(df)
        
        # Save results
        print_section("SAVING PREPROCESSED DATA")
        print(f"Final shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
        print(f"Saving to: {output_file}")
        
        df.write_parquet(output_file)
        
        print(f"✓ Saved successfully!")
        
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

