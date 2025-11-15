"""
Standalone preprocessing script
Runs only the preprocessing steps and saves the result to a parquet file
"""

import sys
import time
from datetime import datetime
import polars as pl

from src.preprocessing import (
    load_data,
    generate_clase_ternaria,
    eliminate_features,
    data_quality_fixes,
    data_drifting_correction
)


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def main():
    """Main preprocessing execution"""
    start_time = time.time()
    print_section(f"PREPROCESSING ONLY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Input and output paths
        input_file = "data/competencia_02_target.parquet"
        output_file = "data/preprocessed_data.parquet"
        
        print(f"\nInput:  {input_file}")
        print(f"Output: {output_file}")
        
        # =====================================================================
        # PREPROCESSING STEPS
        # =====================================================================
        print_section("STEP 1: LOAD DATA")
        df = load_data(input_file)
        
        print_section("STEP 2: GENERATE CLASE_TERNARIA")
        df = generate_clase_ternaria(df)
        
        print_section("STEP 3: ELIMINATE FEATURES")
        df = eliminate_features(df)
        
        print_section("STEP 4: DATA QUALITY FIXES (MICE)")
        df = data_quality_fixes(df)
        
        print_section("STEP 5: DATA DRIFTING CORRECTION (IPC)")
        df = data_drifting_correction(df)
        
        # =====================================================================
        # SAVE RESULTS
        # =====================================================================
        print_section("SAVING PREPROCESSED DATA")
        print(f"Final shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Saving to: {output_file}")
        
        df.write_parquet(output_file)
        
        print(f"✓ Saved successfully!")
        
        # =====================================================================
        # COMPLETION
        # =====================================================================
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        print_section("PREPROCESSING COMPLETED SUCCESSFULLY")
        print(f"\nExecution time: {minutes:02d}m {seconds:02d}s")
        print(f"Output file: {output_file}")
        print(f"\nNext step: Run 'python run_feature_engineering.py' to add features")
        
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

