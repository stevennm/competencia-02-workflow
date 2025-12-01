"""
Step 0: Join competencia_02 and competencia_03 datasets
Reads both raw datasets and concatenates them into a single file
"""

import sys
import time
from datetime import datetime
import polars as pl

# Fix encoding for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def main():
    """Main execution"""
    start_time = time.time()
    print_section(f"JOIN DATASETS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        input_file_02 = "data/competencia_02_crudo.csv.gz"
        input_file_03 = "data/competencia_03_crudo.csv.gz"
        output_file = "data/competencia_combined.csv.gz"
        
        print(f"\nInput 1:  {input_file_02}")
        print(f"Input 2:  {input_file_03}")
        print(f"Output:   {output_file}")
        
        # Load datasets
        print_section("LOADING DATASETS")
        
        df_02 = pl.read_csv(input_file_02, infer_schema_length=None)
        print(f"competencia_02: {df_02.shape[0]:,} rows, {df_02.shape[1]} columns")
        
        df_03 = pl.read_csv(input_file_03, infer_schema_length=None)
        print(f"competencia_03: {df_03.shape[0]:,} rows, {df_03.shape[1]} columns")
        
        # Concatenate
        print_section("CONCATENATING")
        df_combined = pl.concat([df_02, df_03])
        print(f"Combined: {df_combined.shape[0]:,} rows, {df_combined.shape[1]} columns")
        
        # Save
        print_section("SAVING")
        df_combined.write_csv(output_file)
        print(f"✓ Saved to {output_file}")
        
        elapsed_time = time.time() - start_time
        print(f"\nExecution time: {int(elapsed_time // 60):02d}m {int(elapsed_time % 60):02d}s")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
