"""
Step 1: Generate clase_ternaria from raw data
Reads competencia_combined.csv.gz and creates competencia_02_target.parquet

IMPORTANT: 
- Uses infer_schema_length=None for robust type inference
- This analyzes the ENTIRE file to detect correct types
- Recommended for VM environments with sufficient RAM
- Prevents loss of columns due to type inference issues
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


def generate_clase_ternaria(df: pl.DataFrame) -> pl.DataFrame:
    """
    Generate clase_ternaria column based on customer continuation
    
    Logic from workflow-jueves notebook:
    - CONTINUA: Customer continues in the next period
    - BAJA+1: Customer leaves in the next period
    - BAJA+2: Customer leaves after 2 periods
    
    Args:
        df: Input DataFrame with numero_de_cliente and foto_mes
        
    Returns:
        DataFrame with clase_ternaria column added
    """
    print("\nGenerating clase_ternaria...")
    
    # Calculate consecutive period (periodo0)
    df = df.with_columns([
        (pl.col("foto_mes") // 100 * 12 + pl.col("foto_mes") % 100).alias("periodo0")
    ])
    
    # Sort by customer and period
    df = df.sort(["numero_de_cliente", "periodo0"])
    
    # Calculate periodo_ultimo and periodo_anteultimo
    periodo_ultimo = df["periodo0"].max()
    periodo_anteultimo = periodo_ultimo - 1
    
    print(f"  periodo_ultimo: {periodo_ultimo}")
    print(f"  periodo_anteultimo: {periodo_anteultimo}")
    
    # Calculate leads (next 1 and 2 periods) for each customer
    df = df.with_columns([
        pl.col("periodo0").shift(-1).over("numero_de_cliente").alias("periodo1"),
        pl.col("periodo0").shift(-2).over("numero_de_cliente").alias("periodo2")
    ])
    
    # Initialize clase_ternaria as CONTINUA for most records
    df = df.with_columns([
        pl.when(pl.col("periodo0") < periodo_anteultimo)
        .then(pl.lit("CONTINUA"))
        .otherwise(pl.lit(None))
        .alias("clase_ternaria")
    ])
    
    # Calculate BAJA+1: customer leaves in next period
    df = df.with_columns([
        pl.when(
            (pl.col("periodo0") < periodo_ultimo) &
            ((pl.col("periodo1").is_null()) | (pl.col("periodo0") + 1 < pl.col("periodo1")))
        )
        .then(pl.lit("BAJA+1"))
        .otherwise(pl.col("clase_ternaria"))
        .alias("clase_ternaria")
    ])
    
    # Calculate BAJA+2: customer leaves after 2 periods
    df = df.with_columns([
        pl.when(
            (pl.col("periodo0") < periodo_anteultimo) &
            (pl.col("periodo0") + 1 == pl.col("periodo1")) &
            ((pl.col("periodo2").is_null()) | (pl.col("periodo0") + 2 < pl.col("periodo2")))
        )
        .then(pl.lit("BAJA+2"))
        .otherwise(pl.col("clase_ternaria"))
        .alias("clase_ternaria")
    ])
    
    # Drop temporary periodo columns
    df = df.drop(["periodo0", "periodo1", "periodo2"])
    
    # Sort by foto_mes, clase_ternaria, numero_de_cliente
    df = df.sort(["foto_mes", "clase_ternaria", "numero_de_cliente"])
    
    # Print class distribution
    class_counts = df.group_by(["foto_mes", "clase_ternaria"]).agg(pl.count().alias("N"))
    print("\nClass distribution by month (first 10 rows):")
    print(class_counts.sort(["foto_mes", "clase_ternaria"]).head(10))
    
    total_by_class = df.group_by("clase_ternaria").agg(pl.count().alias("N"))
    print("\nTotal by class:")
    print(total_by_class)
    
    return df


def main():
    """Main execution"""
    start_time = time.time()
    print_section(f"GENERATE CLASE_TERNARIA - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Input and output paths
        input_file = "data/competencia_combined.csv.gz"
        output_file = "data/competencia_02_target.parquet"
        
        print(f"\nInput:  {input_file}")
        print(f"Output: {output_file}")
        
        # Load raw data
        print_section("LOADING RAW DATA")
        print("Reading CSV (this may take a while)...")
        print("Using robust schema inference (analyzes entire file)...")
        
        # Use infer_schema_length=None for most robust type inference
        # This reads the entire file to infer types correctly
        # Good for VM with sufficient RAM
        df = pl.read_csv(
            input_file,
            infer_schema_length=None,  # Analyze entire file (most robust)
            ignore_errors=False         # Fail loudly if there are parsing issues
        )
        print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
        
        # Generate clase_ternaria
        print_section("GENERATING CLASE_TERNARIA")
        df = generate_clase_ternaria(df)
        
        # Save result
        print_section("SAVING RESULT")
        print(f"Final shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
        print(f"Saving to: {output_file}")
        
        df.write_parquet(
            output_file,
            compression="zstd",
            compression_level=3
        )
        
        print(f"✓ Saved successfully!")
        
        # Completion
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        print_section("COMPLETED SUCCESSFULLY")
        print(f"\nExecution time: {minutes:02d}m {seconds:02d}s")
        print(f"Output file: {output_file}")
        print(f"\nNext step: Run '02_preprocessing.py' to preprocess the data")
        
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

