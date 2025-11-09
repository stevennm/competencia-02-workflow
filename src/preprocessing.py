"""
Data preprocessing module
Handles data loading, clase_ternaria generation, and placeholder functions
"""

import polars as pl
from typing import List


def load_data(file_path: str) -> pl.DataFrame:
    """
    Load the dataset from parquet file
    
    Args:
        file_path: Path to the parquet file
        
    Returns:
        Polars DataFrame with the dataset
    """
    print(f"Loading dataset from {file_path}...")
    df = pl.read_parquet(file_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def generate_clase_ternaria(df: pl.DataFrame) -> pl.DataFrame:
    """
    Generate clase_ternaria column based on customer continuation
    
    Classes:
    - CONTINUA: Customer continues in the next period
    - BAJA+1: Customer leaves in the next period
    - BAJA+2: Customer leaves after 2 periods
    
    Args:
        df: Input DataFrame with numero_de_cliente and foto_mes
        
    Returns:
        DataFrame with clase_ternaria column added
    """
    print("Generating clase_ternaria...")
    
    # Calculate consecutive period (periodo0)
    df = df.with_columns([
        (pl.col("foto_mes") // 100 * 12 + pl.col("foto_mes") % 100).alias("periodo0")
    ])
    
    # Sort by customer and period
    df = df.sort(["numero_de_cliente", "periodo0"])
    
    # Calculate periodo_ultimo and periodo_anteultimo
    periodo_ultimo = df["periodo0"].max()
    periodo_anteultimo = periodo_ultimo - 1
    
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
    print("\nClass distribution by month:")
    print(class_counts.sort(["foto_mes", "clase_ternaria"]))
    
    return df


def eliminate_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    TODO: Implement feature elimination based on exploratory data analysis
    
    This should eliminate features that are not useful for the model.
    Features to eliminate may differ from Competition 1.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with eliminated features
    """
    print("\n[PLACEHOLDER] Feature elimination - to be implemented")
    # TODO: Add feature elimination logic here after EDA
    # Example:
    # features_to_drop = ["feature1", "feature2", ...]
    # df = df.drop(features_to_drop)
    return df


def data_quality_fixes(df: pl.DataFrame) -> pl.DataFrame:
    """
    TODO: Fix data quality issues
    
    Repair attributes where ALL values are zero for a certain month.
    Several repair strategies:
    1. Do nothing (leave as 0, knowing it's incorrect)
    2. Replace damaged values with None/NA
    3. Interpolate damaged values using previous and next month
    4. Calculate using a model (e.g., MICE library)
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with fixed data quality issues
    """
    print("\n[PLACEHOLDER] Data quality fixes - to be implemented")
    # TODO: Identify (attribute, month) pairs that are damaged
    # TODO: Implement repair strategy
    # Example:
    # for col in numeric_columns:
    #     # Find months where all values are 0
    #     # Apply repair strategy
    return df


def data_drifting_correction(df: pl.DataFrame) -> pl.DataFrame:
    """
    TODO: Correct data drifting
    
    Correct natural drifting in data, particularly monetary values
    affected by high inflation.
    
    Possible methods:
    1. Do nothing
    2. Adjust monetary values by indices:
       - IPC (Consumer Price Index)
       - Official Dollar rate
       - Blue Dollar rate
       - UVA (Unit of Acquisition Value)
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with drifting corrected
    """
    print("\n[PLACEHOLDER] Data drifting correction - to be implemented")
    # TODO: Implement drifting correction
    # Example:
    # monetary_features = [col for col in df.columns if "m" in col[:2]]
    # # Apply inflation adjustment using IPC or other index
    return df


def preprocess_data(file_path: str) -> pl.DataFrame:
    """
    Main preprocessing pipeline
    
    Args:
        file_path: Path to the input parquet file
        
    Returns:
        Preprocessed DataFrame
    """
    # Load data
    df = load_data(file_path)
    
    # Generate clase_ternaria
    #df = generate_clase_ternaria(df)
    
    # Apply placeholder transformations
    df = eliminate_features(df)
    df = data_quality_fixes(df)
    df = data_drifting_correction(df)
    
    print(f"\nPreprocessing complete: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df

