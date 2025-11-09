"""
Data preprocessing module
Handles data loading, clase_ternaria generation, and placeholder functions
"""

import polars as pl
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
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
    Fix data quality issues using MICE (Multiple Imputation by Chained Equations)
    
    Specifically targets 202006 which has many zero values that should be imputed.
    Uses sklearn's IterativeImputer which implements a MICE-like algorithm.
    
    Strategy:
    1. Identify numeric columns with suspicious zeros in 202006
    2. Treat zeros as missing values for 202006 only
    3. Use MICE to impute based on:
       - Customer's values in other months
       - Relationships between features
       - Other customers' patterns
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with fixed data quality issues
    """
    print("\n[DATA QUALITY] Fixing 202006 with MICE imputation...")
    
    # Check if 202006 exists in the dataset
    if 202006 not in df.select("foto_mes").unique().to_series().to_list():
        print("  202006 not found in dataset, skipping...")
        return df
    
    # Identify numeric columns (exclude identifiers and target)
    exclude_cols = ["numero_de_cliente", "foto_mes", "clase_ternaria"]
    numeric_cols = [col for col in df.columns 
                   if col not in exclude_cols 
                   and df[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]]
    
    if not numeric_cols:
        print("  No numeric columns to fix")
        return df
    
    print(f"  Processing {len(numeric_cols)} numeric columns...")
    
    # Analyze 202006 for problematic columns
    df_202006 = df.filter(pl.col("foto_mes") == 202006)
    
    # Find columns where >80% of values are 0 in 202006 (more selective)
    zero_ratios = {}
    for col in numeric_cols:
        zero_count = df_202006.filter(pl.col(col) == 0).shape[0]
        zero_ratio = zero_count / df_202006.shape[0]
        if zero_ratio > 0.8:  # More than 80% zeros (was 50%)
            zero_ratios[col] = zero_ratio
    
    if not zero_ratios:
        print("  No problematic columns found in 202006")
        return df
    
    print(f"  Found {len(zero_ratios)} columns with >50% zeros in 202006")
    print(f"  Top problematic: {list(zero_ratios.keys())[:5]}")
    
    # Extract data for imputation (include some context months)
    context_months = [202005, 202006, 202007]
    df_context = df.filter(pl.col("foto_mes").is_in(context_months))
    
    # Prepare data for sklearn (convert to numpy)
    # Keep track of indices
    df_context = df_context.with_row_count(name="row_idx")
    
    # Get the data
    X = df_context.select(numeric_cols).to_numpy()
    
    # Mark zeros as missing ONLY for 202006 and problematic columns
    X_impute = X.copy()
    mask_202006 = df_context.select("foto_mes").to_series() == 202006
    
    for idx, col in enumerate(numeric_cols):
        if col in zero_ratios:
            # Replace zeros with NaN for this column in 202006
            X_impute[mask_202006.to_numpy(), idx] = np.where(
                X_impute[mask_202006.to_numpy(), idx] == 0,
                np.nan,
                X_impute[mask_202006.to_numpy(), idx]
            )
    
    # Count missing values
    n_missing = np.isnan(X_impute).sum()
    print(f"  Marked {n_missing:,} values for imputation")
    
    if n_missing == 0:
        print("  No values to impute")
        return df
    
    # Apply MICE imputation
    print("  Running MICE imputation (this may take a few minutes)...")
    imputer = IterativeImputer(
        max_iter=3,  # Reduced from 10 to 3 for faster execution
        random_state=102191,
        verbose=0,
        skip_complete=True  # Skip columns with no missing values
    )
    
    X_imputed = imputer.fit_transform(X_impute)
    
    print("  Imputation complete!")
    
    # Create DataFrame with imputed values
    df_imputed = pl.DataFrame({
        col: X_imputed[:, idx] 
        for idx, col in enumerate(numeric_cols)
    })
    
    # Add back identifiers
    df_imputed = df_imputed.with_columns([
        df_context.select("row_idx").to_series(),
        df_context.select("numero_de_cliente").to_series(),
        df_context.select("foto_mes").to_series(),
    ])
    
    # Update only the 202006 records in the original dataframe
    df_202006_fixed = df_imputed.filter(pl.col("foto_mes") == 202006)
    
    # Remove 202006 from original df and add the fixed version
    df_without_202006 = df.filter(pl.col("foto_mes") != 202006)
    
    # Get non-numeric columns from original 202006
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
    df_202006_original = df.filter(pl.col("foto_mes") == 202006).select(non_numeric_cols)
    
    # Merge fixed numeric columns with original non-numeric columns
    df_202006_complete = df_202006_original.join(
        df_202006_fixed.select(["numero_de_cliente"] + numeric_cols),
        on="numero_de_cliente",
        how="left"
    )
    
    # Combine back
    df_fixed = pl.concat([df_without_202006, df_202006_complete])
    
    # Sort to restore original order
    df_fixed = df_fixed.sort(["foto_mes", "numero_de_cliente"])
    
    print(f"  ✓ Fixed {len(zero_ratios)} columns in 202006")
    
    return df_fixed


def data_drifting_correction(df: pl.DataFrame) -> pl.DataFrame:
    """
    Correct data drifting using IPC (Consumer Price Index) inflation adjustment
    
    Adjusts monetary values to account for inflation, bringing all values
    to the most recent month's purchasing power.
    
    Strategy:
    1. Load IPC indicators (monthly % change)
    2. Calculate cumulative inflation multipliers
    3. Identify monetary columns (start with 'm', exclude certain patterns)
    4. Adjust values: value_adjusted = value_original * multiplier
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with inflation-adjusted monetary values
    """
    print("\n[DATA DRIFTING] Correcting inflation with IPC indicators...")
    
    # Load IPC indicators
    try:
        df_ipc = pl.read_csv("data/indicadores.csv")
        print(f"  Loaded IPC data: {len(df_ipc)} months")
    except Exception as e:
        print(f"  WARNING: Could not load indicadores.csv: {e}")
        print("  Skipping drifting correction...")
        return df
    
    # Calculate cumulative multipliers
    # We'll adjust everything to the last month (most recent)
    base_month = df_ipc["foto_mes"].max()
    print(f"  Base month for adjustment: {base_month}")
    
    # Calculate cumulative inflation from each month to base month
    # Formula: if prices went up 3.8% from month1 to month2, 
    # to convert month1 values to month2: multiply by 1.038
    
    df_ipc = df_ipc.sort("foto_mes")
    
    # Calculate multiplier for each month
    # Start from the end and work backwards
    multipliers = {}
    current_multiplier = 1.0  # Base month has multiplier 1.0
    
    # Get sorted months in reverse (from newest to oldest)
    months = df_ipc["foto_mes"].to_list()
    ipc_values = df_ipc["ipc"].to_list()
    
    # Set base month multiplier
    multipliers[base_month] = 1.0
    
    # Work backwards from base month
    for i in range(len(months) - 1, 0, -1):
        month = months[i]
        prev_month = months[i - 1]
        ipc = ipc_values[i]  # IPC shows change FROM prev_month TO month
        
        # To convert prev_month values to base month purchasing power,
        # we need to account for inflation from prev_month to month
        # If IPC is 3.8%, prices went up, so we multiply old values by 1.038
        multiplier_step = 1 + (ipc / 100)
        
        if month in multipliers:
            multipliers[prev_month] = multipliers[month] * multiplier_step
        else:
            # This shouldn't happen if we process in order
            multipliers[prev_month] = current_multiplier * multiplier_step
    
    print(f"  Calculated multipliers for {len(multipliers)} months")
    print(f"  Example: 201901 -> {multipliers.get(201901, 1.0):.4f}x")
    print(f"  Example: 202001 -> {multipliers.get(202001, 1.0):.4f}x")
    print(f"  Example: 202106 -> {multipliers.get(202106, 1.0):.4f}x")
    
    # Identify monetary columns
    # Typically start with 'm' (e.g., mpasivos_margen, mcuentas_saldo)
    # Exclude: foto_mes, and columns that might not be monetary
    monetary_cols = []
    for col in df.columns:
        if (col.startswith("m") and 
            col not in ["foto_mes"] and
            df[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]):
            monetary_cols.append(col)
    
    if not monetary_cols:
        print("  No monetary columns found to adjust")
        return df
    
    print(f"  Found {len(monetary_cols)} monetary columns to adjust")
    print(f"  Examples: {monetary_cols[:5]}")
    
    # Create a mapping DataFrame
    df_multipliers = pl.DataFrame({
        "foto_mes": list(multipliers.keys()),
        "ipc_multiplier": list(multipliers.values())
    })
    
    # Join multipliers with main dataframe
    df = df.join(df_multipliers, on="foto_mes", how="left")
    
    # Fill missing multipliers with 1.0 (no adjustment)
    df = df.with_columns([
        pl.col("ipc_multiplier").fill_null(1.0)
    ])
    
    # Apply adjustment to all monetary columns
    print("  Applying inflation adjustment...")
    
    # Store a sample before adjustment for validation
    sample_col = monetary_cols[0]  # First monetary column
    sample_months = [201901, 202001, 202106, base_month]
    
    print(f"\n  Validation - {sample_col} (sample values):")
    print(f"  {'Month':<10} {'Before':<15} {'Multiplier':<12} {'After':<15}")
    print(f"  {'-'*52}")
    
    for month in sample_months:
        if month in df["foto_mes"].unique().to_list():
            # Get original value (before adjustment)
            sample_before = df.filter(pl.col("foto_mes") == month).select(sample_col).head(1).item()
            multiplier = multipliers.get(month, 1.0)
            sample_after = sample_before * multiplier
            print(f"  {month:<10} ${sample_before:<14,.2f} {multiplier:<11.4f}x ${sample_after:<14,.2f}")
    
    print()
    
    # Now apply the adjustment
    for col in monetary_cols:
        df = df.with_columns([
            (pl.col(col) * pl.col("ipc_multiplier")).alias(col)
        ])
    
    # Drop the temporary multiplier column
    df = df.drop("ipc_multiplier")
    
    print(f"  ✓ Adjusted {len(monetary_cols)} columns for inflation")
    print(f"  All values normalized to {base_month} purchasing power")
    
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

