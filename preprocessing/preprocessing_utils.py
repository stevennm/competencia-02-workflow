"""
Data preprocessing module
Handles data loading, clase_ternaria generation, and placeholder functions
"""

import polars as pl
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from typing import List, Tuple


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


def add_canaritos(df: pl.DataFrame, n_canaritos: int, seed: int = 102191) -> Tuple[pl.DataFrame, List[str]]:
    """
    Add canary features (random noise) at the BEGINNING of the dataset
    
    CRITICAL: Canaries MUST be the first columns (mandatory for zLightGBM)
    
    Canaries are random features with no predictive power. zLightGBM uses them
    as a reference to detect overfitting: if a real feature has less importance
    than a canary, it's a sign of overfitting.
    
    Args:
        df: Input DataFrame
        n_canaritos: Number of canary features to add
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (DataFrame with canaries, list of canary column names)
    """
    print(f"\n{'='*70}")
    print(f"ADDING {n_canaritos} CANARY FEATURES")
    print(f"{'='*70}")
    
    np.random.seed(seed)
    n_rows = df.shape[0]
    
    # Create canary features (uniform random [0, 1])
    canaritos_dict = {}
    for i in range(n_canaritos):
        canaritos_dict[f"canarito_{i+1}"] = np.random.uniform(0, 1, n_rows)
    
    # Create DataFrame with canaries
    df_canaritos = pl.DataFrame(canaritos_dict)
    
    # Concatenate: CANARIES FIRST, then original data
    df_with_canaritos = pl.concat([df_canaritos, df], how="horizontal")
    
    canaritos_names = list(canaritos_dict.keys())
    
    print(f"✓ Original shape: {df.shape}")
    print(f"✓ New shape: {df_with_canaritos.shape}")
    print(f"✓ First 5 columns: {df_with_canaritos.columns[:5]}")
    print(f"✓ Canaries: {canaritos_names[0]}...{canaritos_names[-1]}")
    
    return df_with_canaritos, canaritos_names


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
    
    Targets months with data quality issues:
    - 201905: Many zero values that should be imputed
    - 201910: Many zero values that should be imputed
    - 202006: Many zero values that should be imputed (COVID period)
    
    Uses sklearn's IterativeImputer which implements a MICE-like algorithm.
    
    Strategy:
    1. Identify numeric columns with suspicious zeros in problematic months
    2. Treat zeros as missing values for those months only
    3. Use MICE to impute based on:
       - Customer's values in other months
       - Relationships between features
       - Other customers' patterns
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with fixed data quality issues
    """
    print("\n[DATA QUALITY] Fixing problematic months with MICE imputation...")
    
    # Define problematic months
    problematic_months = [201905, 201910, 202006]
    
    # Check which problematic months exist in the dataset
    available_months = df.select("foto_mes").unique().to_series().to_list()
    months_to_fix = [m for m in problematic_months if m in available_months]
    
    if not months_to_fix:
        print(f"  None of the problematic months {problematic_months} found in dataset, skipping...")
        return df
    
    print(f"  Target months for MICE: {months_to_fix}")
    
    # Identify numeric columns (exclude identifiers and target)
    exclude_cols = ["numero_de_cliente", "foto_mes", "clase_ternaria"]
    numeric_cols = [col for col in df.columns 
                   if col not in exclude_cols 
                   and df[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]]
    
    # Store original dtypes to restore after imputation
    original_dtypes = {col: df[col].dtype for col in numeric_cols}
    
    if not numeric_cols:
        print("  No numeric columns to fix")
        return df
    
    print(f"  Processing {len(numeric_cols)} numeric columns...")
    
    # Analyze problematic months for columns with many zeros
    zero_ratios = {}
    for month in months_to_fix:
        df_month = df.filter(pl.col("foto_mes") == month)
        
        # Find columns where >80% of values are 0 in this month
        for col in numeric_cols:
            zero_count = df_month.filter(pl.col(col) == 0).shape[0]
            zero_ratio = zero_count / df_month.shape[0]
            if zero_ratio > 0.8:  # More than 80% zeros
                # Store with month info
                key = f"{col}_{month}"
                zero_ratios[key] = {
                    'col': col,
                    'month': month,
                    'ratio': zero_ratio
                }
    
    if not zero_ratios:
        print(f"  No problematic columns found in months {months_to_fix}")
        return df
    
    # Group by column to see which columns are problematic
    problematic_cols = set([info['col'] for info in zero_ratios.values()])
    print(f"  Found {len(problematic_cols)} columns with >80% zeros across problematic months")
    print(f"  Sample columns: {list(problematic_cols)[:5]}")
    
    # Extract data for imputation (include context months around problematic ones)
    # For 201905: include 201904, 201905, 201906
    # For 201910: include 201909, 201910, 201911
    # For 202006: include 202005, 202006, 202007
    context_months = set()
    for month in months_to_fix:
        context_months.add(month - 1)  # Previous month
        context_months.add(month)      # Current month
        context_months.add(month + 1)  # Next month
    
    context_months = sorted([m for m in context_months if m in available_months])
    print(f"  Using context months: {context_months}")
    
    df_context = df.filter(pl.col("foto_mes").is_in(context_months))
    
    # Prepare data for sklearn (convert to numpy)
    # Keep track of indices
    df_context = df_context.with_row_count(name="row_idx")
    
    # Get the data
    X = df_context.select(numeric_cols).to_numpy()
    foto_mes_array = df_context.select("foto_mes").to_series().to_numpy()
    
    # Mark zeros as missing for problematic columns in their respective months
    X_impute = X.copy()
    
    for key, info in zero_ratios.items():
        col = info['col']
        month = info['month']
        col_idx = numeric_cols.index(col)
        
        # Create mask for this specific month
        mask_month = foto_mes_array == month
        
        # Replace zeros with NaN for this column in this month
        X_impute[mask_month, col_idx] = np.where(
            X_impute[mask_month, col_idx] == 0,
            np.nan,
            X_impute[mask_month, col_idx]
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
    
    # Check for extreme values after imputation
    max_val = np.nanmax(np.abs(X_imputed))
    if max_val > 1e15:
        print(f"  ⚠ WARNING: Detected extreme values after imputation (max: {max_val:.2e})")
        print(f"  This might cause issues. Consider reviewing the data or imputation settings.")
    
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
    
    # Restore original data types (MICE converts to float64)
    # For integer columns, round and handle overflow/NaN values
    cast_exprs = []
    for col in numeric_cols:
        if col in df_202006_complete.columns:
            original_dtype = original_dtypes[col]
            if original_dtype in [pl.Int64, pl.Int32]:
                # For integer columns: clip extreme values, round, fill NaN with 0, and cast
                # Int64 range: -9223372036854775808 to 9223372036854775807
                max_int64 = 9223372036854775807
                min_int64 = -9223372036854775808
                cast_exprs.append(
                    pl.col(col)
                    .clip(min_int64, max_int64)  # Clip to int64 range
                    .round(0)
                    .fill_nan(0)
                    .cast(original_dtype, strict=False)
                )
            else:
                # For float columns: just cast back
                cast_exprs.append(pl.col(col).cast(original_dtype, strict=False))
    
    if cast_exprs:
        df_202006_complete = df_202006_complete.with_columns(cast_exprs)
    
    # Ensure column order matches before concatenating
    df_202006_complete = df_202006_complete.select(df.columns)
    
    # Combine back
    df_fixed = pl.concat([df_without_202006, df_202006_complete])
    
    # Sort to restore original order
    df_fixed = df_fixed.sort(["foto_mes", "numero_de_cliente"])
    
    print(f"  [OK] Fixed {len(zero_ratios)} columns in 202006")
    
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
    
    print(f"  [OK] Adjusted {len(monetary_cols)} columns for inflation")
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

