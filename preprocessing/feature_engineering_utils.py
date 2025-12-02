"""
Feature Engineering module
Handles intra-month features, historical lags, deltas, and trends
"""

import polars as pl
import numpy as np
import duckdb
from typing import List, Dict
from tqdm import tqdm


def add_intra_month_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create features within the same record, without using historical information
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with intra-month features added
    """
    print("\nAdding intra-month features...")
    
    df = df.with_columns([
        # Extract month (1-12) from foto_mes for seasonality
        (pl.col("foto_mes") % 100).alias("kmes"),
        
        # Normalized ctrx_quarter accounting for customer tenure
        pl.col("ctrx_quarter").cast(pl.Float64).alias("ctrx_quarter_normalizado")
    ])
    
    # Adjust ctrx_quarter_normalizado based on customer tenure
    df = df.with_columns([
        pl.when(pl.col("cliente_antiguedad") == 1)
        .then(pl.col("ctrx_quarter_normalizado") * 5.0)
        .when(pl.col("cliente_antiguedad") == 2)
        .then(pl.col("ctrx_quarter_normalizado") * 2.0)
        .when(pl.col("cliente_antiguedad") == 3)
        .then(pl.col("ctrx_quarter_normalizado") * 1.2)
        .otherwise(pl.col("ctrx_quarter_normalizado"))
        .alias("ctrx_quarter_normalizado")
    ])
    
    # Payroll over age ratio (from Irish master's thesis)
    df = df.with_columns([
        (pl.col("mpayroll") / pl.col("cliente_edad")).alias("mpayroll_sobre_edad")
    ])
    
    print(f"Intra-month features added. Current columns: {df.shape[1]}")
    
    return df


def add_lag_features(df: pl.DataFrame, cols_lagueables: List[str], 
                     lags: List[int] = [1, 2]) -> pl.DataFrame:
    """
    Add lag features for specified columns
    
    Args:
        df: Input DataFrame (must be sorted by numero_de_cliente, foto_mes)
        cols_lagueables: List of columns to create lags for
        lags: List of lag orders to create
        
    Returns:
        DataFrame with lag features added
    """
    print(f"\nAdding lag features (lags: {lags})...")
    
    for lag in lags:
        print(f"  Creating lag {lag}...")
        lag_exprs = []
        for col in cols_lagueables:
            lag_exprs.append(
                pl.col(col).shift(lag).over("numero_de_cliente").alias(f"{col}_lag{lag}")
            )
        df = df.with_columns(lag_exprs)
    
    print(f"Lag features added. Current columns: {df.shape[1]}")
    
    return df


def add_delta_features(df: pl.DataFrame, cols_lagueables: List[str], 
                       lags: List[int] = [1, 2]) -> pl.DataFrame:
    """
    Add delta features (difference from lag)
    
    Args:
        df: Input DataFrame with lag features already created
        cols_lagueables: List of base columns
        lags: List of lag orders
        
    Returns:
        DataFrame with delta features added
    """
    print(f"\nAdding delta features (lags: {lags})...")
    
    delta_exprs = []
    for col in cols_lagueables:
        for lag in lags:
            lag_col = f"{col}_lag{lag}"
            if lag_col in df.columns:
                delta_exprs.append(
                    (pl.col(col) - pl.col(lag_col)).alias(f"{col}_delta{lag}")
                )
    
    df = df.with_columns(delta_exprs)
    
    print(f"Delta features added. Current columns: {df.shape[1]}")
    
    return df


def calculate_trend_features_polars(df: pl.DataFrame, cols: List[str], 
                                     ventana: int = 6,
                                     tendencia: bool = True,
                                     minimo: bool = False,
                                     maximo: bool = False,
                                     promedio: bool = False,
                                     ratioavg: bool = False,
                                     ratiomax: bool = False) -> pl.DataFrame:
    """
    Calculate trend features using DuckDB for fast REGR_SLOPE calculation
    
    Args:
        df: Input DataFrame (must be sorted by numero_de_cliente, foto_mes)
        cols: List of columns to calculate trends for
        ventana: Window size in months
        tendencia: Whether to calculate trend (slope)
        minimo: Whether to calculate minimum
        maximo: Whether to calculate maximum
        promedio: Whether to calculate average
        ratioavg: Whether to calculate ratio to average
        ratiomax: Whether to calculate ratio to maximum
        
    Returns:
        DataFrame with trend features added
    """
    print(f"\nCalculating trend features (window: {ventana} months)...")
    print(f"  Features: tend={tendencia}, min={minimo}, max={maximo}, avg={promedio}, ratioavg={ratioavg}, ratiomax={ratiomax}")
    
    # First, add simple rolling features using Polars (fast)
    for col in cols:
        if col not in df.columns:
            continue
            
        new_cols = []
        
        if promedio or ratioavg:
            new_cols.append(
                pl.col(col).rolling_mean(window_size=ventana, min_samples=2)
                .over("numero_de_cliente")
                .alias(f"{col}_avg{ventana}")
            )
        
        if minimo:
            new_cols.append(
                pl.col(col).rolling_min(window_size=ventana, min_samples=2)
                .over("numero_de_cliente")
                .alias(f"{col}_min{ventana}")
            )
        
        if maximo or ratiomax:
            new_cols.append(
                pl.col(col).rolling_max(window_size=ventana, min_samples=2)
                .over("numero_de_cliente")
                .alias(f"{col}_max{ventana}")
            )
        
        if new_cols:
            df = df.with_columns(new_cols)
    
    # Now calculate trends using DuckDB (much faster than numpy loops)
    if tendencia:
        print(f"  Calculating trends with DuckDB for {len(cols)} columns...")
        df = calculate_trends_duckdb(df, cols, ventana)
    
    # Add ratio features
    for col in cols:
        if ratioavg and f"{col}_avg{ventana}" in df.columns:
            df = df.with_columns([
                (pl.col(col) / pl.col(f"{col}_avg{ventana}")).alias(f"{col}_ratioavg{ventana}")
            ])
        
        if ratiomax and f"{col}_max{ventana}" in df.columns:
            df = df.with_columns([
                (pl.col(col) / pl.col(f"{col}_max{ventana}")).alias(f"{col}_ratiomax{ventana}")
            ])
    
    print(f"Trend features added. Current columns: {df.shape[1]}")
    
    return df


def calculate_trends_duckdb(df: pl.DataFrame, cols: List[str], ventana: int) -> pl.DataFrame:
    """
    Calculate linear regression slopes using DuckDB's REGR_SLOPE function
    
    Args:
        df: Input DataFrame
        cols: List of columns to calculate trends for
        ventana: Window size
        
    Returns:
        DataFrame with trend columns added
    """
    con = duckdb.connect(':memory:')
    
    # Build trend expressions for SQL
    trend_exprs = []
    for col in cols:
        if col in df.columns:
            trend_exprs.append(f"""
                REGR_SLOPE("{col}", row_num) OVER (
                    PARTITION BY numero_de_cliente 
                    ORDER BY foto_mes 
                    ROWS BETWEEN {ventana - 1} PRECEDING AND CURRENT ROW
                ) as "{col}_tend{ventana}"
            """)
    
    # Use CTE to calculate row numbers first, then apply REGR_SLOPE
    query = f"""
    WITH numbered AS (
        SELECT 
            *,
            ROW_NUMBER() OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) as row_num
        FROM df
    )
    SELECT 
        * EXCLUDE (row_num),
        {', '.join(trend_exprs)}
    FROM numbered
    """
    
    result = con.execute(query).pl()
    con.close()
    
    return result


def add_historical_features(df: pl.DataFrame, cols_lagueables: List[str],
                           config: Dict) -> pl.DataFrame:
    """
    Add all historical features: lags, deltas, and trends
    
    Args:
        df: Input DataFrame
        cols_lagueables: List of columns to create features for
        config: Configuration dictionary with FE_hist settings
        
    Returns:
        DataFrame with historical features added
    """
    print("\n" + "="*50)
    print("ADDING HISTORICAL FEATURES")
    print("="*50)
    
    # Sort by customer and month
    print("Sorting by numero_de_cliente and foto_mes...")
    df = df.sort(["numero_de_cliente", "foto_mes"])
    
    # Add lags
    df = add_lag_features(df, cols_lagueables, lags=[1, 2, 3, 6, 12])
    
    # Add deltas
    df = add_delta_features(df, cols_lagueables, lags=[1, 2, 3, 6, 12])
    
    # Update cols_lagueables to include lag columns for trend calculation
    cols_for_trends = [col for col in df.columns 
                       if col not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]]
    
    # Add trends if configured
    if config.get("FE_hist", {}).get("Tendencias", {}).get("run", False):
        tend_config = config["FE_hist"]["Tendencias"]
        df = calculate_trend_features_polars(
            df, 
            cols_lagueables,  # Only calculate trends on original features, not lags
            ventana=tend_config["ventana"],
            tendencia=tend_config["tendencia"],
            minimo=tend_config["minimo"],
            maximo=tend_config["maximo"],
            promedio=tend_config["promedio"],
            ratioavg=tend_config["ratioavg"],
            ratiomax=tend_config["ratiomax"]
        )
    
    print(f"\nHistorical features complete. Final columns: {df.shape[1]}")
    
    return df

