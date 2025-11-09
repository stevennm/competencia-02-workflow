"""
Feature Engineering module
Handles intra-month features, historical lags, deltas, and trends
"""

import polars as pl
import numpy as np
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
    Calculate trend features using rolling windows
    
    Translates the R fhistC function to Polars.
    For each column, calculates over a rolling window:
    - Trend: linear regression slope
    - Min, max, average
    - Ratios
    
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
    
    # We'll process each column
    for col in tqdm(cols, desc="Processing columns"):
        if col not in df.columns:
            continue
            
        new_cols = []
        
        # Rolling average
        if promedio or ratioavg:
            new_cols.append(
                pl.col(col).rolling_mean(window_size=ventana, min_periods=2)
                .over("numero_de_cliente")
                .alias(f"{col}_avg{ventana}")
            )
        
        # Rolling min
        if minimo:
            new_cols.append(
                pl.col(col).rolling_min(window_size=ventana, min_periods=2)
                .over("numero_de_cliente")
                .alias(f"{col}_min{ventana}")
            )
        
        # Rolling max
        if maximo or ratiomax:
            new_cols.append(
                pl.col(col).rolling_max(window_size=ventana, min_periods=2)
                .over("numero_de_cliente")
                .alias(f"{col}_max{ventana}")
            )
        
        # Add these columns to dataframe
        if new_cols:
            df = df.with_columns(new_cols)
        
        # Trend calculation (linear regression slope) - more complex
        if tendencia:
            # We need to calculate the slope of linear regression over rolling window
            # Formula: slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
            # where x = [1, 2, 3, ..., ventana] and y = values
            df = df.with_columns([
                calculate_rolling_trend(pl.col(col), ventana)
                .over("numero_de_cliente")
                .alias(f"{col}_tend{ventana}")
            ])
        
        # Ratio features
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


def calculate_rolling_trend(col_expr: pl.Expr, window_size: int) -> pl.Expr:
    """
    Calculate rolling linear regression slope (trend)
    
    This implements the least squares formula for slope:
    slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
    
    Args:
        col_expr: Polars expression for the column
        window_size: Window size
        
    Returns:
        Polars expression for the trend
    """
    # Create indices [1, 2, 3, ..., window_size] for each window
    # For a window of size n, x values are always [1, 2, ..., n]
    # So we can precompute: Σx = n(n+1)/2, Σx² = n(n+1)(2n+1)/6
    
    n = window_size
    sum_x = n * (n + 1) / 2
    sum_x_squared = n * (n + 1) * (2 * n + 1) / 6
    denominator = n * sum_x_squared - sum_x * sum_x
    
    # Calculate Σy (rolling sum of values)
    sum_y = col_expr.rolling_sum(window_size=window_size, min_periods=2)
    
    # Calculate Σxy (need to multiply each value by its position in window)
    # This is more complex - we need a custom approach
    # For simplicity, we'll use a map_batches approach
    
    def calc_trend_numpy(s: pl.Series) -> pl.Series:
        """Calculate trend using numpy for a series"""
        arr = s.to_numpy()
        result = np.full(len(arr), np.nan, dtype=np.float64)
        
        for i in range(window_size - 1, len(arr)):
            window = arr[i - window_size + 1:i + 1]
            # Remove NaN values
            valid_mask = ~np.isnan(window)
            valid_values = window[valid_mask]
            
            if len(valid_values) >= 2:
                # x values for valid data points
                x_vals = np.arange(1, len(valid_values) + 1)
                n_valid = len(valid_values)
                
                # Calculate slope
                sum_x_local = np.sum(x_vals)
                sum_y_local = np.sum(valid_values)
                sum_xy = np.sum(x_vals * valid_values)
                sum_x2_local = np.sum(x_vals ** 2)
                
                denom = n_valid * sum_x2_local - sum_x_local ** 2
                if denom != 0:
                    slope = (n_valid * sum_xy - sum_x_local * sum_y_local) / denom
                    result[i] = slope
        
        return pl.Series(result)
    
    return col_expr.map_batches(calc_trend_numpy)


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
    df = add_lag_features(df, cols_lagueables, lags=[1, 2])
    
    # Add deltas
    df = add_delta_features(df, cols_lagueables, lags=[1, 2])
    
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

