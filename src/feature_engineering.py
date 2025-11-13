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
                pl.col(col).rolling_mean(window_size=ventana, min_periods=2)
                .over("numero_de_cliente")
                .alias(f"{col}_avg{ventana}")
            )
        
        if minimo:
            new_cols.append(
                pl.col(col).rolling_min(window_size=ventana, min_periods=2)
                .over("numero_de_cliente")
                .alias(f"{col}_min{ventana}")
            )
        
        if maximo or ratiomax:
            new_cols.append(
                pl.col(col).rolling_max(window_size=ventana, min_periods=2)
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


def add_advanced_features(df: pl.DataFrame, cols_lagueables: List[str]) -> pl.DataFrame:
    """
    Add advanced features: volatility, acceleration, momentum, interactions, etc.
    
    Args:
        df: Input DataFrame (must have lag, delta, and trend features already)
        cols_lagueables: List of base columns
        
    Returns:
        DataFrame with advanced features added
    """
    print("\n" + "="*50)
    print("ADDING ADVANCED FEATURES")
    print("="*50)
    
    # Select key columns for advanced features (most important ones)
    # Focus on monetary and transaction columns
    key_cols = []
    for col in cols_lagueables:
        # Monetary columns (start with 'm')
        if col.startswith('m') and col in df.columns:
            key_cols.append(col)
        # Transaction/count columns (start with 'c')
        elif col.startswith('c') and col in df.columns:
            key_cols.append(col)
        # Visa/Master columns
        elif ('Visa_' in col or 'Master_' in col) and col in df.columns:
            key_cols.append(col)
    
    # Limit to avoid explosion (take most common ones)
    key_cols = list(set(key_cols))[:50]  # Top 50 key columns
    
    print(f"Creating advanced features for {len(key_cols)} key columns...")
    
    # =========================================================================
    # 1. VOLATILITY FEATURES (Desviación estándar y coeficiente de variación)
    # =========================================================================
    print("\n1. Adding volatility features (std, CV, range)...")
    volatility_exprs = []
    
    for col in key_cols:
        avg_col = f"{col}_avg6"
        max_col = f"{col}_max6"
        min_col = f"{col}_min6"
        
        # Standard deviation (using approximation: std ≈ sqrt((max-min)²/12) for uniform)
        # Better: use proper rolling std
        if col in df.columns:
            volatility_exprs.append(
                pl.col(col).rolling_std(window_size=6, min_periods=2)
                .over("numero_de_cliente")
                .alias(f"{col}_std6")
            )
        
        # Coefficient of variation (CV = std / mean)
        if avg_col in df.columns:
            volatility_exprs.append(
                (pl.col(f"{col}_std6") / (pl.col(avg_col).abs() + 1)).alias(f"{col}_cv6")
            )
        
        # Range (max - min)
        if max_col in df.columns and min_col in df.columns:
            volatility_exprs.append(
                (pl.col(max_col) - pl.col(min_col)).alias(f"{col}_rango6")
            )
    
    if volatility_exprs:
        df = df.with_columns(volatility_exprs)
    
    print(f"  Added {len(volatility_exprs)} volatility features")
    
    # =========================================================================
    # 2. ACCELERATION FEATURES (Segunda derivada - cambio en el cambio)
    # =========================================================================
    print("\n2. Adding acceleration features (2nd derivative)...")
    accel_exprs = []
    
    for col in key_cols:
        delta1_col = f"{col}_delta1"
        delta2_col = f"{col}_delta2"
        tend6_col = f"{col}_tend6"
        
        # Acceleration: delta1 - delta2
        if delta1_col in df.columns and delta2_col in df.columns:
            accel_exprs.append(
                (pl.col(delta1_col) - pl.col(delta2_col)).alias(f"{col}_accel")
            )
        
        # Change in trend
        if tend6_col in df.columns:
            accel_exprs.append(
                (pl.col(tend6_col) - pl.col(tend6_col).shift(1).over("numero_de_cliente"))
                .alias(f"{col}_cambio_tend")
            )
    
    if accel_exprs:
        df = df.with_columns(accel_exprs)
    
    print(f"  Added {len(accel_exprs)} acceleration features")
    
    # =========================================================================
    # 3. MOMENTUM & DIRECTION FEATURES (Dirección consistente - MUST HAVE)
    # =========================================================================
    print("\n3. Adding momentum & direction features...")
    momentum_exprs = []
    
    for col in key_cols:
        delta1_col = f"{col}_delta1"
        delta2_col = f"{col}_delta2"
        
        if delta1_col in df.columns and delta2_col in df.columns:
            # Momentum: suma de deltas
            momentum_exprs.append(
                (pl.col(delta1_col) + pl.col(delta2_col)).alias(f"{col}_momentum")
            )
            
            # Dirección consistente: ¿mismo signo? (FEATURE OBLIGATORIA)
            momentum_exprs.append(
                (pl.col(delta1_col).sign() == pl.col(delta2_col).sign())
                .cast(pl.Int32)
                .alias(f"{col}_direccion_consistente")
            )
            
            # Dirección: -1 (baja), 0 (estable), 1 (sube)
            momentum_exprs.append(
                pl.col(delta1_col).sign().alias(f"{col}_direccion")
            )
    
    if momentum_exprs:
        df = df.with_columns(momentum_exprs)
    
    print(f"  Added {len(momentum_exprs)} momentum & direction features")
    
    # =========================================================================
    # 4. RELATIVE ACTIVITY FEATURES (Actividad vs promedio histórico)
    # =========================================================================
    print("\n4. Adding relative activity features...")
    relative_exprs = []
    
    for col in key_cols:
        avg_col = f"{col}_avg6"
        max_col = f"{col}_max6"
        
        # Ratio vs promedio
        if avg_col in df.columns and col in df.columns:
            relative_exprs.append(
                (pl.col(col) / (pl.col(avg_col).abs() + 1)).alias(f"{col}_vs_promedio")
            )
        
        # Ratio vs máximo
        if max_col in df.columns and col in df.columns:
            relative_exprs.append(
                (pl.col(col) / (pl.col(max_col).abs() + 1)).alias(f"{col}_vs_maximo")
            )
    
    if relative_exprs:
        df = df.with_columns(relative_exprs)
    
    print(f"  Added {len(relative_exprs)} relative activity features")
    
    # =========================================================================
    # 5. OUTLIER & ANOMALY FEATURES (Z-scores y detección de outliers)
    # =========================================================================
    print("\n5. Adding outlier detection features...")
    outlier_exprs = []
    
    for col in key_cols:
        avg_col = f"{col}_avg6"
        std_col = f"{col}_std6"
        
        # Z-score
        if avg_col in df.columns and std_col in df.columns and col in df.columns:
            outlier_exprs.append(
                ((pl.col(col) - pl.col(avg_col)) / (pl.col(std_col) + 1))
                .alias(f"{col}_zscore")
            )
            
            # Es outlier? (|z| > 2)
            outlier_exprs.append(
                (pl.col(f"{col}_zscore").abs() > 2)
                .cast(pl.Int32)
                .alias(f"{col}_es_outlier")
            )
    
    if outlier_exprs:
        df = df.with_columns(outlier_exprs)
    
    # Count recent outliers
    outlier_count_exprs = []
    for col in key_cols:
        outlier_col = f"{col}_es_outlier"
        if outlier_col in df.columns:
            outlier_count_exprs.append(
                pl.col(outlier_col).rolling_sum(window_size=3, min_periods=1)
                .over("numero_de_cliente")
                .alias(f"{col}_outliers_recientes")
            )
    
    if outlier_count_exprs:
        df = df.with_columns(outlier_count_exprs)
    
    print(f"  Added {len(outlier_exprs) + len(outlier_count_exprs)} outlier features")
    
    # =========================================================================
    # 6. PRODUCT INTERACTION FEATURES (Ratios entre productos)
    # =========================================================================
    print("\n6. Adding product interaction features...")
    interaction_exprs = []
    
    # Ratios de uso entre tarjetas
    if "ctarjeta_visa_transacciones" in df.columns and "ctarjeta_master_transacciones" in df.columns:
        interaction_exprs.append(
            (pl.col("ctarjeta_visa_transacciones") / 
             (pl.col("ctarjeta_master_transacciones") + 1))
            .alias("ratio_transacciones_visa_master")
        )
    
    # Proporción de saldo en caja de ahorro
    if "mcaja_ahorro" in df.columns and "mcuentas_saldo" in df.columns:
        interaction_exprs.append(
            (pl.col("mcaja_ahorro") / (pl.col("mcuentas_saldo") + 1))
            .alias("prop_saldo_caja_ahorro")
        )
    
    # Proporción de saldo en cuenta corriente
    if "mcuenta_corriente" in df.columns and "mcuentas_saldo" in df.columns:
        interaction_exprs.append(
            (pl.col("mcuenta_corriente") / (pl.col("mcuentas_saldo") + 1))
            .alias("prop_saldo_cuenta_corriente")
        )
    
    # Ratio comisiones vs saldo (¿paga mucho por poco?)
    if "mcomisiones" in df.columns and "mcuentas_saldo" in df.columns:
        interaction_exprs.append(
            (pl.col("mcomisiones") / (pl.col("mcuentas_saldo") + 1))
            .alias("ratio_comisiones_saldo")
        )
    
    # Ratio ingresos/egresos
    if "mtransferencias_recibidas" in df.columns and "mtransferencias_emitidas" in df.columns:
        interaction_exprs.append(
            (pl.col("mtransferencias_recibidas") / 
             (pl.col("mtransferencias_emitidas") + 1))
            .alias("ratio_ingresos_egresos")
        )
    
    # Utilización de crédito Visa
    if "Visa_msaldototal" in df.columns and "Visa_mlimitecompra" in df.columns:
        interaction_exprs.append(
            (pl.col("Visa_msaldototal") / (pl.col("Visa_mlimitecompra") + 1))
            .alias("utilizacion_credito_visa")
        )
    
    # Utilización de crédito Master
    if "Master_msaldototal" in df.columns and "Master_mlimitecompra" in df.columns:
        interaction_exprs.append(
            (pl.col("Master_msaldototal") / (pl.col("Master_mlimitecompra") + 1))
            .alias("utilizacion_credito_master")
        )
    
    # Ratio payroll vs gastos totales
    if "mpayroll" in df.columns and "mtarjeta_visa_consumo" in df.columns:
        interaction_exprs.append(
            (pl.col("mpayroll") / (pl.col("mtarjeta_visa_consumo") + 1))
            .alias("ratio_payroll_consumo")
        )
    
    # Diversificación: número de productos activos
    product_cols = ["ccaja_ahorro", "ccuenta_corriente", "ctarjeta_visa", 
                    "ctarjeta_master", "cprestamos_personales"]
    existing_product_cols = [c for c in product_cols if c in df.columns]
    if existing_product_cols:
        interaction_exprs.append(
            sum([pl.col(c) > 0 for c in existing_product_cols])
            .cast(pl.Int32)
            .alias("num_productos_activos")
        )
    
    if interaction_exprs:
        df = df.with_columns(interaction_exprs)
    
    print(f"  Added {len(interaction_exprs)} product interaction features")
    
    # =========================================================================
    # 7. MULTI-WINDOW FEATURES (Comparar ventanas de diferentes tamaños)
    # =========================================================================
    print("\n7. Adding multi-window comparison features...")
    multiwindow_exprs = []
    
    for col in key_cols[:30]:  # Limit to top 30 to avoid explosion
        if col in df.columns:
            # Short-term average (3 months)
            multiwindow_exprs.append(
                pl.col(col).rolling_mean(window_size=3, min_periods=2)
                .over("numero_de_cliente")
                .alias(f"{col}_avg3")
            )
    
    if multiwindow_exprs:
        df = df.with_columns(multiwindow_exprs)
    
    # Ratios between windows
    ratio_window_exprs = []
    for col in key_cols[:30]:
        avg3_col = f"{col}_avg3"
        avg6_col = f"{col}_avg6"
        
        if avg3_col in df.columns and avg6_col in df.columns:
            ratio_window_exprs.append(
                (pl.col(avg3_col) / (pl.col(avg6_col).abs() + 1))
                .alias(f"{col}_ratio_avg3_avg6")
            )
    
    if ratio_window_exprs:
        df = df.with_columns(ratio_window_exprs)
    
    print(f"  Added {len(multiwindow_exprs) + len(ratio_window_exprs)} multi-window features")
    
    # =========================================================================
    # 8. BINARY CHANGE FEATURES (Cambios de estado)
    # =========================================================================
    print("\n8. Adding binary change features...")
    binary_exprs = []
    
    # Check for key product columns
    binary_cols = ["ccaja_ahorro", "ccuenta_corriente", "ctarjeta_visa", 
                   "ctarjeta_master", "cprestamos_personales", "internet"]
    
    for col in binary_cols:
        if col in df.columns:
            # Tiene el producto? (binario)
            binary_exprs.append(
                (pl.col(col) > 0).cast(pl.Int32).alias(f"tiene_{col}")
            )
            
            # Cambió de estado?
            binary_exprs.append(
                (pl.col(f"tiene_{col}") != 
                 pl.col(f"tiene_{col}").shift(1).over("numero_de_cliente"))
                .cast(pl.Int32)
                .alias(f"cambio_{col}")
            )
    
    if binary_exprs:
        df = df.with_columns(binary_exprs)
    
    # Count changes in last 6 months
    change_count_exprs = []
    for col in binary_cols:
        cambio_col = f"cambio_{col}"
        if cambio_col in df.columns:
            change_count_exprs.append(
                pl.col(cambio_col).rolling_sum(window_size=6, min_periods=1)
                .over("numero_de_cliente")
                .alias(f"cambios_{col}_6m")
            )
    
    if change_count_exprs:
        df = df.with_columns(change_count_exprs)
    
    print(f"  Added {len(binary_exprs) + len(change_count_exprs)} binary change features")
    
    # =========================================================================
    # 9. AGE & TENURE INTERACTIONS
    # =========================================================================
    print("\n9. Adding age & tenure interaction features...")
    age_exprs = []
    
    if "cliente_edad" in df.columns and "cliente_antiguedad" in df.columns:
        # Ratio edad/antigüedad
        age_exprs.append(
            (pl.col("cliente_edad") / (pl.col("cliente_antiguedad") + 1))
            .alias("edad_sobre_antiguedad")
        )
        
        # Cliente nuevo pero mayor
        age_exprs.append(
            ((pl.col("cliente_edad") > 50) & (pl.col("cliente_antiguedad") < 12))
            .cast(pl.Int32)
            .alias("cliente_mayor_nuevo")
        )
        
        # Cliente joven con antigüedad
        age_exprs.append(
            ((pl.col("cliente_edad") < 30) & (pl.col("cliente_antiguedad") > 24))
            .cast(pl.Int32)
            .alias("cliente_joven_antiguo")
        )
    
    # Interacciones edad con productos
    if "cliente_edad" in df.columns:
        for col in ["ctarjeta_visa", "mcaja_ahorro", "mpayroll"]:
            if col in df.columns:
                age_exprs.append(
                    (pl.col("cliente_edad") * pl.col(col))
                    .alias(f"edad_x_{col}")
                )
    
    if age_exprs:
        df = df.with_columns(age_exprs)
    
    print(f"  Added {len(age_exprs)} age & tenure features")
    
    # =========================================================================
    # 10. SEASONALITY FEATURES (Estacionalidad mejorada)
    # =========================================================================
    print("\n10. Adding enhanced seasonality features...")
    season_exprs = []
    
    if "kmes" in df.columns:
        # Trimestre
        season_exprs.append(
            ((pl.col("kmes") - 1) // 3 + 1).alias("trimestre")
        )
        
        # Semestre
        season_exprs.append(
            ((pl.col("kmes") - 1) // 6 + 1).alias("semestre")
        )
        
        # Es fin de año? (nov, dic)
        season_exprs.append(
            (pl.col("kmes").is_in([11, 12])).cast(pl.Int32).alias("es_fin_anio")
        )
        
        # Es inicio de año? (ene, feb, mar)
        season_exprs.append(
            (pl.col("kmes").is_in([1, 2, 3])).cast(pl.Int32).alias("es_inicio_anio")
        )
    
    if season_exprs:
        df = df.with_columns(season_exprs)
    
    print(f"  Added {len(season_exprs)} seasonality features")
    
    print(f"\n✓ Advanced features complete! Total columns: {df.shape[1]}")
    
    return df


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
    
    # Add advanced features (NEW!)
    df = add_advanced_features(df, cols_lagueables)
    
    print(f"\nHistorical features complete. Final columns: {df.shape[1]}")
    
    return df

