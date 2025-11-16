"""
Step 4: Advanced Features
Loads featured_data.parquet and adds advanced analytical features
These features help capture complex patterns like volatility, momentum, and interactions
"""

import sys
import time
from datetime import datetime
from pathlib import Path
import polars as pl
from typing import List


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def add_advanced_features(df: pl.DataFrame, cols_lagueables: List[str]) -> pl.DataFrame:
    """
    Add advanced features: volatility, acceleration, momentum, interactions, etc.
    
    Args:
        df: Input DataFrame (must have lag, delta, and trend features already)
        cols_lagueables: List of base columns (original features)
        
    Returns:
        DataFrame with advanced features added
    """
    print("\n" + "="*70)
    print("ADDING ADVANCED FEATURES")
    print("="*70)
    
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
    # 1. VOLATILITY FEATURES (Standard deviation and coefficient of variation)
    # =========================================================================
    print("\n1. Adding volatility features (std, CV, range)...")
    
    # Step 1a: Create std and range features first
    volatility_exprs_step1 = []
    
    for col in key_cols:
        max_col = f"{col}_max6"
        min_col = f"{col}_min6"
        
        # Standard deviation over 6 months
        if col in df.columns:
            volatility_exprs_step1.append(
                pl.col(col).rolling_std(window_size=6, min_samples=2)
                .over("numero_de_cliente")
                .alias(f"{col}_std6")
            )
        
        # Range (max - min)
        if max_col in df.columns and min_col in df.columns:
            volatility_exprs_step1.append(
                (pl.col(max_col) - pl.col(min_col)).alias(f"{col}_rango6")
            )
    
    if volatility_exprs_step1:
        df = df.with_columns(volatility_exprs_step1)
    
    # Step 1b: Now create CV features using the std6 columns we just created
    volatility_exprs_step2 = []
    
    for col in key_cols:
        avg_col = f"{col}_avg6"
        std_col = f"{col}_std6"
        
        # Coefficient of variation (CV = std / mean)
        if avg_col in df.columns and std_col in df.columns:
            volatility_exprs_step2.append(
                (pl.col(std_col) / (pl.col(avg_col).abs() + 1)).alias(f"{col}_cv6")
            )
    
    if volatility_exprs_step2:
        df = df.with_columns(volatility_exprs_step2)
    
    print(f"  Added {len(volatility_exprs_step1) + len(volatility_exprs_step2)} volatility features")
    
    # =========================================================================
    # 2. ACCELERATION FEATURES (Second derivative - change in the change)
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
    # 3. MOMENTUM & DIRECTION FEATURES (Consistent direction - MUST HAVE)
    # =========================================================================
    print("\n3. Adding momentum & direction features...")
    momentum_exprs = []
    
    for col in key_cols:
        delta1_col = f"{col}_delta1"
        delta2_col = f"{col}_delta2"
        
        if delta1_col in df.columns and delta2_col in df.columns:
            # Momentum: sum of deltas
            momentum_exprs.append(
                (pl.col(delta1_col) + pl.col(delta2_col)).alias(f"{col}_momentum")
            )
            
            # Consistent direction: same sign? (MUST-HAVE FEATURE)
            momentum_exprs.append(
                (pl.col(delta1_col).sign() == pl.col(delta2_col).sign())
                .cast(pl.Int32)
                .alias(f"{col}_direccion_consistente")
            )
            
            # Direction: -1 (down), 0 (stable), 1 (up)
            momentum_exprs.append(
                pl.col(delta1_col).sign().alias(f"{col}_direccion")
            )
    
    if momentum_exprs:
        df = df.with_columns(momentum_exprs)
    
    print(f"  Added {len(momentum_exprs)} momentum & direction features")
    
    # =========================================================================
    # 4. RELATIVE ACTIVITY FEATURES (Activity vs historical average)
    # =========================================================================
    print("\n4. Adding relative activity features...")
    relative_exprs = []
    
    for col in key_cols:
        avg_col = f"{col}_avg6"
        max_col = f"{col}_max6"
        
        # Ratio vs average
        if avg_col in df.columns and col in df.columns:
            relative_exprs.append(
                (pl.col(col) / (pl.col(avg_col).abs() + 1)).alias(f"{col}_vs_promedio")
            )
        
        # Ratio vs maximum
        if max_col in df.columns and col in df.columns:
            relative_exprs.append(
                (pl.col(col) / (pl.col(max_col).abs() + 1)).alias(f"{col}_vs_maximo")
            )
    
    if relative_exprs:
        df = df.with_columns(relative_exprs)
    
    print(f"  Added {len(relative_exprs)} relative activity features")
    
    # =========================================================================
    # 5. OUTLIER & ANOMALY FEATURES (Z-scores and outlier detection)
    # =========================================================================
    print("\n5. Adding outlier detection features...")
    
    # Step 5a: Calculate z-scores first
    outlier_exprs_step1 = []
    
    for col in key_cols:
        avg_col = f"{col}_avg6"
        std_col = f"{col}_std6"
        
        # Z-score
        if avg_col in df.columns and std_col in df.columns and col in df.columns:
            outlier_exprs_step1.append(
                ((pl.col(col) - pl.col(avg_col)) / (pl.col(std_col) + 1))
                .alias(f"{col}_zscore")
            )
    
    if outlier_exprs_step1:
        df = df.with_columns(outlier_exprs_step1)
    
    # Step 5b: Now create outlier flags using the zscores we just created
    outlier_exprs_step2 = []
    
    for col in key_cols:
        zscore_col = f"{col}_zscore"
        
        # Is outlier? (|z| > 2)
        if zscore_col in df.columns:
            outlier_exprs_step2.append(
                (pl.col(zscore_col).abs() > 2)
                .cast(pl.Int32)
                .alias(f"{col}_es_outlier")
            )
    
    if outlier_exprs_step2:
        df = df.with_columns(outlier_exprs_step2)
    
    # Count recent outliers
    outlier_count_exprs = []
    for col in key_cols:
        outlier_col = f"{col}_es_outlier"
        if outlier_col in df.columns:
            outlier_count_exprs.append(
                pl.col(outlier_col).rolling_sum(window_size=3, min_samples=1)
                .over("numero_de_cliente")
                .alias(f"{col}_outliers_recientes")
            )
    
    if outlier_count_exprs:
        df = df.with_columns(outlier_count_exprs)
    
    print(f"  Added {len(outlier_exprs_step1) + len(outlier_exprs_step2) + len(outlier_count_exprs)} outlier features")
    
    # =========================================================================
    # 6. PRODUCT INTERACTION FEATURES (Ratios between products)
    # =========================================================================
    print("\n6. Adding product interaction features...")
    interaction_exprs = []
    
    # Ratios between cards
    if "ctarjeta_visa_transacciones" in df.columns and "ctarjeta_master_transacciones" in df.columns:
        interaction_exprs.append(
            (pl.col("ctarjeta_visa_transacciones") / 
             (pl.col("ctarjeta_master_transacciones") + 1))
            .alias("ratio_transacciones_visa_master")
        )
    
    # Proportion of balance in savings account
    if "mcaja_ahorro" in df.columns and "mcuentas_saldo" in df.columns:
        interaction_exprs.append(
            (pl.col("mcaja_ahorro") / (pl.col("mcuentas_saldo") + 1))
            .alias("prop_saldo_caja_ahorro")
        )
    
    # Proportion of balance in checking account
    if "mcuenta_corriente" in df.columns and "mcuentas_saldo" in df.columns:
        interaction_exprs.append(
            (pl.col("mcuenta_corriente") / (pl.col("mcuentas_saldo") + 1))
            .alias("prop_saldo_cuenta_corriente")
        )
    
    # Commission ratio vs balance (paying too much for too little?)
    if "mcomisiones" in df.columns and "mcuentas_saldo" in df.columns:
        interaction_exprs.append(
            (pl.col("mcomisiones") / (pl.col("mcuentas_saldo") + 1))
            .alias("ratio_comisiones_saldo")
        )
    
    # Income/expense ratio
    if "mtransferencias_recibidas" in df.columns and "mtransferencias_emitidas" in df.columns:
        interaction_exprs.append(
            (pl.col("mtransferencias_recibidas") / 
             (pl.col("mtransferencias_emitidas") + 1))
            .alias("ratio_ingresos_egresos")
        )
    
    # Visa credit utilization
    if "Visa_msaldototal" in df.columns and "Visa_mlimitecompra" in df.columns:
        interaction_exprs.append(
            (pl.col("Visa_msaldototal") / (pl.col("Visa_mlimitecompra") + 1))
            .alias("utilizacion_credito_visa")
        )
    
    # Master credit utilization
    if "Master_msaldototal" in df.columns and "Master_mlimitecompra" in df.columns:
        interaction_exprs.append(
            (pl.col("Master_msaldototal") / (pl.col("Master_mlimitecompra") + 1))
            .alias("utilizacion_credito_master")
        )
    
    # Payroll vs total expenses ratio
    if "mpayroll" in df.columns and "mtarjeta_visa_consumo" in df.columns:
        interaction_exprs.append(
            (pl.col("mpayroll") / (pl.col("mtarjeta_visa_consumo") + 1))
            .alias("ratio_payroll_consumo")
        )
    
    # Diversification: number of active products
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
    # 7. MULTI-WINDOW FEATURES (Compare windows of different sizes)
    # =========================================================================
    print("\n7. Adding multi-window comparison features...")
    multiwindow_exprs = []
    
    for col in key_cols[:30]:  # Limit to top 30 to avoid explosion
        if col in df.columns:
            # Short-term average (3 months)
            multiwindow_exprs.append(
                pl.col(col).rolling_mean(window_size=3, min_samples=2)
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
    # 8. BINARY CHANGE FEATURES (State changes)
    # =========================================================================
    print("\n8. Adding binary change features...")
    
    # Step 8a: Create binary flags first
    binary_exprs_step1 = []
    
    # Check for key product columns
    binary_cols = ["ccaja_ahorro", "ccuenta_corriente", "ctarjeta_visa", 
                   "ctarjeta_master", "cprestamos_personales", "internet"]
    
    for col in binary_cols:
        if col in df.columns:
            # Has the product? (binary)
            binary_exprs_step1.append(
                (pl.col(col) > 0).cast(pl.Int32).alias(f"tiene_{col}")
            )
    
    if binary_exprs_step1:
        df = df.with_columns(binary_exprs_step1)
    
    # Step 8b: Now create change flags using the binary columns we just created
    binary_exprs_step2 = []
    
    for col in binary_cols:
        tiene_col = f"tiene_{col}"
        if tiene_col in df.columns:
            # State changed?
            binary_exprs_step2.append(
                (pl.col(tiene_col) != 
                 pl.col(tiene_col).shift(1).over("numero_de_cliente"))
                .cast(pl.Int32)
                .alias(f"cambio_{col}")
            )
    
    if binary_exprs_step2:
        df = df.with_columns(binary_exprs_step2)
    
    # Count changes in last 6 months
    change_count_exprs = []
    for col in binary_cols:
        cambio_col = f"cambio_{col}"
        if cambio_col in df.columns:
            change_count_exprs.append(
                pl.col(cambio_col).rolling_sum(window_size=6, min_samples=1)
                .over("numero_de_cliente")
                .alias(f"cambios_{col}_6m")
            )
    
    if change_count_exprs:
        df = df.with_columns(change_count_exprs)
    
    print(f"  Added {len(binary_exprs_step1) + len(binary_exprs_step2) + len(change_count_exprs)} binary change features")
    
    # =========================================================================
    # 9. AGE & TENURE INTERACTIONS
    # =========================================================================
    print("\n9. Adding age & tenure interaction features...")
    age_exprs = []
    
    if "cliente_edad" in df.columns and "cliente_antiguedad" in df.columns:
        # Age/tenure ratio
        age_exprs.append(
            (pl.col("cliente_edad") / (pl.col("cliente_antiguedad") + 1))
            .alias("edad_sobre_antiguedad")
        )
        
        # New but older client
        age_exprs.append(
            ((pl.col("cliente_edad") > 50) & (pl.col("cliente_antiguedad") < 12))
            .cast(pl.Int32)
            .alias("cliente_mayor_nuevo")
        )
        
        # Young client with tenure
        age_exprs.append(
            ((pl.col("cliente_edad") < 30) & (pl.col("cliente_antiguedad") > 24))
            .cast(pl.Int32)
            .alias("cliente_joven_antiguo")
        )
    
    # Age interactions with products
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
    # 10. SEASONALITY FEATURES (Enhanced seasonality)
    # =========================================================================
    print("\n10. Adding enhanced seasonality features...")
    season_exprs = []
    
    if "kmes" in df.columns:
        # Quarter
        season_exprs.append(
            ((pl.col("kmes") - 1) // 3 + 1).alias("trimestre")
        )
        
        # Semester
        season_exprs.append(
            ((pl.col("kmes") - 1) // 6 + 1).alias("semestre")
        )
        
        # Is end of year? (nov, dec)
        season_exprs.append(
            (pl.col("kmes").is_in([11, 12])).cast(pl.Int32).alias("es_fin_anio")
        )
        
        # Is beginning of year? (jan, feb, mar)
        season_exprs.append(
            (pl.col("kmes").is_in([1, 2, 3])).cast(pl.Int32).alias("es_inicio_anio")
        )
    
    if season_exprs:
        df = df.with_columns(season_exprs)
    
    print(f"  Added {len(season_exprs)} seasonality features")
    
    print(f"\n✓ Advanced features complete! Total columns: {df.shape[1]}")
    
    return df


def main():
    """Main advanced features execution"""
    start_time = time.time()
    print_section(f"ADVANCED FEATURES - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Input and output paths
        input_file = "data/featured_data.parquet"
        output_file = "data/advanced_featured_data.parquet"
        
        # Check if input exists
        if not Path(input_file).exists():
            raise FileNotFoundError(
                f"Input file not found: {input_file}\n"
                f"Run 03_feature_engineering.py first"
            )
        
        # Load data
        print(f"\nLoading data from {input_file}...")
        df = pl.read_parquet(input_file)
        print(f"✓ Data loaded: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
        
        # Identify base columns (original features before any engineering)
        # These are the columns that don't have lag, delta, tend, etc. suffixes
        cols_lagueables = [
            col for col in df.columns 
            if not any(suffix in col for suffix in 
                      ['_lag', '_delta', '_tend', '_avg', '_max', '_min', 
                       '_ratio', 'canarito', 'clase_ternaria'])
            and col not in ['numero_de_cliente', 'foto_mes', 'kmes']
        ]
        
        print(f"Base features identified: {len(cols_lagueables)}")
        
        # Add advanced features
        df = add_advanced_features(df, cols_lagueables)
        
        # Save result
        print(f"\nSaving to {output_file}...")
        df.write_parquet(output_file)
        print(f"✓ Saved: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
        
        # Summary
        elapsed_time = time.time() - start_time
        print_section("ADVANCED FEATURES COMPLETED")
        print(f"  Execution time: {elapsed_time:.1f} seconds")
        print(f"  Output: {output_file}")
        print(f"  Final dataset: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: {str(e)}")
        print(f"{'='*70}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

