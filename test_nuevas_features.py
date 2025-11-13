"""
Script de prueba para verificar las nuevas features
"""

import polars as pl
from src.feature_engineering import add_advanced_features

# Crear datos de prueba
print("Creando datos de prueba...")

# Simular datos con las columnas necesarias
data = {
    "numero_de_cliente": [1, 1, 1, 2, 2, 2],
    "foto_mes": [202101, 202102, 202103, 202101, 202102, 202103],
    "mcaja_ahorro": [1000, 1100, 900, 2000, 1900, 1800],
    "ctarjeta_visa_transacciones": [10, 12, 8, 20, 18, 16],
    "mpayroll": [5000, 5100, 5200, 3000, 3100, 3200],
}

df = pl.DataFrame(data)

# Agregar features básicas que necesita add_advanced_features
print("\nAgregando features básicas (lags, deltas, trends)...")

# Lags
df = df.sort(["numero_de_cliente", "foto_mes"])
for col in ["mcaja_ahorro", "ctarjeta_visa_transacciones", "mpayroll"]:
    df = df.with_columns([
        pl.col(col).shift(1).over("numero_de_cliente").alias(f"{col}_lag1"),
        pl.col(col).shift(2).over("numero_de_cliente").alias(f"{col}_lag2"),
    ])

# Deltas
for col in ["mcaja_ahorro", "ctarjeta_visa_transacciones", "mpayroll"]:
    df = df.with_columns([
        (pl.col(col) - pl.col(f"{col}_lag1")).alias(f"{col}_delta1"),
        (pl.col(col) - pl.col(f"{col}_lag2")).alias(f"{col}_delta2"),
    ])

# Trends (simples)
for col in ["mcaja_ahorro", "ctarjeta_visa_transacciones", "mpayroll"]:
    df = df.with_columns([
        pl.col(col).rolling_mean(window_size=3, min_periods=2).over("numero_de_cliente").alias(f"{col}_avg6"),
        pl.col(col).rolling_min(window_size=3, min_periods=2).over("numero_de_cliente").alias(f"{col}_min6"),
        pl.col(col).rolling_max(window_size=3, min_periods=2).over("numero_de_cliente").alias(f"{col}_max6"),
    ])

# Tendencia (slope) - simulada
for col in ["mcaja_ahorro", "ctarjeta_visa_transacciones", "mpayroll"]:
    df = df.with_columns([
        pl.lit(0.5).alias(f"{col}_tend6")  # Valor dummy
    ])

print(f"\nDataFrame antes de advanced features: {df.shape[1]} columnas")
print(df.columns[:10])

# Probar add_advanced_features
print("\n" + "="*70)
print("PROBANDO ADD_ADVANCED_FEATURES")
print("="*70)

cols_lagueables = ["mcaja_ahorro", "ctarjeta_visa_transacciones", "mpayroll"]

try:
    df_advanced = add_advanced_features(df, cols_lagueables)
    
    print("\n" + "="*70)
    print("✅ ÉXITO!")
    print("="*70)
    print(f"\nDataFrame después de advanced features: {df_advanced.shape[1]} columnas")
    print(f"Features agregadas: {df_advanced.shape[1] - df.shape[1]}")
    
    # Mostrar algunas de las nuevas features
    print("\n📊 Ejemplos de nuevas features:")
    new_cols = [c for c in df_advanced.columns if c not in df.columns]
    for i, col in enumerate(new_cols[:20], 1):
        print(f"  {i}. {col}")
    
    if len(new_cols) > 20:
        print(f"  ... y {len(new_cols) - 20} más")
    
    # Verificar features obligatorias
    print("\n🎯 Verificando features obligatorias (dirección consistente):")
    direccion_features = [c for c in new_cols if "direccion_consistente" in c]
    for feat in direccion_features:
        print(f"  ✓ {feat}")
    
    # Mostrar algunas filas
    print("\n📋 Muestra de datos (primeras 3 filas, últimas 10 columnas):")
    print(df_advanced.select(df_advanced.columns[-10:]).head(3))
    
except Exception as e:
    print("\n" + "="*70)
    print("❌ ERROR!")
    print("="*70)
    print(f"\n{str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("FIN DEL TEST")
print("="*70)



