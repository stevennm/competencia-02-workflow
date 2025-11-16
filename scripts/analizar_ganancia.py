"""
Script simple para analizar ganancia de predicciones
Cambia las variables de configuracion abajo y ejecuta
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURACION - CAMBIA ESTO
# ============================================================================

EXPERIMENTO = "model_until_202104_p202106_zlgbm"                       # Nombre del experimento
TARGET_FILE = "data/competencia_02_target.parquet"             # Archivo con labels reales
MES_COMPARAR = 202106                                          # Mes contra el que comparar
GAIN_BAJA2 = 780000                                            # Ganancia por BAJA+2
COST_ENVIO = 20000                                             # Costo por envío
MAX_ENVIOS = 30000                                             # Rango maximo para buscar optimo

# ============================================================================

# Cargar predicciones
print(f"Cargando predicciones de: output/{EXPERIMENTO}/prediccion.txt")
df_pred = pl.read_csv(f"output/{EXPERIMENTO}/prediccion.txt", separator="\t")
print(f"  {len(df_pred):,} predicciones")

# Cargar target (filtrar por mes especifico)
print(f"Cargando target de: {TARGET_FILE}")
print(f"  Filtrando mes: {MES_COMPARAR}")
df_target = pl.read_parquet(TARGET_FILE).filter(pl.col("foto_mes") == MES_COMPARAR)
print(f"  {len(df_target):,} registros del mes {MES_COMPARAR}")

# Filtrar predicciones del mes a comparar
df_pred_mes = df_pred.filter(pl.col("foto_mes") == MES_COMPARAR)
print(f"  Predicciones del mes {MES_COMPARAR}: {len(df_pred_mes):,}")

# Join
df = df_pred_mes.join(
    df_target.select(["numero_de_cliente", "foto_mes", "clase_ternaria"]),
    on=["numero_de_cliente", "foto_mes"],
    how="inner"
)
print(f"  Registros matched: {len(df):,}")

# Calcular ganancia según métrica oficial de Michelina
# GAIN_BAJA2 = 780000 ya es neto (800000 - 20000)
df = df.with_columns([
    pl.when(pl.col("clase_ternaria") == "BAJA+2")
    .then(pl.lit(GAIN_BAJA2))        # +780000 (ganancia neta cuando aciertas)
    .otherwise(pl.lit(-COST_ENVIO))  # -20000 (pierdes el costo cuando fallas)
    .alias("gan")
])

# Ordenar por probabilidad y calcular ganancia acumulada
df = df.sort("prob", descending=True)
df = df.with_columns([
    pl.col("gan").cum_sum().alias("gan_acum")
])
df = df.with_row_index(name="envios", offset=1)

# Encontrar optimo (limitado al rango relevante)
df_relevant = df.filter(pl.col("envios") <= MAX_ENVIOS)
max_gain = df_relevant["gan_acum"].max()
optimal = df_relevant.filter(pl.col("gan_acum") == max_gain).row(0, named=True)

print(f"\nOPTIMO:")
print(f"  Ganancia maxima: ${optimal['gan_acum']:,.0f}")
print(f"  Envios optimos: {optimal['envios']:,}")
print(f"  Probabilidad: {optimal['prob']:.6f}")

# Graficar (recortar hasta 20000 envios)
df_plot = df.filter(pl.col("envios") <= 20000)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df_plot["envios"], df_plot["gan_acum"], linewidth=2, color='steelblue')
ax.axvline(x=optimal['envios'], color='green', linestyle='--', linewidth=2, label=f"Optimo: {optimal['envios']:,} envios")
ax.axhline(y=optimal['gan_acum'], color='red', linestyle='--', alpha=0.5, label=f"Max ganancia: ${optimal['gan_acum']:,.0f}")
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
ax.set_xlabel('Envios', fontsize=12)
ax.set_ylabel('Ganancia Acumulada ($)', fontsize=12)
ax.set_title(f'{EXPERIMENTO}\nCurva de Ganancia', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# Formato de numeros enteros (sin notacion cientifica)
ax.ticklabel_format(style='plain', axis='both')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

plt.tight_layout()
output_file = f"scripts/ganancia_{EXPERIMENTO}.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nGrafico guardado: {output_file}")

