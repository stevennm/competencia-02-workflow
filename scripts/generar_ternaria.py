import polars as pl
import sys

# Fix encoding para Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')


# 3. Leer el CSV directamente con inferencia extendida
# infer_schema_length=None analiza todo el archivo para inferir tipos correctamente
df = pl.read_csv(
    "data/competencia_02_crudo.csv.gz",
    infer_schema_length=None  # Analiza todo el archivo para inferir tipos
)

# 4. Función para agregar la columna clase_ternaria
def agregar_clase_ternaria_optimizada(df: pl.DataFrame) -> pl.DataFrame:
    """
    Versión optimizada usando Polars
    """
    return (
        df.lazy()
        .sort(["numero_de_cliente", "foto_mes"])
        .with_columns([
            pl.when(
                pl.col("foto_mes").shift(-1).over("numero_de_cliente").is_null() &
                (pl.col("foto_mes") != pl.col("foto_mes").max())
            ).then(pl.lit("BAJA+1"))
            .when(
                pl.col("foto_mes").shift(-2).over("numero_de_cliente").is_null() &
                (pl.col("foto_mes") <= pl.col("foto_mes").max() - 2)
            ).then(pl.lit("BAJA+2"))
            .otherwise(pl.lit("CONTINUA"))
            .alias("clase_ternaria")
        ])
        .collect()
    )

# 5. Agregar la columna clase_ternaria al dataframe
df = agregar_clase_ternaria_optimizada(df)

# save as csv
df.write_parquet(
    "data/competencia_02_target.parquet",
    compression="zstd",  # Mejor compresión
    compression_level=3   # Balance velocidad/tamaño
)

