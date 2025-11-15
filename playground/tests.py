"""
Explorador simple de columnas para datasets grandes
Lee solo el schema sin cargar datos en memoria
"""
import polars as pl

# ============================================================================
# CONFIGURACION
# ============================================================================
DATASET_PATH = "data/final_dataset.parquet"

# Columnas originales del dataset (competencia_02_target.parquet)
# NOTA: cprestamos_personales y mprestamos_personales aparecen en featured_data
# pero NO en el dataset original. Se crean en algún punto del pipeline (posiblemente RF)
ORIGINAL_COLUMNS = {
    "numero_de_cliente", "foto_mes", "active_quarter", "cliente_vip", "internet",
    "cliente_edad", "cliente_antiguedad", "mrentabilidad", "mrentabilidad_annual",
    "mcomisiones", "mactivos_margen", "mpasivos_margen", "cproductos", "tcuentas",
    "ccuenta_corriente", "mcuenta_corriente_adicional", "mcuenta_corriente",
    "ccaja_ahorro", "mcaja_ahorro", "mcaja_ahorro_adicional", "mcaja_ahorro_dolares",
    "cdescubierto_preacordado", "mcuentas_saldo", "ctarjeta_debito",
    "ctarjeta_debito_transacciones", "mautoservicio", "ctarjeta_visa",
    "ctarjeta_visa_transacciones", "mtarjeta_visa_consumo", "ctarjeta_master",
    "ctarjeta_master_transacciones", "mtarjeta_master_consumo", "cprestamos_prendarios",
    "mprestamos_prendarios", "cprestamos_hipotecarios", "mprestamos_hipotecarios",
    "cplazo_fijo", "mplazo_fijo_dolares", "mplazo_fijo_pesos", "cinversion1",
    "minversion1_pesos", "minversion1_dolares", "cinversion2", "minversion2",
    "cseguro_vida", "cseguro_auto", "cseguro_vivienda", "cseguro_accidentes_personales",
    "ccaja_seguridad", "cpayroll_trx", "mpayroll", "mpayroll2", "cpayroll2_trx",
    "ccuenta_debitos_automaticos", "mcuenta_debitos_automaticos",
    "ctarjeta_visa_debitos_automaticos", "mttarjeta_visa_debitos_automaticos",
    "ctarjeta_master_debitos_automaticos", "mttarjeta_master_debitos_automaticos",
    "cpagodeservicios", "mpagodeservicios", "cpagomiscuentas", "mpagomiscuentas",
    "ccajeros_propios_descuentos", "mcajeros_propios_descuentos",
    "ctarjeta_visa_descuentos", "mtarjeta_visa_descuentos", "ctarjeta_master_descuentos",
    "mtarjeta_master_descuentos", "ccomisiones_mantenimiento", "mcomisiones_mantenimiento",
    "ccomisiones_otras", "mcomisiones_otras", "cforex", "cforex_buy", "mforex_buy",
    "cforex_sell", "mforex_sell", "ctransferencias_recibidas", "mtransferencias_recibidas",
    "ctransferencias_emitidas", "mtransferencias_emitidas", "cextraccion_autoservicio",
    "mextraccion_autoservicio", "ccheques_depositados", "mcheques_depositados",
    "ccheques_emitidos", "mcheques_emitidos", "ccheques_depositados_rechazados",
    "mcheques_depositados_rechazados", "ccheques_emitidos_rechazados",
    "mcheques_emitidos_rechazados", "tcallcenter", "ccallcenter_transacciones",
    "thomebanking", "chomebanking_transacciones", "ccajas_transacciones",
    "ccajas_consultas", "ccajas_depositos", "ccajas_extracciones", "ccajas_otras",
    "catm_trx", "matm", "catm_trx_other", "matm_other", "ctrx_quarter", "tmobile_app",
    "cmobile_app_trx", "Master_delinquency", "Master_status", "Master_mfinanciacion_limite",
    "Master_Fvencimiento", "Master_Finiciomora", "Master_msaldototal", "Master_msaldopesos",
    "Master_msaldodolares", "Master_mconsumospesos", "Master_mconsumosdolares",
    "Master_mlimitecompra", "Master_madelantopesos", "Master_madelantodolares",
    "Master_fultimo_cierre", "Master_mpagado", "Master_mpagospesos", "Master_mpagosdolares",
    "Master_fechaalta", "Master_mconsumototal", "Master_cconsumos",
    "Master_cadelantosefectivo", "Master_mpagominimo", "Visa_delinquency", "Visa_status",
    "Visa_mfinanciacion_limite", "Visa_Fvencimiento", "Visa_Finiciomora",
    "Visa_msaldototal", "Visa_msaldopesos", "Visa_msaldodolares", "Visa_mconsumospesos",
    "Visa_mconsumosdolares", "Visa_mlimitecompra", "Visa_madelantopesos",
    "Visa_madelantodolares", "Visa_fultimo_cierre", "Visa_mpagado", "Visa_mpagospesos",
    "Visa_mpagosdolares", "Visa_fechaalta", "Visa_mconsumototal", "Visa_cconsumos",
    "Visa_cadelantosefectivo", "Visa_mpagominimo", "clase_ternaria",
    # Las siguientes 2 aparecen en featured_data pero NO en competencia_02_target
    # Posiblemente creadas por un código anterior o importadas de otro dataset
    "cprestamos_personales", "mprestamos_personales"
}

# ============================================================================

def categorize_columns(columns):
    """Categoriza las columnas por tipo de feature"""
    categories = {
        'Metadata': [],
        'Original': [],
        'Lag': [],
        'Delta': [],
        'Trend': [],
        'Ratio': [],
        'MaxMin': [],
        'Engineered': []
    }
    
    for col in columns:
        col_lower = col.lower()
        
        # Metadata
        if col in ['numero_de_cliente', 'foto_mes', 'clase_ternaria']:
            categories['Metadata'].append(col)
        # Columnas originales del dataset base
        elif col in ORIGINAL_COLUMNS:
            categories['Original'].append(col)
        # Lag features
        elif '_lag' in col_lower or col_lower.startswith('lag'):
            categories['Lag'].append(col)
        # Delta features
        elif '_delta' in col_lower or col_lower.startswith('delta'):
            categories['Delta'].append(col)
        # Trend features
        elif '_trend' in col_lower or col_lower.startswith('trend'):
            categories['Trend'].append(col)
        # Ratio features
        elif '_ratio' in col_lower or col_lower.startswith('ratio'):
            categories['Ratio'].append(col)
        # MaxMin features
        elif 'maxmin' in col_lower or '_max' in col_lower or '_min' in col_lower:
            categories['MaxMin'].append(col)
        # Otras features engineered
        else:
            categories['Engineered'].append(col)
    
    return categories

# Leer solo el schema (sin cargar datos)
print("="*70)
print(f"EXPLORANDO: {DATASET_PATH}")
print("="*70)

schema = pl.read_parquet_schema(DATASET_PATH)
columns = list(schema.keys())

print(f"\nTotal columnas: {len(columns)}")
print(f"Tipos de datos: {set(schema.values())}")

# Categorizar columnas
categories = categorize_columns(columns)

# Mostrar resumen
print("\n" + "="*70)
print("CATEGORIAS DE FEATURES")
print("="*70)

for cat_name, cols in categories.items():
    if cols:  # Solo mostrar categorias no vacias
        print(f"\n{cat_name.upper()} ({len(cols)} features):")
        # Mostrar primeras 10
        for col in sorted(cols)[:10]:
            dtype = schema[col]
            print(f"  - {col:40s} [{dtype}]")
        if len(cols) > 10:
            print(f"  ... y {len(cols) - 10} más")

# Resumen final
print("\n" + "="*70)
print("RESUMEN")
print("="*70)
for cat_name, cols in categories.items():
    if cols:
        print(f"  {cat_name:15s}: {len(cols):4d} features")
print(f"  {'TOTAL':15s}: {len(columns):4d} features")