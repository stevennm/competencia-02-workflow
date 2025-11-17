# Cómo Replicar el Ensemble Final

Este documento explica cómo configurar y ejecutar los 3 modelos que usé para el ensemble final de la competencia 202108.

## 📋 Prerequisitos

Primero hay que correr el pipeline de preprocessing completo:

```bash
python preprocessing/run_all.py
```

Esto genera `data/final_dataset.parquet` con todas las features necesarias.

---

## 🎯 Los 3 Modelos del Ensemble

### **Modelo 1: Undersampling 0.05 (comp2_entrega2_202108_under0.05_zlgbm)**

Configurar en `src/config.py`:

```python
"experimento": "comp2_entrega2_202108_under0.05",
"undersampling": 0.05,
```

Correr:
```bash
uv run run_zlgbm.py
```

---

### **Modelo 2: Undersampling 0.1 (comp2_entrega3_202108_under0.1_zlgbm)**

Cambiar en `src/config.py`:

```python
"experimento": "comp2_entrega3_202108_under0.1",
"undersampling": 0.10,
```

Correr:
```bash
uv run run_zlgbm.py
```

---

### **Modelo 3: Undersampling 0.05 + Zero-to-Null (comp2_entrega4_202108_under0.05_zero_zlgbm)**

Cambiar en `src/config.py`:

```python
"experimento": "comp2_entrega4_202108_under0.05_zero",
"undersampling": 0.05,
```

**Importante:** Para este modelo, descomentar las líneas 163-175 de `run_zlgbm.py` que reemplazan columnas 100% ceros con nulls:

```python
# Replace 100% zero columns with nulls
zero_cols = []
for col in df.columns:
    if col not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]:
        if df[col].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
            if (df[col] == 0).all():
                zero_cols.append(col)

if zero_cols:
    logger.info(f"Replacing {len(zero_cols)} 100% zero columns with nulls:")
    for col in zero_cols:
        logger.info(f"  - {col}")
    df = df.with_columns([pl.lit(None).alias(col) for col in zero_cols])
```

Correr:
```bash
uv run run_zlgbm.py
```

---

## 🔄 Generar el Ensemble

Una vez que los 3 modelos terminaron de ejecutarse, abrir el notebook:

```bash
jupyter notebook scripts/ensamble.ipynb
```

Ejecutar hasta la celda 11 (inclusive) para generar el ensemble de 3 modelos sin entrega1.

El archivo final queda en:
```
output/kaggle/recortado/ensemble_3models_no_entrega1_202108.csv
```

Este es el que entregado con 11,000 envíos.
