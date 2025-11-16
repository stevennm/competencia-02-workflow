# Experimentos con Pesos por Mes

## 🎯 Objetivo

Comparar diferentes estrategias de ponderación temporal para determinar si dar más peso a meses recientes mejora la predicción.

## 📊 Estrategias Implementadas

### 1. **Equal** (Baseline)
- Todos los meses tienen peso = 1.0
- No hay preferencia temporal
- Modelo estándar

### 2. **Step** (Decaimiento por escalones)
- Últimos 6 meses: peso = 1.0
- Siguientes 6 meses: peso = 0.7
- Resto: peso = 0.4
- Idea: Comportamiento cambia cada 6 meses

### 3. **Linear** (Decaimiento lineal)
- Más reciente (202104): peso = 1.0
- Más antiguo (201901): peso = 0.3
- Gradual entre ambos
- Idea: Importancia decrece uniformemente

### 4. **Exponential** (Decaimiento exponencial)
- Factor de decay: 0.95 por mes
- Últimos meses tienen MUCHO más peso
- Ejemplo:
  - 202104 (último): peso = 1.000
  - 202103 (1 mes atrás): peso = 0.950
  - 202102 (2 meses atrás): peso = 0.903
  - 201901 (28 meses atrás): peso = 0.239
- Idea: Comportamiento reciente es mucho más relevante

## 🚀 Uso

```bash
# Ejecutar los 4 experimentos
python run_experiments.py experiments_weights.txt
```

## 📈 Output Esperado

```
======================================================================
  EXPERIMENT: model-equal
======================================================================
✓ Config updated: model-equal (month_weights=equal)

Training data:
  Samples: 125,000
  ...

Month weighting strategy: equal
  (All weights = 1.0)

======================================================================
  EXPERIMENT: model-step
======================================================================
✓ Config updated: model-step (month_weights=step)

Month weighting strategy: step
  Month weights:
    201901: 0.400
    201902: 0.400
    201903: 0.400
    ...
    202007: 0.700
    202008: 0.700
    ...
    202102: 1.000
    202103: 1.000
    202104: 1.000

[... similarmente para linear y exponential ...]
```

## 📊 Cómo Comparar Resultados

Después de ejecutar los 4 modelos, comparar:

1. **Feature importance**: Ver si cambian las features más importantes
```bash
python scripts/analizar_feature_importance.py model-equal_zlgbm
python scripts/analizar_feature_importance.py model-step_zlgbm
python scripts/analizar_feature_importance.py model-linear_zlgbm
python scripts/analizar_feature_importance.py model-exponential_zlgbm
```

2. **Submissions en Kaggle**: Subir y comparar scores
```
output/kaggle/KAmodel-equal_zlgbm_11000.csv
output/kaggle/KAmodel-step_zlgbm_11000.csv
output/kaggle/KAmodel-linear_zlgbm_11000.csv
output/kaggle/KAmodel-exponential_zlgbm_11000.csv
```

3. **Comparar predicciones**: Ver distribución de probabilidades
```python
import polars as pl

pred_equal = pl.read_csv("output/model-equal_zlgbm/prediccion.txt", separator="\t")
pred_step = pl.read_csv("output/model-step_zlgbm/prediccion.txt", separator="\t")
pred_linear = pl.read_csv("output/model-linear_zlgbm/prediccion.txt", separator="\t")
pred_exp = pl.read_csv("output/model-exponential_zlgbm/prediccion.txt", separator="\t")

print(pred_equal['prob'].describe())
print(pred_step['prob'].describe())
print(pred_linear['prob'].describe())
print(pred_exp['prob'].describe())
```

## 💡 Hipótesis a Validar

- ✅ **H1**: Modelos con decay > equal (meses recientes importan más)
- ✅ **H2**: Exponential > Linear > Step (más peso reciente = mejor)
- ✅ **H3**: Features de lags recientes más importantes con decay

## ⚙️ Personalización

Para ajustar los pesos, editar `src/training_zlgbm.py`:

```python
# Step decay
if months_from_end < 6:
    weight = 1.0
elif months_from_end < 12:
    weight = 0.7  # Cambiar este valor
else:
    weight = 0.4  # Cambiar este valor

# Linear decay
weight = 0.3 + (0.7 * i / (n_months - 1))  # Min=0.3, Max=1.0

# Exponential decay
weight = 0.95 ** months_from_end  # Cambiar factor (0.90, 0.95, 0.99)
```

---

**Creado**: 2024-11-15

