# zLightGBM - Guía de Uso

## 🎯 ¿Qué es zLightGBM?

zLightGBM es una versión modificada de LightGBM que usa **canary features** (features aleatorios) para controlar automáticamente el overfitting, eliminando la necesidad de Bayesian Optimization.

## 🚀 Quick Start

```bash
# Ejecutar workflow completo con zLightGBM
uv run run_zlgbm.py
```

## 📋 Requisitos Previos

1. **zLightGBM instalado** (ver `docs/ZLIGHTGBM_QUICKSTART.md`)
2. **Datos preprocesados** con feature engineering:
   ```bash
   # Si no los tienes, ejecutar:
   uv run run_preprocessing.py
   uv run run_feature_engineering.py
   ```
3. **Archivo requerido**: `data/final_dataset.parquet`

## 🔄 Workflow Completo

### Opción A: Script Standalone (Recomendado)

```bash
uv run run_zlgbm.py
```

Este script ejecuta:
1. Carga datos preprocesados
2. Agrega 100 canaritos al inicio
3. Entrena modelo zLightGBM (1 solo modelo)
4. Predice en mes futuro
5. Genera submission Kaggle

**Tiempo estimado:** 20-30 minutos

### Opción B: Paso a Paso (Para debugging)

```python
import polars as pl
from src.config import PARAM
from src.preprocessing import add_canaritos
from src.training_zlgbm import train_zlgbm_final_model
from src.scoring_zlgbm import score_zlgbm_future_data, generate_zlgbm_submission

# 1. Cargar datos
df = pl.read_parquet("data/final_dataset.parquet")

# 2. Agregar canaritos
df, canaritos_names = add_canaritos(df, PARAM["zlgbm"]["qcanaritos"])

# 3. Preparar features (canaritos primero)
campos_buenos_base = [col for col in df.columns 
                     if col not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]
                     and not col.startswith("canarito_")]
campos_buenos = canaritos_names + campos_buenos_base

# 4. Entrenar
train_zlgbm_final_model(df, PARAM, campos_buenos)

# 5. Scoring
df_pred = score_zlgbm_future_data(df, PARAM, campos_buenos)

# 6. Submission
generate_zlgbm_submission(df_pred, PARAM, n_envios=11000)
```

## ⚙️ Configuración

En `src/config.py`, la sección `zlgbm`:

```python
"zlgbm": {
    "qcanaritos": 100,  # Número de canaritos (probar 50-200)
    
    "train_final": {
        "training": [202101, 202102, 202103, 202104],
        "future": [202106],
        "undersampling": 0.50,  # Más conservador que BO
        "ksemillerio": 1,  # Solo 1 modelo
    },
    
    "param": {
        # Hiperparámetros "libres"
        "num_iterations": 9999,
        "num_leaves": 999,
        "learning_rate": 1.0,
        
        # Ajustables
        "feature_fraction": 0.50,  # Probar 0.3-0.7
        "min_data_in_leaf": 20,    # Probar 10-50
        
        # zLightGBM específicos
        "canaritos": 100,
        "gradient_bound": 0.1  # Probar 0.05-0.2
    }
}
```

## 🎛️ Hiperparámetros a Experimentar

### Alta Prioridad

1. **`qcanaritos`**: 50, 100, 150, 200
   - Más canaritos = más control de overfitting
   - Empezar con 100

2. **`gradient_bound`**: 0.05, 0.1, 0.15, 0.2
   - Más bajo = más conservador
   - Empezar con 0.1

3. **`undersampling`**: 0.3, 0.5, 0.7
   - Con zLightGBM usar valores altos
   - Empezar con 0.5

### Media Prioridad

4. **`feature_fraction`**: 0.3, 0.5, 0.7
   - % de features por árbol
   - Empezar con 0.5

5. **`min_data_in_leaf`**: 10, 20, 30, 50
   - Mínimo de registros por hoja
   - Empezar con 20

## 📊 Output Files

```
output/{experimento}_zlgbm/
├── zmodelo.txt                    # Modelo entrenado
├── tb_arboles.txt                 # Estructura de árboles
└── prediccion.txt                 # Predicciones

output/kaggle/
└── KA{experimento}_zlgbm_11000.csv  # Submission Kaggle
```

## 🔍 Análisis de Resultados

### Verificar Importancia de Features

El script automáticamente muestra:
- Top 20 features más importantes
- Marcador 🐤 para canaritos
- Ratio importancia real/canaritos

**Esperado:** Features reales deben tener mayor importancia que canaritos.

### Estructura de Árboles

```
Tree structure analysis:
  Total trees: 555
  Leaves per tree:
    Min: 2
    Median: 643
    Mean: 616.6
    Max: 999
```

**Interpretación:**
- zLightGBM construyó 555 árboles (de máximo 9999)
- Se detuvo automáticamente
- Algunos árboles tienen pocas hojas (2) = no pudo mejorar

## ⚠️ Troubleshooting

### Error: "Canaries must be the first columns"

**Causa:** Los canaritos no están al inicio del dataset.

**Solución:**
```python
# Verificar orden
print(df.columns[:10])  # Deben ser canarito_1, canarito_2, ...

# Si no están al inicio, recrear campos_buenos
campos_buenos = canaritos_names + campos_buenos_base
```

### Error: "Future data is missing canaries"

**Causa:** El mes futuro no tiene canaritos.

**Solución:**
```python
# Los canaritos se agregan a TODO el dataset antes de filtrar
df, canaritos = add_canaritos(df, n_canaritos)
# LUEGO filtrar por mes
df_future = df.filter(pl.col("foto_mes") == 202106)
```

### Warning: "Canaries have similar importance"

**Causa:** Posible overfitting.

**Solución:**
- Aumentar `qcanaritos` (ej: 150 o 200)
- Aumentar `gradient_bound` (ej: 0.15)
- Reducir `feature_fraction` (ej: 0.3)

## 📈 Comparación: BO vs zLightGBM

| Aspecto | Bayesian Opt | zLightGBM |
|---------|--------------|-----------|
| Tiempo | 1-2 horas | 20-30 min |
| Modelos | 30 ensemble | 1 modelo |
| Tuning | Automático | Manual básico |
| Overfitting | Control manual | Control automático |
| Complejidad | Alta | Media |

## 💡 Tips

1. **Primera vez:** Usa configuración default (100 canaritos, gradient_bound 0.1)
2. **Iteración:** Experimenta con diferentes valores
3. **Validación:** Compara resultados con BO en meses pasados
4. **Producción:** Si funciona bien, zLightGBM es más rápido

## 🎓 Recursos

- **Notebook original:** `docs/zlightgbm_50.ipynb`
- **Guía completa:** `docs/ZLIGHTGBM_INTEGRATION.md`
- **Quick start:** `docs/ZLIGHTGBM_QUICKSTART.md`
- **Script de prueba:** `playground/test_zlgbm.py`

---

**¡Éxito con zLightGBM! 🚀**

