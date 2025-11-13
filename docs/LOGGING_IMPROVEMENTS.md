# Mejoras en Logging - Documentación

## 📋 Resumen

Se agregó logging detallado en ambos workflows (`run.py` y `run_zlgbm.py`) para registrar todos los parámetros relevantes de configuración y ejecución.

## 🎯 Información Registrada

### 1. Configuración del Experimento

**Ambos scripts registran:**
- Nombre del experimento
- Semilla primigenia
- Modo de operación (BO vs zLightGBM)
- Meses de entrenamiento
- Meses de testing/validación
- Meses futuros
- Undersampling ratio
- Número de modelos en ensemble

### 2. Hiperparámetros

**`run.py` (Bayesian Optimization):**
```
Hyperparameter Tuning:
  BO iterations: 50
  ksemillerio (models per trial): 3
  repe (repetitions): 2
  Total models per trial: 6
```

**`run_zlgbm.py` (zLightGBM):**
```
zLightGBM Parameters:
  Canaries: 100
  gradient_bound: 0.1
  learning_rate: 1.0
  feature_fraction: 0.50
  min_data_in_leaf: 20
  max_bin: 31
  num_iterations (max): 9999
  num_leaves (max): 999
```

### 3. Datos Cargados

**Ambos scripts registran:**
- Dimensiones del dataset (filas x columnas)
- Distribución de registros por mes
- Número total de features
- Primeras y últimas 5 features

Ejemplo:
```
Loaded data: 2,345,678 rows, 850 columns
Month distribution:
  201901: 123,456 records
  201902: 124,567 records
  ...
  202106: 125,678 records
```

### 4. Entrenamiento

**Ambos scripts registran:**
- Meses utilizados para training
- Undersampling aplicado
- Número de modelos a entrenar
- Hiperparámetros utilizados

### 5. Scoring

**Ambos scripts registran:**
- Meses futuros predichos
- Número de predicciones generadas
- Estadísticas de probabilidades:
  - Mínimo
  - Máximo
  - Promedio
  - Mediana

Ejemplo:
```
Scored 125,678 predictions
Prediction stats:
  Min prob: 0.000123
  Max prob: 0.987654
  Mean prob: 0.045678
  Median prob: 0.023456
```

### 6. Submission

**Ambos scripts registran:**
- Cutoff utilizado (número de envíos)
- Archivo de submission generado

## 📁 Ubicación de Logs

Los logs se guardan en:
```
logs/
├── run_YYYYMMDD_HHMMSS.log          # Workflow normal
└── run_zlgbm_YYYYMMDD_HHMMSS.log    # Workflow zLightGBM
```

## 📊 Ejemplo de Log Completo

### `run.py` (Bayesian Optimization)

```
2025-01-15 10:30:00 - INFO - ======================================================================
2025-01-15 10:30:00 - INFO - EXPERIMENT CONFIGURATION
2025-01-15 10:30:00 - INFO - ======================================================================
2025-01-15 10:30:00 - INFO - Experiment: semillerio_optuna2_moretrials
2025-01-15 10:30:00 - INFO - Seed: 102191
2025-01-15 10:30:00 - INFO - Mode: Bayesian Optimization
2025-01-15 10:30:00 - INFO - 
2025-01-15 10:30:00 - INFO - Training Strategy (BO):
2025-01-15 10:30:00 - INFO -   Training months: [201901, 201902, ..., 202102]
2025-01-15 10:30:00 - INFO -   Testing months: [202104]
2025-01-15 10:30:00 - INFO -   Undersampling: 0.05 (5%)
2025-01-15 10:30:00 - INFO - 
2025-01-15 10:30:00 - INFO - Final Training Strategy:
2025-01-15 10:30:00 - INFO -   Training months: [201901, 201902, ..., 202104]
2025-01-15 10:30:00 - INFO -   Future months: [202106]
2025-01-15 10:30:00 - INFO -   Undersampling: 0.10 (10%)
2025-01-15 10:30:00 - INFO -   Models in ensemble: 30
2025-01-15 10:30:00 - INFO - 
2025-01-15 10:30:00 - INFO - Hyperparameter Tuning:
2025-01-15 10:30:00 - INFO -   BO iterations: 50
2025-01-15 10:30:00 - INFO -   ksemillerio (models per trial): 3
2025-01-15 10:30:00 - INFO -   repe (repetitions): 2
2025-01-15 10:30:00 - INFO -   Total models per trial: 6
2025-01-15 10:30:00 - INFO - 
2025-01-15 10:30:00 - INFO - Feature Engineering:
2025-01-15 10:30:00 - INFO -   RF features: 20 trees
2025-01-15 10:30:00 - INFO -   Historical features: ventana=6
2025-01-15 10:30:00 - INFO - ======================================================================
```

### `run_zlgbm.py` (zLightGBM)

```
2025-01-15 10:30:00 - INFO - ======================================================================
2025-01-15 10:30:00 - INFO - EXPERIMENT CONFIGURATION
2025-01-15 10:30:00 - INFO - ======================================================================
2025-01-15 10:30:00 - INFO - Experiment: semillerio_optuna2_moretrials_zlgbm
2025-01-15 10:30:00 - INFO - Seed: 102191
2025-01-15 10:30:00 - INFO - Mode: zLightGBM (no Bayesian Optimization)
2025-01-15 10:30:00 - INFO - 
2025-01-15 10:30:00 - INFO - Training Strategy:
2025-01-15 10:30:00 - INFO -   Training months: [202101, 202102, 202103, 202104]
2025-01-15 10:30:00 - INFO -   Future months: [202106]
2025-01-15 10:30:00 - INFO -   Undersampling: 0.10 (10%)
2025-01-15 10:30:00 - INFO -   Models in ensemble: 1
2025-01-15 10:30:00 - INFO - 
2025-01-15 10:30:00 - INFO - zLightGBM Parameters:
2025-01-15 10:30:00 - INFO -   Canaries: 100
2025-01-15 10:30:00 - INFO -   gradient_bound: 0.1
2025-01-15 10:30:00 - INFO -   learning_rate: 1.0
2025-01-15 10:30:00 - INFO -   feature_fraction: 0.50
2025-01-15 10:30:00 - INFO -   min_data_in_leaf: 20
2025-01-15 10:30:00 - INFO -   max_bin: 31
2025-01-15 10:30:00 - INFO -   num_iterations (max): 9999
2025-01-15 10:30:00 - INFO -   num_leaves (max): 999
2025-01-15 10:30:00 - INFO - ======================================================================
```

## 🔍 Cómo Usar los Logs

### 1. Verificar Configuración

Antes de ejecutar un experimento largo, revisa el log para confirmar:
- Meses de entrenamiento correctos
- Undersampling apropiado
- Hiperparámetros esperados

### 2. Debugging

Si algo falla, el log contiene:
- Configuración exacta utilizada
- Distribución de datos por mes
- Estadísticas de predicciones

### 3. Comparación de Experimentos

Compara logs de diferentes experimentos para:
- Ver qué parámetros cambiaron
- Identificar mejores configuraciones
- Reproducir resultados

### 4. Análisis Post-Ejecución

Usa el log para:
- Documentar experimentos exitosos
- Entender por qué un experimento falló
- Reproducir resultados exactos

## 💡 Tips

### Buscar en Logs

```bash
# Ver configuración de un experimento
grep "EXPERIMENT CONFIGURATION" -A 30 logs/run_20250115_103000.log

# Ver estadísticas de predicciones
grep "Prediction stats" -A 5 logs/run_20250115_103000.log

# Ver distribución de meses
grep "Month distribution" -A 20 logs/run_20250115_103000.log

# Ver hiperparámetros encontrados por BO
grep "Best parameters found" -A 10 logs/run_20250115_103000.log
```

### Comparar Configuraciones

```bash
# Extraer configuración de múltiples logs
for log in logs/run_*.log; do
    echo "=== $log ==="
    grep "Experiment:" $log
    grep "Training months:" $log | head -1
    grep "Undersampling:" $log | head -1
done
```

### Monitorear Ejecución en Tiempo Real

```bash
# Ver log en tiempo real
tail -f logs/run_YYYYMMDD_HHMMSS.log

# Ver solo líneas importantes
tail -f logs/run_YYYYMMDD_HHMMSS.log | grep -E "(STEP|INFO|ERROR)"
```

## 📈 Beneficios

1. **Reproducibilidad**: Todos los parámetros están registrados
2. **Debugging**: Fácil identificar problemas
3. **Comparación**: Comparar diferentes experimentos
4. **Documentación**: Log sirve como documentación automática
5. **Auditoría**: Trazabilidad completa de experimentos

## ✅ Checklist de Información Registrada

- [x] Nombre del experimento
- [x] Semilla
- [x] Modo (BO vs zLightGBM)
- [x] Meses de entrenamiento
- [x] Meses de testing
- [x] Meses futuros
- [x] Undersampling
- [x] Número de modelos
- [x] Hiperparámetros
- [x] Dimensiones del dataset
- [x] Distribución por mes
- [x] Número de features
- [x] Estadísticas de predicciones
- [x] Cutoff de submission

---

**Logging mejorado implementado en ambos workflows! 📊**

