# Ensemble de Semilleros (ksemillerio) - zLightGBM

## 📋 **¿Qué es el Ensemble de Semilleros?**

El ensemble de semilleros es una técnica para entrenar **múltiples modelos** con diferentes **semillas aleatorias**, cada uno viendo una **muestra diferente** de la clase negativa (CONTINUA) mediante undersampling.

## 🎯 **¿Por qué usar Ensemble?**

### **Ventajas:**

1. **Reduce varianza**: Promediar 5 modelos suaviza predicciones extremas
2. **Aprovecha más datos**: Cada modelo ve 10% de CONTINUA diferente → 5 modelos ven ~50% total
3. **Más robusto**: Si un modelo se equivoca, los otros 4 lo compensan
4. **Mejor ranking**: Mejora AUC y ordenamiento de clientes

### **Desventajas:**

1. **5x más tiempo de entrenamiento**: Cada modelo tarda ~35-60 minutos
2. **5x más espacio en disco**: Cada modelo ocupa ~50-100 MB
3. **Más complejo**: Más archivos para gestionar

---

## ⚙️ **Configuración**

### **En `src/config.py`:**

```python
"train_final": {
    "training": [201901, ..., 202106],
    "future": [202108],
    "undersampling": 0.10,  # 10% de la clase negativa por modelo
    "ksemillerio": 5,       # ← Número de modelos en el ensemble
    "month_weights": "equal",
}
```

**Valores posibles para `ksemillerio`:**
- `1`: Un solo modelo (sin ensemble)
- `2-5`: Ensemble de 2 a 5 modelos (recomendado: 5)

---

## 🔢 **Semillas Predefinidas**

Las semillas están hardcodeadas en el código (igual que en el notebook R):

```python
SEMILLAS = [123479, 123491, 123493, 123499, 123503]
```

Estas son **números primos** cercanos que garantizan independencia entre los undersampling.

---

## 🚀 **¿Cómo Funciona?**

### **1. Training (`src/training_zlgbm.py`):**

Para cada semilla (ej: 5 modelos):

1. **Genera undersampling diferente**: `np.random.seed(semilla)` → cada modelo ve diferentes CONTINUA
2. **Entrena modelo independiente**: LightGBM con esa semilla
3. **Guarda modelo con sufijo**: `zmodelo_123479.txt`, `zmodelo_123491.txt`, etc.
4. **Guarda feature importance**: `feature_importance_123479.txt`, etc.

**Archivos generados (ksemillerio=5):**
```
output/experimento/
├── zmodelo_123479.txt
├── zmodelo_123491.txt
├── zmodelo_123493.txt
├── zmodelo_123499.txt
├── zmodelo_123503.txt
├── feature_importance_123479.txt
├── feature_importance_123491.txt
├── feature_importance_123493.txt
├── feature_importance_123499.txt
├── feature_importance_123503.txt
├── month_weights.txt
└── prediccion.txt
```

### **2. Scoring (`src/scoring_zlgbm.py`):**

1. **Carga todos los modelos**: Lee los 5 archivos `zmodelo_*.txt`
2. **Predice con cada modelo**: Genera 5 vectores de probabilidades
3. **Promedia predicciones**: `avg_predictions = mean(all_predictions)`
4. **Guarda resultado final**: `prediccion.txt` con probabilidades promediadas

---

## 📊 **Ejemplo de Uso**

### **Con ksemillerio=1 (sin ensemble):**

```bash
# Configurar en config.py: ksemillerio = 1
python run_zlgbm.py
```

**Salida:**
```
TRAINING 1 MODEL(S) FOR ENSEMBLE
Seeds to use: [123479]

MODEL 1/1 - SEED 123479
Training data (seed 123479):
  Samples: 63,720
  ...
✓ Model 1/1 saved: 85.23 MB

ENSEMBLE TRAINING COMPLETED
  Models trained: 1
  Model file: zmodelo.txt (85.23 MB)
```

### **Con ksemillerio=5 (ensemble completo):**

```bash
# Configurar en config.py: ksemillerio = 5
python run_zlgbm.py
```

**Salida:**
```
TRAINING 5 MODEL(S) FOR ENSEMBLE
Seeds to use: [123479, 123491, 123493, 123499, 123503]

MODEL 1/5 - SEED 123479
Training data (seed 123479):
  Samples: 63,720
  ...
✓ Model 1/5 saved: 85.23 MB

MODEL 2/5 - SEED 123491
Training data (seed 123491):
  Samples: 63,698
  ...
✓ Model 2/5 saved: 84.91 MB

... (3 more models) ...

ENSEMBLE TRAINING COMPLETED
  Models trained: 5
  Model files:
    - zmodelo_123479.txt (85.23 MB)
    - zmodelo_123491.txt (84.91 MB)
    - zmodelo_123493.txt (85.45 MB)
    - zmodelo_123499.txt (84.78 MB)
    - zmodelo_123503.txt (85.12 MB)
```

**Scoring:**
```
LOADING 5 MODEL(S) FOR ENSEMBLE PREDICTION

Model 1/5 (seed=123479):
  Loading from output/experimento/zmodelo_123479.txt...
  ✓ Loaded: 1247 trees
  Predicting...
  ✓ Predictions: min=0.000123, max=0.987654, mean=0.023456

... (4 more models) ...

AVERAGING ENSEMBLE PREDICTIONS
Ensemble statistics:
  Models: 5
  Final predictions:
    Min: 0.000145
    Max: 0.985432
    Mean: 0.023401
    Median: 0.012345
  Prediction std dev (across models):
    Mean: 0.001234
    Max: 0.012345
```

---

## 🎯 **Recomendaciones**

### **Para Experimentación Rápida:**
- **ksemillerio = 1**: Más rápido, bueno para iterar
- **undersampling = 0.05**: Aún más rápido (41% menos tiempo)

### **Para Submission Final:**
- **ksemillerio = 5**: Máxima robustez
- **undersampling = 0.10**: Balance entre velocidad y performance

### **Para Máxima Performance (si tenés tiempo):**
- **ksemillerio = 5**
- **undersampling = 0.15**: Más datos por modelo
- **month_weights = "exponential"**: Más peso a meses recientes

---

## ⏱️ **Tiempos Estimados**

Asumiendo ~35 minutos por modelo con undersampling=0.10:

| ksemillerio | Tiempo Total Training | Tiempo Total Pipeline |
|-------------|----------------------|----------------------|
| 1 | ~35 min | ~175 min |
| 2 | ~70 min | ~210 min |
| 3 | ~105 min | ~245 min |
| 5 | ~175 min | ~315 min |

---

## 🔍 **Validación**

El código incluye validaciones automáticas:

1. **Verificación de semillas**: Valida que ksemillerio ≤ 5
2. **Verificación de archivos**: Chequea que todos los modelos se guardaron
3. **Verificación de tamaño**: Valida que los archivos no estén vacíos
4. **Análisis de canaries**: Verifica que no haya overfitting en cada modelo

---

## 🐛 **Troubleshooting**

### **Error: "Model not found: zmodelo_123491.txt"**
- **Causa**: El training no completó todos los modelos
- **Solución**: Revisar logs de training, verificar que no haya errores

### **Error: "ksemillerio (10) exceeds available seeds (5)"**
- **Causa**: Configuraste ksemillerio > 5
- **Solución**: Cambiar a ksemillerio ≤ 5 en config.py

### **Predicciones muy similares entre modelos (std dev muy bajo)**
- **Causa**: Los modelos son muy parecidos (mismo undersampling efectivo)
- **Solución**: Normal si undersampling es alto, no es problema

---

## 📚 **Referencias**

- Notebook R original: `workflow-jueves.ipynb`
- Semillas originales: Línea 5 de `src/config.py`
- Paper zLightGBM: [Pendiente]

---

**Última actualización:** 2025-01-16

