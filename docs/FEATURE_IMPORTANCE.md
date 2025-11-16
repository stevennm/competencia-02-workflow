# Feature Importance Output

## 📊 Overview

Los experimentos ahora generan automáticamente análisis de feature importance que te ayudan a entender qué features son más importantes para el modelo.

## 📁 Archivos Generados

Después de entrenar un modelo, encontrarás estos archivos en `output/<experimento>/`:

### 1. `feature_importance.txt`
Archivo con todas las features ordenadas por importancia:

```
feature                    importance    is_canary    rank    importance_pct    importance_cumsum_pct
mpayroll_sobre_edad        1234567.8     False        1       2.45              2.45
ctrx_quarter_lag1          987654.3      False        2       1.96              4.41
cliente_edad_lag1          876543.2      False        3       1.74              6.15
canarito_1                 12345.6       True         150     0.02              95.23
...
```

**Columnas:**
- `feature`: Nombre de la feature
- `importance`: Importancia (gain total en el modelo)
- `is_canary`: Si es un canarito (True) o feature real (False)
- `rank`: Ranking de importancia (1 = más importante)
- `importance_pct`: Porcentaje de importancia relativa
- `importance_cumsum_pct`: Porcentaje acumulado

### 2. `feature_importance_analysis.txt`
Análisis detallado con:
- Resumen estadístico
- Top 100 features
- Análisis por tipo de feature (lag, delta, trend, etc.)

### 3. `feature_importance_plots.png`
Visualizaciones con 4 gráficos:
1. **Top 20 features**: Barras horizontales
2. **Importancia acumulada**: Curva mostrando cuántas features necesitas para capturar X% de importancia
3. **Distribución por tipo**: Conteo de features por categoría
4. **Pie chart**: Porcentaje de importancia por tipo

## 🚀 Uso

### Durante el entrenamiento (automático)

El feature importance se genera automáticamente cuando entrenas:

```bash
uv run run_zlgbm.py
```

**Output en consola:**
```
✓ Feature importance saved to output/seg-001_zlgbm/feature_importance.txt

Top 20 features by importance:
  Rank  Feature                                  Importance        %  Cumsum %
  -------------------------------------------------------------------------------
    1.    mpayroll_sobre_edad                     1,234,567.8    2.45%     2.45%
    2.    ctrx_quarter_lag1                         987,654.3    1.96%     4.41%
   ...

Canary analysis:
  Canary features: 200
  Real features: 1800
  Avg canary importance: 123.4
  Avg real feature importance: 456.7
  Ratio (real/canary): 3.70x
  Canaries in top 100: 2
  Canaries in top 200: 8
  ✓ Real features are more important than canaries (good!)
```

### Análisis post-entrenamiento (manual)

Para un análisis más detallado después del entrenamiento:

```bash
uv run python scripts/analizar_feature_importance.py <nombre_experimento>
```

**Ejemplo:**
```bash
uv run python scripts/analizar_feature_importance.py seg-001_zlgbm
```

**Output:**
```
======================================================================
  FEATURE IMPORTANCE ANALYSIS - seg-001_zlgbm
======================================================================

Loading feature importance from: output/seg-001_zlgbm/feature_importance.txt
✓ Loaded 2000 features

======================================================================
  SUMMARY STATISTICS
======================================================================

Total features: 2000
  Canaries: 200 (10.0%)
  Real features: 1800 (90.0%)

Importance statistics:
  Total importance: 50,345,678.9
  Canary importance: 2,468,123.4 (4.90%)
  Real importance: 47,877,555.5 (95.10%)

Average importance:
  Canaries: 12,340.6
  Real features: 26,598.6
  Ratio (real/canary): 2.16x

======================================================================
  TOP FEATURES
======================================================================

Top  10 features:
  Canaries: 0 (0.0%)
  Cumulative importance: 15.23%

Top  20 features:
  Canaries: 0 (0.0%)
  Cumulative importance: 24.56%

Top  50 features:
  Canaries: 1 (2.0%)
  Cumulative importance: 45.78%

Top 100 features:
  Canaries: 3 (3.0%)
  Cumulative importance: 65.34%

======================================================================
  FEATURE TYPE ANALYSIS (REAL FEATURES ONLY)
======================================================================

Feature type summary:
Type            Count    Total Imp      Avg Imp   % Total
----------------------------------------------------------------------
lag              600    15,234,567.8    25,390.9    31.84%
delta            600    12,456,789.0    20,761.3    26.03%
trend            153     8,765,432.1    57,292.0    18.31%
original         150     6,543,210.9    43,621.4    13.67%
ratio            100     2,345,678.0    23,456.8     4.90%
intra_month        3     1,234,567.8   411,522.6     2.58%
rf_leaf          194     1,297,309.9     6,687.2     2.71%

Top 5 features by type:

LAG:
    1. ctrx_quarter_lag1                         987,654.3
    2. cliente_edad_lag1                         876,543.2
    3. mpayroll_lag1                             765,432.1
   ...

✓ Detailed analysis saved to output/seg-001_zlgbm/feature_importance_analysis.txt
✓ Visualization saved to output/seg-001_zlgbm/feature_importance_plots.png

======================================================================
  ANALYSIS COMPLETE
======================================================================

Output files created in: output/seg-001_zlgbm/
  - feature_importance.txt (raw data)
  - feature_importance_analysis.txt (detailed analysis)
  - feature_importance_plots.png (visualizations)
```

## 🔍 Interpretación

### ¿Qué buscar?

#### 1. **Ratio real/canary > 2.0x**
- ✅ **Bueno**: Las features reales son significativamente más importantes
- ⚠️ **Malo**: Si el ratio es < 1.5x, puede haber overfitting

#### 2. **Canaritos en top features**
- ✅ **Bueno**: < 5% de canaritos en top 100
- ⚠️ **Malo**: > 10% de canaritos en top 100

#### 3. **Importancia acumulada**
- Típicamente el **20% de features** capturan **80% de importancia**
- Si necesitas > 50% de features para llegar a 80%, las features son muy dispersas

#### 4. **Tipos de features dominantes**
- **Lags y deltas**: Suelen ser los más importantes (información histórica)
- **Trends**: Capturan patrones temporales
- **RF leaves**: Capturan interacciones complejas
- **Original**: Features sin transformar

### Ejemplo de buen modelo:

```
✓ Ratio real/canary: 3.70x  (mucho mayor a 2.0)
✓ Canaries in top 100: 2    (solo 2%)
✓ Top 20 features capturan 25% de importancia
✓ Lags y deltas dominan (60% de importancia total)
```

### Ejemplo de mal modelo (overfitting):

```
⚠ Ratio real/canary: 1.2x   (muy bajo)
⚠ Canaries in top 100: 15   (15%)
⚠ Top 50 features capturan solo 30% de importancia
⚠ Features RF dominan excesivamente (50% de importancia)
```

## 📈 Casos de Uso

### 1. Feature Selection
Identificar features poco útiles para eliminar en futuras iteraciones:

```python
# Leer el archivo
df = pd.read_csv("output/seg-001_zlgbm/feature_importance.txt", sep="\t")

# Features que capturan el 95% de importancia
top_features = df[df['importance_cumsum_pct'] <= 95.0]['feature'].tolist()

# Puedes usar solo estas en el próximo experimento
```

### 2. Feature Engineering Insights
Ver qué tipos de features funcionan mejor:

```
Si "trend" features tienen alta importancia → crear más variaciones de trends
Si "ratio" features son poco útiles → eliminar o repensar
```

### 3. Debugging
Detectar problemas:

```
Si canaritos tienen alta importancia → modelo está overfitting
Si solo 1-2 features dominan → revisar data leakage
Si RF leaves tienen 0 importancia → RF no agrega valor
```

## 🎯 Tips

1. **Compara entre experimentos**: Guarda los feature importance de cada experimento para ver la evolución
2. **Documenta cambios**: Si eliminas features, anota el impacto en performance
3. **No elimines muy agresivamente**: Features con baja importancia individual pueden ser útiles en conjunto
4. **Revisa los canaritos**: Si algún canarito aparece en top 50, revisa tu código

## 🔗 Integración con otros análisis

El feature importance se complementa con:
- `tb_arboles.txt`: Estructura de árboles del modelo
- `tb_ganancias.txt`: Análisis de ganancia por corte
- Logs de entrenamiento: Para ver evolución durante training

---

**Última actualización**: 2024-11-15

