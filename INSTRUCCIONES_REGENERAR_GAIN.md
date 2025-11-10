# Instrucciones para Regenerar Gain Analysis

## Problema Identificado

El gráfico de ensemble mostraba solo 5 líneas en la leyenda cuando en realidad hay **30 modelos** en el ensemble. Esto era solo un problema de visualización - el código solo etiquetaba las primeras 5 semillas para mantener la leyenda limpia, pero graficaba todas.

## Cambios Realizados

### 1. Archivo modificado: `src/gain_analysis.py`

**Antes:**
- Graficaba todas las semillas pero solo etiquetaba las primeras 5
- La leyenda mostraba "Seed 1", "Seed 2", ... "Seed 5"
- No era claro cuántas semillas había en total

**Después:**
- Grafica todas las 30 semillas con líneas delgadas de colores
- La leyenda muestra "Individual Seeds (n=30)" en lugar de listar cada una
- Más limpio y claro visualmente
- El título ahora usa el nombre del experimento dinámicamente

### 2. Script creado: `regenerate_gain_analysis.py`

Este script regenera ambos gráficos de gain:
- `gain_curve.png` - Curva de ganancia del modelo promedio
- `gain_curve_ensemble.png` - Curva con todas las semillas individuales + promedio

## Cómo Ejecutar

### Opción 1: Desde Python directamente

```bash
python regenerate_gain_analysis.py
```

### Opción 2: Desde tu entorno virtual (si usas uno)

```bash
# Activar entorno virtual primero
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate  # Windows

# Ejecutar script
python regenerate_gain_analysis.py
```

### Opción 3: Desde Jupyter/IPython

```python
%run regenerate_gain_analysis.py
```

## Verificación

Después de ejecutar, verifica que los archivos se regeneraron:

```bash
# Deberían tener timestamp reciente
ls -la output/semillerio_optuna/analysis/
```

Archivos que se regeneran:
- ✅ `output/semillerio_optuna/analysis/gain_curve.png`
- ✅ `output/semillerio_optuna/analysis/gain_curve_ensemble.png`
- ✅ `output/semillerio_optuna/analysis/gain_by_cutoff.csv`
- ✅ `output/semillerio_optuna/analysis/gain_ensemble_stats.csv`

## Qué Esperar en el Nuevo Gráfico

### gain_curve_ensemble.png

**Visualización mejorada:**
- 30 líneas de colores (una por cada semilla) - delgadas y semi-transparentes
- 1 línea negra gruesa (promedio del ensemble)
- Leyenda limpia: "Individual Seeds (n=30)" y "Ensemble (Average)"
- Punto rojo marcando el máximo de ganancia
- Título con el nombre del experimento: "EXP semillerio_optuna"

**Interpretación:**
- Si las 30 líneas están muy dispersas → alta variabilidad entre semillas
- Si las 30 líneas están juntas → baja variabilidad (bueno!)
- La línea negra (ensemble) debería estar en el medio de todas

## Datos Verificados

✅ Hay **30 modelos** en `output/semillerio_optuna/modelitos/`
✅ Todos tienen **2,026 features**
✅ El archivo `predicciones_ensemble.parquet` contiene 30 columnas `prob_seed_1` a `prob_seed_30`
✅ El archivo `gain_ensemble_stats.csv` muestra estadísticas correctas (min, max, std) de las 30 semillas

## Troubleshooting

### Error: "No module named 'src'"

Asegúrate de ejecutar desde el directorio raíz del proyecto:
```bash
cd c:\Users\rubmartins\Documents\MAESTRIA\VERSION_WORKFLOW_JUEVES
python regenerate_gain_analysis.py
```

### Error: "File not found: data/final_dataset.parquet"

El script necesita el dataset completo. Si no lo tienes, usa:
```python
# En regenerate_gain_analysis.py, cambiar línea 18 a:
df = pl.read_parquet("data/featured_data.parquet")
```

### Error: "predicciones_ensemble.parquet not found"

Necesitas ejecutar primero el scoring:
```bash
python -c "from src.scoring import score_future_data; from src.config import PARAM; import polars as pl; df = pl.read_parquet('data/final_dataset.parquet'); campos_buenos = [c for c in df.columns if c not in ['numero_de_cliente', 'foto_mes', 'clase_ternaria']]; score_future_data(df, PARAM, campos_buenos)"
```

O simplemente vuelve a ejecutar el pipeline completo.

## Próximos Pasos

Después de regenerar los gráficos:

1. **Compara** el nuevo gráfico con el anterior
2. **Verifica** que ahora se vean las 30 líneas de colores
3. **Analiza** la variabilidad entre semillas (dispersión de las líneas)
4. **Confirma** que el ensemble (línea negra) tiene mejor ganancia que la mayoría de semillas individuales

## Notas Adicionales

- El cambio en `gain_analysis.py` es permanente - todos los futuros experimentos usarán esta visualización mejorada
- No afecta el cálculo de ganancias, solo la visualización
- Los archivos CSV con estadísticas siguen siendo los mismos

