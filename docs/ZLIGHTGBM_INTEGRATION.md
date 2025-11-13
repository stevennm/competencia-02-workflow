# Integración de zLightGBM en el Workflow Python

## 📋 Resumen

zLightGBM es una versión modificada de LightGBM con dos hiperparámetros nuevos:
- **`canaritos`** (aliases: canarito, qcanaritos, canaries, canary): Cantidad de features aleatorios para control de overfitting
- **`gradient_bound`** (alias: gradient_max): Cota para los scores de las hojas (learning rate adaptativo)

## 🔧 Instalación en VM

### Paso 1: Instalar zLightGBM (solo una vez por VM)

```bash
# Conectarse por SSH a la VM

# 1. Ir al home
cd

# 2. Eliminar instalación anterior de LightGBM (si existe)
rm -rf LightGBM

# 3. Clonar el repositorio de zLightGBM
git clone --recursive https://github.com/dmecoyfin/LightGBM

# 4. Activar el entorno virtual
source ~/.venv/bin/activate

# 5. Desinstalar LightGBM estándar
pip uninstall --yes lightgbm

# 6. Instalar zLightGBM (compilación desde fuente)
cd ~/LightGBM
sh ./build-python.sh install
```

**Importante:** 
- ✅ Solo se hace UNA VEZ por VM
- ✅ La librería se sigue llamando `lightgbm` (mismo import)
- ✅ Tiene los mismos métodos + los nuevos hiperparámetros

### Verificar instalación

```python
import lightgbm as lgb
print(lgb.__version__)
# Debería mostrar algo como: 4.6.0.99
```

---

## 🎯 Diferencias Clave: Workflow Original vs zLightGBM

### Workflow Original (con BO)
```
1. Preprocessing
2. Feature Engineering
3. Bayesian Optimization (Optuna) → Encuentra mejores hiperparámetros
4. Train Final Models con hiperparámetros optimizados
5. Scoring
```

### Workflow con zLightGBM (sin BO)
```
1. Preprocessing
2. Feature Engineering
3. Agregar Canaritos al Dataset ← NUEVO
4. Train Final Models con zLightGBM (sin BO) ← SIMPLIFICADO
5. Scoring
```

**¿Por qué no hace falta BO?**
- zLightGBM controla el overfitting automáticamente con canaritos
- `gradient_bound` actúa como learning rate adaptativo
- Se pueden usar hiperparámetros "libres" (num_leaves=999, num_iterations=9999)
- El modelo se detiene solo cuando no puede mejorar

---

## 📐 Adaptación del Código

### 1. Agregar Canaritos al Dataset

Los canaritos **DEBEN** ser las primeras columnas del dataset.

```python
def add_canaritos(df: pl.DataFrame, n_canaritos: int = 100) -> pl.DataFrame:
    """
    Agrega canaritos (features aleatorios) al inicio del dataset
    
    Args:
        df: DataFrame de Polars
        n_canaritos: Cantidad de canaritos a agregar
        
    Returns:
        DataFrame con canaritos al inicio
    """
    print(f"Agregando {n_canaritos} canaritos al dataset...")
    
    # Generar canaritos (valores aleatorios uniformes [0, 1])
    n_rows = df.shape[0]
    canaritos_dict = {}
    
    for i in range(n_canaritos):
        canaritos_dict[f"canarito_{i+1}"] = np.random.uniform(0, 1, n_rows)
    
    # Crear DataFrame de canaritos
    df_canaritos = pl.DataFrame(canaritos_dict)
    
    # Concatenar horizontalmente: canaritos primero, luego el resto
    df_with_canaritos = pl.concat([df_canaritos, df], how="horizontal")
    
    print(f"✓ Dataset shape: {df_with_canaritos.shape}")
    print(f"✓ Primeras columnas: {df_with_canaritos.columns[:5]}")
    
    return df_with_canaritos
```

### 2. Actualizar campos_buenos

```python
# ANTES de agregar canaritos
campos_buenos_base = [col for col in df.columns 
                      if col not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]]

# Agregar canaritos
n_canaritos = 100
df = add_canaritos(df, n_canaritos)

# DESPUÉS de agregar canaritos
canaritos_cols = [f"canarito_{i+1}" for i in range(n_canaritos)]
campos_buenos = canaritos_cols + campos_buenos_base
```

### 3. Configurar Hiperparámetros de zLightGBM

```python
# En src/config.py

PARAM = {
    # ... otros parámetros ...
    
    # zLightGBM Configuration
    "zlgbm": {
        "use_zlgbm": True,  # Flag para activar zLightGBM
        "qcanaritos": 100,   # Cantidad de canaritos
        
        "param": {
            # Parámetros estándar
            "objective": "binary",
            "metric": "custom",
            "first_metric_only": True,
            "boost_from_average": True,
            "feature_pre_filter": False,
            "verbosity": -100,
            "force_row_wise": True,
            
            # Hiperparámetros "libres" (zLightGBM se detiene solo)
            "num_iterations": 9999,  # Máximo, pero se detiene antes
            "num_leaves": 999,        # Máximo, pero hace menos splits
            "learning_rate": 1.0,     # Se deja en 1.0 para que gradient_bound funcione
            
            # Hiperparámetros a ajustar manualmente
            "min_data_in_leaf": 20,   # Default de LightGBM (probar otros valores)
            "feature_fraction": 0.50, # Equilibrado (probar 0.3-0.7)
            "max_bin": 31,
            
            # Hiperparámetros NUEVOS de zLightGBM
            "canaritos": 100,         # DEBE coincidir con qcanaritos
            "gradient_bound": 0.1     # Default de zLightGBM (probar 0.05-0.2)
        }
    },
    
    # Final Training (simplificado para zLightGBM)
    "train_final": {
        "future": [202106],
        "training": [202101, 202102, 202103, 202104],
        "undersampling": 0.50,  # Más conservador con zLightGBM
        "ksemillerio": 1,       # Con zLightGBM, un solo modelo puede ser suficiente
        "param_mejores": None   # No se usa con zLightGBM
    }
}
```

### 4. Modificar Función de Entrenamiento

```python
# En src/training.py

def train_final_models_zlgbm(df: pl.DataFrame, config: Dict, 
                              campos_buenos: List[str]) -> None:
    """
    Train final models using zLightGBM (sin Bayesian Optimization)
    
    Args:
        df: Input DataFrame (ya debe tener canaritos)
        config: Configuration dictionary
        campos_buenos: List of feature columns (incluyendo canaritos)
    """
    print("\n" + "="*50)
    print("TRAINING FINAL MODELS WITH zLightGBM")
    print("="*50)
    
    # Verificar que canaritos estén al inicio
    n_canaritos = config["zlgbm"]["qcanaritos"]
    expected_canaritos = [f"canarito_{i+1}" for i in range(n_canaritos)]
    actual_first_cols = campos_buenos[:n_canaritos]
    
    if actual_first_cols != expected_canaritos:
        raise ValueError(
            f"ERROR: Canaritos deben ser las primeras {n_canaritos} columnas!\n"
            f"Expected: {expected_canaritos[:5]}...\n"
            f"Got: {actual_first_cols[:5]}..."
        )
    
    print(f"✓ Canaritos verificados: {n_canaritos} al inicio del dataset")
    
    # Preparar datos
    df = df.with_columns([
        pl.when(pl.col("clase_ternaria").is_in(["BAJA+2", "BAJA+1"]))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("clase01")
    ])
    
    # Undersampling
    training_months = config["train_final"]["training"]
    undersampling = config["train_final"]["undersampling"]
    
    np.random.seed(config["semilla_primigenia"])
    azar = np.random.uniform(0, 1, df.shape[0])
    df = df.with_columns([pl.Series("azar", azar)])
    
    df = df.with_columns([
        pl.when(
            (pl.col("foto_mes").is_in(training_months)) &
            ((pl.col("azar") <= undersampling) | 
             (pl.col("clase_ternaria").is_in(["BAJA+1", "BAJA+2"])))
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("training")
    ])
    
    # Filtrar datos de entrenamiento
    df_train = df.filter(pl.col("training") == 1)
    
    campos_buenos_valid = [col for col in campos_buenos
                          if col in df.columns and 
                          col not in ["clase_ternaria", "clase01", "azar", "training"]]
    
    X_train = df_train.select(campos_buenos_valid).to_numpy()
    y_train = df_train.select("clase01").to_numpy().ravel()
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"  Positives: {y_train.sum()}")
    print(f"  Features: {X_train.shape[1]} (including {n_canaritos} canaritos)")
    
    # Crear dataset de LightGBM
    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    
    # Obtener parámetros de zLightGBM
    lgb_params = config["zlgbm"]["param"].copy()
    lgb_params["seed"] = config["semilla_primigenia"]
    
    print(f"\nzLightGBM parameters:")
    print(f"  canaritos: {lgb_params['canaritos']}")
    print(f"  gradient_bound: {lgb_params['gradient_bound']}")
    print(f"  feature_fraction: {lgb_params['feature_fraction']}")
    print(f"  min_data_in_leaf: {lgb_params['min_data_in_leaf']}")
    
    # Entrenar modelo
    print("\nTraining zLightGBM model...")
    print("(El modelo se detendrá automáticamente cuando no pueda mejorar)")
    
    modelo = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=lgb_params["num_iterations"]
    )
    
    # Información del modelo
    n_trees = modelo.num_trees()
    print(f"\n✓ Training complete!")
    print(f"  Trees built: {n_trees} (de máximo {lgb_params['num_iterations']})")
    print(f"  zLightGBM se detuvo automáticamente")
    
    # Guardar modelo
    experimento = config["experimento"]
    output_dir = Path(f"output/{experimento}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_file = output_dir / "zmodelo.txt"
    modelo.save_model(str(model_file))
    print(f"\n✓ Model saved to {model_file}")
    
    # Guardar información de árboles
    import pandas as pd
    tree_df = modelo.trees_to_dataframe()
    tree_file = output_dir / "tb_arboles.txt"
    tree_df.to_csv(tree_file, sep="\t", index=False)
    print(f"✓ Tree structure saved to {tree_file}")
    
    # Análisis de hojas por árbol
    leaves_per_tree = tree_df.groupby('tree_index')['leaf_index'].max() + 1
    print(f"\nTree structure summary:")
    print(f"  Total trees: {n_trees}")
    print(f"  Leaves per tree (summary):")
    print(f"    Min: {leaves_per_tree.min()}")
    print(f"    Median: {leaves_per_tree.median():.0f}")
    print(f"    Max: {leaves_per_tree.max()}")
    print(f"    Mean: {leaves_per_tree.mean():.1f}")
```

### 5. Modificar Scoring para zLightGBM

```python
def score_future_data_zlgbm(df: pl.DataFrame, config: Dict, 
                            campos_buenos: List[str]) -> pl.DataFrame:
    """
    Score future data using zLightGBM model
    
    Args:
        df: Input DataFrame (ya debe tener canaritos)
        config: Configuration dictionary
        campos_buenos: List of feature columns (incluyendo canaritos)
        
    Returns:
        DataFrame with predictions
    """
    print("\n" + "="*50)
    print("SCORING FUTURE DATA (zLightGBM)")
    print("="*50)
    
    # Filtrar mes futuro
    future_months = config["train_final"]["future"]
    df_future = df.filter(pl.col("foto_mes").is_in(future_months))
    
    print(f"Future data: {df_future.shape[0]} rows")
    
    # Verificar que canaritos estén presentes
    n_canaritos = config["zlgbm"]["qcanaritos"]
    expected_canaritos = [f"canarito_{i+1}" for i in range(n_canaritos)]
    
    missing_canaritos = [c for c in expected_canaritos if c not in df_future.columns]
    if missing_canaritos:
        raise ValueError(f"ERROR: Faltan canaritos en future data: {missing_canaritos[:5]}")
    
    # Preparar features
    campos_buenos_valid = [col for col in campos_buenos 
                          if col in df_future.columns and 
                          col not in ["clase_ternaria", "numero_de_cliente", "foto_mes"]]
    
    X_future = df_future.select(campos_buenos_valid).to_numpy()
    
    # Cargar modelo
    experimento = config["experimento"]
    model_file = Path(f"output/{experimento}/zmodelo.txt")
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")
    
    print(f"Loading model from {model_file}...")
    modelo = lgb.Booster(model_file=str(model_file))
    
    # Predecir
    print("Making predictions...")
    predictions = modelo.predict(X_future)
    
    # Crear DataFrame de predicciones
    df_pred = df_future.select(["numero_de_cliente", "foto_mes"]).with_columns([
        pl.Series("prob", predictions)
    ])
    
    # Guardar predicciones
    output_dir = Path(f"output/{experimento}")
    pred_file = output_dir / "prediccion.txt"
    df_pred.write_csv(pred_file, separator="\t")
    print(f"✓ Predictions saved to {pred_file}")
    
    return df_pred
```

---

## 🔄 Modificar run.py

```python
# En run.py, modificar el main()

def main():
    # ... código existente ...
    
    # Determinar si usar zLightGBM
    use_zlgbm = PARAM.get("zlgbm", {}).get("use_zlgbm", False)
    
    if use_zlgbm:
        print_section("MODO: zLightGBM (sin Bayesian Optimization)")
        
        # PASO ADICIONAL: Agregar canaritos
        print_section("STEP X: AGREGAR CANARITOS")
        n_canaritos = PARAM["zlgbm"]["qcanaritos"]
        df = add_canaritos(df, n_canaritos)
        
        # Actualizar campos_buenos
        canaritos_cols = [f"canarito_{i+1}" for i in range(n_canaritos)]
        campos_buenos_base = [col for col in df.columns 
                             if col not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]
                             and not col.startswith("canarito_")]
        campos_buenos = canaritos_cols + campos_buenos_base
        
        print(f"Total features: {len(campos_buenos)}")
        print(f"  Canaritos: {n_canaritos}")
        print(f"  Real features: {len(campos_buenos_base)}")
        
        # SKIP BAYESIAN OPTIMIZATION
        print_section("SKIPPING BAYESIAN OPTIMIZATION (usando zLightGBM)")
        
        # TRAIN FINAL MODELS
        print_section("STEP: TRAIN FINAL MODEL (zLightGBM)")
        train_final_models_zlgbm(df, PARAM, campos_buenos)
        
        # SCORING
        print_section("STEP: SCORE FUTURE DATA")
        df_pred = score_future_data_zlgbm(df, PARAM, campos_buenos)
        
    else:
        # Workflow normal con BO
        # ... código existente ...
```

---

## 🎛️ Hiperparámetros a Experimentar

Con zLightGBM, los hiperparámetros críticos son:

### 1. `canaritos` (qcanaritos)
- **Rango:** 50-200
- **Efecto:** Más canaritos = más control de overfitting
- **Recomendación inicial:** 100

### 2. `gradient_bound`
- **Rango:** 0.05-0.2
- **Efecto:** Learning rate adaptativo
- **Recomendación inicial:** 0.1 (default)

### 3. `feature_fraction`
- **Rango:** 0.3-0.7
- **Efecto:** % de features usados por árbol
- **Recomendación inicial:** 0.5

### 4. `min_data_in_leaf`
- **Rango:** 10-50
- **Efecto:** Mínimo de registros por hoja
- **Recomendación inicial:** 20 (default)

### 5. `undersampling`
- **Rango:** 0.3-0.7
- **Efecto:** % de negativos en training
- **Recomendación inicial:** 0.5 (más conservador que con BO)

---

## 📊 Ventajas de zLightGBM

1. ✅ **No necesita Bayesian Optimization** → Más rápido
2. ✅ **Control automático de overfitting** → Menos tuning manual
3. ✅ **Se detiene solo** → No necesitas early stopping
4. ✅ **Hiperparámetros "libres"** → num_leaves=999, num_iterations=9999
5. ✅ **Menos modelos en ensemble** → Un solo modelo puede ser suficiente

---

## ⚠️ Consideraciones Importantes

1. **Canaritos DEBEN estar al inicio**
   - No opcional, es mandatorio
   - Verificar siempre con assertions

2. **Mismo número de canaritos en train y score**
   - `qcanaritos` debe coincidir con `canaritos` en params
   - Future data también debe tener canaritos

3. **Learning rate = 1.0**
   - Para que `gradient_bound` funcione correctamente
   - No usar learning_rate < 1.0

4. **Undersampling más conservador**
   - Con zLightGBM, usar 0.5 en vez de 0.1
   - Menos datos pero mejor calidad

5. **Feature Engineering sigue siendo importante**
   - zLightGBM no reemplaza buenos features
   - Lags, deltas, RF features siguen siendo valiosos

---

## 🚀 Plan de Implementación

### Fase 1: Setup (1 sesión)
1. Instalar zLightGBM en VM
2. Crear funciones `add_canaritos()`
3. Actualizar `config.py` con parámetros zLightGBM

### Fase 2: Adaptación (1-2 sesiones)
4. Crear `train_final_models_zlgbm()`
5. Crear `score_future_data_zlgbm()`
6. Modificar `run.py` para soportar ambos modos

### Fase 3: Testing (1 sesión)
7. Probar con un experimento pequeño
8. Verificar que canaritos estén correctos
9. Comparar resultados con workflow original

### Fase 4: Optimización (2-3 sesiones)
10. Experimentar con `qcanaritos` (50, 100, 150, 200)
11. Experimentar con `gradient_bound` (0.05, 0.1, 0.15, 0.2)
12. Experimentar con `feature_fraction` (0.3, 0.5, 0.7)
13. Analizar curvas de ganancia

---

## 📚 Referencias

- Notebook original: `docs/zlightgbm_50.ipynb`
- Repositorio zLightGBM: https://github.com/dmecoyfin/LightGBM
- Documentación LightGBM: https://lightgbm.readthedocs.io/

---

**Próximos pasos:** ¿Quieres que empiece a implementar las funciones o prefieres revisar primero la estrategia?

