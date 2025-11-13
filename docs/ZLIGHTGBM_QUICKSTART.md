# zLightGBM - Quick Start Guide

## 🎯 ¿Qué es zLightGBM?

zLightGBM es una versión modificada de LightGBM con **control automático de overfitting** mediante dos hiperparámetros nuevos:

1. **`canaritos`**: Features aleatorios que sirven de referencia para detectar overfitting
2. **`gradient_bound`**: Cota adaptativa para los scores (learning rate inteligente)

**Ventaja principal:** No necesitas Bayesian Optimization, el modelo se regula solo.

---

## 🚀 Instalación (Solo una vez por VM)

```bash
# 1. Conectar por SSH a la VM

# 2. Ir al home
cd

# 3. Limpiar instalación anterior
rm -rf LightGBM

# 4. Clonar zLightGBM
git clone --recursive https://github.com/dmecoyfin/LightGBM

# 5. Activar entorno virtual
source ~/.venv/bin/activate

# 6. Desinstalar LightGBM estándar
pip uninstall --yes lightgbm

# 7. Instalar zLightGBM
cd ~/LightGBM
sh ./build-python.sh install

# 8. Verificar (en Python)
python -c "import lightgbm as lgb; print(lgb.__version__)"
# Debe mostrar: 4.6.0.99 o similar
```

---

## 📋 Diferencias vs Workflow Normal

### Workflow Normal (con BO)
```
Preprocessing → Feature Eng → BO (Optuna) → Train → Score
                                 ↑
                          Encuentra mejores
                          hiperparámetros
```

### Workflow zLightGBM (sin BO)
```
Preprocessing → Feature Eng → Canaritos → Train (zLGBM) → Score
                                  ↑
                           Control automático
                           de overfitting
```

**Tiempo ahorrado:** ~30-60 minutos (no hace falta BO)

---

## 🔑 Conceptos Clave

### 1. Canaritos
- Features **aleatorios** sin información real
- **DEBEN** estar al **inicio** del dataset (mandatorio)
- zLightGBM los usa como referencia: si un feature real tiene menos importancia que un canarito, algo anda mal

### 2. Gradient Bound
- Cota para los scores de las hojas
- Actúa como learning rate adaptativo
- Default: 0.1 (probar 0.05-0.2)

### 3. Hiperparámetros "Libres"
- `num_iterations`: 9999 (se detiene solo antes)
- `num_leaves`: 999 (hace menos splits si es necesario)
- `learning_rate`: 1.0 (para que gradient_bound funcione)

---

## 📝 Checklist de Implementación

### ✅ Paso 1: Instalar zLightGBM
- [ ] Ejecutar comandos de instalación en VM
- [ ] Verificar versión: `lightgbm.__version__`
- [ ] Probar script: `python playground/test_zlgbm.py`

### ✅ Paso 2: Configurar Parámetros
En `src/config.py`:
```python
"zlgbm": {
    "use_zlgbm": True,
    "qcanaritos": 100,
    "param": {
        "canaritos": 100,
        "gradient_bound": 0.1,
        "feature_fraction": 0.5,
        "min_data_in_leaf": 20,
        "num_iterations": 9999,
        "num_leaves": 999,
        "learning_rate": 1.0,
        # ... otros params
    }
}
```

### ✅ Paso 3: Agregar Canaritos
```python
def add_canaritos(df, n_canaritos=100):
    canaritos_dict = {}
    for i in range(n_canaritos):
        canaritos_dict[f"canarito_{i+1}"] = np.random.uniform(0, 1, len(df))
    
    df_canaritos = pl.DataFrame(canaritos_dict)
    return pl.concat([df_canaritos, df], how="horizontal")
```

### ✅ Paso 4: Entrenar con zLightGBM
```python
# Verificar que canaritos estén al inicio
assert campos_buenos[:n_canaritos] == [f"canarito_{i+1}" for i in range(n_canaritos)]

# Entrenar
params = {
    "canaritos": 100,
    "gradient_bound": 0.1,
    # ... otros params
}
modelo = lgb.train(params, dtrain)
```

### ✅ Paso 5: Scoring
```python
# Future data TAMBIÉN debe tener canaritos
df_future = add_canaritos(df_future, n_canaritos)
predictions = modelo.predict(X_future)
```

---

## ⚠️ Errores Comunes

### Error 1: "Canaritos no están al inicio"
```python
# ✗ MAL
campos_buenos = ["feature1", "canarito_1", "canarito_2", ...]

# ✓ BIEN
campos_buenos = ["canarito_1", "canarito_2", ..., "feature1", ...]
```

### Error 2: "Diferentes canaritos en train vs score"
```python
# ✗ MAL
df_train = add_canaritos(df_train, 100)
df_future = add_canaritos(df_future, 50)  # Diferente cantidad!

# ✓ BIEN
n_canaritos = 100
df_train = add_canaritos(df_train, n_canaritos)
df_future = add_canaritos(df_future, n_canaritos)
```

### Error 3: "Learning rate < 1.0"
```python
# ✗ MAL
params = {
    "learning_rate": 0.01,  # No funciona bien con gradient_bound
    "gradient_bound": 0.1
}

# ✓ BIEN
params = {
    "learning_rate": 1.0,  # Dejar en 1.0
    "gradient_bound": 0.1  # Este controla el learning
}
```

---

## 🎛️ Hiperparámetros a Experimentar

### Prioridad Alta
1. **`qcanaritos`**: 50, 100, 150, 200
   - Más canaritos = más control de overfitting
   - Empezar con 100

2. **`gradient_bound`**: 0.05, 0.1, 0.15, 0.2
   - Más bajo = más conservador
   - Empezar con 0.1

3. **`undersampling`**: 0.3, 0.5, 0.7
   - Con zLightGBM usar valores más altos (0.5)
   - Menos datos pero mejor calidad

### Prioridad Media
4. **`feature_fraction`**: 0.3, 0.5, 0.7
   - % de features por árbol
   - Empezar con 0.5

5. **`min_data_in_leaf`**: 10, 20, 30, 50
   - Mínimo de registros por hoja
   - Empezar con 20 (default)

---

## 📊 Comparación: BO vs zLightGBM

| Aspecto | Con BO (Optuna) | Con zLightGBM |
|---------|-----------------|---------------|
| **Tiempo** | 1-2 horas | 20-30 min |
| **Hiperparámetros** | Optimizados | Fijos + canaritos |
| **Overfitting** | Control manual | Control automático |
| **Ensemble** | 30 modelos | 1-5 modelos |
| **Complejidad** | Alta | Media |
| **Flexibilidad** | Máxima | Alta |

**Recomendación:** 
- Usar **zLightGBM** para iteraciones rápidas
- Usar **BO** para submission final (si tienes tiempo)

---

## 🧪 Testing

```bash
# Test rápido
python playground/test_zlgbm.py

# Debe mostrar:
# ✓ zLightGBM instalado correctamente
# ✓ Canaritos están correctamente al inicio
# ✓ Entrenamiento exitoso
# ✓ Workflow completo exitoso
```

---

## 📚 Documentación Completa

- **Guía detallada:** `docs/ZLIGHTGBM_INTEGRATION.md`
- **Notebook original:** `docs/zlightgbm_50.ipynb`
- **Script de prueba:** `playground/test_zlgbm.py`

---

## 💡 Tips del Profesor

1. **Canaritos al inicio es MANDATORIO**
   - No es opcional
   - Verificar siempre con assertions

2. **Undersampling más conservador**
   - Con zLightGBM: 0.5
   - Con BO: 0.1

3. **Un solo modelo puede ser suficiente**
   - No hace falta ensemble de 30 modelos
   - zLightGBM es más robusto

4. **Feature Engineering sigue siendo clave**
   - zLightGBM no reemplaza buenos features
   - Lags, deltas, RF features son importantes

5. **Experimentar es importante**
   - Probar diferentes valores de canaritos
   - Probar diferentes valores de gradient_bound
   - Analizar curvas de ganancia

---

## 🎓 Preguntas Frecuentes

**P: ¿Puedo usar zLightGBM sin canaritos?**
R: Técnicamente sí (canaritos=0), pero entonces es LightGBM normal.

**P: ¿Cuántos canaritos usar?**
R: Empezar con 100. Probar 50-200 según el tamaño del dataset.

**P: ¿zLightGBM es mejor que BO?**
R: Depende. zLightGBM es más rápido, BO es más flexible. Probar ambos.

**P: ¿Puedo combinar zLightGBM con BO?**
R: Sí, pero pierde sentido. zLightGBM ya controla overfitting automáticamente.

**P: ¿Los canaritos consumen mucha memoria?**
R: No. 100 canaritos = 100 columnas de floats. Negligible vs el resto del dataset.

---

**¡Éxito con zLightGBM! 🚀**

