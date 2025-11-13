# Analizar y proponer nuevas features
_Exported on 11/12/2025 at 00:33:48 GMT-3 from Cursor (2.0.69)_

---

**User**

analizá mis features... que otras features se te ocurren que podemos agregarle? fijate tambien los files de salidas conl as features que ya tenemos (por ejemplo el experimento semillerio_optuna2 tiene importancias)

---

**Cursor**

I'll analyze your features by examining the codebase, particularly the preprocessing and feature engineering code, as well as the feature importance outputs.

Excelente! Ahora voy a analizar tus features actuales y sugerirte nuevas ideas. Déjame ver también los archivos de feature engineering para entender mejor qué ya tienes.

Perfecto! Ahora tengo una visión completa de tus features. Déjame analizar y sugerirte nuevas ideas:

## 📊 **ANÁLISIS DE TUS FEATURES ACTUALES**

Según el archivo de importancias y tu código, tienes:

### **Features Existentes:**
1. **Intra-mes básicas:** `kmes`, `ctrx_quarter_normalizado`, `mpayroll_sobre_edad`
2. **Lags:** lag1, lag2 de todas las variables
3. **Deltas:** delta1, delta2 (diferencias con lags)
4. **Tendencias (ventana 6):** tend6, min6, max6, avg6, ratioavg6, ratiomax6
5. **Random Forest features:** rf_XXX_YYY (20 árboles, features binarias de hojas)

### **Top Features según importancia:**
- `ctrx_quarter_normalizado` (209K gain) 🔥
- `rf_014_007` (158K)
- `mcaja_ahorro`, `ctarjeta_visa` (transacciones y montos)
- Ratios y tendencias de productos bancarios

---

## 💡 **NUEVAS IDEAS DE FEATURES**

### **1. FEATURES DE VOLATILIDAD Y ESTABILIDAD** ⭐⭐⭐
```python
# Desviación estándar sobre ventana móvil
pl.col("mcaja_ahorro").rolling_std(window_size=6).over("numero_de_cliente").alias("mcaja_ahorro_std6")

# Coeficiente de variación (CV = std/mean)
(pl.col("mcaja_ahorro_std6") / pl.col("mcaja_ahorro_avg6")).alias("mcaja_ahorro_cv6")

# Rango intercuartil (IQR) - robustez a outliers
(pl.col("mcaja_ahorro_max6") - pl.col("mcaja_ahorro_min6")).alias("mcaja_ahorro_rango6")
```
**Por qué:** Los clientes que van a dar de baja suelen tener comportamiento más errático antes de irse.

---

### **2. FEATURES DE ACELERACIÓN (2da DERIVADA)** ⭐⭐⭐
```python
# Aceleración = delta de deltas
(pl.col("mcaja_ahorro_delta1") - pl.col("mcaja_ahorro_delta2")).alias("mcaja_ahorro_accel")

# Cambio en la tendencia
(pl.col("mcaja_ahorro_tend6") - pl.col("mcaja_ahorro_tend6").shift(1).over("numero_de_cliente")).alias("mcaja_ahorro_cambio_tend")
```
**Por qué:** Captura cambios de comportamiento acelerados (ej: cliente que primero reduce actividad lentamente, luego rápido).

---

### **3. FEATURES DE MOMENTUM Y DIRECCIÓN** ⭐⭐
```python
# Momentum: suma de deltas recientes
(pl.col("mcaja_ahorro_delta1") + pl.col("mcaja_ahorro_delta2")).alias("mcaja_ahorro_momentum")

# Dirección consistente: ¿delta1 y delta2 tienen mismo signo?
(pl.col("mcaja_ahorro_delta1").sign() == pl.col("mcaja_ahorro_delta2").sign()).cast(pl.Int32).alias("mcaja_ahorro_direccion_consistente")

# Conteo de meses consecutivos con crecimiento/decrecimiento
```
**Por qué:** Clientes con tendencia consistente a la baja son más riesgosos.

---

### **4. FEATURES DE INTERACCIÓN ENTRE PRODUCTOS** ⭐⭐⭐
```python
# Ratio de uso entre productos
(pl.col("ctarjeta_visa_transacciones") / (pl.col("ctarjeta_master_transacciones") + 1)).alias("ratio_visa_master")

# Diversificación de productos (entropía simplificada)
# Proporción de saldo en cada producto
pl.col("mcaja_ahorro") / (pl.col("mcuentas_saldo") + 1).alias("prop_caja_ahorro")

# Concentración: ¿usa principalmente un solo producto?
pl.max_horizontal(["prop_caja_ahorro", "prop_cuenta_corriente", ...]).alias("concentracion_producto")
```
**Por qué:** Clientes con un solo producto son más propensos a irse.

---

### **5. FEATURES DE ACTIVIDAD RELATIVA** ⭐⭐⭐
```python
# Actividad vs promedio histórico propio
(pl.col("ctrx_quarter") / pl.col("ctrx_quarter_avg6")).alias("ctrx_quarter_vs_promedio")

# Percentil de actividad en los últimos 6 meses
# (requiere rank over window)

# Días desde última transacción (si tienes fechas)
```
**Por qué:** Caída en actividad relativa es señal de desenganche.

---

### **6. FEATURES DE ESTACIONALIDAD MEJORADAS** ⭐⭐
```python
# Diferencia vs mismo mes año anterior (si tienes datos)
(pl.col("mcaja_ahorro") - pl.col("mcaja_ahorro").shift(12).over("numero_de_cliente")).alias("mcaja_ahorro_yoy")

# Promedio del mes específico (ej: promedio de todos los eneros del cliente)
pl.col("mcaja_ahorro").mean().over(["numero_de_cliente", "kmes"]).alias("mcaja_ahorro_avg_mes")

# Desviación del patrón estacional
(pl.col("mcaja_ahorro") - pl.col("mcaja_ahorro_avg_mes")).alias("mcaja_ahorro_desv_estacional")
```

---

### **7. FEATURES DE EDAD/ANTIGÜEDAD RELATIVA** ⭐⭐
```python
# Edad normalizada por percentiles
pl.col("cliente_edad").rank().over("foto_mes").alias("cliente_edad_percentil")

# Ratio edad/antigüedad (clientes nuevos pero mayores)
(pl.col("cliente_edad") / (pl.col("cliente_antiguedad") + 1)).alias("edad_sobre_antiguedad")

# Interacción edad con productos
(pl.col("cliente_edad") * pl.col("ctarjeta_visa")).alias("edad_x_visa")
```

---

### **8. FEATURES DE CAMBIOS BINARIOS** ⭐⭐
```python
# ¿Cambió de estado? (ej: tenía producto, ahora no)
(pl.col("ccaja_ahorro") > 0).cast(pl.Int32).alias("tiene_caja_ahorro")
(pl.col("tiene_caja_ahorro") != pl.col("tiene_caja_ahorro").shift(1).over("numero_de_cliente")).alias("cambio_caja_ahorro")

# Conteo de cambios en últimos N meses
pl.col("cambio_caja_ahorro").rolling_sum(window_size=6).over("numero_de_cliente").alias("cambios_caja_ahorro_6m")
```

---

### **9. FEATURES DE RATIOS COMPLEJOS** ⭐⭐⭐
```python
# Ratio ingresos/egresos
(pl.col("mtransferencias_recibidas") / (pl.col("mtransferencias_emitidas") + 1)).alias("ratio_ingresos_egresos")

# Ratio comisiones/saldo (¿paga mucho por poco saldo?)
(pl.col("mcomisiones") / (pl.col("mcuentas_saldo") + 1)).alias("ratio_comisiones_saldo")

# Utilización de límite de crédito
(pl.col("Visa_msaldototal") / (pl.col("Visa_mlimitecompra") + 1)).alias("utilizacion_credito_visa")
```

---

### **10. FEATURES DE OUTLIERS Y ANOMALÍAS** ⭐⭐
```python
# Z-score respecto a historia propia
((pl.col("mcaja_ahorro") - pl.col("mcaja_ahorro_avg6")) / (pl.col("mcaja_ahorro_std6") + 1)).alias("mcaja_ahorro_zscore")

# ¿Es outlier? (|z-score| > 2)
(pl.col("mcaja_ahorro_zscore").abs() > 2).cast(pl.Int32).alias("mcaja_ahorro_es_outlier")

# Conteo de outliers recientes
pl.col("mcaja_ahorro_es_outlier").rolling_sum(window_size=3).over("numero_de_cliente").alias("outliers_recientes")
```

---

### **11. FEATURES DE VENTANAS MÚLTIPLES** ⭐⭐
```python
# Comparar tendencias de diferentes ventanas
pl.col("mcaja_ahorro").rolling_mean(window_size=3).over("numero_de_cliente").alias("mcaja_ahorro_avg3")
(pl.col("mcaja_ahorro_avg3") / pl.col("mcaja_ahorro_avg6")).alias("ratio_avg3_avg6")

# Tendencia corto vs largo plazo
(pl.col("mcaja_ahorro_tend3") - pl.col("mcaja_ahorro_tend6")).alias("divergencia_tendencias")
```

---

### **12. FEATURES DE SECUENCIAS** ⭐
```python
# Meses consecutivos con valor > 0
# Meses consecutivos con crecimiento
# Longitud de racha actual (streak)
```

---

## 🎯 **RECOMENDACIONES PRIORIZADAS**

### **Implementar PRIMERO (mayor impacto esperado):**
1. **Volatilidad** (std, CV) - muy predictivo de churn
2. **Aceleración** - captura cambios bruscos
3. **Interacciones entre productos** - diversificación
4. **Ratios complejos** (comisiones/saldo, utilización crédito)

### **Implementar SEGUNDO:**
5. Momentum y dirección
6. Actividad relativa
7. Features de outliers

### **Considerar si tienes tiempo:**
8. Estacionalidad mejorada
9. Ventanas múltiples
10. Features binarias de cambios

---

## 📝 **CÓDIGO DE EJEMPLO PARA IMPLEMENTAR**

Te puedo ayudar a agregar estas features a tu `feature_engineering.py`. ¿Querés que te prepare una función `add_advanced_features()` con las más importantes?

**Nota importante:** Con 2027 features actuales, agregar muchas más puede causar overfitting. Te recomiendo:
- Agregar 50-100 features nuevas bien pensadas
- Usar feature selection después (ej: eliminar features con gain < threshold)
- Probar en validación si mejoran el modelo

¿Querés que implemente alguna de estas categorías en particular?

```python
# Desviación estándar sobre ventana móvil
pl.col("mcaja_ahorro").rolling_std(window_size=6).over("numero_de_cliente").alias("mcaja_ahorro_std6")

# Coeficiente de variación (CV = std/mean)
(pl.col("mcaja_ahorro_std6") / pl.col("mcaja_ahorro_avg6")).alias("mcaja_ahorro_cv6")

# Rango intercuartil (IQR) - robustez a outliers
(pl.col("mcaja_ahorro_max6") - pl.col("mcaja_ahorro_min6")).alias("mcaja_ahorro_rango6")
```

```python
# Aceleración = delta de deltas
(pl.col("mcaja_ahorro_delta1") - pl.col("mcaja_ahorro_delta2")).alias("mcaja_ahorro_accel")

# Cambio en la tendencia
(pl.col("mcaja_ahorro_tend6") - pl.col("mcaja_ahorro_tend6").shift(1).over("numero_de_cliente")).alias("mcaja_ahorro_cambio_tend")
```

```python
# Momentum: suma de deltas recientes
(pl.col("mcaja_ahorro_delta1") + pl.col("mcaja_ahorro_delta2")).alias("mcaja_ahorro_momentum")

# Dirección consistente: ¿delta1 y delta2 tienen mismo signo?
(pl.col("mcaja_ahorro_delta1").sign() == pl.col("mcaja_ahorro_delta2").sign()).cast(pl.Int32).alias("mcaja_ahorro_direccion_consistente")

# Conteo de meses consecutivos con crecimiento/decrecimiento
```

```python
# Ratio de uso entre productos
(pl.col("ctarjeta_visa_transacciones") / (pl.col("ctarjeta_master_transacciones") + 1)).alias("ratio_visa_master")

# Diversificación de productos (entropía simplificada)
# Proporción de saldo en cada producto
pl.col("mcaja_ahorro") / (pl.col("mcuentas_saldo") + 1).alias("prop_caja_ahorro")

# Concentración: ¿usa principalmente un solo producto?
pl.max_horizontal(["prop_caja_ahorro", "prop_cuenta_corriente", ...]).alias("concentracion_producto")
```

```python
# Actividad vs promedio histórico propio
(pl.col("ctrx_quarter") / pl.col("ctrx_quarter_avg6")).alias("ctrx_quarter_vs_promedio")

# Percentil de actividad en los últimos 6 meses
# (requiere rank over window)

# Días desde última transacción (si tienes fechas)
```

```python
# Diferencia vs mismo mes año anterior (si tienes datos)
(pl.col("mcaja_ahorro") - pl.col("mcaja_ahorro").shift(12).over("numero_de_cliente")).alias("mcaja_ahorro_yoy")

# Promedio del mes específico (ej: promedio de todos los eneros del cliente)
pl.col("mcaja_ahorro").mean().over(["numero_de_cliente", "kmes"]).alias("mcaja_ahorro_avg_mes")

# Desviación del patrón estacional
(pl.col("mcaja_ahorro") - pl.col("mcaja_ahorro_avg_mes")).alias("mcaja_ahorro_desv_estacional")
```

```python
# Edad normalizada por percentiles
pl.col("cliente_edad").rank().over("foto_mes").alias("cliente_edad_percentil")

# Ratio edad/antigüedad (clientes nuevos pero mayores)
(pl.col("cliente_edad") / (pl.col("cliente_antiguedad") + 1)).alias("edad_sobre_antiguedad")

# Interacción edad con productos
(pl.col("cliente_edad") * pl.col("ctarjeta_visa")).alias("edad_x_visa")
```

```python
# ¿Cambió de estado? (ej: tenía producto, ahora no)
(pl.col("ccaja_ahorro") > 0).cast(pl.Int32).alias("tiene_caja_ahorro")
(pl.col("tiene_caja_ahorro") != pl.col("tiene_caja_ahorro").shift(1).over("numero_de_cliente")).alias("cambio_caja_ahorro")

# Conteo de cambios en últimos N meses
pl.col("cambio_caja_ahorro").rolling_sum(window_size=6).over("numero_de_cliente").alias("cambios_caja_ahorro_6m")
```

```python
# Ratio ingresos/egresos
(pl.col("mtransferencias_recibidas") / (pl.col("mtransferencias_emitidas") + 1)).alias("ratio_ingresos_egresos")

# Ratio comisiones/saldo (¿paga mucho por poco saldo?)
(pl.col("mcomisiones") / (pl.col("mcuentas_saldo") + 1)).alias("ratio_comisiones_saldo")

# Utilización de límite de crédito
(pl.col("Visa_msaldototal") / (pl.col("Visa_mlimitecompra") + 1)).alias("utilizacion_credito_visa")
```

```python
# Z-score respecto a historia propia
((pl.col("mcaja_ahorro") - pl.col("mcaja_ahorro_avg6")) / (pl.col("mcaja_ahorro_std6") + 1)).alias("mcaja_ahorro_zscore")

# ¿Es outlier? (|z-score| > 2)
(pl.col("mcaja_ahorro_zscore").abs() > 2).cast(pl.Int32).alias("mcaja_ahorro_es_outlier")

# Conteo de outliers recientes
pl.col("mcaja_ahorro_es_outlier").rolling_sum(window_size=3).over("numero_de_cliente").alias("outliers_recientes")
```

```python
# Comparar tendencias de diferentes ventanas
pl.col("mcaja_ahorro").rolling_mean(window_size=3).over("numero_de_cliente").alias("mcaja_ahorro_avg3")
(pl.col("mcaja_ahorro_avg3") / pl.col("mcaja_ahorro_avg6")).alias("ratio_avg3_avg6")

# Tendencia corto vs largo plazo
(pl.col("mcaja_ahorro_tend3") - pl.col("mcaja_ahorro_tend6")).alias("divergencia_tendencias")
```

```python
# Meses consecutivos con valor > 0
# Meses consecutivos con crecimiento
# Longitud de racha actual (streak)
```

