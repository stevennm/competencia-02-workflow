# 🚀 NUEVAS FEATURES AVANZADAS

## Resumen
Se agregaron **~500-700 features nuevas** organizadas en 10 categorías diferentes.

---

## 📊 CATEGORÍAS DE FEATURES

### 1. **VOLATILITY FEATURES** (Volatilidad y Estabilidad)
- `{col}_std6`: Desviación estándar en ventana de 6 meses
- `{col}_cv6`: Coeficiente de variación (CV = std/mean)
- `{col}_rango6`: Rango (max - min)

**Por qué:** Clientes con comportamiento errático antes de dar de baja.

---

### 2. **ACCELERATION FEATURES** (Segunda Derivada)
- `{col}_accel`: Aceleración (delta1 - delta2)
- `{col}_cambio_tend`: Cambio en la tendencia

**Por qué:** Captura cambios de comportamiento acelerados.

---

### 3. **MOMENTUM & DIRECTION FEATURES** ⭐ (OBLIGATORIA)
- `{col}_momentum`: Suma de deltas (delta1 + delta2)
- `{col}_direccion_consistente`: ¿Delta1 y delta2 tienen mismo signo? (0/1)
- `{col}_direccion`: Dirección del cambio (-1, 0, 1)

**Por qué:** Tendencia consistente a la baja = mayor riesgo de churn.

---

### 4. **RELATIVE ACTIVITY FEATURES** (Actividad Relativa)
- `{col}_vs_promedio`: Valor actual / promedio histórico
- `{col}_vs_maximo`: Valor actual / máximo histórico

**Por qué:** Caída en actividad relativa = desenganche.

---

### 5. **OUTLIER & ANOMALY FEATURES** (Detección de Anomalías)
- `{col}_zscore`: Z-score respecto a historia propia
- `{col}_es_outlier`: ¿Es outlier? (|z| > 2)
- `{col}_outliers_recientes`: Conteo de outliers en últimos 3 meses

**Por qué:** Comportamiento anómalo puede indicar problemas.

---

### 6. **PRODUCT INTERACTION FEATURES** (Interacciones entre Productos)
- `ratio_transacciones_visa_master`: Visa / Master transacciones
- `prop_saldo_caja_ahorro`: Proporción de saldo en caja de ahorro
- `prop_saldo_cuenta_corriente`: Proporción en cuenta corriente
- `ratio_comisiones_saldo`: Comisiones / Saldo total
- `ratio_ingresos_egresos`: Transferencias recibidas / emitidas
- `utilizacion_credito_visa`: Saldo / Límite Visa
- `utilizacion_credito_master`: Saldo / Límite Master
- `ratio_payroll_consumo`: Payroll / Consumo tarjeta
- `num_productos_activos`: Cantidad de productos activos

**Por qué:** Diversificación de productos reduce riesgo de churn.

---

### 7. **MULTI-WINDOW FEATURES** (Ventanas Múltiples)
- `{col}_avg3`: Promedio de 3 meses
- `{col}_ratio_avg3_avg6`: Ratio entre promedio corto y largo plazo

**Por qué:** Comparar tendencias de corto vs largo plazo.

---

### 8. **BINARY CHANGE FEATURES** (Cambios de Estado)
- `tiene_{producto}`: ¿Tiene el producto activo? (0/1)
- `cambio_{producto}`: ¿Cambió de estado este mes?
- `cambios_{producto}_6m`: Conteo de cambios en 6 meses

**Productos monitoreados:**
- caja_ahorro
- cuenta_corriente
- tarjeta_visa
- tarjeta_master
- prestamos_personales
- internet

**Por qué:** Cambios frecuentes = inestabilidad.

---

### 9. **AGE & TENURE INTERACTIONS** (Edad y Antigüedad)
- `edad_sobre_antiguedad`: Edad / Antigüedad
- `cliente_mayor_nuevo`: Edad > 50 y Antigüedad < 12 meses
- `cliente_joven_antiguo`: Edad < 30 y Antigüedad > 24 meses
- `edad_x_ctarjeta_visa`: Interacción edad × uso de visa
- `edad_x_mcaja_ahorro`: Interacción edad × saldo caja ahorro
- `edad_x_mpayroll`: Interacción edad × payroll

**Por qué:** Segmentos demográficos tienen comportamientos diferentes.

---

### 10. **ENHANCED SEASONALITY FEATURES** (Estacionalidad Mejorada)
- `trimestre`: Trimestre del año (1-4)
- `semestre`: Semestre del año (1-2)
- `es_fin_anio`: ¿Es noviembre o diciembre? (0/1)
- `es_inicio_anio`: ¿Es enero, febrero o marzo? (0/1)

**Por qué:** Patrones estacionales en comportamiento bancario.

---

## 🎯 FEATURES APLICADAS A:

Las features se aplican a las **50 columnas más importantes** que cumplan:
- Columnas monetarias (empiezan con `m`)
- Columnas de transacciones/conteos (empiezan con `c`)
- Columnas de Visa/Master

**Ejemplos de columnas clave:**
- `mcaja_ahorro`
- `mcuentas_saldo`
- `ctarjeta_visa_transacciones`
- `mpayroll`
- `Visa_msaldototal`
- `Master_mlimitecompra`
- etc.

---

## 📈 ESTIMACIÓN DE FEATURES NUEVAS

| Categoría | Features por columna | Total estimado |
|-----------|---------------------|----------------|
| Volatility | 3 | ~150 |
| Acceleration | 2 | ~100 |
| Momentum & Direction | 3 | ~150 |
| Relative Activity | 2 | ~100 |
| Outliers | 3 | ~150 |
| Product Interactions | 9 | 9 |
| Multi-window | 2 | ~60 |
| Binary Changes | 3 × 6 productos | 18 |
| Age & Tenure | 6 | 6 |
| Seasonality | 4 | 4 |
| **TOTAL** | | **~750 features** |

---

## 🔧 IMPLEMENTACIÓN

La función `add_advanced_features()` se llama automáticamente desde `add_historical_features()` en el pipeline.

**Orden de ejecución:**
1. Lags (lag1, lag2)
2. Deltas (delta1, delta2)
3. Trends (tend6, min6, max6, avg6, ratioavg6, ratiomax6)
4. **Advanced Features** ← NUEVO!

---

## ⚠️ CONSIDERACIONES

1. **Overfitting:** Con ~2700 features totales, es importante:
   - Usar regularización en LightGBM
   - Monitorear validación vs training
   - Considerar feature selection post-entrenamiento

2. **Tiempo de cómputo:** Las nuevas features agregan ~10-20% de tiempo extra

3. **Memoria:** Polars maneja eficientemente, pero monitorear uso de RAM

---

## 🚀 PRÓXIMOS PASOS

1. Correr el pipeline completo
2. Revisar feature importances
3. Identificar cuáles de las nuevas features tienen mayor gain
4. Iterar: agregar variaciones de las más exitosas

---

## 💡 FEATURES DESTACADAS (Esperamos alto impacto)

1. ✅ `{col}_direccion_consistente` - OBLIGATORIA, captura tendencias sostenidas
2. ✅ `{col}_cv6` - Volatilidad es muy predictiva de churn
3. ✅ `ratio_comisiones_saldo` - Clientes que pagan mucho por poco
4. ✅ `utilizacion_credito_visa/master` - Uso de límite de crédito
5. ✅ `num_productos_activos` - Diversificación
6. ✅ `{col}_accel` - Cambios acelerados en comportamiento
7. ✅ `{col}_zscore` - Detección de anomalías

---

**Fecha de implementación:** 2025-11-12
**Autor:** Cursor AI Assistant
**Versión:** 1.0



