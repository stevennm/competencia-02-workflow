# Bucket Sync Implementation Summary

## ✅ Archivos Creados/Modificados

### 🆕 Nuevos Archivos

1. **`src/bucket_utils.py`** (NUEVO)
   - ✅ `get_bucket_path()` - Obtiene ruta del bucket
   - ✅ `copy_to_bucket()` - Copia archivo/directorio al bucket
   - ✅ `copy_experiment_to_bucket()` - Copia experimento completo
   - ✅ `sync_output_to_bucket()` - Sincroniza todos los outputs
   - ✅ `get_bucket_info()` - Info sobre configuración del bucket

2. **`docs/BUCKET_SYNC.md`** (NUEVO)
   - ✅ Documentación completa
   - ✅ Ejemplos de uso
   - ✅ Troubleshooting
   - ✅ Mejores prácticas

### 📝 Modificados

3. **`src/config.py`**
   - ✅ Agregada sección `bucket` con configuración
   - ✅ `enabled`: activar/desactivar
   - ✅ `base_path`: ruta del bucket (ej: `~/buckets/b1`)
   - ✅ `copy_on_create`: copiar inmediatamente (futuro)

4. **`run.py`**
   - ✅ Import de `bucket_utils`
   - ✅ Log de configuración del bucket al inicio
   - ✅ STEP 12: Sync to bucket al final
   - ✅ Copia automática de todos los outputs

5. **`run_zlgbm.py`**
   - ✅ Import de `bucket_utils`
   - ✅ Log de configuración del bucket al inicio
   - ✅ STEP 7: Sync to bucket al final
   - ✅ Copia automática de todos los outputs

## 🎯 Funcionalidad Implementada

### Configuración en `config.py`

```python
"bucket": {
    "enabled": True,  # Activar/desactivar
    "base_path": "~/buckets/b1",  # Ruta base del bucket
    "copy_on_create": True,  # Copiar inmediatamente
}
```

### Qué se copia automáticamente

1. **Directorio del experimento completo**
   - `output/{experimento}/` → `~/buckets/b1/exp/{experimento}/`
   - Incluye: modelos, predicciones, análisis, etc.

2. **Directorio Kaggle**
   - `output/kaggle/` → `~/buckets/b1/exp/kaggle/`
   - Incluye: submissions CSV

3. **Log de ejecución**
   - `logs/run_YYYYMMDD_HHMMSS.log` → `~/buckets/b1/logs/`
   - Solo el log más reciente

### Estructura del Bucket

```
~/buckets/b1/
├── exp/
│   ├── experimento1/
│   │   ├── zmodelo.txt
│   │   ├── tb_arboles.txt
│   │   ├── prediccion.txt
│   │   ├── modelitos/
│   │   │   ├── modelo_001.txt
│   │   │   ├── modelo_002.txt
│   │   │   └── ...
│   │   └── analysis/
│   │       ├── gain_curve.png
│   │       └── ensemble_gain.png
│   └── kaggle/
│       ├── KA_experimento1_11000.csv
│       └── KA_experimento2_11000.csv
└── logs/
    ├── run_20250115_103000.log
    └── run_zlgbm_20250115_120000.log
```

## 🚀 Cómo Usar

### 1. Configurar el Bucket

Editar `src/config.py`:

```python
"bucket": {
    "enabled": True,
    "base_path": "~/buckets/b1",  # Cambiar según necesites
}
```

### 2. Crear el Bucket (si no existe)

```bash
mkdir -p ~/buckets/b1
```

### 3. Ejecutar Workflow

```bash
# Workflow normal
uv run run.py

# Workflow zLightGBM
uv run run_zlgbm.py
```

### 4. Verificar Copiado

Al finalizar verás:

```
======================================================================
  STEP 12: SYNC TO BUCKET
======================================================================
✓ Copied experiment to bucket: output/experimento → ~/buckets/b1/exp/experimento
✓ Copied to bucket: kaggle/ → ~/buckets/b1/exp/kaggle/
✓ Copied to bucket: run_20250115_103000.log → ~/buckets/b1/logs/
======================================================================
BUCKET SYNC COMPLETE
======================================================================
```

## 📊 Logging

### Al inicio del workflow:

```
Bucket Configuration:
  Enabled: Yes
  Base path: /home/user/buckets/b1
  Exp path: /home/user/buckets/b1/exp
  Bucket exists: True
```

### Al final del workflow:

```
======================================================================
SYNCING TO BUCKET
======================================================================
✓ Copied experiment to bucket: output/experimento → /home/user/buckets/b1/exp/experimento
✓ Copied to bucket: kaggle/ → /home/user/buckets/b1/exp/kaggle/
✓ Copied to bucket: run_20250115_103000.log → /home/user/buckets/b1/logs/
======================================================================
BUCKET SYNC COMPLETE
======================================================================
```

## 💡 Casos de Uso

### Cambiar Bucket

```python
# En config.py
"bucket": {
    "enabled": True,
    "base_path": "~/buckets/b2",  # Usar otro bucket
}
```

### Desactivar Bucket

```python
# En config.py
"bucket": {
    "enabled": False,  # No copiar
    "base_path": "~/buckets/b1",
}
```

### Copiar Manualmente

```python
from src.config import PARAM
from src.bucket_utils import sync_output_to_bucket
import logging

logger = logging.getLogger(__name__)
sync_output_to_bucket(PARAM, logger)
```

## 🔧 API de Funciones

### `sync_output_to_bucket(config, logger=None)`

Sincroniza todos los outputs al bucket.

**Copia:**
- Directorio del experimento
- Directorio kaggle
- Último log

### `copy_to_bucket(source_path, config, bucket_subdir="", logger=None)`

Copia archivo o directorio específico.

**Ejemplo:**
```python
copy_to_bucket("output/experimento/zmodelo.txt", PARAM, "exp/experimento")
```

### `copy_experiment_to_bucket(experimento, config, logger=None)`

Copia directorio completo del experimento.

### `get_bucket_info(config)`

Obtiene información del bucket.

**Returns:**
```python
{
    "enabled": True,
    "base_path": "/home/user/buckets/b1",
    "exists": True,
    "exp_path": "/home/user/buckets/b1/exp",
    "logs_path": "/home/user/buckets/b1/logs"
}
```

## ⚠️ Consideraciones

### Espacio en Disco

- Modelos: ~100-500 MB por experimento
- Logs: ~1-10 MB por ejecución
- Submissions: ~1-5 MB

### Tiempo de Copiado

- Experimento pequeño: ~5-10 segundos
- Experimento grande (30 modelos): ~30-60 segundos

### Requisitos

1. **Bucket debe existir**: `mkdir -p ~/buckets/b1`
2. **Permisos de escritura**: `chmod 755 ~/buckets/b1`
3. **Espacio suficiente**: Verificar con `df -h ~/buckets/`

## 🔍 Troubleshooting

### Error: "Bucket path does not exist"

```bash
mkdir -p ~/buckets/b1
```

### Error: "Permission denied"

```bash
chmod 755 ~/buckets/b1
```

### Bucket sync no aparece

Verificar en `config.py`:
```python
"bucket": {
    "enabled": True,  # Debe ser True
}
```

## 📈 Beneficios

1. ✅ **Backup automático** - Todos los outputs respaldados
2. ✅ **Persistencia** - Archivos persisten aunque se borre workspace
3. ✅ **Compartir** - Fácil acceso desde otras VMs
4. ✅ **Organización** - Estructura clara y consistente
5. ✅ **Trazabilidad** - Logs también se copian
6. ✅ **Configuración flexible** - Activar/desactivar fácilmente
7. ✅ **Múltiples buckets** - Cambiar bucket según necesidad

## 🎯 Flujo Completo

```
1. Configurar bucket en config.py
   ↓
2. Crear bucket: mkdir -p ~/buckets/b1
   ↓
3. Ejecutar workflow: uv run run.py
   ↓
4. Workflow genera outputs en output/
   ↓
5. Al finalizar, automáticamente:
   - Copia output/{experimento}/ → ~/buckets/b1/exp/{experimento}/
   - Copia output/kaggle/ → ~/buckets/b1/exp/kaggle/
   - Copia logs/latest.log → ~/buckets/b1/logs/
   ↓
6. Verificar en bucket: ls -la ~/buckets/b1/exp/
```

## 📚 Documentación

- **Guía completa**: `docs/BUCKET_SYNC.md`
- **Módulo**: `src/bucket_utils.py`
- **Configuración**: `src/config.py`

## 🎉 Resumen

✅ **Implementación completa** de bucket sync
✅ **Automático** - Se ejecuta al final de cada workflow
✅ **Configurable** - Activar/desactivar, cambiar bucket
✅ **Logging completo** - Todo registrado en logs
✅ **Documentación** - Guía completa y ejemplos
✅ **Flexible** - API para uso manual también

---

**¡Bucket sync implementado y listo para usar! 💾**

