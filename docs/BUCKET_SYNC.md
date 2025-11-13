# Bucket Sync - Documentación

## 📋 Resumen

Los workflows automáticamente copian todos los outputs al bucket de la VM al finalizar la ejecución.

## ⚙️ Configuración

### En `src/config.py`:

```python
PARAM = {
    # ...
    
    # Bucket configuration (for VM storage)
    "bucket": {
        "enabled": True,  # Set to False to disable bucket copying
        "base_path": "~/buckets/b1",  # Base bucket path (change as needed)
        "copy_on_create": True,  # Copy files immediately after creation
    },
    
    # ...
}
```

### Parámetros:

- **`enabled`**: `True` para activar, `False` para desactivar
- **`base_path`**: Ruta base del bucket (siempre empieza con `~/buckets/`)
  - Ejemplos: `~/buckets/b1`, `~/buckets/b2`, `~/buckets/mi_bucket`
- **`copy_on_create`**: Si copiar archivos inmediatamente (futuro)

## 📁 Estructura de Copiado

### Origen (workspace local):
```
output/
├── {experimento}/
│   ├── zmodelo.txt
│   ├── tb_arboles.txt
│   ├── prediccion.txt
│   ├── modelitos/
│   └── analysis/
└── kaggle/
    └── KA{experimento}_11000.csv

logs/
└── run_YYYYMMDD_HHMMSS.log
```

### Destino (bucket):
```
~/buckets/b1/
├── exp/
│   ├── {experimento}/
│   │   ├── zmodelo.txt
│   │   ├── tb_arboles.txt
│   │   ├── prediccion.txt
│   │   ├── modelitos/
│   │   └── analysis/
│   └── kaggle/
│       └── KA{experimento}_11000.csv
└── logs/
    └── run_YYYYMMDD_HHMMSS.log
```

## 🚀 Uso

### Automático

Los workflows copian automáticamente al finalizar:

```bash
# Workflow normal
uv run run.py

# Workflow zLightGBM
uv run run_zlgbm.py
```

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

### Manual (desde Python)

```python
from src.config import PARAM
from src.bucket_utils import sync_output_to_bucket
import logging

logger = logging.getLogger(__name__)

# Sincronizar todo
sync_output_to_bucket(PARAM, logger)
```

### Copiar archivo específico

```python
from src.config import PARAM
from src.bucket_utils import copy_to_bucket

# Copiar un archivo
copy_to_bucket("output/experimento/zmodelo.txt", PARAM, "exp/experimento")

# Copiar un directorio
copy_to_bucket("output/experimento/modelitos", PARAM, "exp/experimento")
```

## 🔧 Funciones Disponibles

### `sync_output_to_bucket(config, logger=None)`

Sincroniza todos los outputs al bucket:
- Directorio del experimento completo
- Directorio kaggle
- Último archivo de log

### `copy_to_bucket(source_path, config, bucket_subdir="", logger=None)`

Copia un archivo o directorio específico al bucket.

**Args:**
- `source_path`: Ruta del archivo/directorio a copiar
- `config`: Diccionario de configuración (PARAM)
- `bucket_subdir`: Subdirectorio dentro del bucket (ej: "exp/experimento")
- `logger`: Logger opcional

**Returns:** `True` si se copió exitosamente

### `copy_experiment_to_bucket(experimento, config, logger=None)`

Copia el directorio completo de un experimento.

### `get_bucket_info(config)`

Obtiene información sobre la configuración del bucket.

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

## 💡 Casos de Uso

### 1. Desactivar bucket temporalmente

```python
# En config.py
"bucket": {
    "enabled": False,  # Desactivado
    "base_path": "~/buckets/b1",
}
```

### 2. Cambiar bucket

```python
# En config.py
"bucket": {
    "enabled": True,
    "base_path": "~/buckets/b2",  # Usar bucket b2
}
```

### 3. Copiar solo ciertos archivos

```python
from src.bucket_utils import copy_to_bucket
from src.config import PARAM

# Solo copiar modelo
copy_to_bucket("output/experimento/zmodelo.txt", PARAM, "exp/experimento")

# Solo copiar submission
copy_to_bucket("output/kaggle/KA_experimento_11000.csv", PARAM, "exp/kaggle")
```

### 4. Verificar bucket antes de ejecutar

```python
from src.bucket_utils import get_bucket_info
from src.config import PARAM

info = get_bucket_info(PARAM)
print(f"Bucket enabled: {info['enabled']}")
print(f"Bucket path: {info['base_path']}")
print(f"Bucket exists: {info['exists']}")
```

## 📊 Logging

El bucket sync se registra en el log:

```
2025-01-15 10:30:00 - INFO - Bucket Configuration:
2025-01-15 10:30:00 - INFO -   Enabled: Yes
2025-01-15 10:30:00 - INFO -   Base path: /home/user/buckets/b1
2025-01-15 10:30:00 - INFO -   Exp path: /home/user/buckets/b1/exp
2025-01-15 10:30:00 - INFO -   Bucket exists: True

...

2025-01-15 12:45:00 - INFO - ======================================================================
2025-01-15 12:45:00 - INFO - SYNCING TO BUCKET
2025-01-15 12:45:00 - INFO - ======================================================================
2025-01-15 12:45:00 - INFO - ✓ Copied experiment to bucket: output/experimento → /home/user/buckets/b1/exp/experimento
2025-01-15 12:45:00 - INFO - ✓ Copied to bucket: kaggle/ → /home/user/buckets/b1/exp/kaggle/
2025-01-15 12:45:00 - INFO - ✓ Copied to bucket: run_20250115_103000.log → /home/user/buckets/b1/logs/
2025-01-15 12:45:00 - INFO - ======================================================================
2025-01-15 12:45:00 - INFO - BUCKET SYNC COMPLETE
2025-01-15 12:45:00 - INFO - ======================================================================
```

## ⚠️ Consideraciones

### 1. Espacio en Disco

Los archivos se copian, no se mueven. Asegúrate de tener suficiente espacio:
- Modelos: ~100-500 MB por experimento
- Logs: ~1-10 MB por ejecución
- Submissions: ~1-5 MB

### 2. Tiempo de Copiado

El copiado es rápido pero agrega tiempo al final:
- Experimento pequeño: ~5-10 segundos
- Experimento grande (30 modelos): ~30-60 segundos

### 3. Bucket debe existir

El directorio base del bucket debe existir:
```bash
# Crear bucket si no existe
mkdir -p ~/buckets/b1
```

### 4. Permisos

Asegúrate de tener permisos de escritura en el bucket:
```bash
# Verificar permisos
ls -la ~/buckets/
```

## 🔍 Troubleshooting

### Error: "Bucket path does not exist"

**Causa:** El directorio del bucket no existe.

**Solución:**
```bash
mkdir -p ~/buckets/b1
```

### Error: "Permission denied"

**Causa:** No tienes permisos de escritura.

**Solución:**
```bash
chmod 755 ~/buckets/b1
```

### Warning: "Source does not exist, skipping bucket copy"

**Causa:** El archivo/directorio origen no existe.

**Solución:** Normal si el archivo no se generó (ej: sin BO no hay BO_log.txt)

### Bucket sync no aparece

**Causa:** Bucket desactivado en config.

**Solución:**
```python
# En config.py
"bucket": {
    "enabled": True,  # Cambiar a True
    # ...
}
```

## 📈 Beneficios

1. **Backup automático**: Todos los outputs se respaldan
2. **Persistencia**: Los archivos persisten aunque se borre el workspace
3. **Compartir**: Fácil acceso desde otras VMs o notebooks
4. **Organización**: Estructura clara en el bucket
5. **Trazabilidad**: Logs también se copian

## 🎯 Mejores Prácticas

1. **Siempre activar en VM**: En producción, siempre tener bucket enabled
2. **Verificar espacio**: Monitorear espacio en bucket regularmente
3. **Limpiar antiguos**: Borrar experimentos viejos del bucket
4. **Usar buckets separados**: Un bucket por tipo de experimento
5. **Revisar logs**: Verificar que el sync fue exitoso

## 📚 Referencias

- Módulo: `src/bucket_utils.py`
- Configuración: `src/config.py`
- Workflows: `run.py`, `run_zlgbm.py`

---

**¡Bucket sync implementado! 💾**

