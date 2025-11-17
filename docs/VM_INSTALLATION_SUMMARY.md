# VM Installation Summary

## ✅ Cambios Realizados

### 1. **`requirements.txt`** (NUEVO)
- ✅ Creado con todas las dependencias del proyecto
- ✅ **SIN lightgbm** (usamos zLightGBM instalado por separado)
- ✅ Listo para `pip install -r requirements.txt`

### 2. **`pyproject.toml`** (MODIFICADO)
- ✅ Comentado `lightgbm>=4.0.0`
- ✅ Agregado comentario explicativo
- ✅ Mantiene todas las demás dependencias

### 3. **`VM_SETUP.md`** (NUEVO)
- ✅ Instrucciones completas para setup en VM
- ✅ Paso a paso para instalar zLightGBM
- ✅ Cómo instalar dependencias del proyecto
- ✅ Troubleshooting común

## 🚀 Cómo Usar en la VM

### Paso 1: Pull los cambios

```bash
cd ~/workflow-jueves-python
git pull
```

### Paso 2: Instalar dependencias

```bash
# Activar entorno global (donde está zLightGBM)
source ~/.venv/bin/activate

# Instalar todas las dependencias
pip install -r requirements.txt
```

### Paso 3: Verificar

```bash
# Verificar zLightGBM
python -c "import lightgbm; print('zLightGBM OK:', lightgbm.__version__)"

# Verificar otras dependencias
python -c "import polars, numpy, optuna; print('✓ Dependencias OK')"
```

### Paso 4: Ejecutar

```bash
# Asegurarse de estar en el directorio correcto
cd ~/workflow-jueves-python

# Activar venv
source ~/.venv/bin/activate

# Ejecutar workflow
python run_zlgbm.py
```

## 📦 Dependencias en requirements.txt

```
polars>=0.19.0
numpy>=1.24.0
optuna>=3.0.0
tqdm>=4.65.0
pyarrow>=22.0.0
duckdb>=0.9.0
matplotlib>=3.10.7
altair>=5.5.0
scikit-learn>=1.3.0
optuna-dashboard>=0.19.0
matplotlib-venn>=1.1.2
```

**Nota:** lightgbm NO está incluido porque usamos zLightGBM

## 🔄 Workflow Completo en VM

```bash
# 1. Primera vez: Instalar zLightGBM (solo una vez)
cd
rm -rf LightGBM
git clone --recursive https://github.com/dmecoyfin/LightGBM
source ~/.venv/bin/activate
pip uninstall --yes lightgbm
cd ~/LightGBM
sh ./build-python.sh install

# 2. Clonar proyecto (solo una vez)
cd ~
git clone <tu-repo-url> workflow-jueves-python
cd workflow-jueves-python

# 3. Instalar dependencias (solo una vez, o cuando cambien)
source ~/.venv/bin/activate
pip install -r requirements.txt

# 4. Ejecutar (cada vez que quieras correr)
cd ~/workflow-jueves-python
source ~/.venv/bin/activate
python run_zlgbm.py
```

## ⚠️ Importante

1. **Siempre activar el venv**: `source ~/.venv/bin/activate`
2. **NO reinstalar lightgbm**: Sobrescribiría zLightGBM
3. **Si actualizas el repo**: Solo hacer `git pull` y `pip install -r requirements.txt` si cambió

## 🎯 Ventajas de este Setup

- ✅ **Separación clara**: zLightGBM separado de otras dependencias
- ✅ **Reproducible**: `requirements.txt` garantiza mismas versiones
- ✅ **Fácil actualización**: `git pull` + `pip install -r requirements.txt`
- ✅ **Documentado**: `VM_SETUP.md` con instrucciones completas
- ✅ **Sin conflictos**: No sobrescribe zLightGBM

## 📚 Archivos de Referencia

- **`requirements.txt`**: Lista de dependencias para pip
- **`VM_SETUP.md`**: Guía completa de setup en VM
- **`pyproject.toml`**: Config del proyecto (sin lightgbm)

---

**¡Todo listo para usar en la VM! 🎉**

## 🔍 Verificación Rápida

Después de instalar, verifica que todo funciona:

```bash
source ~/.venv/bin/activate

# Test rápido
python << 'EOF'
import lightgbm
import polars
import numpy
import optuna
import matplotlib
import sklearn

print("✓ zLightGBM:", lightgbm.__version__)
print("✓ Polars:", polars.__version__)
print("✓ NumPy:", numpy.__version__)
print("✓ Optuna:", optuna.__version__)
print("\n🎉 ¡Todo instalado correctamente!")
EOF
```

Si ves todos los ✓, estás listo para ejecutar los workflows!

