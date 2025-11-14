# VM Setup Instructions

## 🚀 Quick Setup for VM

### 1. Install zLightGBM (First Time Only)

```bash
cd
rm -rf LightGBM
git clone --recursive https://github.com/dmecoyfin/LightGBM
source ~/.venv/bin/activate
pip uninstall --yes lightgbm

# Install Python package
cd ~/LightGBM
sh ./build-python.sh install
```

### 2. Clone Project Repository

```bash
cd ~
git clone <your-repo-url> workflow-jueves-python
cd workflow-jueves-python
```

### 3. Install Python Dependencies

```bash
# Activate the global venv (where zLightGBM is installed)
source ~/.venv/bin/activate

# Install all dependencies EXCEPT lightgbm (already have zLightGBM)
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Check zLightGBM is installed
python -c "import lightgbm; print('LightGBM version:', lightgbm.__version__)"

# Check other packages
python -c "import polars, numpy, optuna; print('✓ All packages OK')"
```

### 5. Run Workflows

```bash
# Make sure you're in the project directory
cd ~/workflow-jueves-python

# Activate venv
source ~/.venv/bin/activate

# Run zLightGBM workflow
python run_zlgbm.py

# Or run normal workflow
python run.py
```

## 📋 Dependencies Installed

From `requirements.txt`:
- polars>=0.19.0
- numpy>=1.24.0
- optuna>=3.0.0
- tqdm>=4.65.0
- pyarrow>=22.0.0
- duckdb>=0.9.0
- matplotlib>=3.10.7
- altair>=5.5.0
- scikit-learn>=1.3.0
- optuna-dashboard>=0.19.0
- matplotlib-venn>=1.1.2

**Note:** `lightgbm` is NOT in requirements.txt because we use zLightGBM installed separately.

## 🔄 Updating the Project

When you pull new changes:

```bash
cd ~/workflow-jueves-python
git pull

# If requirements.txt changed, reinstall
source ~/.venv/bin/activate
pip install -r requirements.txt
```

## ⚠️ Important Notes

1. **Always activate the venv**: `source ~/.venv/bin/activate`
2. **Don't reinstall lightgbm**: It will overwrite zLightGBM
3. **If zLightGBM breaks**: Re-run the installation steps from section 1

## 🆘 Troubleshooting

### Error: "No module named 'lightgbm'"

```bash
# Reinstall zLightGBM
cd ~/LightGBM
source ~/.venv/bin/activate
sh ./build-python.sh install
```

### Error: "No module named 'polars'" (or other package)

```bash
source ~/.venv/bin/activate
pip install -r requirements.txt
```

### Check what's installed

```bash
source ~/.venv/bin/activate
pip list
```

---

**Ready to go! 🎉**

