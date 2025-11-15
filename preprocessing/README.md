# Preprocessing Pipeline

This folder contains **all** preprocessing and feature engineering code in one place.
All scripts are self-contained and can run independently.

## 📁 Files

### Main Scripts (run these):

1. **`01_generate_clase_ternaria.py`**
   - Reads: `data/competencia_02_crudo.csv.gz`
   - Creates: `data/competencia_02_target.parquet`
   - Generates the `clase_ternaria` column (CONTINUA, BAJA+1, BAJA+2)
   - **Uses `infer_schema_length=None`** for robust type inference (analyzes entire file)
   - Logic verified against workflow-jueves notebook ✓

2. **`02_preprocessing.py`**
   - Reads: `data/competencia_02_target.parquet`
   - Creates: `data/preprocessed_data.parquet`
   - Steps:
     - Eliminates problematic features (cprestamos_personales, mprestamos_personales)
     - Data quality fixes (MICE imputation for 202006)
     - Data drifting correction (IPC adjustment)

3. **`03_feature_engineering.py`**
   - Reads: `data/preprocessed_data.parquet`
   - Creates: `data/featured_data.parquet`
   - Steps:
     - Intra-month features (3 features)
     - Historical features (lags, deltas)
     - Trend features (trends, min, max, avg, ratios)
   - Configuration embedded from workflow-jueves notebook

4. **`run_all.py`**
   - Master script that runs all 3 steps in sequence
   - Stops if any step fails
   - Provides progress and timing information

### Utility Modules (imported by scripts):

5. **`preprocessing_utils.py`**
   - Core preprocessing functions
   - Copied from `src/preprocessing.py`
   - Contains: MICE, IPC correction, clase_ternaria generation

6. **`feature_engineering_utils.py`**
   - Core feature engineering functions
   - Copied from `src/feature_engineering.py`
   - Contains: lags, deltas, trends, ratios

7. **`__init__.py`**
   - Makes preprocessing a Python package
   - Allows importing functions: `from preprocessing import data_quality_fixes`

### Documentation:

8. **`README.md`** - This file
9. **`show_structure.py`** - Displays pipeline structure visually

## 🚀 Usage

### Run complete pipeline:
```bash
python preprocessing/run_all.py
```

### Run individual steps:
```bash
# Step 1: Generate clase_ternaria
python preprocessing/01_generate_clase_ternaria.py

# Step 2: Preprocessing
python preprocessing/02_preprocessing.py

# Step 3: Feature Engineering
python preprocessing/03_feature_engineering.py
```

### From other scripts (as a package):
```python
from preprocessing import (
    data_quality_fixes,
    add_historical_features
)
```

## ⚙️ Configuration

Feature engineering configuration is embedded in `03_feature_engineering.py`:

```python
FE_CONFIG = {
    "FE_hist": {
        "Tendencias": {
            "run": True,
            "ventana": 6,              # 6-month window
            "tendencia": True,          # Linear trend (slope)
            "minimo": True,             # Rolling minimum
            "maximo": True,             # Rolling maximum
            "promedio": True,           # Rolling average
            "ratioavg": True,           # Ratio to average
            "ratiomax": True            # Ratio to maximum
        }
    }
}
```

This configuration comes from `config_old.py` and follows the workflow-jueves notebook.

## 📊 Output Files

After running the complete pipeline:

| File | Description | Columns |
|------|-------------|---------|
| `data/competencia_02_target.parquet` | Raw data + clase_ternaria | 153 |
| `data/preprocessed_data.parquet` | Cleaned data (no prestamos_personales) | 151 |
| `data/featured_data.parquet` | With all engineered features | ~2000 |

## 🔧 Key Features

- **Self-contained**: All preprocessing code in one folder
- **Independent scripts**: Can run each step separately for debugging
- **No external dependencies**: Uses local modules (preprocessing_utils, feature_engineering_utils)
- **Embedded configuration**: No dependency on `src/config.py` (which is only for experiments)
- **Simple and clean**: Each script does one thing well
- **Based on best practices**: Following workflow-jueves notebook recommendations
- **Removes problematic features**: Eliminates cprestamos_personales and mprestamos_personales (data drifting)
- **Verified logic**: All functions tested against original implementations ✓

## 📝 Notes

- Original code from `src/preprocessing.py` and `src/feature_engineering.py` copied to local modules
- Configuration is embedded to keep `src/config.py` focused on experiment parameters
- Feature elimination follows the "salsa mágica" from the notebook (removes data drifting columns)
- All scripts can be run from project root: `python preprocessing/script_name.py`

### ⚠️ RAM Considerations:

**`01_generate_clase_ternaria.py`**:
- Uses `infer_schema_length=None` for maximum robustness
- This analyzes the **entire CSV** to infer correct data types
- Prevents column loss due to type inference issues (like cprestamos_personales)
- Recommended RAM: 8GB+ for large datasets
- If you have memory constraints, you can change to `infer_schema_length=50000` in the script

## 🔍 Verification

Logic verified against:
- `src/preprocessing.py` - generate_clase_ternaria() ✓
- `workflow-jueves.ipynb` - clase_ternaria calculation ✓
- `config_old.py` - FE configuration ✓

