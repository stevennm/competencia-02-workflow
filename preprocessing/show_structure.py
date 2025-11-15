"""
Muestra la estructura del pipeline de preprocessing
"""

print("""
================================================================================
                    PREPROCESSING PIPELINE STRUCTURE
================================================================================

preprocessing/                                  [SELF-CONTAINED PACKAGE]
|
+-- MAIN SCRIPTS (run these):
|   |
|   +-- 01_generate_clase_ternaria.py         Step 1: Generate clase_ternaria
|   |   +- Input:  data/competencia_02_crudo.csv.gz (154 cols)
|   |   +- Output: data/competencia_02_target.parquet (153 cols + clase_ternaria)
|   |
|   +-- 02_preprocessing.py                   Step 2: Clean & preprocess
|   |   +- Input:  data/competencia_02_target.parquet
|   |   +- Eliminates: cprestamos_personales, mprestamos_personales
|   |   +- Applies: MICE imputation, IPC correction
|   |   +- Output: data/preprocessed_data.parquet (151 cols)
|   |
|   +-- 03_feature_engineering.py             Step 3: Feature engineering
|   |   +- Input:  data/preprocessed_data.parquet
|   |   +- Adds: Intra-month (3), Lags (306), Deltas (306)
|   |   +- Adds: Trends, MaxMin, Ratios (~930)
|   |   +- Output: data/featured_data.parquet (~2000 cols)
|   |
|   +-- run_all.py                            Master script (runs all 3)
|
+-- UTILITY MODULES (imported by scripts):
|   |
|   +-- preprocessing_utils.py               [FROM src/preprocessing.py]
|   |   +- generate_clase_ternaria()
|   |   +- data_quality_fixes() (MICE)
|   |   +- data_drifting_correction() (IPC)
|   |   +- add_canaritos()
|   |
|   +-- feature_engineering_utils.py         [FROM src/feature_engineering.py]
|   |   +- add_intra_month_features()
|   |   +- add_historical_features()
|   |   +- add_lag_features(), add_delta_features()
|   |   +- calculate_trend_features_polars()
|   |
|   +-- __init__.py                           Makes it a Python package
|
+-- DOCUMENTATION:
    |
    +-- README.md                             Complete documentation
    +-- show_structure.py                     This file

================================================================================

USAGE:

  Run complete pipeline:
    python preprocessing/run_all.py

  Run individual steps:
    python preprocessing/01_generate_clase_ternaria.py
    python preprocessing/02_preprocessing.py
    python preprocessing/03_feature_engineering.py

  Import as package (from other scripts):
    from preprocessing import data_quality_fixes, add_historical_features

================================================================================

KEY FEATURES:

  [OK] Self-contained - All preprocessing code in one folder
  [OK] Independent scripts - Can run separately for debugging
  [OK] Local modules - No dependency on src/ (uses local copies)
  [OK] Embedded configuration - From config_old.py / workflow-jueves
  [OK] Eliminates data drifting features - cprestamos_personales
  [OK] Clean and simple code - Each script does one thing
  [OK] Verified logic - Tested against original implementations

================================================================================

VERIFICATION:

  Logic verified against:
    - src/preprocessing.py (generate_clase_ternaria) [OK]
    - workflow-jueves.ipynb (clase_ternaria calc)    [OK]
    - config_old.py (FE configuration)               [OK]

================================================================================
""")

