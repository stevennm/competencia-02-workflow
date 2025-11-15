# Deprecated Files

This folder contains old/duplicated code that has been replaced by the new `preprocessing/` package.

## Why These Files Are Here

These files are kept for reference but are **no longer used** in the active codebase.

## Files

### From `src/`:

- **`preprocessing.py`** 
  - Replaced by: `preprocessing/preprocessing_utils.py`
  - Reason: Now part of self-contained preprocessing package

- **`feature_engineering.py`**
  - Replaced by: `preprocessing/feature_engineering_utils.py`
  - Reason: Now part of self-contained preprocessing package

- **`rf_features.py`**
  - Replaced by: Integrated into `preprocessing/feature_engineering_utils.py`
  - Reason: Consolidated with feature engineering

- **`config_old.py`**
  - Replaced by: Configuration embedded in `preprocessing/03_feature_engineering.py`
  - Reason: Config is now embedded in scripts, not separate

### From project root:

- **`run_preprocessing.py`**
  - Replaced by: `preprocessing/02_preprocessing.py`
  - Reason: New organized pipeline in preprocessing/

- **`run_feature_engineering.py`**
  - Replaced by: `preprocessing/03_feature_engineering.py`
  - Reason: New organized pipeline in preprocessing/

## Current Structure

The active codebase now has:

```
preprocessing/          # All preprocessing code (self-contained)
  ├── 01_generate_clase_ternaria.py
  ├── 02_preprocessing.py
  ├── 03_feature_engineering.py
  ├── preprocessing_utils.py
  └── feature_engineering_utils.py

src/                    # Only experiment/modeling code
  ├── config.py         # Experiment configuration
  ├── training_zlgbm.py
  ├── scoring_zlgbm.py
  └── bucket_utils.py
```

## Can I Delete This Folder?

Yes, after you've verified that the new `preprocessing/` pipeline works correctly.

**Recommendation**: Keep it for a few weeks, then delete once you're confident the new pipeline is stable.

