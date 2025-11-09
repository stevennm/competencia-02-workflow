# Changes Summary - Standalone Scripts

## What Was Created

Three new files have been added to enable running the workflow in parts:

### 1. `run_preprocessing.py`
**Purpose:** Run only the preprocessing steps

**Steps included:**
- Load data
- Generate clase_ternaria
- Eliminate features
- Data quality fixes (MICE imputation for 202006)
- Data drifting correction (IPC inflation adjustment)

**Usage:**
```bash
python run_preprocessing.py
```

**Output:** `data/preprocessed_data.parquet`

---

### 2. `run_feature_engineering.py`
**Purpose:** Run only the feature engineering steps

**Steps included:**
- Load preprocessed data
- Add intra-month features
- Add historical features (lags, deltas, trends)
- Add RF features (optional, commented out)

**Usage:**
```bash
python run_feature_engineering.py
```

**Input:** `data/preprocessed_data.parquet`  
**Output:** `data/featured_data.parquet`

---

### 3. `STANDALONE_SCRIPTS.md`
**Purpose:** Complete documentation for the standalone scripts

**Contents:**
- Overview and rationale
- Detailed usage instructions
- Workflow comparisons
- Example scenarios
- Configuration tips
- Troubleshooting guide

---

## Benefits

✅ **Modularity**: Run preprocessing and feature engineering independently  
✅ **Faster iteration**: Skip expensive preprocessing when testing features  
✅ **Debugging**: Isolate and test specific pipeline stages  
✅ **Checkpoints**: Save intermediate results as parquet files  
✅ **Flexibility**: Mix different preprocessing/feature combinations  

---

## Quick Start

### Run Everything (Original Workflow)
```bash
python run.py
```

### Run By Parts (New Workflow)
```bash
# Step 1: Preprocessing
python run_preprocessing.py

# Step 2: Feature Engineering
python run_feature_engineering.py

# Step 3: Use in run.py (uncomment lines 80-81 to load featured_data.parquet)
python run.py
```

---

## File Structure

```
VERSION_WORKFLOW_JUEVES/
├── run.py                          # Complete end-to-end workflow
├── run_preprocessing.py            # NEW: Preprocessing only
├── run_feature_engineering.py      # NEW: Feature engineering only
├── STANDALONE_SCRIPTS.md           # NEW: Documentation
├── CHANGES_SUMMARY.md              # NEW: This file
├── data/
│   ├── competencia_02_target.parquet      # Original input
│   ├── preprocessed_data.parquet          # NEW: After preprocessing
│   └── featured_data.parquet              # NEW: After feature engineering
└── src/
    ├── preprocessing.py            # Preprocessing functions
    ├── feature_engineering.py      # Feature engineering functions
    └── ...
```

---

## What Changed in run.py

✅ **Added commented lines** showing how to load intermediate datasets:
- Lines 55-56: Load `preprocessed_data.parquet` (skip preprocessing)
- Lines 80-81: Load `featured_data.parquet` (skip preprocessing + FE)

✅ **Simple and elegant**: Just uncomment the lines you need
✅ **No breaking changes**: Default behavior unchanged

## What Wasn't Changed

✅ No changes to existing functions in `src/`  
✅ No changes to configuration or dependencies  
✅ Backward compatible - all existing code still works  

---

## Next Steps

1. **Test the preprocessing script:**
   ```bash
   python run_preprocessing.py
   ```

2. **Test the feature engineering script:**
   ```bash
   python run_feature_engineering.py
   ```

3. **Verify outputs:**
   ```python
   import polars as pl
   
   # Check preprocessed data
   df_prep = pl.read_parquet('data/preprocessed_data.parquet')
   print(f"Preprocessed: {df_prep.shape}")
   
   # Check featured data
   df_feat = pl.read_parquet('data/featured_data.parquet')
   print(f"Featured: {df_feat.shape}")
   ```

4. **Read the documentation:**
   See `STANDALONE_SCRIPTS.md` for detailed usage examples

---

## Notes

- Both scripts use the same configuration from `src/config.py`
- The scripts include detailed progress printing
- Error handling is included with full stack traces
- Execution time is reported at the end
- All intermediate data is saved in parquet format for efficiency

---

## Questions?

For more details, see:
- `STANDALONE_SCRIPTS.md` - Complete usage guide
- `docs/README.md` - Overall project documentation
- `docs/QUICK_START.md` - Quick start guide

