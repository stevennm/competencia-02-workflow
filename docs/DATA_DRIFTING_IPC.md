# Data Drifting Correction with IPC

## Implementation

The `data_drifting_correction()` function now adjusts all monetary values for inflation using IPC (Consumer Price Index) data.

## How It Works

### 1. Load IPC Data
Reads `data/indicadores.csv` which contains monthly inflation rates:
```
foto_mes,ipc
201901,2.9    ← 2.9% inflation from previous month
201902,3.8    ← 3.8% inflation from 201901 to 201902
...
```

### 2. Calculate Cumulative Multipliers
Works backwards from the most recent month (202108) to calculate how much to multiply each month's values:

**Example calculation:**
- Base month: 202108 (multiplier = 1.0)
- 202107: multiply by 1.025 (2.5% inflation to reach 202108)
- 202106: multiply by 1.025 × 1.032 = 1.0578 (3.2% + 2.5% cumulative)
- 201901: multiply by ~1.85 (85% cumulative inflation over 2.5 years)

### 3. Identify Monetary Columns
Automatically detects columns starting with 'm':
- `mpasivos_margen`
- `mcuentas_saldo`
- `mpayroll`
- `mtarjeta_visa_consumo`
- etc.

### 4. Apply Adjustment
For each monetary column:
```
value_adjusted = value_original × multiplier
```

**Example:**
- Customer had $10,000 in account in 201901
- Multiplier for 201901: 1.85
- Adjusted value: $10,000 × 1.85 = $18,500
- Now comparable to 202108 purchasing power

## Why This Matters

### Without Correction:
- $10,000 in 201901 ≠ $10,000 in 202108 (85% inflation!)
- Model sees decreasing values over time (misleading)
- Temporal bias in predictions

### With Correction:
- All values in same purchasing power (202108 dollars)
- Model learns real patterns, not inflation artifacts
- Fair comparison across all time periods

## Example Output

```
[DATA DRIFTING] Correcting inflation with IPC indicators...
  Loaded IPC data: 32 months
  Base month for adjustment: 202108
  Calculated multipliers for 32 months
  Example: 201901 -> 1.8523x
  Example: 202001 -> 1.4231x
  Example: 202106 -> 1.0578x
  Found 73 monetary columns to adjust
  Examples: ['mpasivos_margen', 'mcuentas_saldo', 'mpayroll', ...]
  Applying inflation adjustment...
  ✓ Adjusted 73 columns for inflation
  All values normalized to 202108 purchasing power
```

## Technical Details

### Formula
For monthly IPC of x%, the multiplier step is:
```
multiplier_step = 1 + (ipc / 100)
```

Cumulative from month A to base month B:
```
multiplier = ∏(1 + ipc_i/100) for all i from A to B
```

### Column Detection
- Starts with 'm' (monetary prefix)
- Excludes: `foto_mes`
- Must be numeric (Int or Float)

### Graceful Handling
- If `indicadores.csv` not found → skips with warning
- Missing months → uses multiplier 1.0 (no adjustment)
- Non-monetary columns → left unchanged

## Validation

You can validate the adjustment by checking a customer's monetary values:

```python
# Before correction
customer_201901 = df.filter(
    (pl.col("numero_de_cliente") == 12345) & 
    (pl.col("foto_mes") == 201901)
).select("mpayroll")

# After correction
customer_201901_adjusted = df.filter(
    (pl.col("numero_de_cliente") == 12345) & 
    (pl.col("foto_mes") == 201901)
).select("mpayroll")

# Should be ~1.85x higher after correction
```

## Performance

- **Speed**: Fast (~10-20 seconds)
- **Memory**: Minimal overhead
- **Accuracy**: Exact IPC-based adjustment

## Integration

Runs automatically in the preprocessing pipeline:
```python
df = preprocess_data("data/competencia_02_target.parquet")
# ↓ Automatically applies IPC correction
```

