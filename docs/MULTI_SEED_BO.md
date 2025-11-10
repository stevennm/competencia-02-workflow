# Multi-Seed Bayesian Optimization

Optional feature for more robust hyperparameter optimization using ensemble of models with different seeds.

## 📖 Overview

By default, each Bayesian Optimization trial trains **one model** with one seed. This is fast but can be noisy.

With multi-seed support, each trial can train **multiple models** with different seeds and average the results. This is slower but more robust.

## 🎛️ Configuration

Edit `src/config.py`:

```python
"hipeparametertuning": {
    "BO_iteraciones": 50,
    "ksemillerio": 1,  # Number of seeds per trial
    "repe": 1          # Number of repetitions
}
```

### Parameters

**`ksemillerio`** - Number of models with different seeds to ensemble
- `1` = Single model (default, fast)
- `3-5` = Small ensemble (moderate speed, good robustness)
- `10+` = Large ensemble (slow, very robust)

**`repe`** - Number of times to repeat the measurement and average
- `1` = Single measurement (default, fast)
- `3` = Triple measurement (slower, reduces noise)
- `5+` = Multiple measurements (very slow, very stable)

**Total models per trial** = `ksemillerio × repe`

## 📊 Examples

### Default (Fast Mode)
```python
"ksemillerio": 1,
"repe": 1
```
- 1 model per trial
- Fast optimization
- Good for initial exploration

### Moderate Robustness
```python
"ksemillerio": 3,
"repe": 1
```
- 3 models per trial (3x slower)
- Averages predictions from 3 seeds
- Reduces seed variance

### High Robustness
```python
"ksemillerio": 5,
"repe": 3
```
- 15 models per trial (15x slower!)
- 5-seed ensemble, measured 3 times
- Very stable results
- Use for final optimization

## ⏱️ Time Impact

Example with 50 BO iterations:

| Config | Models/Trial | Total Models | Relative Time |
|--------|-------------|--------------|---------------|
| `1, 1` | 1 | 50 | 1x (baseline) |
| `3, 1` | 3 | 150 | 3x |
| `5, 1` | 5 | 250 | 5x |
| `3, 3` | 9 | 450 | 9x |
| `5, 3` | 15 | 750 | 15x |

## 🎯 When to Use

### Use `ksemillerio=1, repe=1` (Default):
- ✅ Initial experiments
- ✅ Fast iteration
- ✅ Exploring hyperparameter space
- ✅ Limited compute time

### Use `ksemillerio=3-5, repe=1`:
- ✅ Final optimization
- ✅ When you have more time
- ✅ Want more robust parameters
- ✅ Reducing seed sensitivity

### Use `ksemillerio=5, repe=3`:
- ✅ Production-critical optimization
- ✅ Maximum robustness needed
- ✅ Plenty of compute time
- ✅ Final competition submission

## 💡 Tips

1. **Start simple**: Use `1,1` for initial runs
2. **Increase gradually**: Try `3,1` if results are noisy
3. **Final polish**: Use `5,3` for final optimization
4. **Monitor time**: Check if improvement justifies the time cost
5. **Compare results**: Run with different configs and compare

## 🔍 How It Works

### Single Seed (Default)
```
Trial 1: Train model with seed=102191 → Gain = 325M
Trial 2: Train model with seed=102191 → Gain = 310M
...
```

### With ksemillerio=3
```
Trial 1:
  - Train with seed=123456 → Pred1
  - Train with seed=234567 → Pred2
  - Train with seed=345678 → Pred3
  - Average predictions → Gain = 327M
  
Trial 2:
  - Train with seed=456789 → Pred1
  - Train with seed=567890 → Pred2
  - Train with seed=678901 → Pred3
  - Average predictions → Gain = 315M
...
```

### With ksemillerio=3, repe=3
```
Trial 1:
  Repetition 1: 3-seed ensemble → Gain1 = 327M
  Repetition 2: 3-seed ensemble → Gain2 = 325M
  Repetition 3: 3-seed ensemble → Gain3 = 326M
  Average gains → Final Gain = 326M
  
Trial 2:
  Repetition 1: 3-seed ensemble → Gain1 = 315M
  Repetition 2: 3-seed ensemble → Gain2 = 314M
  Repetition 3: 3-seed ensemble → Gain3 = 316M
  Average gains → Final Gain = 315M
...
```

## 📈 Expected Benefits

- **Reduced variance**: Less sensitive to random seed
- **More stable**: Consistent results across runs
- **Better generalization**: Finds parameters that work across seeds
- **Confidence**: Know parameters are robust, not lucky

## ⚠️ Trade-offs

- **Time**: Linearly increases with `ksemillerio × repe`
- **Compute**: Requires more resources
- **Diminishing returns**: Beyond 5-10 seeds, improvement is small
- **Complexity**: More parameters to tune

## 🎓 Professor's Approach

The professor implemented this feature but uses `ksemillerio=1, repe=1` in the notebook because:
- Bayesian Optimization is already expensive
- Best hyperparameters are relatively stable across seeds
- Multi-seed ensemble is used in **final training** (30 seeds) instead

This is the recommended approach:
1. **BO with single seed** (fast) → Find best hyperparameters
2. **Final training with 30 seeds** (robust) → Stable predictions

## 🚀 Quick Start

**Default (no changes needed):**
```python
"ksemillerio": 1,
"repe": 1
```

**Try moderate robustness:**
```python
"ksemillerio": 3,
"repe": 1
```

**Maximum robustness (slow!):**
```python
"ksemillerio": 5,
"repe": 3
```

Then run:
```bash
python run.py
```

The script will automatically detect and use the multi-seed configuration!

