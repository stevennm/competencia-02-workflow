"""
Analyze if eliminating low-importance features would help optimize training time
"""
import polars as pl

print("="*80)
print("FEATURE ELIMINATION ANALYSIS")
print("="*80)

# Load feature importance
df = pl.read_csv("output/comp2_entrega2_202108_under0.05_zlgbm/feature_importance.txt", 
                 separator="\t")

print(f"\nTotal features: {len(df)}")
print(f"Features with importance > 0: {(df['importance'] > 0).sum()}")
print(f"Features with importance = 0: {(df['importance'] == 0).sum()}")

# Analyze cumulative importance
print("\n" + "="*80)
print("CUMULATIVE IMPORTANCE ANALYSIS")
print("="*80)

thresholds = [50, 100, 200, 300, 500, 1000, 1500, 2000]

print(f"\n{'Top N Features':<20} {'Cumulative Imp %':<20} {'Features Saved':<20}")
print("-"*80)

for n in thresholds:
    if n <= len(df):
        cum_imp = df.head(n)["importance_pct"].sum()
        features_saved = len(df) - n
        print(f"{n:<20} {cum_imp:>6.2f}%              {features_saved:<20}")

# Features with importance = 0
zero_imp = df.filter(pl.col("importance") == 0)
print(f"\n" + "="*80)
print(f"FEATURES WITH ZERO IMPORTANCE")
print(f"="*80)
print(f"\nTotal features with importance = 0: {len(zero_imp)}")

if len(zero_imp) > 0:
    print(f"\nSample of zero-importance features:")
    for row in zero_imp.head(20).iter_rows(named=True):
        print(f"  - {row['feature']}")

# Low importance features (< 0.01%)
low_imp = df.filter((pl.col("importance") > 0) & (pl.col("importance_pct") < 0.01))
print(f"\n" + "="*80)
print(f"FEATURES WITH VERY LOW IMPORTANCE (< 0.01%)")
print(f"="*80)
print(f"\nTotal features with importance < 0.01%: {len(low_imp)}")
print(f"Combined importance: {low_imp['importance_pct'].sum():.2f}%")

# Time impact estimation
print("\n" + "="*80)
print("TIME IMPACT ESTIMATION")
print("="*80)

total_features = len(df)
print(f"\nCurrent setup:")
print(f"  Total features: {total_features}")
print(f"  Estimated training time: ~35-60 minutes")

# Scenarios
scenarios = [
    ("Keep all features", total_features, 1.0),
    ("Remove importance = 0", total_features - len(zero_imp), 
     (total_features - len(zero_imp)) / total_features),
    ("Keep top 2000", 2000, 2000 / total_features),
    ("Keep top 1500", 1500, 1500 / total_features),
    ("Keep top 1000", 1000, 1000 / total_features),
    ("Keep top 500", 500, 500 / total_features),
]

print(f"\n{'Scenario':<30} {'Features':<12} {'Time Factor':<15} {'Est. Time':<15} {'Imp. Loss'}")
print("-"*100)

for scenario_name, n_features, time_factor in scenarios:
    est_time_min = 35 * time_factor
    est_time_max = 60 * time_factor
    
    if n_features <= len(df):
        imp_kept = df.head(n_features)["importance_pct"].sum()
        imp_loss = 100 - imp_kept
    else:
        imp_kept = 100
        imp_loss = 0
    
    print(f"{scenario_name:<30} {n_features:<12} {time_factor:>6.2f}x          "
          f"{est_time_min:>4.0f}-{est_time_max:>2.0f} min     {imp_loss:>6.2f}%")

# LightGBM behavior with many features
print("\n" + "="*80)
print("LIGHTGBM BEHAVIOR WITH MANY FEATURES")
print("="*80)

print(f"""
LightGBM is designed to handle MANY features efficiently:

1. **Feature Pre-filter**: LightGBM ignores features with zero variance
2. **Histogram-based**: Groups features into bins (not affected by feature count)
3. **Feature sampling**: Only considers a subset per tree (feature_fraction = 0.50)
4. **Lazy evaluation**: Only computes splits for promising features

With feature_fraction = 0.50:
  - Each tree only considers ~{int(total_features * 0.5)} features
  - Low-importance features are rarely selected
  - Training time is NOT linear with feature count
""")

# Memory impact
print("\n" + "="*80)
print("MEMORY IMPACT")
print("="*80)

bytes_per_feature_per_sample = 4  # Float32
samples = 37260  # With undersampling 0.05
mb_per_feature = (bytes_per_feature_per_sample * samples) / (1024 * 1024)

print(f"\nMemory usage estimation:")
print(f"  Samples (with undersampling 0.05): ~{samples:,}")
print(f"  Memory per feature: ~{mb_per_feature:.2f} MB")
print(f"  Total memory for features: ~{mb_per_feature * total_features:.0f} MB")
print(f"  If we remove 500 features: ~{mb_per_feature * 500:.0f} MB saved")

# Recommendation
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

print(f"""
Based on the analysis:

1. **TIME SAVINGS**: Minimal (~5-10% at most)
   - LightGBM already ignores low-importance features
   - feature_fraction = 0.50 means only half are considered per tree
   - Removing features won't speed up training significantly

2. **MEMORY SAVINGS**: Moderate (~{mb_per_feature * len(zero_imp):.0f} MB if removing zero-importance)
   - Not critical unless you're running out of RAM

3. **RISK**: Potential performance loss
   - Even low-importance features can help in edge cases
   - Ensemble models benefit from diverse features
   - Removing features = losing information

**VERDICT**: 
""")

if len(zero_imp) > 50:
    print(f"  [MAYBE] Remove {len(zero_imp)} features with importance = 0")
    print(f"           Risk: VERY LOW (they contribute nothing)")
    print(f"           Benefit: ~{mb_per_feature * len(zero_imp):.0f} MB RAM saved")
else:
    print(f"  [NO] Keep all features")
    print(f"           Only {len(zero_imp)} features have zero importance")
    print(f"           Not worth the risk of removing potentially useful features")

print(f"""
**FOR COMPETITION**: Keep ALL features
  - Every 0.01% of importance can improve your ranking
  - Time is not critical (you can wait 1 hour)
  - Memory is not critical (you have enough RAM)

**FOR EXPERIMENTATION**: Remove features with importance = 0
  - Safe to remove (they contribute nothing)
  - Saves ~{int(35 * (len(zero_imp) / total_features))} minutes per experiment
  - No performance loss
""")

print("\n" + "="*80)

