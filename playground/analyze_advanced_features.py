"""
Analyze if advanced features are useful in the model
"""
import polars as pl

print("="*80)
print("ANALYZING ADVANCED FEATURES IMPORTANCE")
print("="*80)

# Load feature importance
df = pl.read_csv("output/comp2_entrega2_202108_under0.05_zlgbm/feature_importance.txt", 
                 separator="\t")

# Define advanced feature patterns
advanced_patterns = {
    "Volatility (_std6)": "_std6",
    "Volatility (_cv6)": "_cv6",
    "Momentum": "_momentum",
    "Acceleration": "_accel",
    "Cambio Tendencia": "_cambio_tend",
    "Rango": "_rango6"
}

print(f"\nTotal features: {len(df)}")
print(f"Features with importance > 0: {(df['importance'] > 0).sum()}")

# Analyze each type of advanced feature
print("\n" + "="*80)
print("ADVANCED FEATURES BY TYPE")
print("="*80)

total_advanced_importance = 0
total_advanced_count = 0

for name, pattern in advanced_patterns.items():
    features = df.filter(pl.col("feature").str.contains(pattern))
    features_with_imp = features.filter(pl.col("importance") > 0)
    total_imp = features["importance_pct"].sum()
    
    total_advanced_importance += total_imp
    total_advanced_count += len(features)
    
    print(f"\n{name}:")
    print(f"  Total: {len(features)}")
    print(f"  With importance > 0: {len(features_with_imp)}")
    print(f"  Total importance: {total_imp:.2f}%")
    
    if len(features_with_imp) > 0:
        # Show top 3
        top3 = features.sort("importance", descending=True).head(3)
        print(f"  Top 3:")
        for row in top3.iter_rows(named=True):
            print(f"    #{row['rank']:>4} {row['feature']:<45} {row['importance_pct']:>6.2f}%")

# Overall summary
print("\n" + "="*80)
print("OVERALL ADVANCED FEATURES SUMMARY")
print("="*80)

print(f"\nTotal advanced features: {total_advanced_count}")
print(f"Total importance of advanced features: {total_advanced_importance:.2f}%")

# Compare with other feature types
print("\n" + "="*80)
print("COMPARISON WITH OTHER FEATURE TYPES")
print("="*80)

# RF features
rf_features = df.filter(pl.col("feature").str.starts_with("rf_"))
rf_imp = rf_features["importance_pct"].sum()

# Intra-month features (kmes)
intra_features = df.filter(
    pl.col("feature").str.contains("ctrx_quarter") | 
    pl.col("feature").str.contains("kmes")
)
intra_imp = intra_features["importance_pct"].sum()

# Historical features (lag, delta, tend, avg, min, max, ratio)
historical_patterns = ["_lag", "_delta", "_tend6", "_avg", "_min6", "_max6", "_ratio"]
historical_features = df.filter(
    pl.col("feature").str.contains("|".join(historical_patterns)) &
    ~pl.col("feature").str.contains("|".join(advanced_patterns.values()))
)
historical_imp = historical_features["importance_pct"].sum()

# Base features (no suffix)
base_features = df.filter(
    ~pl.col("feature").str.contains("_") &
    ~pl.col("feature").str.starts_with("rf_")
)
base_imp = base_features["importance_pct"].sum()

print(f"\n{'Feature Type':<30} {'Count':<10} {'Total Importance':<20}")
print("-"*80)
print(f"{'RF Features':<30} {len(rf_features):<10} {rf_imp:>6.2f}%")
print(f"{'Intra-month (kmes)':<30} {len(intra_features):<10} {intra_imp:>6.2f}%")
print(f"{'Historical (lag/delta/etc)':<30} {len(historical_features):<10} {historical_imp:>6.2f}%")
print(f"{'Advanced (std/cv/momentum)':<30} {total_advanced_count:<10} {total_advanced_importance:>6.2f}%")
print(f"{'Base features':<30} {len(base_features):<10} {base_imp:>6.2f}%")

# Top 50 analysis
print("\n" + "="*80)
print("ADVANCED FEATURES IN TOP RANKS")
print("="*80)

top50 = df.head(50)
top100 = df.head(100)
top200 = df.head(200)

advanced_in_top50 = top50.filter(
    pl.col("feature").str.contains("|".join(advanced_patterns.values()))
)
advanced_in_top100 = top100.filter(
    pl.col("feature").str.contains("|".join(advanced_patterns.values()))
)
advanced_in_top200 = top200.filter(
    pl.col("feature").str.contains("|".join(advanced_patterns.values()))
)

print(f"\nAdvanced features in top 50: {len(advanced_in_top50)}")
print(f"Advanced features in top 100: {len(advanced_in_top100)}")
print(f"Advanced features in top 200: {len(advanced_in_top200)}")

if len(advanced_in_top50) > 0:
    print(f"\nTop 50 advanced features:")
    for row in advanced_in_top50.iter_rows(named=True):
        print(f"  #{row['rank']:>3} {row['feature']:<50} {row['importance_pct']:>6.2f}%")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if total_advanced_importance > 5:
    print(f"\n[OK] Advanced features are USEFUL ({total_advanced_importance:.1f}% of total importance)")
elif total_advanced_importance > 2:
    print(f"\n[MODERATE] Advanced features have moderate impact ({total_advanced_importance:.1f}% of total importance)")
else:
    print(f"\n[LOW] Advanced features have low impact ({total_advanced_importance:.1f}% of total importance)")

print("\n" + "="*80)

