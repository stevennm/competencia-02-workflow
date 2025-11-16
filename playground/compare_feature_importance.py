"""
Compare feature importance between two experiments
"""
import polars as pl

print("="*80)
print("COMPARING FEATURE IMPORTANCE")
print("="*80)

# Load both files
exp1_file = "output/comp2_entrega2_202108_under0.05_zlgbm/feature_importance.txt"
exp2_file = "output/comp2_entrega2_test202106_under0.05_zlgbm/feature_importance.txt"

print(f"\nExperiment 1: {exp1_file}")
print(f"Experiment 2: {exp2_file}")

df1 = pl.read_csv(exp1_file, separator="\t")
df2 = pl.read_csv(exp2_file, separator="\t")

print(f"\nExp1 features: {len(df1)}")
print(f"Exp2 features: {len(df2)}")

# Get top 50 from each
top1 = df1.head(50)
top2 = df2.head(50)

print("\n" + "="*80)
print("TOP 10 COMPARISON")
print("="*80)

print(f"\n{'Rank':<6} {'Exp1 (202108)':<45} {'Imp%':<8} | {'Exp2 (202106)':<45} {'Imp%':<8}")
print("-"*80)

for i in range(10):
    feat1 = top1[i, "feature"]
    imp1 = top1[i, "importance_pct"]
    feat2 = top2[i, "feature"]
    imp2 = top2[i, "importance_pct"]
    
    marker = "  "
    if feat1 == feat2:
        marker = "=="
    
    print(f"{i+1:<6} {feat1:<45} {imp1:>6.2f}% {marker} {feat2:<45} {imp2:>6.2f}%")

# Check canaries
print("\n" + "="*80)
print("CANARY ANALYSIS")
print("="*80)

canary1 = df1.filter(pl.col("is_canary"))
canary2 = df2.filter(pl.col("is_canary"))

print(f"\nExp1 (202108):")
print(f"  Canary importance: {canary1['importance'].sum():,.0f}")
print(f"  Avg canary importance: {canary1['importance'].mean():,.2f}")

print(f"\nExp2 (202106):")
print(f"  Canary importance: {canary2['importance'].sum():,.0f}")
print(f"  Avg canary importance: {canary2['importance'].mean():,.2f}")

# Compare RF features dominance
print("\n" + "="*80)
print("RF FEATURES ANALYSIS")
print("="*80)

rf1 = df1.filter(pl.col("feature").str.starts_with("rf_"))
rf2 = df2.filter(pl.col("feature").str.starts_with("rf_"))

print(f"\nExp1 (202108):")
print(f"  Total RF importance: {rf1['importance_pct'].sum():.2f}%")
print(f"  RF features in top 10: {top1.head(10).filter(pl.col('feature').str.starts_with('rf_')).shape[0]}")
print(f"  RF features in top 20: {top1.head(20).filter(pl.col('feature').str.starts_with('rf_')).shape[0]}")
print(f"  RF features in top 50: {top1.head(50).filter(pl.col('feature').str.starts_with('rf_')).shape[0]}")

print(f"\nExp2 (202106):")
print(f"  Total RF importance: {rf2['importance_pct'].sum():.2f}%")
print(f"  RF features in top 10: {top2.head(10).filter(pl.col('feature').str.starts_with('rf_')).shape[0]}")
print(f"  RF features in top 20: {top2.head(20).filter(pl.col('feature').str.starts_with('rf_')).shape[0]}")
print(f"  RF features in top 50: {top2.head(50).filter(pl.col('feature').str.starts_with('rf_')).shape[0]}")

# Top RF feature comparison
print("\n" + "="*80)
print("TOP RF FEATURE COMPARISON")
print("="*80)

top_rf1 = df1.filter(pl.col("feature").str.starts_with("rf_")).head(5)
top_rf2 = df2.filter(pl.col("feature").str.starts_with("rf_")).head(5)

print(f"\nExp1 (202108) - Top 5 RF features:")
for row in top_rf1.iter_rows(named=True):
    print(f"  #{row['rank']:>3} {row['feature']:<20} {row['importance_pct']:>6.2f}%")

print(f"\nExp2 (202106) - Top 5 RF features:")
for row in top_rf2.iter_rows(named=True):
    print(f"  #{row['rank']:>3} {row['feature']:<20} {row['importance_pct']:>6.2f}%")

# Key business features comparison
print("\n" + "="*80)
print("KEY BUSINESS FEATURES COMPARISON")
print("="*80)

key_features = [
    "ctrx_quarter_normalizado",
    "mcaja_ahorro",
    "mtarjeta_visa_consumo",
    "cpayroll_trx",
    "mcuentas_saldo"
]

print(f"\n{'Feature':<35} {'Exp1 Rank':<12} {'Exp1 %':<10} | {'Exp2 Rank':<12} {'Exp2 %':<10}")
print("-"*80)

for feat in key_features:
    row1 = df1.filter(pl.col("feature") == feat)
    row2 = df2.filter(pl.col("feature") == feat)
    
    if len(row1) > 0 and len(row2) > 0:
        rank1 = row1[0, "rank"]
        imp1 = row1[0, "importance_pct"]
        rank2 = row2[0, "rank"]
        imp2 = row2[0, "importance_pct"]
        
        diff = rank1 - rank2
        marker = ""
        if diff < 0:
            marker = f"(+{abs(diff)} better)"
        elif diff > 0:
            marker = f"(-{diff} worse)"
        
        print(f"{feat:<35} #{rank1:<11} {imp1:>6.2f}%   | #{rank2:<11} {imp2:>6.2f}%   {marker}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nKey differences:")
print(f"  1. RF feature dominance: Exp1 has {rf1['importance_pct'].sum():.1f}% vs Exp2 {rf2['importance_pct'].sum():.1f}%")
print(f"  2. Top RF feature (rf_019_004): Exp1 has {df1.filter(pl.col('feature') == 'rf_019_004')[0, 'importance_pct']:.2f}% vs Exp2 {df2.filter(pl.col('feature') == 'rf_019_004')[0, 'importance_pct']:.2f}%")
print(f"  3. ctrx_quarter_normalizado: Exp1 rank #{df1.filter(pl.col('feature') == 'ctrx_quarter_normalizado')[0, 'rank']} vs Exp2 rank #{df2.filter(pl.col('feature') == 'ctrx_quarter_normalizado')[0, 'rank']}")

print("\n" + "="*80)

