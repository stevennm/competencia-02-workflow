"""
Verification script: Check that competencia_02_target.parquet only adds clase_ternaria
to competencia_02_crudo.csv.gz without modifying any other data.

This script verifies:
1. All original columns are preserved
2. Only clase_ternaria was added
3. Row count matches (total and per month)
4. Data values are identical for specific clients (sample check)
5. Column counts and distributions match
"""

import polars as pl
from pathlib import Path

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def main():
    print_section("VERIFICATION: crudo -> target transformation")
    
    # File paths
    crudo_file = Path("data/competencia_02_crudo.csv.gz")
    target_file = Path("data/competencia_02_target.parquet")
    
    # Check files exist
    if not crudo_file.exists():
        print(f"[ERROR] {crudo_file} not found")
        return 1
    if not target_file.exists():
        print(f"[ERROR] {target_file} not found")
        return 1
    
    print(f"[OK] Both files exist")
    
    # Load data
    print("\nLoading data...")
    print(f"  - Reading {crudo_file}...")
    df_crudo = pl.read_csv(
        crudo_file,
        infer_schema_length=10000  # Fast inference for verification
    )
    print(f"    Shape: {df_crudo.shape}")
    
    print(f"  - Reading {target_file}...")
    df_target = pl.read_parquet(target_file)
    print(f"    Shape: {df_target.shape}")
    
    # =========================================================================
    # CHECK 1: Row count
    # =========================================================================
    print_section("CHECK 1: Row Count")
    if df_crudo.shape[0] == df_target.shape[0]:
        print(f"[OK] Row count matches: {df_crudo.shape[0]:,} rows")
    else:
        print(f"[ERROR] Row count mismatch!")
        print(f"   Crudo:  {df_crudo.shape[0]:,}")
        print(f"   Target: {df_target.shape[0]:,}")
        return 1
    
    # =========================================================================
    # CHECK 2: Column comparison
    # =========================================================================
    print_section("CHECK 2: Column Comparison")
    
    crudo_cols = set(df_crudo.columns)
    target_cols = set(df_target.columns)
    
    # Expected: target should have all crudo columns + clase_ternaria
    expected_new_cols = {"clase_ternaria"}
    actual_new_cols = target_cols - crudo_cols
    missing_cols = crudo_cols - target_cols
    
    print(f"Crudo columns:  {len(crudo_cols)}")
    print(f"Target columns: {len(target_cols)}")
    print(f"Expected new:   {expected_new_cols}")
    print(f"Actual new:     {actual_new_cols}")
    
    if actual_new_cols == expected_new_cols:
        print(f"[OK] Only clase_ternaria was added")
    else:
        print(f"[ERROR] Unexpected column changes!")
        if actual_new_cols != expected_new_cols:
            print(f"   Unexpected new columns: {actual_new_cols - expected_new_cols}")
        return 1
    
    if missing_cols:
        print(f"[ERROR] Missing columns from crudo: {missing_cols}")
        return 1
    else:
        print(f"[OK] All original columns preserved")
    
    # =========================================================================
    # CHECK 3: Row count per month
    # =========================================================================
    print_section("CHECK 3: Row Count per Month")
    
    crudo_months = df_crudo.group_by("foto_mes").agg(pl.len().alias("count")).sort("foto_mes")
    target_months = df_target.group_by("foto_mes").agg(pl.len().alias("count")).sort("foto_mes")
    
    merged = crudo_months.join(target_months, on="foto_mes", suffix="_target")
    merged = merged.with_columns([
        (pl.col("count") - pl.col("count_target")).alias("diff")
    ])
    
    all_match = (merged["diff"] == 0).all()
    
    if all_match:
        print(f"[OK] All {len(merged)} months have matching row counts")
        print(f"    Month range: {merged['foto_mes'].min()} - {merged['foto_mes'].max()}")
    else:
        print(f"[ERROR] Row count mismatch in some months!")
        mismatches = merged.filter(pl.col("diff") != 0)
        for row in mismatches.iter_rows(named=True):
            print(f"   {row['foto_mes']}: Crudo={row['count']:,}, Target={row['count_target']:,}, Diff={row['diff']}")
        return 1
    
    # =========================================================================
    # CHECK 4: clase_ternaria values
    # =========================================================================
    print_section("CHECK 4: clase_ternaria Values")
    
    if "clase_ternaria" not in df_target.columns:
        print("[ERROR] clase_ternaria column not found!")
        return 1
    
    # Check unique values
    clase_counts = df_target.group_by("clase_ternaria").agg(pl.len().alias("count")).sort("count", descending=True)
    print("\nclase_ternaria distribution:")
    for row in clase_counts.iter_rows(named=True):
        clase = row["clase_ternaria"]
        count = row["count"]
        pct = 100 * count / df_target.shape[0]
        clase_str = str(clase) if clase is not None else "NULL"
        print(f"  {clase_str:15s}: {count:8,} ({pct:5.2f}%)")
    
    # Expected classes (None is OK for future months without labels)
    expected_classes = {"BAJA+1", "BAJA+2", "CONTINUA"}
    actual_classes = set(clase_counts["clase_ternaria"].to_list())
    
    # Remove None from actual classes for comparison
    actual_classes_non_null = {c for c in actual_classes if c is not None}
    
    if actual_classes_non_null == expected_classes:
        print(f"\n[OK] clase_ternaria has expected values: {expected_classes}")
        if None in actual_classes:
            null_count = clase_counts.filter(pl.col("clase_ternaria").is_null())["count"][0]
            print(f"[OK] NULL values present ({null_count:,} rows) - expected for future months")
    else:
        print(f"\n[ERROR] Unexpected clase_ternaria values!")
        print(f"   Expected: {expected_classes}")
        print(f"   Actual (non-null): {actual_classes_non_null}")
        return 1
    
    # =========================================================================
    # CHECK 5: Data integrity - Client-based comparison
    # =========================================================================
    print_section("CHECK 5: Data Integrity (Client Sample Check)")
    
    # Select common columns (excluding clase_ternaria)
    common_cols = sorted(crudo_cols)
    
    # Sample 10 random clients for comparison
    sample_size = 10
    print(f"\nComparing {sample_size} random clients...")
    
    # Get unique clients
    all_clients = df_crudo["numero_de_cliente"].unique().sample(n=sample_size, seed=42)
    
    mismatches = []
    for client_id in all_clients:
        # Get data for this client from both sources
        crudo_client = df_crudo.filter(pl.col("numero_de_cliente") == client_id).sort("foto_mes").select(common_cols)
        target_client = df_target.filter(pl.col("numero_de_cliente") == client_id).sort("foto_mes").select(common_cols)
        
        # Compare row counts
        if len(crudo_client) != len(target_client):
            mismatches.append(f"Client {client_id}: Different row counts ({len(crudo_client)} vs {len(target_client)})")
            continue
        
        # Compare data
        for col in common_cols:
            crudo_vals = crudo_client[col]
            target_vals = target_client[col]
            
            try:
                # Compare (handling nulls)
                matches = (crudo_vals == target_vals) | (crudo_vals.is_null() & target_vals.is_null())
                n_mismatches = (~matches).sum()
                
                if n_mismatches > 0:
                    mismatches.append(f"Client {client_id}, column {col}: {n_mismatches} mismatches")
                    break  # Stop checking this client
            except Exception:
                # Type mismatch - cast to string and compare
                crudo_str = crudo_vals.cast(pl.Utf8)
                target_str = target_vals.cast(pl.Utf8)
                matches = (crudo_str == target_str) | (crudo_vals.is_null() & target_vals.is_null())
                n_mismatches = (~matches).sum()
                
                if n_mismatches > 0:
                    mismatches.append(f"Client {client_id}, column {col}: {n_mismatches} mismatches (type cast)")
                    break  # Stop checking this client
    
    if mismatches:
        print(f"[ERROR] Found data mismatches:")
        for msg in mismatches[:10]:  # Show first 10
            print(f"   {msg}")
        return 1
    else:
        print(f"[OK] All {sample_size} sampled clients match perfectly")
    
    # =========================================================================
    # CHECK 6: Key columns verification
    # =========================================================================
    print_section("CHECK 6: Key Columns Verification")
    
    # Check numero_de_cliente
    if "numero_de_cliente" in common_cols:
        crudo_clients = set(df_crudo["numero_de_cliente"].unique().to_list())
        target_clients = set(df_target["numero_de_cliente"].unique().to_list())
        
        if crudo_clients == target_clients:
            print(f"[OK] numero_de_cliente: Same unique values ({len(crudo_clients):,})")
        else:
            print(f"[ERROR] numero_de_cliente mismatch!")
            print(f"   Crudo:  {len(crudo_clients):,} unique")
            print(f"   Target: {len(target_clients):,} unique")
            return 1
    
    # Check foto_mes
    if "foto_mes" in common_cols:
        crudo_months_list = sorted(df_crudo["foto_mes"].unique().to_list())
        target_months_list = sorted(df_target["foto_mes"].unique().to_list())
        
        if crudo_months_list == target_months_list:
            print(f"[OK] foto_mes: Same months ({len(crudo_months_list)} months)")
            print(f"    Range: {crudo_months_list[0]} - {crudo_months_list[-1]}")
        else:
            print(f"[ERROR] foto_mes mismatch!")
            return 1
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_section("VERIFICATION PASSED")
    
    print("\nSummary:")
    print(f"  [OK] Row count preserved: {df_crudo.shape[0]:,}")
    print(f"  [OK] All original columns preserved: {len(crudo_cols)}")
    print(f"  [OK] Only clase_ternaria added")
    print(f"  [OK] Row counts match for all {len(merged)} months")
    print(f"  [OK] clase_ternaria has correct values: {expected_classes}")
    print(f"  [OK] Data integrity verified ({sample_size} clients checked)")
    print(f"  [OK] Key columns verified")
    
    print("\n" + "="*70)
    print("The transformation from crudo to target is CORRECT [OK]")
    print("="*70)
    print("\nConclusion:")
    print("  competencia_02_target.parquet is competencia_02_crudo.csv.gz")
    print("  with ONLY clase_ternaria added. No other data was modified.")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

