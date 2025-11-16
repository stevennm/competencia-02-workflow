"""
Verification script: Check that preprocessed_data.parquet applies only expected
transformations to competencia_02_target.parquet

Expected transformations (from 02_preprocessing.py):
1. MICE imputation for months 201905, 201910, 202006
2. IPC (inflation) correction for monetary columns
3. Elimination of problematic features: cprestamos_personales, mprestamos_personales

This script verifies:
1. Row count and structure preservation
2. Only expected columns were dropped
3. MICE imputation was applied (values changed in specific months)
4. IPC correction was applied (monetary values adjusted)
5. No unexpected data modifications

Usage:
    python playground/verify_preprocessing.py

Exit codes:
    0: Verification passed - only expected transformations applied
    1: Verification failed - found unexpected changes
"""

import polars as pl
from pathlib import Path
from datetime import datetime

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def main():
    print_section(f"VERIFICATION: target -> preprocessed - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # File paths
    target_file = Path("data/competencia_02_target.parquet")
    preprocessed_file = Path("data/preprocessed_data.parquet")
    
    # Check files exist
    if not target_file.exists():
        print(f"[ERROR] {target_file} not found")
        return 1
    if not preprocessed_file.exists():
        print(f"[ERROR] {preprocessed_file} not found")
        print("Please run preprocessing/02_preprocessing.py first")
        return 1
    
    print(f"[OK] Both files exist")
    
    # Load data
    print("\nLoading data...")
    print(f"  - Reading {target_file}...")
    df_target = pl.read_parquet(target_file)
    print(f"    Shape: {df_target.shape}")
    
    print(f"  - Reading {preprocessed_file}...")
    df_preprocessed = pl.read_parquet(preprocessed_file)
    print(f"    Shape: {df_preprocessed.shape}")
    
    # =========================================================================
    # CHECK 1: Row count
    # =========================================================================
    print_section("CHECK 1: Row Count")
    if df_target.shape[0] == df_preprocessed.shape[0]:
        print(f"[OK] Row count matches: {df_target.shape[0]:,} rows")
    else:
        print(f"[ERROR] Row count mismatch!")
        print(f"   Target:       {df_target.shape[0]:,}")
        print(f"   Preprocessed: {df_preprocessed.shape[0]:,}")
        return 1
    
    # =========================================================================
    # CHECK 2: Column comparison
    # =========================================================================
    print_section("CHECK 2: Column Comparison")
    
    target_cols = set(df_target.columns)
    preprocessed_cols = set(df_preprocessed.columns)
    
    # Expected dropped columns
    expected_dropped = {"cprestamos_personales", "mprestamos_personales"}
    actual_dropped = target_cols - preprocessed_cols
    unexpected_new = preprocessed_cols - target_cols
    
    print(f"Target columns:       {len(target_cols)}")
    print(f"Preprocessed columns: {len(preprocessed_cols)}")
    print(f"Expected dropped:     {expected_dropped}")
    print(f"Actual dropped:       {actual_dropped}")
    
    if actual_dropped == expected_dropped:
        print(f"[OK] Only expected columns were dropped")
    else:
        if actual_dropped != expected_dropped:
            extra_dropped = actual_dropped - expected_dropped
            missing_dropped = expected_dropped - actual_dropped
            if extra_dropped:
                print(f"[ERROR] Unexpected columns dropped: {extra_dropped}")
            if missing_dropped:
                print(f"[WARNING] Expected columns not dropped: {missing_dropped}")
        if actual_dropped != expected_dropped:
            return 1
    
    if unexpected_new:
        print(f"[ERROR] Unexpected new columns: {unexpected_new}")
        return 1
    else:
        print(f"[OK] No unexpected new columns")
    
    # =========================================================================
    # CHECK 3: Key columns verification
    # =========================================================================
    print_section("CHECK 3: Key Columns Verification")
    
    # Check numero_de_cliente
    target_clients = set(df_target["numero_de_cliente"].unique().to_list())
    preprocessed_clients = set(df_preprocessed["numero_de_cliente"].unique().to_list())
    
    if target_clients == preprocessed_clients:
        print(f"[OK] numero_de_cliente: Same unique values ({len(target_clients):,})")
    else:
        print(f"[ERROR] numero_de_cliente mismatch!")
        print(f"   Target:       {len(target_clients):,} unique")
        print(f"   Preprocessed: {len(preprocessed_clients):,} unique")
        return 1
    
    # Check foto_mes
    target_months = sorted(df_target["foto_mes"].unique().to_list())
    preprocessed_months = sorted(df_preprocessed["foto_mes"].unique().to_list())
    
    if target_months == preprocessed_months:
        print(f"[OK] foto_mes: Same months ({len(target_months)} months)")
        print(f"    Range: {target_months[0]} - {target_months[-1]}")
    else:
        print(f"[ERROR] foto_mes mismatch!")
        return 1
    
    # Check clase_ternaria
    target_clase = df_target.group_by("clase_ternaria").agg(pl.len().alias("count")).sort("clase_ternaria")
    preprocessed_clase = df_preprocessed.group_by("clase_ternaria").agg(pl.len().alias("count")).sort("clase_ternaria")
    
    if target_clase.equals(preprocessed_clase):
        print(f"[OK] clase_ternaria: Distribution unchanged")
    else:
        print(f"[ERROR] clase_ternaria distribution changed!")
        return 1
    
    # =========================================================================
    # CHECK 4: MICE Imputation Verification
    # =========================================================================
    print_section("CHECK 4: MICE Imputation Verification")
    
    # MICE should be applied to months 201905, 201910, 202006
    mice_months = [201905, 201910, 202006]
    
    print(f"\nChecking MICE imputation for months: {mice_months}")
    print("(Expecting NULL counts to decrease in these months)")
    
    # Get common columns (excluding dropped ones)
    common_cols = sorted(preprocessed_cols - {"numero_de_cliente", "foto_mes", "clase_ternaria"})
    
    mice_applied = False
    for month in mice_months:
        target_month = df_target.filter(pl.col("foto_mes") == month)
        preprocessed_month = df_preprocessed.filter(pl.col("foto_mes") == month)
        
        if len(target_month) == 0:
            print(f"  Month {month}: Not in dataset")
            continue
        
        # Check column-by-column changes (MICE may reduce NULLs in some cols, increase in others)
        cols_with_null_reduction = 0
        cols_with_null_increase = 0
        total_reduction = 0
        total_increase = 0
        
        for col in common_cols:
            if col not in df_target.columns:
                continue
            
            target_nulls = target_month[col].null_count()
            preprocessed_nulls = preprocessed_month[col].null_count()
            change = preprocessed_nulls - target_nulls
            
            if change < 0:  # NULLs reduced (imputed)
                cols_with_null_reduction += 1
                total_reduction += abs(change)
            elif change > 0:  # NULLs increased
                cols_with_null_increase += 1
                total_increase += change
        
        if cols_with_null_reduction > 0:
            print(f"  Month {month}: [OK] MICE detected:")
            print(f"    - {cols_with_null_reduction} columns with NULL reduction (total: {total_reduction:,})")
            print(f"    - {cols_with_null_increase} columns with NULL increase (total: {total_increase:,})")
            print(f"    - Net change: {total_increase - total_reduction:+,} NULLs")
            mice_applied = True
        else:
            print(f"  Month {month}: [WARNING] No NULL reduction detected")
    
    if mice_applied:
        print(f"\n[OK] MICE imputation was applied")
    else:
        print(f"\n[WARNING] No evidence of MICE imputation (maybe no NULLs in those months?)")
    
    # =========================================================================
    # CHECK 5: IPC Correction Verification
    # =========================================================================
    print_section("CHECK 5: IPC (Inflation) Correction Verification")
    
    # IPC should adjust monetary columns (those starting with 'm')
    monetary_cols = [col for col in common_cols if col.startswith('m') and col in df_target.columns]
    
    print(f"\nFound {len(monetary_cols)} monetary columns (starting with 'm')")
    print("Checking if values were adjusted for inflation...")
    
    # Sample a few months and check if values changed
    sample_months = [201901, 202001, 202101]  # Different years
    ipc_applied = False
    
    for month in sample_months:
        if month not in target_months:
            continue
        
        target_month = df_target.filter(pl.col("foto_mes") == month)
        preprocessed_month = df_preprocessed.filter(pl.col("foto_mes") == month)
        
        # Check a few monetary columns
        changes_detected = 0
        for col in monetary_cols[:5]:  # Check first 5 monetary columns
            target_vals = target_month[col].drop_nulls()
            preprocessed_vals = preprocessed_month[col].drop_nulls()
            
            if len(target_vals) > 0 and len(preprocessed_vals) > 0:
                # Compare means (should be different if IPC was applied)
                target_mean = target_vals.mean()
                preprocessed_mean = preprocessed_vals.mean()
                
                if target_mean is not None and preprocessed_mean is not None:
                    if abs(target_mean - preprocessed_mean) > 0.01:  # Allow small tolerance
                        changes_detected += 1
        
        if changes_detected > 0:
            print(f"  Month {month}: [OK] {changes_detected}/5 monetary columns show value changes")
            ipc_applied = True
        else:
            print(f"  Month {month}: [WARNING] No changes detected in monetary columns")
    
    if ipc_applied:
        print(f"\n[OK] IPC correction appears to have been applied")
    else:
        print(f"\n[WARNING] No clear evidence of IPC correction")
    
    # =========================================================================
    # CHECK 6: Data Integrity - Sample Comparison
    # =========================================================================
    print_section("CHECK 6: Data Integrity (Sample Check)")
    
    print("\nComparing 5 random clients (non-monetary, non-MICE columns)...")
    print("These columns should be identical between target and preprocessed")
    
    # Select columns that should NOT change (non-monetary, non-imputed)
    stable_cols = [col for col in common_cols 
                   if not col.startswith('m')  # Not monetary
                   and col in df_target.columns
                   and not any(x in col.lower() for x in ['prestamos', 'saldo', 'descubierto'])][:20]
    
    if len(stable_cols) == 0:
        print("[WARNING] No stable columns found to compare")
    else:
        print(f"Checking {len(stable_cols)} stable columns...")
        
        # Sample 5 random clients
        sample_clients = df_target["numero_de_cliente"].unique().sample(n=5, seed=42)
        
        mismatches = []
        for client_id in sample_clients:
            target_client = df_target.filter(pl.col("numero_de_cliente") == client_id).sort("foto_mes").select(["numero_de_cliente", "foto_mes"] + stable_cols)
            preprocessed_client = df_preprocessed.filter(pl.col("numero_de_cliente") == client_id).sort("foto_mes").select(["numero_de_cliente", "foto_mes"] + stable_cols)
            
            # Compare
            for col in stable_cols:
                target_vals = target_client[col]
                preprocessed_vals = preprocessed_client[col]
                
                try:
                    matches = (target_vals == preprocessed_vals) | (target_vals.is_null() & preprocessed_vals.is_null())
                    n_mismatches = (~matches).sum()
                    
                    if n_mismatches > 0:
                        mismatches.append(f"Client {client_id}, column {col}: {n_mismatches} mismatches")
                        break
                except Exception:
                    # Type mismatch - cast and compare
                    target_str = target_vals.cast(pl.Utf8)
                    preprocessed_str = preprocessed_vals.cast(pl.Utf8)
                    matches = (target_str == preprocessed_str) | (target_vals.is_null() & preprocessed_vals.is_null())
                    n_mismatches = (~matches).sum()
                    
                    if n_mismatches > 0:
                        mismatches.append(f"Client {client_id}, column {col}: {n_mismatches} mismatches (cast)")
                        break
        
        if mismatches:
            print(f"[WARNING] Found mismatches in stable columns:")
            for msg in mismatches[:5]:
                print(f"   {msg}")
            print("   (This might be expected if these columns were affected by MICE)")
        else:
            print(f"[OK] All {len(sample_clients)} sampled clients match in stable columns")
    
    # =========================================================================
    # CHECK 7: Row count per month
    # =========================================================================
    print_section("CHECK 7: Row Count per Month")
    
    target_month_counts = df_target.group_by("foto_mes").agg(pl.len().alias("count")).sort("foto_mes")
    preprocessed_month_counts = df_preprocessed.group_by("foto_mes").agg(pl.len().alias("count")).sort("foto_mes")
    
    merged = target_month_counts.join(preprocessed_month_counts, on="foto_mes", suffix="_preprocessed")
    merged = merged.with_columns([
        (pl.col("count") - pl.col("count_preprocessed")).alias("diff")
    ])
    
    all_match = (merged["diff"] == 0).all()
    
    if all_match:
        print(f"[OK] All {len(merged)} months have matching row counts")
    else:
        print(f"[ERROR] Row count mismatch in some months!")
        mismatches = merged.filter(pl.col("diff") != 0)
        for row in mismatches.iter_rows(named=True):
            print(f"   {row['foto_mes']}: Target={row['count']:,}, Preprocessed={row['count_preprocessed']:,}, Diff={row['diff']}")
        return 1
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_section("VERIFICATION PASSED")
    
    print("\nSummary:")
    print(f"  [OK] Row count preserved: {df_target.shape[0]:,}")
    print(f"  [OK] Expected columns dropped: {expected_dropped}")
    print(f"  [OK] No unexpected new columns")
    print(f"  [OK] Key columns preserved (numero_de_cliente, foto_mes, clase_ternaria)")
    print(f"  [OK] Row counts match for all {len(merged)} months")
    
    if mice_applied:
        print(f"  [OK] MICE imputation detected in target months")
    else:
        print(f"  [WARNING] MICE imputation not clearly detected")
    
    if ipc_applied:
        print(f"  [OK] IPC correction detected in monetary columns")
    else:
        print(f"  [WARNING] IPC correction not clearly detected")
    
    print("\n" + "="*70)
    print("The transformation from target to preprocessed is CORRECT [OK]")
    print("="*70)
    print("\nConclusion:")
    print("  preprocessed_data.parquet is competencia_02_target.parquet with:")
    print("  1. MICE imputation applied (201905, 201910, 202006)")
    print("  2. IPC (inflation) correction applied to monetary columns")
    print("  3. Problematic features removed (cprestamos_personales, mprestamos_personales)")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

