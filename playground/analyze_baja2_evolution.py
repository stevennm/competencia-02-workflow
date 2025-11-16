"""
Analysis script: Track BAJA+2 evolution over time

This script analyzes how BAJA+2 (customers who will leave after 2 periods)
evolve over time in the dataset.

Analysis includes:
1. BAJA+2 count and percentage per month
2. Trend visualization over time
3. Comparison with BAJA+1 and CONTINUA
4. Statistical summary

Usage:
    python playground/analyze_baja2_evolution.py

Output:
    - Console report with statistics
    - CSV file with monthly breakdown (optional)
"""

import polars as pl
from pathlib import Path
from datetime import datetime

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def format_month(foto_mes: int) -> str:
    """Convert foto_mes (YYYYMM) to readable format"""
    year = foto_mes // 100
    month = foto_mes % 100
    return f"{year}-{month:02d}"

def main():
    print_section(f"BAJA+2 EVOLUTION ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    target_file = Path("data/competencia_02_target.parquet")
    
    if not target_file.exists():
        print(f"[ERROR] {target_file} not found")
        print("Please run preprocessing/01_generate_clase_ternaria.py first")
        return 1
    
    print(f"\nLoading data from: {target_file}")
    df = pl.read_parquet(target_file)
    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # =========================================================================
    # OVERALL STATISTICS
    # =========================================================================
    print_section("OVERALL STATISTICS")
    
    # Count by clase_ternaria
    clase_counts = df.group_by("clase_ternaria").agg(pl.len().alias("count")).sort("count", descending=True)
    
    print("\nclase_ternaria distribution (all data):")
    print(f"{'Class':<15} {'Count':>10} {'Percentage':>12}")
    print("-" * 40)
    
    total_rows = df.shape[0]
    for row in clase_counts.iter_rows(named=True):
        clase = row["clase_ternaria"]
        count = row["count"]
        pct = 100 * count / total_rows
        clase_str = str(clase) if clase is not None else "NULL"
        print(f"{clase_str:<15} {count:>10,} {pct:>11.2f}%")
    
    # =========================================================================
    # MONTHLY EVOLUTION
    # =========================================================================
    print_section("MONTHLY EVOLUTION")
    
    # Filter out NULL values for monthly analysis
    df_labeled = df.filter(pl.col("clase_ternaria").is_not_null())
    
    # Calculate monthly statistics
    monthly_stats = df_labeled.group_by("foto_mes").agg([
        pl.len().alias("total"),
        (pl.col("clase_ternaria") == "BAJA+2").sum().alias("baja2_count"),
        (pl.col("clase_ternaria") == "BAJA+1").sum().alias("baja1_count"),
        (pl.col("clase_ternaria") == "CONTINUA").sum().alias("continua_count")
    ]).sort("foto_mes")
    
    # Calculate percentages
    monthly_stats = monthly_stats.with_columns([
        (100 * pl.col("baja2_count") / pl.col("total")).alias("baja2_pct"),
        (100 * pl.col("baja1_count") / pl.col("total")).alias("baja1_pct"),
        (100 * pl.col("continua_count") / pl.col("total")).alias("continua_pct"),
        ((pl.col("baja1_count") + pl.col("baja2_count")) / pl.col("total") * 100).alias("total_baja_pct")
    ])
    
    print("\nMonthly breakdown:")
    print(f"{'Month':<10} {'Total':>10} {'BAJA+2':>10} {'%':>7} {'BAJA+1':>10} {'%':>7} {'Total Baja':>7}")
    print("-" * 75)
    
    for row in monthly_stats.iter_rows(named=True):
        month_str = format_month(row['foto_mes'])
        print(f"{month_str:<10} {row['total']:>10,} "
              f"{row['baja2_count']:>10,} {row['baja2_pct']:>6.2f}% "
              f"{row['baja1_count']:>10,} {row['baja1_pct']:>6.2f}% "
              f"{row['total_baja_pct']:>6.2f}%")
    
    # =========================================================================
    # TREND ANALYSIS
    # =========================================================================
    print_section("TREND ANALYSIS")
    
    # Calculate overall statistics
    baja2_mean = monthly_stats["baja2_pct"].mean()
    baja2_std = monthly_stats["baja2_pct"].std()
    baja2_min = monthly_stats["baja2_pct"].min()
    baja2_max = monthly_stats["baja2_pct"].max()
    
    baja1_mean = monthly_stats["baja1_pct"].mean()
    total_baja_mean = monthly_stats["total_baja_pct"].mean()
    
    print(f"\nBAJA+2 Statistics:")
    print(f"  Mean:        {baja2_mean:>6.2f}%")
    print(f"  Std Dev:     {baja2_std:>6.2f}%")
    print(f"  Min:         {baja2_min:>6.2f}% (Month: {format_month(monthly_stats.filter(pl.col('baja2_pct') == baja2_min)['foto_mes'][0])})")
    print(f"  Max:         {baja2_max:>6.2f}% (Month: {format_month(monthly_stats.filter(pl.col('baja2_pct') == baja2_max)['foto_mes'][0])})")
    
    print(f"\nComparison:")
    print(f"  BAJA+1 Mean: {baja1_mean:>6.2f}%")
    print(f"  BAJA+2 Mean: {baja2_mean:>6.2f}%")
    print(f"  Total Baja:  {total_baja_mean:>6.2f}%")
    print(f"  Ratio B2/B1: {baja2_mean/baja1_mean:>6.2f}x")
    
    # =========================================================================
    # TEMPORAL PATTERNS
    # =========================================================================
    print_section("TEMPORAL PATTERNS")
    
    # Split into years
    monthly_stats = monthly_stats.with_columns([
        (pl.col("foto_mes") // 100).alias("year")
    ])
    
    yearly_stats = monthly_stats.group_by("year").agg([
        pl.col("baja2_pct").mean().alias("baja2_pct_mean"),
        pl.col("baja1_pct").mean().alias("baja1_pct_mean"),
        pl.col("total_baja_pct").mean().alias("total_baja_pct_mean"),
        pl.len().alias("n_months")
    ]).sort("year")
    
    print("\nYearly averages:")
    print(f"{'Year':<10} {'Months':>8} {'BAJA+2 %':>10} {'BAJA+1 %':>10} {'Total Baja %':>13}")
    print("-" * 60)
    
    for row in yearly_stats.iter_rows(named=True):
        print(f"{row['year']:<10} {row['n_months']:>8} "
              f"{row['baja2_pct_mean']:>9.2f}% {row['baja1_pct_mean']:>9.2f}% "
              f"{row['total_baja_pct_mean']:>12.2f}%")
    
    # Check for trend (first half vs second half)
    n_months = len(monthly_stats)
    mid_point = n_months // 2
    
    first_half_mean = monthly_stats[:mid_point]["baja2_pct"].mean()
    second_half_mean = monthly_stats[mid_point:]["baja2_pct"].mean()
    
    first_month = monthly_stats["foto_mes"][0]
    mid_month = monthly_stats["foto_mes"][mid_point-1]
    mid_next_month = monthly_stats["foto_mes"][mid_point]
    last_month = monthly_stats["foto_mes"][-1]
    
    print(f"\nTrend Analysis (First vs Second Half):")
    print(f"  First half ({format_month(first_month)} - {format_month(mid_month)}):  {first_half_mean:.2f}%")
    print(f"  Second half ({format_month(mid_next_month)} - {format_month(last_month)}): {second_half_mean:.2f}%")
    
    if second_half_mean > first_half_mean:
        change = second_half_mean - first_half_mean
        print(f"  Trend: INCREASING (+{change:.2f}pp)")
    elif second_half_mean < first_half_mean:
        change = first_half_mean - second_half_mean
        print(f"  Trend: DECREASING (-{change:.2f}pp)")
    else:
        print(f"  Trend: STABLE")
    
    # =========================================================================
    # RECENT MONTHS FOCUS
    # =========================================================================
    print_section("RECENT MONTHS (Last 6)")
    
    recent_months = monthly_stats.tail(6)
    
    print(f"\n{'Month':<10} {'BAJA+2':>10} {'%':>7} {'Change':>8}")
    print("-" * 40)
    
    prev_pct = None
    for row in recent_months.iter_rows(named=True):
        month_str = format_month(row['foto_mes'])
        change_str = ""
        if prev_pct is not None:
            change = row['baja2_pct'] - prev_pct
            change_str = f"{change:+.2f}pp"
        
        print(f"{month_str:<10} {row['baja2_count']:>10,} {row['baja2_pct']:>6.2f}% {change_str:>8}")
        prev_pct = row['baja2_pct']
    
    # =========================================================================
    # SAVE DETAILED REPORT (Optional)
    # =========================================================================
    print_section("EXPORT")
    
    output_file = Path("playground/baja2_evolution_report.csv")
    monthly_stats.select([
        "foto_mes",
        "total",
        "baja2_count",
        "baja2_pct",
        "baja1_count",
        "baja1_pct",
        "continua_count",
        "continua_pct",
        "total_baja_pct"
    ]).write_csv(output_file)
    
    print(f"\nDetailed report saved to: {output_file}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_section("SUMMARY")
    
    first_month_summary = monthly_stats["foto_mes"][0]
    last_month_summary = monthly_stats["foto_mes"][-1]
    
    print(f"\nKey Findings:")
    print(f"  - Dataset covers {len(monthly_stats)} months ({format_month(first_month_summary)} to {format_month(last_month_summary)})")
    print(f"  - Average BAJA+2 rate: {baja2_mean:.2f}%")
    print(f"  - BAJA+2 is {'more' if baja2_mean > baja1_mean else 'less'} common than BAJA+1 ({baja2_mean/baja1_mean:.2f}x)")
    print(f"  - Volatility (std dev): {baja2_std:.2f}pp")
    
    # Check for anomalies in last month
    last_month_data = monthly_stats.tail(1)
    if last_month_data["baja2_count"][0] == 0 or last_month_data["total"][0] < 10000:
        print(f"  - NOTE: Last month ({format_month(last_month_summary)}) has incomplete data (only {last_month_data['total'][0]:,} records)")
    
    if second_half_mean > first_half_mean * 1.1:
        print(f"  - WARNING: BAJA+2 rate is increasing over time!")
    elif second_half_mean < first_half_mean * 0.9:
        print(f"  - NOTE: BAJA+2 rate is decreasing over time")
    else:
        print(f"  - BAJA+2 rate is relatively stable over time")
    
    print("\n" + "="*70)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

