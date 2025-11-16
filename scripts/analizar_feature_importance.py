"""
Feature Importance Analysis
Loads feature_importance.txt from experiment and creates visualizations and analysis
"""

import sys
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def analyze_feature_importance(experiment_name: str):
    """
    Analyze feature importance from an experiment
    
    Args:
        experiment_name: Name of the experiment (e.g., 'seg-001_zlgbm')
    """
    print_section(f"FEATURE IMPORTANCE ANALYSIS - {experiment_name}")
    
    # Load feature importance file
    importance_file = Path(f"output/{experiment_name}/feature_importance.txt")
    
    if not importance_file.exists():
        print(f"\n❌ ERROR: File not found: {importance_file}")
        print(f"   Make sure the experiment has been run and feature importance was saved.")
        return 1
    
    print(f"\nLoading feature importance from: {importance_file}")
    df = pl.read_csv(importance_file, separator="\t")
    
    print(f"✓ Loaded {len(df)} features")
    
    # Summary statistics
    print_section("SUMMARY STATISTICS")
    
    canary_df = df.filter(pl.col('is_canary'))
    real_df = df.filter(~pl.col('is_canary'))
    
    print(f"\nTotal features: {len(df)}")
    print(f"  Canaries: {len(canary_df)} ({len(canary_df)/len(df)*100:.1f}%)")
    print(f"  Real features: {len(real_df)} ({len(real_df)/len(df)*100:.1f}%)")
    
    print(f"\nImportance statistics:")
    total_importance = df['importance'].sum()
    canary_importance = canary_df['importance'].sum()
    real_importance = real_df['importance'].sum()
    
    print(f"  Total importance: {total_importance:,.1f}")
    print(f"  Canary importance: {canary_importance:,.1f} ({canary_importance/total_importance*100:.2f}%)")
    print(f"  Real importance: {real_importance:,.1f} ({real_importance/total_importance*100:.2f}%)")
    
    print(f"\nAverage importance:")
    canary_avg = canary_df['importance'].mean()
    real_avg = real_df['importance'].mean()
    print(f"  Canaries: {canary_avg:,.1f}")
    print(f"  Real features: {real_avg:,.1f}")
    print(f"  Ratio (real/canary): {real_avg / canary_avg:.2f}x")
    
    # Top features analysis
    print_section("TOP FEATURES")
    
    for n in [10, 20, 50, 100]:
        top_n = df.head(n)
        n_canaries = top_n.filter(pl.col('is_canary')).shape[0]
        cumsum = top_n['importance_cumsum_pct'][-1]
        print(f"\nTop {n:>3d} features:")
        print(f"  Canaries: {n_canaries} ({n_canaries/n*100:.1f}%)")
        print(f"  Cumulative importance: {cumsum:.2f}%")
    
    # Top 50 features detailed
    print_section("TOP 50 FEATURES DETAILED")
    
    top_50 = df.head(50)
    print(f"\n{'Rank':<5} {'Feature':<40} {'Importance':>12} {'%':>8} {'Cumsum %':>10} {'Type':>10}")
    print(f"{'-'*92}")
    
    for row in top_50.iter_rows(named=True):
        feat_type = "Canary 🐤" if row['is_canary'] else "Real"
        print(f"{row['rank']:>3d}. {row['feature']:<40s} "
              f"{row['importance']:>12,.1f} {row['importance_pct']:>7.2f}% "
              f"{row['importance_cumsum_pct']:>9.2f}% {feat_type:>10s}")
    
    # Feature type analysis (for real features)
    print_section("FEATURE TYPE ANALYSIS (REAL FEATURES ONLY)")
    
    # Categorize features
    def categorize_feature(feat_name):
        if feat_name.startswith("canarito_"):
            return "canary"
        elif "_lag" in feat_name:
            return "lag"
        elif "_delta" in feat_name:
            return "delta"
        elif "_tend" in feat_name:
            return "trend"
        elif "_min" in feat_name or "_max" in feat_name:
            return "minmax"
        elif "_avg" in feat_name or "_promedio" in feat_name:
            return "average"
        elif "_ratio" in feat_name:
            return "ratio"
        elif feat_name.startswith("rf_"):
            return "rf_leaf"
        elif feat_name in ["kmes", "ctrx_quarter_normalizado", "mpayroll_sobre_edad"]:
            return "intra_month"
        else:
            return "original"
    
    # Add feature type column
    real_df = real_df.with_columns([
        pl.col('feature').map_elements(categorize_feature, return_dtype=pl.Utf8).alias('feature_type')
    ])
    
    # Group by feature type
    type_analysis = real_df.group_by('feature_type').agg([
        pl.col('importance').sum().alias('total_importance'),
        pl.col('importance').mean().alias('avg_importance'),
        pl.count().alias('count')
    ]).sort('total_importance', descending=True)
    
    # Add percentage
    type_analysis = type_analysis.with_columns([
        (pl.col('total_importance') / real_importance * 100).alias('pct_total')
    ])
    
    print("\nFeature type summary:")
    print(f"{'Type':<15} {'Count':>8} {'Total Imp':>15} {'Avg Imp':>12} {'% Total':>10}")
    print(f"{'-'*70}")
    for row in type_analysis.iter_rows(named=True):
        print(f"{row['feature_type']:<15} {row['count']:>8} {row['total_importance']:>15,.1f} "
              f"{row['avg_importance']:>12,.1f} {row['pct_total']:>9.2f}%")
    
    # Top features by type
    print("\nTop 5 features by type:")
    for feat_type_row in type_analysis.iter_rows(named=True):
        feat_type = feat_type_row['feature_type']
        type_features = real_df.filter(pl.col('feature_type') == feat_type).head(5)
        if len(type_features) > 0:
            print(f"\n{feat_type.upper()}:")
            for row in type_features.iter_rows(named=True):
                print(f"  {row['rank']:>3d}. {row['feature']:<40s} {row['importance']:>12,.1f}")
    
    # Save detailed analysis
    output_dir = Path(f"output/{experiment_name}")
    
    # Save categorized features
    analysis_file = output_dir / "feature_importance_analysis.txt"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write("FEATURE IMPORTANCE ANALYSIS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Total features: {len(df)}\n")
        f.write(f"Canaries: {len(canary_df)}\n")
        f.write(f"Real features: {len(real_df)}\n\n")
        
        f.write("FEATURE TYPE SUMMARY\n")
        f.write("-"*70 + "\n")
        for row in type_analysis.iter_rows(named=True):
            f.write(f"{row['feature_type']:<15} {row['count']:>8} {row['total_importance']:>15,.1f} "
                   f"{row['avg_importance']:>12,.1f} {row['pct_total']:>9.2f}%\n")
        f.write("\n\n")
        
        f.write("TOP 100 FEATURES\n")
        f.write("-"*70 + "\n")
        for row in df.head(100).iter_rows(named=True):
            f.write(f"{row['rank']:>3d}. {row['feature']:<40s} {row['importance']:>12,.1f} "
                   f"{row['importance_pct']:>7.2f}% {row['importance_cumsum_pct']:>9.2f}%\n")
    
    print(f"\n✓ Detailed analysis saved to {analysis_file}")
    
    # Create visualization (if matplotlib available)
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Feature Importance Analysis - {experiment_name}', fontsize=14, fontweight='bold')
        
        # 1. Top 20 features
        ax1 = axes[0, 0]
        top_20 = df.head(20).sort('importance')
        colors = ['#FF6B6B' if x else '#4ECDC4' for x in top_20['is_canary'].to_list()]
        ax1.barh(range(len(top_20)), top_20['importance'].to_list(), color=colors)
        ax1.set_yticks(range(len(top_20)))
        features = top_20['feature'].to_list()
        ax1.set_yticklabels([f"{feat[:35]}..." if len(feat) > 35 else feat for feat in features], fontsize=8)
        ax1.set_xlabel('Importance (Gain)')
        ax1.set_title('Top 20 Features')
        # Custom legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#FF6B6B', label='Canary'),
                          Patch(facecolor='#4ECDC4', label='Real')]
        ax1.legend(handles=legend_elements, loc='lower right')
        
        # 2. Cumulative importance
        ax2 = axes[0, 1]
        cumsum_values = df['importance_cumsum_pct'].to_list()
        ax2.plot(range(1, len(df)+1), cumsum_values)
        ax2.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50%')
        ax2.axhline(y=80, color='orange', linestyle='--', alpha=0.5, label='80%')
        ax2.axhline(y=95, color='g', linestyle='--', alpha=0.5, label='95%')
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('Cumulative Importance (%)')
        ax2.set_title('Cumulative Feature Importance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Feature type distribution
        ax3 = axes[1, 0]
        type_counts_sorted = type_analysis.sort('count', descending=True)
        feat_types = type_counts_sorted['feature_type'].to_list()
        counts = type_counts_sorted['count'].to_list()
        ax3.bar(range(len(feat_types)), counts)
        ax3.set_xticks(range(len(feat_types)))
        ax3.set_xticklabels(feat_types, rotation=45, ha='right')
        ax3.set_ylabel('Count')
        ax3.set_title('Feature Count by Type')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Feature type importance
        ax4 = axes[1, 1]
        type_importance_sorted = type_analysis.sort('pct_total', descending=True)
        labels = type_importance_sorted['feature_type'].to_list()
        sizes = type_importance_sorted['pct_total'].to_list()
        colors = plt.cm.Set3(range(len(labels)))
        wedges, texts, autotexts = ax4.pie(sizes, labels=labels,
                                            autopct='%1.1f%%', startangle=90, colors=colors)
        ax4.set_title('Importance Distribution by Feature Type')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_dir / "feature_importance_plots.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to {plot_file}")
        
        # Close figure to free memory
        plt.close(fig)
        
    except Exception as e:
        print(f"\n⚠ Could not create visualization: {e}")
    
    print_section("ANALYSIS COMPLETE")
    print(f"\nOutput files created in: output/{experiment_name}/")
    print(f"  - feature_importance.txt (raw data)")
    print(f"  - feature_importance_analysis.txt (detailed analysis)")
    print(f"  - feature_importance_plots.png (visualizations)")
    
    return 0


def main():
    """Main execution"""
    if len(sys.argv) < 2:
        print("Usage: python scripts/analizar_feature_importance.py <experiment_name>")
        print("\nExample:")
        print("  python scripts/analizar_feature_importance.py seg-001_zlgbm")
        return 1
    
    experiment_name = sys.argv[1]
    return analyze_feature_importance(experiment_name)


if __name__ == "__main__":
    sys.exit(main())

