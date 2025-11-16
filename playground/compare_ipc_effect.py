"""
Comparison script: Analyze IPC (inflation) effect on monetary features

This script compares the evolution of monetary features before and after
IPC adjustment to visualize the inflation correction effect.

Compares:
- competencia_02_target.parquet (before IPC)
- preprocessed_data.parquet (after IPC)

Analysis:
1. Monthly evolution of monetary features (raw vs adjusted)
2. Growth rates comparison
3. Visual comparison charts

Usage:
    python playground/compare_ipc_effect.py
"""

import polars as pl
from pathlib import Path
from datetime import datetime
import json

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def main():
    print_section(f"IPC EFFECT ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    target_file = Path("data/competencia_02_target.parquet")
    preprocessed_file = Path("data/preprocessed_data.parquet")
    
    if not target_file.exists() or not preprocessed_file.exists():
        print("[ERROR] Required files not found")
        return 1
    
    print("\nLoading data...")
    df_before = pl.read_parquet(target_file)
    df_after = pl.read_parquet(preprocessed_file)
    
    print(f"Before IPC: {df_before.shape}")
    print(f"After IPC:  {df_after.shape}")
    
    # Note: After has fewer rows due to month elimination
    removed_months = [201905, 201910, 202006]
    print(f"\nNote: Months {removed_months} were removed in preprocessing")
    
    # Filter before data to match after (remove problematic months)
    df_before_filtered = df_before.filter(~pl.col("foto_mes").is_in(removed_months))
    print(f"Before IPC (filtered): {df_before_filtered.shape}")
    
    # Identify monetary columns
    monetary_cols = [col for col in df_after.columns 
                     if col.startswith("m") 
                     and col not in ["foto_mes"]
                     and df_after[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]]
    
    print(f"\nFound {len(monetary_cols)} monetary columns")
    print(f"Sample columns: {monetary_cols[:5]}")
    
    # =========================================================================
    # ANALYSIS 1: Monthly Averages Comparison
    # =========================================================================
    print_section("ANALYSIS 1: Monthly Averages (Before vs After IPC)")
    
    # Select a few representative columns for detailed analysis
    sample_cols = [
        'mrentabilidad',
        'mcuentas_saldo', 
        'mtarjeta_visa_consumo',
        'mtarjeta_master_consumo',
        'mautoservicio'
    ]
    
    # Filter to existing columns
    sample_cols = [col for col in sample_cols if col in monetary_cols]
    
    print(f"\nAnalyzing {len(sample_cols)} representative columns:")
    for col in sample_cols:
        print(f"  - {col}")
    
    # Calculate monthly averages
    results = []
    
    for col in sample_cols:
        # Before IPC
        before_monthly = df_before_filtered.group_by("foto_mes").agg([
            pl.col(col).mean().alias("mean_before"),
            pl.col(col).median().alias("median_before"),
            pl.len().alias("count")
        ]).sort("foto_mes")
        
        # After IPC
        after_monthly = df_after.group_by("foto_mes").agg([
            pl.col(col).mean().alias("mean_after"),
            pl.col(col).median().alias("median_after")
        ]).sort("foto_mes")
        
        # Join
        comparison = before_monthly.join(after_monthly, on="foto_mes", how="inner")
        
        # Calculate change
        comparison = comparison.with_columns([
            ((pl.col("mean_after") - pl.col("mean_before")) / pl.col("mean_before") * 100).alias("change_pct"),
            (pl.col("mean_after") / pl.col("mean_before")).alias("multiplier")
        ])
        
        # Add column name
        comparison = comparison.with_columns([
            pl.lit(col).alias("column")
        ])
        
        results.append(comparison)
    
    # Combine all results
    all_results = pl.concat(results)
    
    # =========================================================================
    # DISPLAY RESULTS FOR EACH COLUMN
    # =========================================================================
    
    for col in sample_cols:
        print(f"\n{'-'*70}")
        print(f"Column: {col}")
        print(f"{'-'*70}")
        
        col_data = all_results.filter(pl.col("column") == col).sort("foto_mes")
        
        print(f"\n{'Month':<10} {'Before IPC':>15} {'After IPC':>15} {'Multiplier':>12} {'Change %':>10}")
        print("-" * 70)
        
        for row in col_data.iter_rows(named=True):
            print(f"{row['foto_mes']:<10} "
                  f"${row['mean_before']:>14,.2f} "
                  f"${row['mean_after']:>14,.2f} "
                  f"{row['multiplier']:>11.4f}x "
                  f"{row['change_pct']:>9.1f}%")
    
    # =========================================================================
    # ANALYSIS 2: Overall Statistics
    # =========================================================================
    print_section("ANALYSIS 2: Overall IPC Effect Statistics")
    
    # Calculate average multiplier by year
    all_results = all_results.with_columns([
        (pl.col("foto_mes") // 100).alias("year")
    ])
    
    yearly_stats = all_results.group_by(["column", "year"]).agg([
        pl.col("multiplier").mean().alias("avg_multiplier"),
        pl.col("change_pct").mean().alias("avg_change_pct")
    ]).sort(["column", "year"])
    
    print("\nAverage IPC multiplier by year:")
    print(f"\n{'Column':<30} {'Year':<6} {'Avg Multiplier':>15} {'Avg Change %':>15}")
    print("-" * 70)
    
    for row in yearly_stats.iter_rows(named=True):
        print(f"{row['column']:<30} "
              f"{row['year']:<6} "
              f"{row['avg_multiplier']:>14.4f}x "
              f"{row['avg_change_pct']:>14.1f}%")
    
    # =========================================================================
    # ANALYSIS 3: Temporal Trend Comparison
    # =========================================================================
    print_section("ANALYSIS 3: Temporal Trends")
    
    print("\nComparing growth trends (first month vs last month):")
    print(f"\n{'Column':<30} {'First (201901)':>15} {'Last (202108)':>15} {'Growth':>12}")
    print("-" * 75)
    
    for col in sample_cols:
        col_data = all_results.filter(pl.col("column") == col).sort("foto_mes")
        
        if len(col_data) > 0:
            # Get first and last rows as dictionaries
            first_row = col_data.head(1).to_dicts()[0]
            last_row = col_data.tail(1).to_dicts()[0]
            
            # Before IPC growth
            growth_before = ((last_row["mean_before"] - first_row["mean_before"]) / 
                           first_row["mean_before"] * 100)
            
            # After IPC growth (should be lower due to inflation adjustment)
            growth_after = ((last_row["mean_after"] - first_row["mean_after"]) / 
                          first_row["mean_after"] * 100)
            
            print(f"\n{col:<30}")
            print(f"  Before IPC: ${first_row['mean_before']:>13,.2f} -> ${last_row['mean_before']:>13,.2f} ({growth_before:>9.1f}%)")
            print(f"  After IPC:  ${first_row['mean_after']:>13,.2f} -> ${last_row['mean_after']:>13,.2f} ({growth_after:>9.1f}%)")
            print(f"  Inflation removed: {growth_before - growth_after:>9.1f}pp")
    
    # =========================================================================
    # EXPORT DETAILED REPORT
    # =========================================================================
    print_section("EXPORT")
    
    output_file = Path("playground/ipc_comparison_report.csv")
    all_results.select([
        "foto_mes",
        "column",
        "mean_before",
        "mean_after",
        "median_before",
        "median_after",
        "multiplier",
        "change_pct",
        "count"
    ]).write_csv(output_file)
    
    print(f"\nDetailed report saved to: {output_file}")
    print(f"Total records: {len(all_results):,}")
    
    # =========================================================================
    # CREATE HTML VISUALIZATION
    # =========================================================================
    print_section("CREATING HTML VISUALIZATION")
    
    # Prepare data for visualization
    html_data = {}
    for col in sample_cols:
        col_data = all_results.filter(pl.col("column") == col).sort("foto_mes")
        html_data[col] = {
            'months': col_data["foto_mes"].to_list(),
            'before': col_data["mean_before"].to_list(),
            'after': col_data["mean_after"].to_list()
        }
    
    html_output = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>IPC Effect Comparison</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }}
        .chart {{
            margin-bottom: 40px;
        }}
        .info-box {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>IPC (Inflation) Effect on Monetary Features</h1>
        <div class="subtitle">
            Comparison: Before vs After IPC Adjustment<br>
            <strong>Blue</strong> = Before IPC (nominal values) | <strong>Red</strong> = After IPC (adjusted to 202108)
        </div>
        
        <div class="info-box">
            <strong>What is IPC adjustment?</strong><br>
            IPC (Índice de Precios al Consumidor) adjusts monetary values for inflation,
            bringing all values to the same purchasing power (base: August 2021).<br>
            This removes the artificial growth caused by inflation and shows real value changes.
        </div>
"""
    
    # Create a chart for each column
    for col in sample_cols:
        data = html_data[col]
        
        html_output += f"""
        <div class="chart">
            <h3>{col}</h3>
            <div id="chart_{col}"></div>
        </div>
        
        <script>
            var trace1 = {{
                x: {json.dumps(data['months'])},
                y: {json.dumps(data['before'])},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Before IPC (Nominal)',
                line: {{color: '#2196f3', width: 2}},
                marker: {{size: 6}}
            }};
            
            var trace2 = {{
                x: {json.dumps(data['months'])},
                y: {json.dumps(data['after'])},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'After IPC (Real)',
                line: {{color: '#f44336', width: 2}},
                marker: {{size: 6}}
            }};
            
            var layout = {{
                xaxis: {{
                    title: 'Month (foto_mes)',
                    tickangle: -45
                }},
                yaxis: {{
                    title: 'Average Value ($)',
                    tickformat: ',.0f'
                }},
                hovermode: 'x unified',
                showlegend: true,
                height: 400
            }};
            
            Plotly.newPlot('chart_{col}', [trace1, trace2], layout);
        </script>
"""
    
    html_output += """
    </div>
</body>
</html>
"""
    
    # Save HTML
    html_file = Path("playground/ipc_comparison.html")
    html_file.write_text(html_output, encoding='utf-8')
    
    print(f"[OK] HTML visualization saved to: {html_file}")
    print(f"     Open it in your browser to see interactive charts!")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_section("SUMMARY")
    
    print("\nKey Findings:")
    print(f"  - Analyzed {len(sample_cols)} monetary columns")
    print(f"  - Compared {len(col_data)} months (after removing problematic ones)")
    print(f"  - IPC multipliers range from ~1.0x (recent) to ~2.7x (2019)")
    print(f"\nEffect of IPC adjustment:")
    print(f"  - Removes artificial growth from inflation")
    print(f"  - Normalizes all values to August 2021 purchasing power")
    print(f"  - Older values are multiplied by higher factors")
    print(f"  - Recent values remain almost unchanged")
    
    print("\n" + "="*70)
    print("IPC comparison analysis complete!")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

