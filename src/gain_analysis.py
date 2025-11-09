"""
Gain analysis module
Creates gain curves to visualize expected gain per number of clients sent
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict


def create_gain_curve(df: pl.DataFrame, df_pred: pl.DataFrame, config: Dict) -> None:
    """
    Create gain curve showing expected gain per number of clients
    
    This only works when the future month has actual clase_ternaria labels
    (i.e., not 202107 or 202108 which are truly future)
    
    Args:
        df: Full DataFrame with clase_ternaria
        df_pred: Prediction DataFrame with probabilities
        config: Configuration dictionary
    """
    print("\n" + "="*70)
    print("GENERATING GAIN CURVE ANALYSIS")
    print("="*70)
    
    future_months = config["train_final"]["future"]
    
    # Check if future months have clase_ternaria
    df_future = df.filter(pl.col("foto_mes").is_in(future_months))
    
    if df_future.select("clase_ternaria").null_count().item() == len(df_future):
        print("WARNING: Future month has no clase_ternaria labels.")
        print("Gain curve analysis requires actual labels. Skipping...")
        return
    
    # Check if any clase_ternaria is missing
    has_nulls = df_future.filter(pl.col("clase_ternaria").is_null()).shape[0] > 0
    if has_nulls:
        print("WARNING: Some records in future month lack clase_ternaria labels.")
        print("Gain curve analysis skipped.")
        return
    
    print(f"Future month(s): {future_months}")
    print(f"Records with labels: {df_future.shape[0]}")
    
    # Merge predictions with actual labels
    df_analysis = df_future.select(["numero_de_cliente", "foto_mes", "clase_ternaria"]).join(
        df_pred.select(["numero_de_cliente", "foto_mes", "prob"]),
        on=["numero_de_cliente", "foto_mes"],
        how="inner"
    )
    
    print(f"Matched records: {df_analysis.shape[0]}")
    
    # Create binary target and gains
    df_analysis = df_analysis.with_columns([
        pl.when(pl.col("clase_ternaria") == "BAJA+2")
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("y_test")
    ])
    
    # Calculate gain per client
    ganancia_acierto = 78000  # Gain for correctly predicting BAJA+2
    costo_estimulo = 2000     # Cost of sending stimulus
    
    df_analysis = df_analysis.with_columns([
        pl.when(pl.col("y_test") == 1)
        .then(pl.lit(ganancia_acierto - costo_estimulo))
        .otherwise(pl.lit(-costo_estimulo))
        .alias("ganancia")
    ])
    
    # Sort by probability descending
    df_analysis = df_analysis.sort("prob", descending=True)
    
    # Calculate cumulative gain
    ganancia = df_analysis.select("ganancia").to_numpy().ravel()
    prob = df_analysis.select("prob").to_numpy().ravel()
    ganancia_cum = np.cumsum(ganancia)
    
    # Find maximum gain and its position
    gan_max_idx = np.argmax(ganancia_cum)
    ganancia_max = ganancia_cum[gan_max_idx]
    
    print(f"\nGain Statistics:")
    print(f"  Maximum gain: ${ganancia_max:,.0f}")
    print(f"  Optimal cutoff: {gan_max_idx} clients")
    print(f"  Probability at cutoff: {prob[gan_max_idx]:.6f}")
    
    # Create output directory
    os.makedirs("output/analysis", exist_ok=True)
    
    # Plot 1: Gain vs Number of Clients (full range)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(len(ganancia_cum)), ganancia_cum, linewidth=2, color='steelblue')
    plt.axvline(x=gan_max_idx, color='green', linestyle='--', 
                label=f'Optimal cutoff: {gan_max_idx} clients')
    plt.axhline(y=ganancia_max, color='red', linestyle='--', 
                label=f'Max gain: ${ganancia_max:,.0f}')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.title('Cumulative Gain Curve - Full Range', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Clients Sent', fontsize=10)
    plt.ylabel('Cumulative Gain ($)', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Gain vs Number of Clients (zoomed to relevant range)
    piso_envios = max(0, gan_max_idx - 5000)
    techo_envios = min(len(ganancia_cum), gan_max_idx + 10000)
    
    plt.subplot(1, 2, 2)
    x_range = range(piso_envios, techo_envios)
    plt.plot(x_range, ganancia_cum[piso_envios:techo_envios], 
             linewidth=2, color='steelblue', label='LGBM Gain')
    plt.axvline(x=gan_max_idx, color='green', linestyle='--', 
                label=f'Optimal: {gan_max_idx}')
    plt.axhline(y=ganancia_max, color='red', linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Mark common cutoff points
    for cutoff in [8000, 10000, 11000, 12000, 15000]:
        if piso_envios <= cutoff < techo_envios:
            gain_at_cutoff = ganancia_cum[cutoff]
            plt.scatter([cutoff], [gain_at_cutoff], s=50, zorder=5)
            plt.annotate(f'{cutoff}\n${gain_at_cutoff:,.0f}', 
                        xy=(cutoff, gain_at_cutoff),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=8, alpha=0.7)
    
    plt.title('Cumulative Gain Curve - Detail', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Clients Sent', fontsize=10)
    plt.ylabel('Cumulative Gain ($)', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    output_file = "output/analysis/gain_curve.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Gain curve saved to: {output_file}")
    
    # Save gain statistics to CSV
    gain_stats = []
    for cutoff in [5000, 8000, 10000, 11000, 12000, 15000, 20000]:
        if cutoff < len(ganancia_cum):
            gain_stats.append({
                "cutoff": cutoff,
                "cumulative_gain": ganancia_cum[cutoff],
                "probability": prob[cutoff]
            })
    
    if gain_stats:
        df_stats = pl.DataFrame(gain_stats)
        df_stats.write_csv("output/analysis/gain_by_cutoff.csv")
        print(f"✓ Gain statistics saved to: output/analysis/gain_by_cutoff.csv")
        print("\nGain by cutoff:")
        print(df_stats)

