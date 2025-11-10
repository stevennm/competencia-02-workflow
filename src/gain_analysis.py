"""
Gain analysis module
Creates gain curves to visualize expected gain per number of clients sent
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from typing import Dict
from pathlib import Path


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
    ganancia_acierto = 780000  # Gain for correctly predicting BAJA+2
    costo_estimulo = 20000     # Cost of sending stimulus
    
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
    
    # Create output directory (experiment-specific)
    experimento = config["experimento"]
    os.makedirs(f"output/{experimento}/analysis", exist_ok=True)
    
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
    output_file = f"output/{experimento}/analysis/gain_curve.png"
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
        df_stats.write_csv(f"output/{experimento}/analysis/gain_by_cutoff.csv")
        print(f"✓ Gain statistics saved to: output/{experimento}/analysis/gain_by_cutoff.csv")
        print("\nGain by cutoff:")
        print(df_stats)


def create_ensemble_gain_curve(df: pl.DataFrame, config: Dict) -> None:
    """
    Create gain curve showing all individual seed predictions and ensemble average
    Similar to professor's "Ganancia Acumulada (Semillas y Ensemble)" plot
    
    Args:
        df: Full DataFrame with clase_ternaria
        config: Configuration dictionary
    """
    print("\n" + "="*70)
    print("GENERATING ENSEMBLE GAIN CURVE")
    print("="*70)
    
    # Check if individual predictions file exists (experiment-specific path)
    experimento = config["experimento"]
    pred_file = Path(f"output/{experimento}/predicciones_ensemble.parquet")
    if not pred_file.exists():
        print("WARNING: Individual predictions file not found.")
        print(f"Run scoring first to generate output/{experimento}/predicciones_ensemble.parquet")
        return
    
    # Load individual predictions
    df_preds = pl.read_parquet(pred_file)
    print(f"Loaded predictions for {df_preds.shape[0]} clients")
    
    # Get future month and merge with actual labels
    future_months = config["train_final"]["future"]
    df_future = df.filter(pl.col("foto_mes").is_in(future_months))
    
    # Check if labels exist
    if df_future.select("clase_ternaria").null_count().item() == len(df_future):
        print("WARNING: Future month has no clase_ternaria labels.")
        print("Ensemble gain curve requires actual labels. Skipping...")
        return
    
    # Merge predictions with labels
    df_analysis = df_future.select(["numero_de_cliente", "foto_mes", "clase_ternaria"]).join(
        df_preds,
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
    ganancia_acierto = 780000
    costo_estimulo = 20000
    
    df_analysis = df_analysis.with_columns([
        pl.when(pl.col("y_test") == 1)
        .then(pl.lit(ganancia_acierto - costo_estimulo))
        .otherwise(pl.lit(-costo_estimulo))
        .alias("ganancia")
    ])
    
    # Get probability columns (all columns starting with prob_seed_)
    prob_cols = [col for col in df_analysis.columns if col.startswith("prob_seed_")]
    n_seeds = len(prob_cols)
    
    print(f"Found {n_seeds} individual seed predictions")
    
    if n_seeds == 0:
        print("ERROR: No seed predictions found!")
        return
    
    # Calculate gains for each seed
    ganancia = df_analysis.select("ganancia").to_numpy().ravel()
    all_gains = []
    
    for prob_col in prob_cols:
        # Sort by this seed's probability
        df_sorted = df_analysis.sort(prob_col, descending=True)
        gan_sorted = df_sorted.select("ganancia").to_numpy().ravel()
        gan_cum = np.cumsum(gan_sorted)
        all_gains.append(gan_cum)
    
    # Calculate ensemble average gain
    all_gains_array = np.array(all_gains)
    gan_ensemble = np.mean(all_gains_array, axis=0)
    
    # Find maximum
    max_idx = np.argmax(gan_ensemble)
    max_gain = gan_ensemble[max_idx]
    
    print(f"\nEnsemble Statistics:")
    print(f"  Maximum gain: ${max_gain:,.0f}")
    print(f"  Optimal cutoff: {max_idx} clients")
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Plot individual seeds with colormap
    colors = cm.rainbow(np.linspace(0, 1, n_seeds))
    x_range = range(len(gan_ensemble))
    
    # Plot all seeds with thin lines (no labels for cleaner legend)
    for i, (gan_cum, color) in enumerate(zip(all_gains, colors)):
        plt.plot(x_range, gan_cum, color=color, alpha=0.3, linewidth=0.8)
    
    # Add a dummy line for legend
    plt.plot([], [], color='gray', alpha=0.3, linewidth=0.8, 
            label=f'Individual Seeds (n={n_seeds})')
    
    # Plot ensemble average (thick black line)
    plt.plot(x_range, gan_ensemble, color='black', linewidth=3, 
            label='Ensemble (Average)', zorder=10)
    
    # Mark maximum
    plt.scatter([max_idx], [max_gain], color='red', s=200, zorder=11, marker='o')
    plt.annotate(f'Máximo\n${max_gain:,.0f}', 
                xy=(max_idx, max_gain),
                xytext=(20, 20), textcoords='offset points',
                fontsize=12, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red'),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # Formatting
    plt.xlabel('Clientes ordenados por probabilidad (índice / Envíos)', fontsize=12)
    plt.ylabel('Ganancia Acumulada', fontsize=12)
    plt.title(f'Ganancia Acumulada (Semillas y Ensemble) - EXP {experimento}', 
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Format y-axis with millions
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M' if x != 0 else '0'))
    
    # Legend
    plt.legend(loc='upper right', title='Modelo', fontsize=10)
    
    plt.tight_layout()
    
    # Save (experiment-specific path)
    output_file = f"output/{experimento}/analysis/gain_curve_ensemble.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Ensemble gain curve saved to: {output_file}")
    
    # Save ensemble statistics
    ensemble_stats = []
    for cutoff in [5000, 8000, 10000, 11000, 12000, 15000, 20000]:
        if cutoff < len(gan_ensemble):
            # Calculate statistics across all seeds at this cutoff
            gains_at_cutoff = [gains[cutoff] for gains in all_gains]
            ensemble_stats.append({
                "cutoff": cutoff,
                "ensemble_gain": gan_ensemble[cutoff],
                "min_gain": min(gains_at_cutoff),
                "max_gain": max(gains_at_cutoff),
                "std_gain": np.std(gains_at_cutoff)
            })
    
    if ensemble_stats:
        df_stats = pl.DataFrame(ensemble_stats)
        df_stats.write_csv(f"output/{experimento}/analysis/gain_ensemble_stats.csv")
        print(f"✓ Ensemble statistics saved to: output/{experimento}/analysis/gain_ensemble_stats.csv")
        print("\nEnsemble gain by cutoff:")
        print(df_stats)


def create_validation_gain_curve(pred_df: pl.DataFrame, config: Dict) -> None:
    """
    Create gain curve for validation set (Optuna test month)
    
    Args:
        pred_df: DataFrame with predictions from train_validation_ensemble
        config: Configuration dictionary
    """
    print("\n" + "="*70)
    print("GENERATING VALIDATION GAIN CURVE")
    print("="*70)
    
    experimento = config["experimento"]
    testing_months = config["trainingstrategy"]["testing"]
    
    print(f"Validation month(s): {testing_months}")
    print(f"Records: {pred_df.shape[0]}")
    
    # Process each validation month
    for validation_month in testing_months:
        print(f"\n--- Validation Month: {validation_month} ---")
        
        df_month = pred_df.filter(pl.col("foto_mes") == validation_month)
        
        if df_month.shape[0] == 0:
            print(f"No data for month {validation_month}, skipping...")
            continue
        
        # Add gan column
        df_month = df_month.with_columns([
            pl.lit(-20000.0).alias("gan")
        ])
        df_month = df_month.with_columns([
            pl.when(pl.col("clase_ternaria") == "BAJA+2")
            .then(pl.lit(780000.0))
            .otherwise(pl.col("gan"))
            .alias("gan")
        ])
        
        # Get prediction columns
        pred_cols = [col for col in df_month.columns if col.startswith("pred_seed_")]
        
        if not pred_cols:
            print("No individual predictions found, skipping...")
            continue
        
        print(f"Found {len(pred_cols)} model predictions")
        
        # Create gain curves for each model
        cutoffs = [7000, 8000, 9000, 10000, 11000, 12000, 13000]
        
        # Prepare plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        colors = cm.rainbow(np.linspace(0, 1, len(pred_cols)))
        
        # Store results for ensemble
        ensemble_results = {cutoff: [] for cutoff in cutoffs}
        
        # Plot individual models
        for idx, pred_col in enumerate(pred_cols):
            # Sort by prediction
            df_sorted = df_month.sort(pred_col, descending=True)
            gan_values = df_sorted.select("gan").to_numpy().ravel()
            
            # Calculate cumulative gain
            cumulative_gain = np.cumsum(gan_values)
            n_envios = np.arange(1, len(gan_values) + 1)
            
            # Plot full curve
            ax1.plot(n_envios, cumulative_gain, alpha=0.3, color=colors[idx], linewidth=0.8)
            
            # Calculate gains at cutoffs
            for cutoff in cutoffs:
                if cutoff <= len(gan_values):
                    gain_at_cutoff = cumulative_gain[cutoff - 1]
                    ensemble_results[cutoff].append(gain_at_cutoff)
        
        # Calculate and plot ensemble average
        avg_gains = []
        std_gains = []
        
        for cutoff in cutoffs:
            gains = ensemble_results[cutoff]
            avg_gains.append(np.mean(gains))
            std_gains.append(np.std(gains))
        
        # Plot ensemble average on left plot
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_xlabel('Number of Clients', fontsize=12)
        ax1.set_ylabel('Cumulative Gain ($)', fontsize=12)
        ax1.set_title(f'Individual Model Gains - Validation {validation_month}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.ticklabel_format(style='plain', axis='y')
        
        # Plot ensemble statistics on right plot
        ax2.errorbar(cutoffs, avg_gains, yerr=std_gains, marker='o', linewidth=2, 
                    markersize=8, capsize=5, capthick=2, color='darkblue', label='Ensemble Average ± Std')
        
        # Mark optimal cutoff
        optimal_idx = np.argmax(avg_gains)
        optimal_cutoff = cutoffs[optimal_idx]
        optimal_gain = avg_gains[optimal_idx]
        
        ax2.scatter([optimal_cutoff], [optimal_gain], color='red', s=200, zorder=5, 
                   marker='*', label=f'Optimal: {optimal_cutoff} envíos')
        
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Number of Clients', fontsize=12)
        ax2.set_ylabel('Expected Gain ($)', fontsize=12)
        ax2.set_title(f'Ensemble Gain - Validation {validation_month}', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        ax2.ticklabel_format(style='plain', axis='y')
        
        # Add text with optimal info
        textstr = f'Optimal Cutoff: {optimal_cutoff}\n'
        textstr += f'Expected Gain: ${optimal_gain:,.0f}\n'
        textstr += f'Std Dev: ${std_gains[optimal_idx]:,.0f}\n'
        textstr += f'Models: {len(pred_cols)}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(f"output/{experimento}/validation", exist_ok=True)
        plot_file = f"output/{experimento}/validation/gain_curve_{validation_month}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Gain curve saved to: {plot_file}")
        
        # Save statistics
        stats_data = {
            "cutoff": cutoffs,
            "avg_gain": avg_gains,
            "std_gain": std_gains,
            "n_models": [len(pred_cols)] * len(cutoffs)
        }
        
        df_stats = pl.DataFrame(stats_data)
        stats_file = f"output/{experimento}/validation/gain_stats_{validation_month}.csv"
        df_stats.write_csv(stats_file)
        print(f"✓ Statistics saved to: {stats_file}")
        
        # Print summary
        print(f"\n📊 Validation Gain Summary ({validation_month}):")
        print(f"   Optimal cutoff: {optimal_cutoff} envíos")
        print(f"   Expected gain: ${optimal_gain:,.0f}")
        print(f"   Std deviation: ${std_gains[optimal_idx]:,.0f}")
        print(f"   Models in ensemble: {len(pred_cols)}")
    
    print("\n" + "="*70)

