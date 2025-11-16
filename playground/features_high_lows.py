import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# load data
df = pl.read_parquet("data/competencia_02_target.parquet")

print(f"\nDataset shape: {df.shape}")
print(f"Total columns: {len(df.columns)}")

# Select random features to analyze (excluding foto_mes and identifiers)
exclude_cols = ['foto_mes', 'numero_de_cliente', 'clase_ternaria']
numeric_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]]

# Get all features and sort them alphabetically
sample_features = sorted(numeric_cols)
n_features = len(sample_features)

print(f"\nAnalyzing {n_features} numeric features for mean value drops")
print(f"Looking for anomalies in foto_mes 202006...")

# Calculate mean values per foto_mes for each feature
mean_data = []
for feature in sample_features:
    means_by_month = (
        df.group_by("foto_mes")
        .agg([
            pl.col(feature).mean().alias("mean_value"),
            pl.col(feature).is_null().sum().alias("null_count"),
            pl.len().alias("total_count")
        ])
        .sort("foto_mes")
    )
    
    for row in means_by_month.iter_rows(named=True):
        mean_data.append({
            "foto_mes": row["foto_mes"],
            "feature": feature,
            "mean_value": row["mean_value"] if row["mean_value"] is not None else 0,
            "null_pct": (row["null_count"] / row["total_count"] * 100) if row["total_count"] > 0 else 0
        })

mean_df = pl.DataFrame(mean_data)

# Find features that drop significantly in 202006
foto_mes_list = sorted(df["foto_mes"].unique().to_list())

anomalies = []
for feature in sample_features:
    feature_data = mean_df.filter(pl.col("feature") == feature).sort("foto_mes")
    means = feature_data["mean_value"].to_numpy()
    
    # Calculate overall mean excluding 202006
    other_months_mask = feature_data["foto_mes"].to_numpy() != 202006
    if other_months_mask.sum() > 0:
        mean_other = means[other_months_mask].mean()
        
        # Check 202006 specifically
        june_mask = feature_data["foto_mes"].to_numpy() == 202006
        if june_mask.sum() > 0:
            mean_june = means[june_mask][0]
            
            # Detect if June drops significantly (>80% drop)
            if mean_other > 0 and mean_june < mean_other * 0.2:
                drop_pct = ((mean_other - mean_june) / mean_other) * 100
                anomalies.append({
                    "feature": feature,
                    "mean_june": mean_june,
                    "mean_other": mean_other,
                    "drop_pct": drop_pct
                })

if anomalies:
    print(f"\n⚠️  Found {len(anomalies)} features with significant drops in 202006:")
    anomaly_df = pl.DataFrame(anomalies).sort("drop_pct", descending=True)
    print(anomaly_df.head(20))
else:
    print("\n✓ No significant feature drops detected in 202006")

# Create heatmap of mean values (normalized per feature)
pivot_data = np.zeros((len(sample_features), len(foto_mes_list)))

for i, feature in enumerate(sample_features):
    feature_data = mean_df.filter(pl.col("feature") == feature).sort("foto_mes")
    means = feature_data["mean_value"].to_numpy()
    
    # Normalize to 0-100 scale per feature (to compare relative drops)
    if means.max() > 0:
        pivot_data[i, :] = (means / means.max()) * 100
    else:
        pivot_data[i, :] = 0

# ============= CREATE HTML OUTPUT =============
print("\n📊 Creating interactive HTML heatmap...")

html_output = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Feature Mean Values Heatmap - foto_mes Analysis</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 100%;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 20px;
        }
        .controls {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .control-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        input[type="range"] {
            width: 300px;
        }
        #heatmap-container {
            overflow-x: auto;
            overflow-y: hidden;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        table {
            border-collapse: collapse;
            font-size: 11px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 4px 8px;
            text-align: center;
        }
        th {
            background: #2c3e50;
            color: white;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        .feature-name {
            text-align: left;
            background: #ecf0f1;
            font-weight: bold;
            position: sticky;
            left: 0;
            z-index: 5;
        }
        .highlight-202006 {
            background-color: rgba(52, 152, 219, 0.2) !important;
            border-left: 3px solid #3498db;
            border-right: 3px solid #3498db;
        }
        .cell {
            cursor: pointer;
            transition: all 0.2s;
        }
        .cell:hover {
            transform: scale(1.1);
            box-shadow: 0 0 5px rgba(0,0,0,0.3);
            z-index: 100;
        }
        .tooltip {
            position: fixed;
            background: rgba(0,0,0,0.9);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
            display: none;
        }
        .legend {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .legend-box {
            width: 30px;
            height: 20px;
            border: 1px solid #333;
        }
        .anomaly-list {
            margin-top: 20px;
            padding: 15px;
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            border-radius: 5px;
        }
        .anomaly-list h3 {
            margin-top: 0;
            color: #856404;
        }
        .anomaly-item {
            padding: 5px 0;
            font-size: 13px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔥 Feature Mean Values Heatmap by foto_mes</h1>
        <div class="subtitle">
            Normalized per feature (Red = Low values, Green = High values)<br>
            Blue column highlights foto_mes 202006
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label for="scroll-slider">Scroll Features:</label>
                <input type="range" id="scroll-slider" min="0" max="MAX_SCROLL" value="0" step="10">
                <span id="range-display">Showing 1-50 of TOTAL_FEATURES</span>
            </div>
            <div class="control-group">
                <button onclick="scrollFeatures(-50)">⬆️ Up 50</button>
                <button onclick="scrollFeatures(50)">⬇️ Down 50</button>
            </div>
        </div>
"""

# Add anomaly list if any
if anomalies:
    html_output += """
        <div class="anomaly-list">
            <h3>⚠️ Features with Significant Drops in 202006 (>80% drop)</h3>
"""
    for i, anom in enumerate(sorted(anomalies, key=lambda x: x['drop_pct'], reverse=True)[:20]):
        html_output += f"""
            <div class="anomaly-item">
                {i+1}. <strong>{anom['feature']}</strong>: {anom['drop_pct']:.1f}% drop 
                (avg other months: {anom['mean_other']:.2f}, June 2020: {anom['mean_june']:.2f})
            </div>
"""
    html_output += """
        </div>
"""

html_output += """
        <div id="heatmap-container">
            <table id="heatmap-table">
                <thead>
                    <tr>
                        <th class="feature-name">Feature</th>
"""

# Add foto_mes headers
for fm in foto_mes_list:
    css_class = 'highlight-202006' if fm == 202006 else ''
    html_output += f'                        <th class="{css_class}">{fm}</th>\n'

html_output += """
                    </tr>
                </thead>
                <tbody id="heatmap-body">
                </tbody>
            </table>
        </div>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-box" style="background: rgb(0, 104, 55);"></div>
                <span>100% (Max)</span>
            </div>
            <div class="legend-item">
                <div class="legend-box" style="background: rgb(255, 255, 191);"></div>
                <span>50%</span>
            </div>
            <div class="legend-item">
                <div class="legend-box" style="background: rgb(165, 0, 38);"></div>
                <span>0% (Min)</span>
            </div>
            <div class="legend-item" style="margin-left: 20px;">
                <div style="width: 30px; height: 20px; background: rgba(52, 152, 219, 0.3); border: 2px solid #3498db;"></div>
                <span>202006</span>
            </div>
        </div>
        
        <div class="tooltip" id="tooltip"></div>
    </div>
    
    <script>
        // Data embedded from Python
        const features = FEATURES_JSON;
        const pivotData = PIVOT_DATA_JSON;
        const fotoMesList = FOTO_MES_JSON;
        
        const featuresPerView = 50;
        let currentStart = 0;
        
        function getColor(value) {
            // Red-Yellow-Green color scale
            if (value >= 50) {
                // Green side (50-100)
                const t = (value - 50) / 50;
                const r = Math.round(255 * (1 - t));
                const g = Math.round(104 + 151 * t);
                const b = Math.round(191 * (1 - t) + 55 * t);
                return `rgb(${r}, ${g}, ${b})`;
            } else {
                // Red side (0-50)
                const t = value / 50;
                const r = Math.round(165 + 90 * t);
                const g = Math.round(255 * t);
                const b = Math.round(38 + 153 * t);
                return `rgb(${r}, ${g}, ${b})`;
            }
        }
        
        function updateHeatmap(startIdx) {
            const tbody = document.getElementById('heatmap-body');
            tbody.innerHTML = '';
            
            const endIdx = Math.min(startIdx + featuresPerView, features.length);
            
            for (let i = startIdx; i < endIdx; i++) {
                const row = document.createElement('tr');
                
                // Feature name cell
                const nameCell = document.createElement('td');
                nameCell.className = 'feature-name';
                nameCell.textContent = features[i];
                row.appendChild(nameCell);
                
                // Data cells
                for (let j = 0; j < fotoMesList.length; j++) {
                    const cell = document.createElement('td');
                    const value = pivotData[i][j];
                    cell.className = 'cell';
                    if (fotoMesList[j] === 202006) {
                        cell.classList.add('highlight-202006');
                    }
                    cell.style.backgroundColor = getColor(value);
                    cell.textContent = value.toFixed(0);
                    
                    // Tooltip
                    cell.addEventListener('mouseenter', (e) => {
                        const tooltip = document.getElementById('tooltip');
                        tooltip.style.display = 'block';
                        tooltip.innerHTML = `
                            <strong>${features[i]}</strong><br>
                            foto_mes: ${fotoMesList[j]}<br>
                            Value: ${value.toFixed(2)}%
                        `;
                    });
                    
                    cell.addEventListener('mousemove', (e) => {
                        const tooltip = document.getElementById('tooltip');
                        tooltip.style.left = (e.clientX + 10) + 'px';
                        tooltip.style.top = (e.clientY + 10) + 'px';
                    });
                    
                    cell.addEventListener('mouseleave', () => {
                        document.getElementById('tooltip').style.display = 'none';
                    });
                    
                    row.appendChild(cell);
                }
                
                tbody.appendChild(row);
            }
            
            document.getElementById('range-display').textContent = 
                `Showing ${startIdx + 1}-${endIdx} of ${features.length}`;
        }
        
        function scrollFeatures(delta) {
            const slider = document.getElementById('scroll-slider');
            const newVal = Math.max(0, Math.min(parseInt(slider.max), currentStart + delta));
            slider.value = newVal;
            currentStart = newVal;
            updateHeatmap(currentStart);
        }
        
        document.getElementById('scroll-slider').addEventListener('input', (e) => {
            currentStart = parseInt(e.target.value);
            updateHeatmap(currentStart);
        });
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowUp') {
                e.preventDefault();
                scrollFeatures(-10);
            } else if (e.key === 'ArrowDown') {
                e.preventDefault();
                scrollFeatures(10);
            }
        });
        
        // Initial render
        updateHeatmap(0);
    </script>
</body>
</html>
"""

# Prepare data for JSON
features_json = json.dumps(sample_features)
pivot_data_json = json.dumps(pivot_data.tolist())
foto_mes_json = json.dumps(foto_mes_list)

# Replace placeholders
html_output = html_output.replace('FEATURES_JSON', features_json)
html_output = html_output.replace('PIVOT_DATA_JSON', pivot_data_json)
html_output = html_output.replace('FOTO_MES_JSON', foto_mes_json)
html_output = html_output.replace('TOTAL_FEATURES', str(len(sample_features)))
html_output = html_output.replace('MAX_SCROLL', str(max(0, len(sample_features) - 50)))

# Save HTML file
output_path = Path('playground/feature_heatmap.html')
output_path.write_text(html_output, encoding='utf-8')

print(f"✅ HTML heatmap saved to: {output_path.absolute()}")
print(f"   Open it in your browser to view the interactive heatmap!")

# ============= END HTML OUTPUT =============

# Create scrollable visualization
from matplotlib.widgets import Slider

# Make figure with room for slider
fig = plt.figure(figsize=(18, 10))
ax = plt.subplot(111)

# Calculate how many features to show at once
features_per_view = 50
current_start = [0]  # Use list to make it mutable in nested function

def update_plot(start_idx):
    ax.clear()
    
    end_idx = min(start_idx + features_per_view, len(sample_features))
    visible_data = pivot_data[start_idx:end_idx, :]
    visible_features = sample_features[start_idx:end_idx]
    
    im = ax.imshow(visible_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    
    # Highlight 202006 column
    june_idx = foto_mes_list.index(202006) if 202006 in foto_mes_list else -1
    if june_idx >= 0:
        ax.axvline(x=june_idx, color='blue', linestyle='--', linewidth=3, label='202006')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(foto_mes_list)))
    ax.set_yticks(np.arange(len(visible_features)))
    ax.set_xticklabels(foto_mes_list, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(visible_features, fontsize=8)
    
    # Add title and labels
    ax.set_title(f'Feature Mean Values Heatmap by foto_mes (Normalized per feature)\n(Red = Low values, Green = High values, Blue line = 202006)\nShowing features {start_idx+1}-{end_idx} of {len(sample_features)}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('foto_mes', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    
    # Add grid
    ax.set_xticks(np.arange(len(foto_mes_list)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(visible_features)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    if june_idx >= 0:
        ax.legend(loc='upper right')
    
    # Add colorbar if it doesn't exist
    if not hasattr(fig, 'colorbar_added'):
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Mean Value (% of max)', rotation=270, labelpad=20)
        fig.colorbar_added = True
    
    fig.canvas.draw_idle()

# Initial plot
update_plot(0)

# Add slider for scrolling if needed
if len(sample_features) > features_per_view:
    slider_ax = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(
        slider_ax, 
        'Scroll Features', 
        0, 
        len(sample_features) - features_per_view, 
        valinit=0, 
        valstep=10
    )
    
    def on_slider_change(val):
        update_plot(int(val))
    
    slider.on_changed(on_slider_change)
    
    # Add keyboard scrolling
    def on_key(event):
        if event.key == 'up':
            new_val = max(0, slider.val - 10)
            slider.set_val(new_val)
        elif event.key == 'down':
            new_val = min(len(sample_features) - features_per_view, slider.val + 10)
            slider.set_val(new_val)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    print("\n💡 Use the slider or UP/DOWN arrow keys to scroll through features")

plt.tight_layout()
plt.subplots_adjust(bottom=0.08)
plt.show()