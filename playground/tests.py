# Read parquet files efficiently without loading all data
import polars as pl
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib_venn import venn2, venn3

# Files to compare
files = {
    "final_dataset": "data/final_dataset.parquet",
    "featured_data": "data/featured_data.parquet",
    "dataset_pequeno": "data/dataset_pequeno_2022_03.parquet"
}

# Read schemas
schemas = {}
for name, path in files.items():
    if os.path.exists(path):
        print("="*70)
        print(f"{name.upper()} - {path}")
        print("="*70)
        schema = pl.read_parquet_schema(path)
        schemas[name] = schema
        print(f"Total columns: {len(schema)}")
        print(f"Column names: {list(schema.keys())[:10]}... (showing first 10)")
    else:
        print(f"⚠ File not found: {path}")
        print()

# Compare columns between files
print("\n" + "="*70)
print("COLUMN COMPARISON")
print("="*70)

if len(schemas) >= 2:
    # Get column sets
    col_sets = {name: set(schema.keys()) for name, schema in schemas.items()}
    
    # Print summary
    print("\nColumn counts:")
    for name, cols in col_sets.items():
        print(f"  {name}: {len(cols)} columns")
    
    # Compare all pairs
    names = list(col_sets.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            name1, name2 = names[i], names[j]
            cols1, cols2 = col_sets[name1], col_sets[name2]
            
            print(f"\n{'='*70}")
            print(f"Comparing: {name1} vs {name2}")
            print(f"{'='*70}")
            
            # Columns only in first file
            only_1 = cols1 - cols2
            if only_1:
                print(f"\n✓ Columns ONLY in {name1} ({len(only_1)}):")
                for col in sorted(only_1)[:20]:  # Show first 20
                    print(f"  - {col}")
                if len(only_1) > 20:
                    print(f"  ... and {len(only_1) - 20} more")
            
            # Columns only in second file
            only_2 = cols2 - cols1
            if only_2:
                print(f"\n✓ Columns ONLY in {name2} ({len(only_2)}):")
                for col in sorted(only_2)[:20]:  # Show first 20
                    print(f"  - {col}")
                if len(only_2) > 20:
                    print(f"  ... and {len(only_2) - 20} more")
            
            # Common columns
            common = cols1 & cols2
            print(f"\n✓ Common columns: {len(common)}")
else:
    print("Need at least 2 files to compare")

# Create visualizations
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

if len(schemas) >= 2:
    names = list(col_sets.keys())
    
    # 1. Bar chart showing column counts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Chart 1: Total columns per file
    ax1 = axes[0, 0]
    counts = [len(cols) for cols in col_sets.values()]
    bars = ax1.bar(names, counts, color=['#3498db', '#e74c3c', '#2ecc71'][:len(names)])
    ax1.set_ylabel('Number of Columns')
    ax1.set_title('Total Columns per Dataset', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    # Chart 2: Venn diagram (if 2 files)
    if len(names) == 2:
        ax2 = axes[0, 1]
        name1, name2 = names[0], names[1]
        cols1, cols2 = col_sets[name1], col_sets[name2]
        
        venn2([cols1, cols2], set_labels=(name1, name2), ax=ax2)
        ax2.set_title(f'Column Overlap: {name1} vs {name2}', 
                     fontsize=14, fontweight='bold')
    
    # Chart 3: Stacked bar showing unique vs common columns
    ax3 = axes[1, 0]
    
    # Calculate unique and common for each pair
    if len(names) == 2:
        name1, name2 = names[0], names[1]
        cols1, cols2 = col_sets[name1], col_sets[name2]
        
        only_1 = len(cols1 - cols2)
        only_2 = len(cols2 - cols1)
        common = len(cols1 & cols2)
        
        categories = [name1, name2]
        unique_counts = [only_1, only_2]
        common_counts = [common, common]
        
        x = range(len(categories))
        width = 0.6
        
        p1 = ax3.bar(x, unique_counts, width, label='Unique columns', color='#e74c3c')
        p2 = ax3.bar(x, common_counts, width, bottom=unique_counts, 
                    label='Common columns', color='#2ecc71')
        
        ax3.set_ylabel('Number of Columns')
        ax3.set_title('Column Distribution: Unique vs Common', 
                     fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (u, c) in enumerate(zip(unique_counts, common_counts)):
            ax3.text(i, u/2, str(u), ha='center', va='center', 
                    fontweight='bold', color='white')
            ax3.text(i, u + c/2, str(c), ha='center', va='center', 
                    fontweight='bold', color='white')
    
    # Chart 4: Feature categories breakdown
    ax4 = axes[1, 1]
    
    # Categorize columns by prefix/pattern
    def categorize_columns(cols):
        categories = {
            'Original': 0,
            'Lag features': 0,
            'Delta features': 0,
            'Trend features': 0,
            'RF features': 0,
            'Other': 0
        }
        
        for col in cols:
            col_lower = col.lower()
            if col_lower.startswith('rf_'):
                categories['RF features'] += 1
            elif '_lag' in col_lower or col_lower.startswith('lag'):
                categories['Lag features'] += 1
            elif '_delta' in col_lower or col_lower.startswith('delta'):
                categories['Delta features'] += 1
            elif '_trend' in col_lower or col_lower.startswith('trend'):
                categories['Trend features'] += 1
            elif col in ['numero_de_cliente', 'foto_mes', 'clase_ternaria']:
                continue  # Skip metadata
            else:
                categories['Original'] += 1
        
        return categories
    
    # Get categories for final_dataset (or first available)
    target_file = 'final_dataset' if 'final_dataset' in col_sets else names[0]
    categories = categorize_columns(col_sets[target_file])
    
    # Remove zero categories
    categories = {k: v for k, v in categories.items() if v > 0}
    
    colors_cat = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#95a5a6']
    wedges, texts, autotexts = ax4.pie(categories.values(), 
                                        labels=categories.keys(),
                                        autopct='%1.1f%%',
                                        colors=colors_cat[:len(categories)],
                                        startangle=90)
    
    ax4.set_title(f'Feature Categories in {target_file}', 
                 fontsize=14, fontweight='bold')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'playground/column_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    plt.show()
    
    # Print feature category breakdown
    print("\n" + "="*70)
    print(f"FEATURE CATEGORIES IN {target_file.upper()}")
    print("="*70)
    for cat, count in categories.items():
        print(f"  {cat:20s}: {count:4d} features")
    print(f"  {'TOTAL':20s}: {sum(categories.values()):4d} features")
    
else:
    print("Need at least 2 files to create visualizations")