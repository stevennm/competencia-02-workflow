"""
Run complete preprocessing pipeline
Executes all preprocessing steps in sequence:
1. Generate clase_ternaria
2. Preprocessing (eliminate features, MICE, IPC)
3. Feature Engineering (intra-month, lags, deltas, trends)
"""

import sys
import time
from datetime import datetime
import subprocess


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def run_script(script_name: str, step_number: int, total_steps: int) -> bool:
    """
    Run a preprocessing script
    
    Args:
        script_name: Name of the script to run
        step_number: Current step number
        total_steps: Total number of steps
        
    Returns:
        True if successful, False otherwise
    """
    print_section(f"STEP {step_number}/{total_steps}: {script_name}")
    
    try:
        # Run the script using current Python interpreter
        result = subprocess.run(
            [sys.executable, f"preprocessing/{script_name}"],
            check=True,
            capture_output=False,
            text=True
        )
        
        print(f"\n✓ {script_name} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {script_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ Error running {script_name}: {str(e)}")
        return False


def main():
    """Main execution"""
    start_time = time.time()
    print_section(f"COMPLETE PREPROCESSING PIPELINE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nThis will execute the complete preprocessing pipeline:")
    print("  1. Generate clase_ternaria from raw data")
    print("  2. Preprocessing (eliminate features, MICE, IPC)")
    print("  3. Feature Engineering (lags, deltas, trends)")
    
    # Define pipeline steps
    steps = [
        "01_generate_clase_ternaria.py",
        "02_preprocessing.py",
        "03_feature_engineering.py"
    ]
    
    # Execute each step
    for i, script in enumerate(steps, 1):
        success = run_script(script, i, len(steps))
        
        if not success:
            print_section("PIPELINE FAILED")
            print(f"\nPipeline stopped at step {i}: {script}")
            return 1
    
    # Completion
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print_section("COMPLETE PIPELINE FINISHED SUCCESSFULLY")
    print(f"\nTotal execution time: {minutes:02d}m {seconds:02d}s")
    print("\nOutput files created:")
    print("  - data/competencia_02_target.parquet (with clase_ternaria)")
    print("  - data/preprocessed_data.parquet (cleaned data)")
    print("  - data/featured_data.parquet (with all features)")
    print("\nNext step: Use featured_data.parquet for training")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

