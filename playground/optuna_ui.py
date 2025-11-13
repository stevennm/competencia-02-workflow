"""
Script to launch Optuna Dashboard UI
Loads the experiment from the shared database
"""

import sys
import os
import webbrowser
import time
import threading

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PARAM

def main():
    """Launch Optuna Dashboard"""
    
    # Database configuration
    db_path = "db/optuna.db"
    experimento = PARAM["experimento"]
    study_name = f"lgbm_{experimento}"
    
    print("="*70)
    print("  OPTUNA DASHBOARD")
    print("="*70)
    print(f"\nExperiment: {experimento}")
    print(f"Study name: {study_name}")
    print(f"Database: {db_path}")
    
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"\n❌ ERROR: Database not found at {db_path}")
        print("   Run the optimization first to create the database.")
        return 1
    
    print(f"\n✓ Database found")
    
    # Check if optuna-dashboard is installed
    try:
        import optuna_dashboard
    except ImportError:
        print("\n❌ ERROR: optuna-dashboard not installed")
        print("   Install it with: pip install optuna-dashboard")
        return 1
    
    print("✓ optuna-dashboard installed")
    
    # Try to load the study to verify it exists
    try:
        import optuna
        storage = f"sqlite:///{db_path}"
        study = optuna.load_study(study_name=study_name, storage=storage)
        n_trials = len(study.trials)
        print(f"✓ Study loaded: {n_trials} trials found")
        
        if n_trials > 0:
            print(f"\nBest trial:")
            print(f"  Value (gain): ${study.best_value:,.0f}")
            print(f"  Parameters:")
            for key, value in study.best_params.items():
                print(f"    {key}: {value}")
    except KeyError:
        print(f"\n⚠ WARNING: Study '{study_name}' not found in database")
        print("   The dashboard will still open, but this study won't be visible.")
        print("   Run the optimization first to create the study.")
    
    # Launch dashboard
    print("\n" + "="*70)
    print("  LAUNCHING DASHBOARD")
    print("="*70)
    print(f"\nDashboard URL: http://127.0.0.1:8080")
    print("\nPress Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    # Function to open browser after a short delay
    def open_browser():
        time.sleep(2)  # Wait 2 seconds for server to start
        url = "http://127.0.0.1:8080"
        try:
            # Try to use Edge specifically on Windows
            if sys.platform == "win32":
                edge_path = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe"
                if os.path.exists(edge_path):
                    webbrowser.register('edge', None, webbrowser.BackgroundBrowser(edge_path))
                    webbrowser.get('edge').open(url, new=2)  # new=2 opens in new tab
                    print("✓ Opened in Microsoft Edge")
                else:
                    webbrowser.open(url, new=2)
                    print("✓ Opened in default browser")
            else:
                webbrowser.open(url, new=2)
                print("✓ Opened in default browser")
        except Exception as e:
            print(f"⚠ Could not open browser automatically: {e}")
            print(f"  Please open manually: {url}")
    
    # Start browser opener in background thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Import and run dashboard
    from optuna_dashboard import run_server
    
    # Run server (this blocks until Ctrl+C)
    run_server(f"sqlite:///{db_path}", host="127.0.0.1", port=8080)
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("  DASHBOARD STOPPED")
        print("="*70)
        sys.exit(0)

