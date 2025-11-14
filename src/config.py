"""
Configuration file for zLightGBM workflow
"""

PARAM = {
    # Experiment configuration
    "experimento": "zlgbm_test",
    "semilla_primigenia": 102191,
    
    # Bucket configuration (for VM storage)
    "bucket": {
        "enabled": False,  # Set to True to enable bucket copying in VM
        "base_path": "~/buckets/b1",  # Base bucket path (change as needed)
        "copy_on_create": True,  # Copy files immediately after creation
    },
    
    # ========================================================================
    # zLightGBM Configuration
    # ========================================================================
    "zlgbm": {
        "qcanaritos": 100,  # Number of canary features (must match 'canaritos' param)
        
        # Training strategy for zLightGBM
        "train_final": {
            "training": [202101, 202102, 202103, 202104],  # From notebook
            "future": [202106],
            "undersampling": 0.50,  # Conservative (50%)
            "ksemillerio": 1,  # Only 1 model (zLightGBM is robust enough)
        },
        
        # zLightGBM parameters
        "param": {
            # Standard parameters
            "boosting": "gbdt",
            "objective": "binary",
            "metric": "custom",
            "first_metric_only": False,
            "boost_from_average": True,
            "feature_pre_filter": False,
            "force_row_wise": True,
            "verbosity": -100,
            
            # "Free" hyperparameters (zLightGBM stops automatically)
            "num_iterations": 9999,  # Max, but stops earlier
            "num_leaves": 999,       # Max, but makes fewer splits if needed
            "learning_rate": 1.0,    # Keep at 1.0 for gradient_bound to work
            
            # Tunable hyperparameters
            "max_bin": 31,
            "min_data_in_leaf": 20,  # LightGBM default
            "feature_fraction": 0.50,  # Balanced
            
            # zLightGBM specific (NEW!)
            "canaritos": 100,         # MUST match qcanaritos above
            "gradient_bound": 0.1     # Adaptive learning rate (default)
        }
    }
}
