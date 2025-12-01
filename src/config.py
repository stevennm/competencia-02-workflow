"""
Configuration file for zLightGBM workflow
"""

#SEMILLAS: 123479, 123491, 123493, 123499, 123503

PARAM = {
    # Experiment configuration
    "experimento": "comp3_exp1_test202106",
    "semilla_primigenia": 123479,
    
    # Bucket configuration (for VM storage)
    "bucket": {
        "enabled": True,  # Set to True to enable bucket copying in VM
        "base_path": "~/buckets/b1/exp",  # Base bucket path (change as needed)
        "copy_on_create": True,  # Copy files immediately after creation
    },
    
    # ========================================================================
    # zLightGBM Configuration
    # ========================================================================
    "zlgbm": {
        "qcanaritos": 5,  # Number of canary features (must match 'canaritos' param)
        
        # Training strategy for zLightGBM
        "train_final": {
            "training": [
            201901, 
            201902, 201903, 201904,  201906,
            201907, 201908, 201909,  201911, 201912,
            202001, 202002, 202003, 202004, 202005,
            202007, 202008, 202009, 202010, 202011, 202012,
            202101, 202102, 202103, 202104, 202105, 202106, 202107
            ], # From notebook
            "future": [202109],
            "undersampling": 0.1,  # Conservative (50%)
            "ksemillerio": 30,  # Only 1 model (zLightGBM is robust enough)
            
            # Month weighting strategy
            "month_weights": "equal",  # Options: "equal", "step", "linear", "exponential"
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
            "canaritos": 5,         # MUST match qcanaritos above
            "gradient_bound": 0.1     # Adaptive learning rate (default)
        }
    }
}
