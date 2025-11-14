"""
Configuration file for the LightGBM workflow with Bayesian Optimization
"""

PARAM = {
    # Experiment configuration
    "experimento": "zlgbm_test",
    "semilla_primigenia": 102191,
    
    # Bucket configuration (for VM storage)
    "bucket": {
        "enabled": True,  # Set to False to disable bucket copying
        "base_path": "~/buckets/b1",  # Base bucket path (change as needed)
        "copy_on_create": True,  # Copy files immediately after creation
    },
    
    # Feature Engineering - Random Forest
    "FE_rf": {
        "arbolitos": 20,
        "hojas_por_arbol": 16,
        "datos_por_hoja": 100,
        "mtry_ratio": 0.2,
        "train": {
            "training": [202101, 202102, 202103]
        },
        "lgb_param": None  # Will be set dynamically
    },
    
    # Feature Engineering - Historical Tendencies
    "FE_hist": {
        "Tendencias": {
            "run": True,
            "ventana": 6,
            "tendencia": True,
            "minimo": True,
            "maximo": True,
            "promedio": True,
            "ratioavg": True,
            "ratiomax": True
        }
    },
    
    # Training Strategy (for Bayesian Optimization)
    "trainingstrategy": {
        "testing": [202104],
        "training": [
            201901, 201902, 201903, 201904, 201905, 201906,
            201907, 201908, 201909, 201910, 201911, 201912,
            202001, 202002, #202003, 202004, 202005, 202006,
            #202007, 202008, 202009, 202010, 
            202011, 202012,
            202101, 202102 #, 202103, 202104
        ],
        "undersampling": 0.05,
        "positivos": ["BAJA+1", "BAJA+2"]
    },
    
    # Hyperparameter Tuning
    "hipeparametertuning": {
        "BO_iteraciones": 0,  # Set to 0 to skip Bayesian Optimization
        # Multi-seed ensemble during BO (optional, increases robustness but slower)
        "ksemillerio": 3,  # Number of models with different seeds per trial (1=fast, 5=robust)
        "repe": 2          # Number of repetitions to average (1=fast, 3=very robust)
        # Total models per trial = ksemillerio × repe
        # Example: ksemillerio=5, repe=3 → 15 models per trial (15x slower!)
    },
    
    # LightGBM Fixed Parameters
    "lgbm": {
        "param_fijos": {
            "objective": "binary",
            "metric": "custom",
            "first_metric_only": True,
            "boost_from_average": True,
            "feature_pre_filter": False,
            "verbosity": -100,
            "force_row_wise": True,
            "extra_trees": False,
            "max_depth": -1,
            "min_gain_to_split": 0.0,
            "min_sum_hessian_in_leaf": 0.001,
            "lambda_l1": 0.0,
            "lambda_l2": 0.0,
            "bagging_fraction": 1.0,
            "pos_bagging_fraction": 1.0,
            "neg_bagging_fraction": 1.0,
            "is_unbalance": False,
            "scale_pos_weight": 1.0,
            "drop_rate": 0.1,
            "max_drop": 50,
            "skip_drop": 0.5,
            "max_bin": 31
        }
    },
    
    # Final Training Strategy
    "train_final": {
        "future": [202106],
        "training": [
            201901, 201902, 201903, 201904, 201905, 201906,
            201907, 201908, 201909, 201910, 201911, 201912,
            202001, 202002, #202003, 202004, 202005, 202006,
            #202007, 202008, 202009, 202010, 
            202011, 202012,
            202101, 202102, 202103, 202104 #, 202105, 202106
        ],
        "undersampling": 0.10,
        "ksemillerio": 30,  # Number of models in final ensemble
        "param_mejores": {
            # Best hyperparameters from semillerio_optuna2_moretrials (trial 45)
            "num_iterations": 1491,
            "learning_rate": 0.032888026616230304,
            "feature_fraction": 0.2272840312390512,
            "min_data_in_leaf": 3,
            "num_leaves": 680
        },
        "semillas": None  # Will be generated
    },
    
    # Bayesian Optimization
    "BO": {
        "semillas": None  # Will be generated
    },
    
    # ========================================================================
    # zLightGBM Configuration (alternative to Bayesian Optimization)
    # ========================================================================
    "zlgbm": {
        "qcanaritos": 50,  # Number of canary features (must match 'canaritos' param)
        
        # Training strategy for zLightGBM
        "train_final": {
           "training": [
            201901, 
            # 201902, 201903, 201904, 201905, 201906,
            # 201907, 201908, 201909, 201910, 201911, 201912,
            # 202001, 202002, 202003, 202004, 202005, 202006,
            # 202007, 202008, 202009, 202010, 202011, 202012,
            # 202101, 202102, 202103, 202104 #, 202105, 202106
            ],
            "future": [202106],
            "undersampling": 0.10,  # More conservative (50% vs 10%)
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
            "canaritos": 50,         # MUST match qcanaritos above
            "gradient_bound": 0.1     # Adaptive learning rate (default)
        }
    }
}


def setup_rf_lgb_params(PARAM):
    """Setup LightGBM parameters for Random Forest feature engineering"""
    PARAM["FE_rf"]["lgb_param"] = {
        # Parameters that can be changed
        "num_iterations": PARAM["FE_rf"]["arbolitos"],
        "num_leaves": PARAM["FE_rf"]["hojas_por_arbol"],
        "min_data_in_leaf": PARAM["FE_rf"]["datos_por_hoja"],
        "feature_fraction_bynode": PARAM["FE_rf"]["mtry_ratio"],
        
        # To make LightGBM emulate Random Forest
        "boosting": "rf",
        "bagging_fraction": (1.0 - 1.0 / 2.718281828),  # 1 - 1/e
        "bagging_freq": 1,
        "feature_fraction": 1.0,
        
        # Generic LightGBM parameters
        "max_bin": 31,
        "objective": "binary",
        "first_metric_only": True,
        "boost_from_average": True,
        "feature_pre_filter": False,
        "force_row_wise": True,
        "verbosity": -100,
        "max_depth": -1,
        "min_gain_to_split": 0.0,
        "min_sum_hessian_in_leaf": 0.001,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "pos_bagging_fraction": 1.0,
        "neg_bagging_fraction": 1.0,
        "is_unbalance": False,
        "scale_pos_weight": 1.0,
        "drop_rate": 0.1,
        "max_drop": 50,
        "skip_drop": 0.5,
        "extra_trees": False
    }


# Initialize RF parameters
setup_rf_lgb_params(PARAM)

