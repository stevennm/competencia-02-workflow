"""
Preprocessing package
Contains all preprocessing and feature engineering utilities
"""

from .preprocessing_utils import (
    load_data,
    add_canaritos,
    generate_clase_ternaria,
    data_quality_fixes,
    data_drifting_correction
)

from .feature_engineering_utils import (
    add_intra_month_features,
    add_historical_features,
    add_lag_features,
    add_delta_features,
    calculate_trend_features_polars
)

from .rf_features_utils import (
    add_rf_features
)

__all__ = [
    # Preprocessing functions
    'load_data',
    'add_canaritos',
    'generate_clase_ternaria',
    'data_quality_fixes',
    'data_drifting_correction',
    # Feature engineering functions
    'add_intra_month_features',
    'add_historical_features',
    'add_lag_features',
    'add_delta_features',
    'calculate_trend_features_polars',
    # RF features functions
    'add_rf_features'
]

