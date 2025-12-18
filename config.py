"""
Configuration file for Mobile Recommendation System
Contains all constants, paths, and configuration parameters
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
EDA_PLOTS_DIR = OUTPUTS_DIR / "eda_plots"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, EDA_PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data paths
DATASET_PATH = DATA_DIR / "Smartphones_cleaned_dataset.csv"

# Model paths
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
RECOMMENDATION_MODEL_PATH = MODELS_DIR / "recommendation_model.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"
MODEL_METRICS_PATH = MODELS_DIR / "model_metrics.pkl"

# Feature columns for ML
NUMERIC_FEATURES = [
    'price', 'num_cores', 'processor_speed', 'battery_capacity',
    'fast_charging', 'ram_capacity', 'internal_memory', 'screen_size',
    'refresh_rate', 'num_rear_cameras', 'num_front_cameras',
    'primary_camera_rear', 'primary_camera_front', 'resolution_width',
    'resolution_height'
]

CATEGORICAL_FEATURES = [
    'brand_name', 'processor_brand', 'os'
]

BOOLEAN_FEATURES = [
    'has_5g', 'has_nfc', 'has_ir_blaster', 'fast_charging_available',
    'extended_memory_available'
]

TARGET_COLUMN = 'rating'

# Recommendation features (for similarity calculation)
RECOMMENDATION_FEATURES = [
    'price', 'battery_capacity', 'ram_capacity', 'internal_memory',
    'screen_size', 'refresh_rate', 'primary_camera_rear'
]

# ML Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Model configurations
MODEL_CONFIGS = {
    'Linear Regression': {
        'name': 'Linear Regression',
        'params': {}
    },
    'KNN': {
        'name': 'K-Nearest Neighbors',
        'params': {
            'n_neighbors': 5,
            'weights': 'distance'
        }
    },
    'Random Forest': {
        'name': 'Random Forest',
        'params': {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'random_state': RANDOM_STATE
        }
    },
    'Gradient Boosting': {
        'name': 'Gradient Boosting',
        'params': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': RANDOM_STATE
        }
    }
}

# Streamlit UI configuration
APP_TITLE = "📱 Mobile Recommendation System"
APP_SUBTITLE = "AI-Powered Smartphone Recommendations & Rating Predictions"

# Brand options (will be updated from data)
BRAND_OPTIONS = [
    'oneplus', 'samsung', 'motorola', 'realme', 'apple', 'xiaomi',
    'nothing', 'oppo', 'vivo', 'iqoo', 'poco', 'redmi'
]

PROCESSOR_OPTIONS = [
    'snapdragon', 'exynos', 'dimensity', 'bionic', 'helio', 'kirin'
]

OS_OPTIONS = ['android', 'ios']

# UI Color scheme
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'warning': '#ff9800',
    'danger': '#d62728',
    'info': '#17a2b8'
}
