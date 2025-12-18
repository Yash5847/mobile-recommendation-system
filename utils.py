"""
Utility Functions
Helper functions for loading models, preprocessing inputs, and formatting outputs
"""

import pandas as pd
import numpy as np
import joblib
import config
from data_preprocessing import DataPreprocessor
from recommendation_engine import RecommendationEngine

def load_models():
    """
    Load all saved models and preprocessors
    
    Returns:
        dict: Dictionary containing all loaded models
    """
    models = {}
    
    try:
        # Load best ML model
        models['best_model'] = joblib.load(config.BEST_MODEL_PATH)
        
        # Load best model name
        with open(config.MODELS_DIR / 'best_model_name.txt', 'r') as f:
            models['best_model_name'] = f.read().strip()
        
        # Load preprocessor
        preprocessor = DataPreprocessor()
        preprocessor.load_preprocessor()
        models['preprocessor'] = preprocessor
        
        # Load recommendation engine
        rec_engine = RecommendationEngine()
        rec_engine.load_model()
        rec_engine.load_data()
        rec_engine.prepare_features()
        models['recommendation_engine'] = rec_engine
        
        # Load model metrics
        models['metrics'] = joblib.load(config.MODEL_METRICS_PATH)
        
        print("✓ All models loaded successfully")
        return models
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return None


def load_data():
    """
    Load processed smartphone data
    
    Returns:
        DataFrame: Processed smartphone data
    """
    try:
        df = pd.read_csv(config.DATA_DIR / 'processed_data.csv')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def preprocess_user_input(user_input, preprocessor):
    """
    Preprocess user input for prediction
    
    Args:
        user_input: dict with user-provided features
        preprocessor: DataPreprocessor instance
    
    Returns:
        numpy array: Scaled feature vector
    """
    # Create a dataframe from user input
    input_df = pd.DataFrame([user_input])
    
    # Engineer features (same as training)
    # Price category
    input_df['price_category'] = pd.cut(
        input_df['price'],
        bins=[0, 15000, 30000, 50000, np.inf],
        labels=['budget', 'mid-range', 'premium', 'flagship']
    )
    
    # RAM category
    input_df['ram_category'] = pd.cut(
        input_df['ram_capacity'],
        bins=[0, 4, 8, 12, np.inf],
        labels=['low', 'medium', 'high', 'very_high']
    )
    
    # Battery category
    input_df['battery_category'] = pd.cut(
        input_df['battery_capacity'],
        bins=[0, 4000, 5000, 6000, np.inf],
        labels=['small', 'medium', 'large', 'very_large']
    )
    
    # Camera score
    input_df['camera_score'] = (
        input_df['primary_camera_rear'] * 0.7 + 
        input_df['primary_camera_front'] * 0.3
    )
    
    # Performance score
    input_df['performance_score'] = (
        input_df['ram_capacity'] * 0.4 +
        input_df['processor_speed'] * 10 * 0.3 +
        input_df['num_cores'] * 2 * 0.3
    )
    
    # Display score
    input_df['display_score'] = (
        input_df['screen_size'] * 10 * 0.4 +
        input_df['refresh_rate'] * 0.3 +
        (input_df['resolution_width'] * input_df['resolution_height'] / 1000000) * 0.3
    )
    
    # Value score (use median rating for calculation)
    median_rating = 80  # approximate median
    input_df['value_score'] = median_rating / (input_df['price'] / 10000)
    
    # Total storage
    input_df['total_storage'] = input_df['internal_memory'] + input_df.get('extended_upto', 0).fillna(0)
    
    # Prepare features using preprocessor
    X_scaled, _ = preprocessor.prepare_features(input_df, fit=False)
    
    return X_scaled


def format_price(price):
    """Format price in Indian Rupees"""
    if price >= 100000:
        return f"₹{price/100000:.2f}L"
    elif price >= 1000:
        return f"₹{price/1000:.1f}K"
    else:
        return f"₹{price}"


def get_price_category(price):
    """Get price category label"""
    if price < 15000:
        return "Budget"
    elif price < 30000:
        return "Mid-Range"
    elif price < 50000:
        return "Premium"
    else:
        return "Flagship"


def get_rating_color(rating):
    """Get color based on rating"""
    if rating >= 85:
        return "green"
    elif rating >= 75:
        return "orange"
    else:
        return "red"


def get_rating_emoji(rating):
    """Get emoji based on rating"""
    if rating >= 85:
        return "🌟"
    elif rating >= 75:
        return "⭐"
    else:
        return "⚠️"


def format_specs(phone):
    """Format phone specifications for display"""
    specs = []
    
    if 'ram_capacity' in phone:
        specs.append(f"{int(phone['ram_capacity'])}GB RAM")
    
    if 'internal_memory' in phone:
        specs.append(f"{int(phone['internal_memory'])}GB Storage")
    
    if 'battery_capacity' in phone:
        specs.append(f"{int(phone['battery_capacity'])}mAh")
    
    if 'primary_camera_rear' in phone:
        specs.append(f"{int(phone['primary_camera_rear'])}MP Camera")
    
    if 'screen_size' in phone:
        specs.append(f"{phone['screen_size']:.2f}\" Display")
    
    if 'refresh_rate' in phone:
        specs.append(f"{int(phone['refresh_rate'])}Hz")
    
    return " | ".join(specs)


def get_feature_importance_summary(model, feature_names, top_n=10):
    """
    Get feature importance for tree-based models
    
    Args:
        model: Trained model
        feature_names: List of feature names
        top_n: Number of top features to return
    
    Returns:
        DataFrame: Top features and their importance
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        })
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        return importance_df
    else:
        return None


def calculate_match_score(user_prefs, phone):
    """
    Calculate how well a phone matches user preferences
    
    Args:
        user_prefs: dict with user preferences
        phone: Series with phone data
    
    Returns:
        float: Match score (0-100)
    """
    score = 0
    total_weight = 0
    
    # Price match (weight: 30%)
    if 'price' in user_prefs and 'price' in phone:
        price_diff = abs(phone['price'] - user_prefs['price']) / user_prefs['price']
        price_score = max(0, 1 - price_diff) * 30
        score += price_score
        total_weight += 30
    
    # RAM match (weight: 15%)
    if 'ram_capacity' in user_prefs and 'ram_capacity' in phone:
        if phone['ram_capacity'] >= user_prefs['ram_capacity']:
            score += 15
        else:
            score += (phone['ram_capacity'] / user_prefs['ram_capacity']) * 15
        total_weight += 15
    
    # Battery match (weight: 15%)
    if 'battery_capacity' in user_prefs and 'battery_capacity' in phone:
        if phone['battery_capacity'] >= user_prefs['battery_capacity']:
            score += 15
        else:
            score += (phone['battery_capacity'] / user_prefs['battery_capacity']) * 15
        total_weight += 15
    
    # Storage match (weight: 10%)
    if 'internal_memory' in user_prefs and 'internal_memory' in phone:
        if phone['internal_memory'] >= user_prefs['internal_memory']:
            score += 10
        else:
            score += (phone['internal_memory'] / user_prefs['internal_memory']) * 10
        total_weight += 10
    
    # Screen size match (weight: 10%)
    if 'screen_size' in user_prefs and 'screen_size' in phone:
        size_diff = abs(phone['screen_size'] - user_prefs['screen_size']) / user_prefs['screen_size']
        size_score = max(0, 1 - size_diff) * 10
        score += size_score
        total_weight += 10
    
    # Camera match (weight: 10%)
    if 'primary_camera_rear' in user_prefs and 'primary_camera_rear' in phone:
        if phone['primary_camera_rear'] >= user_prefs['primary_camera_rear']:
            score += 10
        else:
            score += (phone['primary_camera_rear'] / user_prefs['primary_camera_rear']) * 10
        total_weight += 10
    
    # Refresh rate match (weight: 10%)
    if 'refresh_rate' in user_prefs and 'refresh_rate' in phone:
        if phone['refresh_rate'] >= user_prefs['refresh_rate']:
            score += 10
        else:
            score += (phone['refresh_rate'] / user_prefs['refresh_rate']) * 10
        total_weight += 10
    
    # Normalize score
    if total_weight > 0:
        final_score = (score / total_weight) * 100
    else:
        final_score = 0
    
    return min(100, final_score)


def get_brand_stats(df):
    """Get statistics by brand"""
    brand_stats = df.groupby('brand_name').agg({
        'model': 'count',
        'price': 'mean',
        'rating': 'mean'
    }).round(2)
    
    brand_stats.columns = ['Model Count', 'Avg Price', 'Avg Rating']
    brand_stats = brand_stats.sort_values('Model Count', ascending=False)
    
    return brand_stats


def get_price_range_stats(df):
    """Get statistics by price range"""
    df['price_range'] = pd.cut(
        df['price'],
        bins=[0, 15000, 30000, 50000, np.inf],
        labels=['Budget (<15K)', 'Mid-Range (15-30K)', 'Premium (30-50K)', 'Flagship (>50K)']
    )
    
    price_stats = df.groupby('price_range').agg({
        'model': 'count',
        'rating': 'mean'
    }).round(2)
    
    price_stats.columns = ['Count', 'Avg Rating']
    
    return price_stats