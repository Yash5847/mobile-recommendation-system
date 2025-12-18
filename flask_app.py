"""
Flask Backend for Mobile Recommendation System
Provides REST API endpoints for ML predictions and recommendations
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import config
from utils import preprocess_user_input, format_price, get_rating_emoji
from data_preprocessing import DataPreprocessor
from recommendation_engine import RecommendationEngine

app = Flask(__name__)
CORS(app)

# Global variables for models
models = {}
df = None

def load_all_models():
    """Load all ML models and data"""
    global models, df
    
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
        
        # Load data
        df = pd.read_csv(config.DATA_DIR / 'processed_data.csv')
        
        print("✅ All models and data loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

# Load models on startup
load_all_models()

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')

@app.route('/recommendation')
def recommendation_page():
    """Render recommendation page"""
    return render_template('recommendation.html')

@app.route('/dashboard')
def dashboard_page():
    """Render dashboard page"""
    return render_template('dashboard.html')

@app.route('/about')
def about_page():
    """Render about page"""
    return render_template('about.html')

@app.route('/api/predict', methods=['POST'])
def predict_rating():
    """
    Predict smartphone rating based on user input
    
    Expected JSON input:
    {
        "price": 25000,
        "ram_capacity": 8,
        "internal_memory": 128,
        "battery_capacity": 5000,
        "screen_size": 6.5,
        "refresh_rate": 120,
        "primary_camera_rear": 50,
        "brand_name": "samsung",
        "processor_brand": "snapdragon",
        "os": "android",
        "has_5g": true
    }
    
    Returns:
    {
        "success": true,
        "predicted_rating": 82.5,
        "rating_category": "Good",
        "model_name": "Gradient Boosting"
    }
    """
    try:
        data = request.get_json()
        
        # Prepare user input
        user_input = {
            'price': float(data.get('price', 25000)),
            'num_cores': 8,
            'processor_speed': 2.5,
            'battery_capacity': float(data.get('battery_capacity', 5000)),
            'fast_charging': 33.0,
            'ram_capacity': float(data.get('ram_capacity', 8)),
            'internal_memory': float(data.get('internal_memory', 128)),
            'screen_size': float(data.get('screen_size', 6.5)),
            'refresh_rate': float(data.get('refresh_rate', 120)),
            'num_rear_cameras': 3,
            'num_front_cameras': 1,
            'primary_camera_rear': float(data.get('primary_camera_rear', 50)),
            'primary_camera_front': 16.0,
            'resolution_width': 1080,
            'resolution_height': 2400,
            'has_5g': 1 if data.get('has_5g', True) else 0,
            'has_nfc': 0,
            'has_ir_blaster': 0,
            'fast_charging_available': 1,
            'extended_memory_available': 0,
            'extended_upto': 0,
            'brand_name': data.get('brand_name', 'samsung'),
            'processor_brand': data.get('processor_brand', 'snapdragon'),
            'os': data.get('os', 'android')
        }
        
        # Preprocess and predict
        X_scaled = preprocess_user_input(user_input, models['preprocessor'])
        predicted_rating = float(models['best_model'].predict(X_scaled)[0])
        predicted_rating = max(0, min(100, predicted_rating))
        
        # Determine rating category
        if predicted_rating >= 85:
            rating_category = "Excellent"
        elif predicted_rating >= 80:
            rating_category = "Very Good"
        elif predicted_rating >= 75:
            rating_category = "Good"
        else:
            rating_category = "Average"
        
        return jsonify({
            'success': True,
            'predicted_rating': round(predicted_rating, 2),
            'rating_category': rating_category,
            'model_name': models['best_model_name'],
            'emoji': get_rating_emoji(predicted_rating)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """
    Get smartphone recommendations based on user preferences
    
    Expected JSON input:
    {
        "price": 25000,
        "battery_capacity": 5000,
        "ram_capacity": 8,
        "internal_memory": 128,
        "screen_size": 6.5,
        "refresh_rate": 120,
        "primary_camera_rear": 50,
        "brand": "any",
        "has_5g": true,
        "top_n": 5
    }
    
    Returns:
    {
        "success": true,
        "recommendations": [
            {
                "brand_name": "Samsung",
                "model": "Galaxy S21",
                "price": 24999,
                "rating": 85.0,
                "match_percentage": 95.5,
                "specs": {...}
            }
        ]
    }
    """
    try:
        data = request.get_json()
        
        # Extract preferences
        rec_prefs = {
            'price': float(data.get('price', 25000)),
            'battery_capacity': float(data.get('battery_capacity', 5000)),
            'ram_capacity': float(data.get('ram_capacity', 8)),
            'internal_memory': float(data.get('internal_memory', 128)),
            'screen_size': float(data.get('screen_size', 6.5)),
            'refresh_rate': float(data.get('refresh_rate', 120)),
            'primary_camera_rear': float(data.get('primary_camera_rear', 50))
        }
        
        top_n = int(data.get('top_n', 5))
        brand_filter = data.get('brand', 'any')
        has_5g = data.get('has_5g', True)
        
        # Get recommendations
        recommendations = models['recommendation_engine'].get_recommendations_by_preferences(
            rec_prefs, top_n=top_n * 2  # Get more to filter
        )
        
        # Apply filters
        if brand_filter and brand_filter.lower() != 'any':
            recommendations = recommendations[recommendations['brand_name'] == brand_filter.lower()]
        
        if has_5g:
            recommendations = recommendations[recommendations['has_5g'] == 1]
        
        # If no results, get top rated in price range
        if len(recommendations) == 0:
            recommendations = models['recommendation_engine'].get_top_rated_phones(
                top_n=top_n, 
                min_price=rec_prefs['price']*0.8, 
                max_price=rec_prefs['price']*1.2
            )
            recommendations['match_percentage'] = 75.0
        
        # Limit to top_n
        recommendations = recommendations.head(top_n)
        
        # Format results
        results = []
        for _, phone in recommendations.iterrows():
            results.append({
                'brand_name': phone['brand_name'].title(),
                'model': phone['model'],
                'price': int(phone['price']),
                'price_formatted': format_price(phone['price']),
                'rating': float(phone['rating']),
                'match_percentage': float(phone.get('match_percentage', 80)),
                'specs': {
                    'ram': int(phone['ram_capacity']),
                    'storage': int(phone['internal_memory']),
                    'battery': int(phone['battery_capacity']),
                    'screen_size': float(phone['screen_size']),
                    'refresh_rate': int(phone['refresh_rate']),
                    'camera_rear': int(phone['primary_camera_rear']),
                    'camera_front': int(phone['primary_camera_front']),
                    'has_5g': bool(phone['has_5g']),
                    'processor': phone['processor_brand']
                }
            })
        
        return jsonify({
            'success': True,
            'count': len(results),
            'recommendations': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/dashboard-data', methods=['GET'])
def get_dashboard_data():
    """
    Get data for dashboard visualizations
    
    Returns:
    {
        "success": true,
        "statistics": {...},
        "price_vs_rating": [...],
        "brand_distribution": {...},
        "battery_vs_rating": [...]
    }
    """
    try:
        # Overall statistics
        statistics = {
            'total_phones': int(len(df)),
            'total_brands': int(df['brand_name'].nunique()),
            'avg_rating': float(df['rating'].mean()),
            'avg_price': float(df['price'].mean()),
            'price_range': {
                'min': int(df['price'].min()),
                'max': int(df['price'].max())
            }
        }
        
        # Price vs Rating data
        price_rating_data = df[['price', 'rating', 'brand_name']].to_dict('records')
        
        # Brand distribution (top 10)
        brand_counts = df['brand_name'].value_counts().head(10)
        brand_distribution = {
            'labels': brand_counts.index.tolist(),
            'values': brand_counts.values.tolist()
        }
        
        # Battery vs Rating data
        battery_rating_data = df[['battery_capacity', 'rating', 'price', 'brand_name']].to_dict('records')
        
        # Price category distribution
        price_bins = [0, 15000, 30000, 50000, np.inf]
        price_labels = ['Budget', 'Mid-Range', 'Premium', 'Flagship']
        df['price_cat'] = pd.cut(df['price'], bins=price_bins, labels=price_labels)
        price_cat_counts = df['price_cat'].value_counts()
        price_categories = {
            'labels': price_cat_counts.index.tolist(),
            'values': price_cat_counts.values.tolist()
        }
        
        # Top rated phones
        top_rated = df.nlargest(5, 'rating')[['brand_name', 'model', 'price', 'rating']].to_dict('records')
        
        # Model performance metrics
        model_metrics = {
            'best_model': models['best_model_name'],
            'test_r2': float(models['metrics'][models['best_model_name']]['test_r2']),
            'test_mae': float(models['metrics'][models['best_model_name']]['test_mae']),
            'test_rmse': float(models['metrics'][models['best_model_name']]['test_rmse'])
        }
        
        return jsonify({
            'success': True,
            'statistics': statistics,
            'price_vs_rating': price_rating_data,
            'brand_distribution': brand_distribution,
            'battery_vs_rating': battery_rating_data,
            'price_categories': price_categories,
            'top_rated': top_rated,
            'model_metrics': model_metrics
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/brands', methods=['GET'])
def get_brands():
    """Get list of all brands"""
    try:
        brands = sorted(df['brand_name'].unique().tolist())
        return jsonify({
            'success': True,
            'brands': brands
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models) > 0,
        'data_loaded': df is not None
    })

if __name__ == '__main__':
    print("="*60)
    print("🚀 Starting Mobile Recommendation System API")
    print("="*60)
    print(f"📊 Dataset: {len(df)} smartphones")
    print(f"🤖 Best Model: {models['best_model_name']}")
    print(f"🌐 Server: http://localhost:5000")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)