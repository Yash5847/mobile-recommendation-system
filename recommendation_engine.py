"""
Recommendation Engine Module
Content-based recommendation system using KNN and cosine similarity
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import config

class RecommendationEngine:
    """
    Content-based recommendation system for smartphones
    Uses KNN and cosine similarity to recommend similar phones
    """
    
    def __init__(self):
        self.knn_model = None
        self.scaler = StandardScaler()
        self.df = None
        self.feature_matrix = None
        self.recommendation_features = config.RECOMMENDATION_FEATURES
        
    def load_data(self, filepath=None):
        """Load smartphone dataset"""
        if filepath is None:
            filepath = config.DATA_DIR / 'processed_data.csv'
        
        self.df = pd.read_csv(filepath)
        print(f"Loaded {len(self.df)} smartphones for recommendation")
        return self.df
    
    def prepare_features(self):
        """Prepare feature matrix for recommendation"""
        print("\n=== Preparing Recommendation Features ===")
        
        # Select recommendation features
        feature_data = self.df[self.recommendation_features].copy()
        
        # Handle missing values
        feature_data = feature_data.fillna(feature_data.median())
        
        # Scale features
        self.feature_matrix = self.scaler.fit_transform(feature_data)
        
        print(f"Feature matrix shape: {self.feature_matrix.shape}")
        print(f"Features used: {', '.join(self.recommendation_features)}")
        
        return self.feature_matrix
    
    def train_knn_model(self, n_neighbors=10):
        """Train KNN model for recommendations"""
        print(f"\n=== Training KNN Model (k={n_neighbors}) ===")
        
        self.knn_model = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric='euclidean',
            algorithm='auto'
        )
        
        self.knn_model.fit(self.feature_matrix)
        print("✓ KNN model trained successfully")
        
        return self.knn_model
    
    def get_recommendations_by_preferences(self, user_preferences, top_n=5):
        """
        Get recommendations based on user preferences
        
        Args:
            user_preferences: dict with keys matching recommendation_features
            top_n: number of recommendations to return
        
        Returns:
            DataFrame with recommended phones
        """
        # Create feature vector from user preferences
        user_vector = []
        for feature in self.recommendation_features:
            user_vector.append(user_preferences.get(feature, self.df[feature].median()))
        
        user_vector = np.array(user_vector).reshape(1, -1)
        
        # Scale user vector
        user_vector_scaled = self.scaler.transform(user_vector)
        
        # Find nearest neighbors
        distances, indices = self.knn_model.kneighbors(user_vector_scaled, n_neighbors=top_n)
        
        # Get recommended phones
        recommendations = self.df.iloc[indices[0]].copy()
        recommendations['similarity_score'] = 1 / (1 + distances[0])  # Convert distance to similarity
        recommendations['match_percentage'] = (recommendations['similarity_score'] / recommendations['similarity_score'].max() * 100)
        
        return recommendations
    
    def get_recommendations_by_phone_id(self, phone_index, top_n=5):
        """
        Get recommendations similar to a specific phone
        
        Args:
            phone_index: index of the phone in the dataset
            top_n: number of recommendations to return (excluding the phone itself)
        
        Returns:
            DataFrame with recommended phones
        """
        # Get phone vector
        phone_vector = self.feature_matrix[phone_index].reshape(1, -1)
        
        # Find nearest neighbors (top_n + 1 because first one will be the phone itself)
        distances, indices = self.knn_model.kneighbors(phone_vector, n_neighbors=top_n + 1)
        
        # Remove the first result (the phone itself)
        indices = indices[0][1:]
        distances = distances[0][1:]
        
        # Get recommended phones
        recommendations = self.df.iloc[indices].copy()
        recommendations['similarity_score'] = 1 / (1 + distances)
        recommendations['match_percentage'] = (recommendations['similarity_score'] / recommendations['similarity_score'].max() * 100)
        
        return recommendations
    
    def get_recommendations_by_filters(self, filters, top_n=10):
        """
        Get recommendations based on filters (price range, brand, etc.)
        
        Args:
            filters: dict with filter criteria
            top_n: number of recommendations to return
        
        Returns:
            DataFrame with filtered and ranked phones
        """
        filtered_df = self.df.copy()
        
        # Apply filters
        if 'min_price' in filters and filters['min_price'] is not None:
            filtered_df = filtered_df[filtered_df['price'] >= filters['min_price']]
        
        if 'max_price' in filters and filters['max_price'] is not None:
            filtered_df = filtered_df[filtered_df['price'] <= filters['max_price']]
        
        if 'brand' in filters and filters['brand'] is not None:
            filtered_df = filtered_df[filtered_df['brand_name'] == filters['brand']]
        
        if 'min_ram' in filters and filters['min_ram'] is not None:
            filtered_df = filtered_df[filtered_df['ram_capacity'] >= filters['min_ram']]
        
        if 'min_battery' in filters and filters['min_battery'] is not None:
            filtered_df = filtered_df[filtered_df['battery_capacity'] >= filters['min_battery']]
        
        if 'has_5g' in filters and filters['has_5g']:
            filtered_df = filtered_df[filtered_df['has_5g'] == 1]
        
        if 'min_rating' in filters and filters['min_rating'] is not None:
            filtered_df = filtered_df[filtered_df['rating'] >= filters['min_rating']]
        
        # Sort by rating and return top N
        filtered_df = filtered_df.sort_values('rating', ascending=False).head(top_n)
        filtered_df['match_percentage'] = 100.0  # All matches are 100% as they meet criteria
        
        return filtered_df
    
    def calculate_similarity_matrix(self):
        """Calculate cosine similarity matrix for all phones"""
        print("\n=== Calculating Similarity Matrix ===")
        
        similarity_matrix = cosine_similarity(self.feature_matrix)
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
        
        return similarity_matrix
    
    def get_top_rated_phones(self, top_n=10, min_price=None, max_price=None):
        """Get top rated phones with optional price filters"""
        filtered_df = self.df.copy()
        
        if min_price is not None:
            filtered_df = filtered_df[filtered_df['price'] >= min_price]
        
        if max_price is not None:
            filtered_df = filtered_df[filtered_df['price'] <= max_price]
        
        top_phones = filtered_df.nlargest(top_n, 'rating')
        return top_phones
    
    def get_best_value_phones(self, top_n=10):
        """Get phones with best value for money"""
        if 'value_score' in self.df.columns:
            best_value = self.df.nlargest(top_n, 'value_score')
        else:
            # Calculate value score on the fly
            self.df['temp_value_score'] = self.df['rating'] / (self.df['price'] / 10000)
            best_value = self.df.nlargest(top_n, 'temp_value_score')
            self.df.drop('temp_value_score', axis=1, inplace=True)
        
        return best_value
    
    def save_model(self):
        """Save recommendation model and scaler"""
        model_data = {
            'knn_model': self.knn_model,
            'scaler': self.scaler,
            'recommendation_features': self.recommendation_features
        }
        
        joblib.dump(model_data, config.RECOMMENDATION_MODEL_PATH)
        print(f"\n✓ Recommendation model saved: {config.RECOMMENDATION_MODEL_PATH}")
    
    def load_model(self):
        """Load saved recommendation model"""
        model_data = joblib.load(config.RECOMMENDATION_MODEL_PATH)
        
        self.knn_model = model_data['knn_model']
        self.scaler = model_data['scaler']
        self.recommendation_features = model_data['recommendation_features']
        
        print("✓ Recommendation model loaded successfully")
        return self


def main():
    """Main recommendation engine pipeline"""
    print("="*60)
    print("MOBILE RECOMMENDATION SYSTEM - RECOMMENDATION ENGINE")
    print("="*60)
    
    # Initialize engine
    engine = RecommendationEngine()
    
    # Load data
    engine.load_data()
    
    # Prepare features
    engine.prepare_features()
    
    # Train KNN model
    engine.train_knn_model(n_neighbors=10)
    
    # Save model
    engine.save_model()
    
    # Test recommendations
    print("\n" + "="*60)
    print("TESTING RECOMMENDATION ENGINE")
    print("="*60)
    
    # Test 1: Recommendations by preferences
    print("\n--- Test 1: Recommendations by User Preferences ---")
    user_prefs = {
        'price': 25000,
        'battery_capacity': 5000,
        'ram_capacity': 8,
        'internal_memory': 128,
        'screen_size': 6.5,
        'refresh_rate': 120,
        'primary_camera_rear': 50
    }
    
    recommendations = engine.get_recommendations_by_preferences(user_prefs, top_n=5)
    print("\nTop 5 Recommendations:")
    print(recommendations[['brand_name', 'model', 'price', 'rating', 'match_percentage']].to_string(index=False))
    
    # Test 2: Similar phones
    print("\n--- Test 2: Phones Similar to Index 0 ---")
    similar_phones = engine.get_recommendations_by_phone_id(0, top_n=5)
    print("\nSimilar Phones:")
    print(similar_phones[['brand_name', 'model', 'price', 'rating', 'match_percentage']].to_string(index=False))
    
    # Test 3: Filtered recommendations
    print("\n--- Test 3: Filtered Recommendations ---")
    filters = {
        'min_price': 15000,
        'max_price': 30000,
        'min_ram': 6,
        'has_5g': True,
        'min_rating': 80
    }
    
    filtered_recs = engine.get_recommendations_by_filters(filters, top_n=5)
    print("\nFiltered Recommendations:")
    print(filtered_recs[['brand_name', 'model', 'price', 'rating', 'ram_capacity']].to_string(index=False))
    
    print("\n" + "="*60)
    print("RECOMMENDATION ENGINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return engine


if __name__ == "__main__":
    main()