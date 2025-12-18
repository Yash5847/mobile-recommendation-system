"""
Mobile Recommendation System - Streamlit Dashboard
Main application with interactive UI for predictions and recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config
import utils
from recommendation_engine import RecommendationEngine

# Page configuration
st.set_page_config(
    page_title="Mobile Recommendation System",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .kpi-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .phone-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .phone-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    .recommendation-badge {
        background: #4CAF50;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.models = None
    st.session_state.df = None

@st.cache_resource
def load_all_models():
    """Load all models and data (cached)"""
    models = utils.load_models()
    df = utils.load_data()
    return models, df

def display_kpi_cards(predicted_rating, price_range, match_score):
    """Display KPI cards"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rating_emoji = utils.get_rating_emoji(predicted_rating)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Predicted Rating</div>
            <div class="kpi-value">{rating_emoji} {predicted_rating:.1f}/100</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Budget Range</div>
            <div class="kpi-value">{price_range}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Match Score</div>
            <div class="kpi-value">{match_score:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)

def display_phone_card(phone, index, show_match=False):
    """Display a phone as a card"""
    brand = phone['brand_name'].title()
    model = phone['model']
    price = utils.format_price(phone['price'])
    rating = phone['rating']
    rating_emoji = utils.get_rating_emoji(rating)
    specs = utils.format_specs(phone)
    
    match_badge = ""
    if show_match and 'match_percentage' in phone:
        match_badge = f'<span class="recommendation-badge">Match: {phone["match_percentage"]:.0f}%</span>'
    
    st.markdown(f"""
    <div class="phone-card">
        <h3>{brand} {model} {match_badge}</h3>
        <p><strong>Price:</strong> {price} | <strong>Rating:</strong> {rating_emoji} {rating:.1f}/100</p>
        <p style="color: #666; font-size: 0.9rem;">{specs}</p>
    </div>
    """, unsafe_allow_html=True)

def create_price_vs_rating_chart(df, highlight_price=None):
    """Create price vs rating scatter plot"""
    fig = px.scatter(
        df, 
        x='price', 
        y='rating',
        color='brand_name',
        hover_data=['model', 'ram_capacity', 'battery_capacity'],
        title='Price vs Rating Analysis',
        labels={'price': 'Price (₹)', 'rating': 'Rating'},
        opacity=0.6
    )
    
    if highlight_price:
        fig.add_vline(x=highlight_price, line_dash="dash", line_color="red", 
                     annotation_text="Your Budget", annotation_position="top")
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_brand_distribution_chart(df):
    """Create brand distribution chart"""
    brand_counts = df['brand_name'].value_counts().head(10)
    
    fig = px.bar(
        x=brand_counts.values,
        y=brand_counts.index,
        orientation='h',
        title='Top 10 Brands by Model Count',
        labels={'x': 'Number of Models', 'y': 'Brand'},
        color=brand_counts.values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_battery_vs_rating_chart(df):
    """Create battery vs rating scatter plot"""
    fig = px.scatter(
        df,
        x='battery_capacity',
        y='rating',
        color='price',
        size='ram_capacity',
        hover_data=['brand_name', 'model'],
        title='Battery Capacity vs Rating',
        labels={'battery_capacity': 'Battery (mAh)', 'rating': 'Rating', 'price': 'Price (₹)'},
        color_continuous_scale='Plasma'
    )
    
    fig.update_layout(height=400)
    return fig

def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">📱 Mobile Recommendation System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Smartphone Recommendations & Rating Predictions</div>', unsafe_allow_html=True)
    
    # Load models
    if not st.session_state.models_loaded:
        with st.spinner('🔄 Loading ML models and data...'):
            try:
                st.session_state.models, st.session_state.df = load_all_models()
                st.session_state.models_loaded = True
                st.success('✅ Models loaded successfully!')
            except Exception as e:
                st.error(f'❌ Error loading models: {e}')
                st.info('Please run the training pipeline first: `python data_preprocessing.py && python model_training.py && python recommendation_engine.py`')
                return
    
    models = st.session_state.models
    df = st.session_state.df
    
    if models is None or df is None:
        st.error('Failed to load models or data. Please check the files.')
        return
    
    # Sidebar for user inputs
    st.sidebar.header('🎯 Your Preferences')
    
    # Price
    min_price = int(df['price'].min())
    max_price = int(df['price'].max())
    price = st.sidebar.slider(
        '💰 Budget (₹)',
        min_value=min_price,
        max_value=max_price,
        value=25000,
        step=1000,
        help='Select your budget range'
    )
    
    # Brand
    brands = ['Any'] + sorted(df['brand_name'].unique().tolist())
    brand = st.sidebar.selectbox('📱 Preferred Brand', brands)
    brand_filter = None if brand == 'Any' else brand
    
    # RAM
    ram_options = sorted(df['ram_capacity'].unique().tolist())
    ram = st.sidebar.select_slider(
        '🧠 RAM (GB)',
        options=ram_options,
        value=8
    )
    
    # Storage
    storage_options = sorted(df['internal_memory'].unique().tolist())
    storage = st.sidebar.select_slider(
        '💾 Storage (GB)',
        options=storage_options,
        value=128
    )
    
    # Battery
    battery = st.sidebar.slider(
        '🔋 Battery Capacity (mAh)',
        min_value=3000,
        max_value=6000,
        value=5000,
        step=100
    )
    
    # Screen size
    screen_size = st.sidebar.slider(
        '📺 Screen Size (inches)',
        min_value=5.0,
        max_value=7.0,
        value=6.5,
        step=0.1
    )
    
    # Refresh rate
    refresh_options = sorted(df['refresh_rate'].unique().tolist())
    refresh_rate = st.sidebar.select_slider(
        '🔄 Refresh Rate (Hz)',
        options=refresh_options,
        value=120
    )
    
    # Camera
    camera = st.sidebar.slider(
        '📷 Primary Camera (MP)',
        min_value=10,
        max_value=200,
        value=50,
        step=5
    )
    
    # 5G
    has_5g = st.sidebar.checkbox('📡 5G Support', value=True)
    
    # Processor
    processor_options = ['Any'] + sorted(df['processor_brand'].unique().tolist())
    processor = st.sidebar.selectbox('⚙️ Processor Brand', processor_options)
    
    # OS
    os_options = ['Any'] + sorted(df['os'].unique().tolist())
    os = st.sidebar.selectbox('💻 Operating System', os_options)
    
    # Predict button
    predict_button = st.sidebar.button('🔮 Get Predictions & Recommendations', type='primary', use_container_width=True)
    
    # Main content
    if predict_button:
        # Prepare user input for prediction
        user_input = {
            'price': price,
            'num_cores': 8,
            'processor_speed': 2.5,
            'battery_capacity': battery,
            'fast_charging': 33.0,
            'ram_capacity': ram,
            'internal_memory': storage,
            'screen_size': screen_size,
            'refresh_rate': refresh_rate,
            'num_rear_cameras': 3,
            'num_front_cameras': 1,
            'primary_camera_rear': camera,
            'primary_camera_front': 16.0,
            'resolution_width': 1080,
            'resolution_height': 2400,
            'has_5g': 1 if has_5g else 0,
            'has_nfc': 0,
            'has_ir_blaster': 0,
            'fast_charging_available': 1,
            'extended_memory_available': 0,
            'extended_upto': 0,
            'brand_name': brand_filter if brand_filter else 'samsung',
            'processor_brand': processor if processor != 'Any' else 'snapdragon',
            'os': os if os != 'Any' else 'android'
        }
        
        # Predict rating
        with st.spinner('🔮 Predicting rating...'):
            X_scaled = utils.preprocess_user_input(user_input, models['preprocessor'])
            predicted_rating = models['best_model'].predict(X_scaled)[0]
            predicted_rating = max(0, min(100, predicted_rating))  # Clip to 0-100
        
        # Get recommendations
        with st.spinner('🔍 Finding best matches...'):
            rec_prefs = {
                'price': price,
                'battery_capacity': battery,
                'ram_capacity': ram,
                'internal_memory': storage,
                'screen_size': screen_size,
                'refresh_rate': refresh_rate,
                'primary_camera_rear': camera
            }
            
            recommendations = models['recommendation_engine'].get_recommendations_by_preferences(
                rec_prefs, top_n=5
            )
            
            # Apply additional filters
            if brand_filter:
                recommendations = recommendations[recommendations['brand_name'] == brand_filter]
            
            if has_5g:
                recommendations = recommendations[recommendations['has_5g'] == 1]
            
            # If no results after filtering, get top rated in price range
            if len(recommendations) == 0:
                recommendations = models['recommendation_engine'].get_top_rated_phones(
                    top_n=5, min_price=price*0.8, max_price=price*1.2
                )
                recommendations['match_percentage'] = 75.0
        
        # Display KPIs
        price_range = utils.get_price_category(price)
        avg_match = recommendations['match_percentage'].mean() if len(recommendations) > 0 else 0
        
        st.markdown('---')
        display_kpi_cards(predicted_rating, price_range, avg_match)
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(['📊 Analytics', '🎯 Top Recommendations', '📱 All Matches', '📈 Insights'])
        
        with tab1:
            st.subheader('📊 Market Analysis')
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = create_price_vs_rating_chart(df, highlight_price=price)
                st.plotly_chart(fig1, use_container_width=True)
                
                fig3 = create_battery_vs_rating_chart(df)
                st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                fig2 = create_brand_distribution_chart(df)
                st.plotly_chart(fig2, use_container_width=True)
                
                # Price distribution
                fig4 = px.histogram(
                    df, x='price', nbins=50,
                    title='Price Distribution in Market',
                    labels={'price': 'Price (₹)', 'count': 'Number of Models'},
                    color_discrete_sequence=['#636EFA']
                )
                fig4.add_vline(x=price, line_dash="dash", line_color="red",
                              annotation_text="Your Budget")
                fig4.update_layout(height=400)
                st.plotly_chart(fig4, use_container_width=True)
        
        with tab2:
            st.subheader('🎯 Top 5 Recommendations for You')
            
            if len(recommendations) > 0:
                for idx, (_, phone) in enumerate(recommendations.head(5).iterrows(), 1):
                    display_phone_card(phone, idx, show_match=True)
            else:
                st.warning('No phones found matching your exact criteria. Try adjusting your preferences.')
        
        with tab3:
            st.subheader('📱 All Matching Phones')
            
            # Apply filters to full dataset
            filtered_df = df.copy()
            filtered_df = filtered_df[
                (filtered_df['price'] >= price * 0.7) & 
                (filtered_df['price'] <= price * 1.3)
            ]
            
            if brand_filter:
                filtered_df = filtered_df[filtered_df['brand_name'] == brand_filter]
            
            if has_5g:
                filtered_df = filtered_df[filtered_df['has_5g'] == 1]
            
            filtered_df = filtered_df[filtered_df['ram_capacity'] >= ram]
            filtered_df = filtered_df.sort_values('rating', ascending=False)
            
            if len(filtered_df) > 0:
                st.dataframe(
                    filtered_df[['brand_name', 'model', 'price', 'rating', 'ram_capacity', 
                                'internal_memory', 'battery_capacity', 'primary_camera_rear']].head(20),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info('No phones found in this price range with your specifications.')
        
        with tab4:
            st.subheader('📈 Model Performance & Insights')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('### 🤖 ML Model Information')
                st.info(f"**Best Model:** {models['best_model_name']}")
                
                if models['best_model_name'] in models['metrics']:
                    metrics = models['metrics'][models['best_model_name']]
                    st.metric('Test R² Score', f"{metrics['test_r2']:.4f}")
                    st.metric('Test MAE', f"{metrics['test_mae']:.4f}")
                    st.metric('Test RMSE', f"{metrics['test_rmse']:.4f}")
                
                st.markdown('### 📊 Your Preferences Summary')
                st.write(f"- **Budget:** {utils.format_price(price)}")
                st.write(f"- **RAM:** {ram}GB")
                st.write(f"- **Storage:** {storage}GB")
                st.write(f"- **Battery:** {battery}mAh")
                st.write(f"- **Camera:** {camera}MP")
                st.write(f"- **5G:** {'Yes' if has_5g else 'No'}")
            
            with col2:
                st.markdown('### 🎯 Recommendation Insights')
                
                if len(recommendations) > 0:
                    avg_price = recommendations['price'].mean()
                    avg_rating = recommendations['rating'].mean()
                    
                    st.metric('Average Price of Recommendations', utils.format_price(avg_price))
                    st.metric('Average Rating', f"{avg_rating:.1f}/100")
                    
                    top_brands = recommendations['brand_name'].value_counts()
                    st.markdown('**Top Recommended Brands:**')
                    for brand, count in top_brands.items():
                        st.write(f"- {brand.title()}: {count} model(s)")
                
                st.markdown('### 💡 Tips')
                st.info("""
                - Higher ratings indicate better overall user satisfaction
                - Consider value for money, not just the lowest price
                - Check if the phone has features important to you (5G, NFC, etc.)
                - Battery capacity above 4500mAh is considered good
                - 120Hz refresh rate provides smoother scrolling
                """)
    
    else:
        # Initial view
        st.info('👈 Set your preferences in the sidebar and click "Get Predictions & Recommendations" to see results!')
        
        # Show some statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric('Total Phones', len(df))
        
        with col2:
            st.metric('Brands', df['brand_name'].nunique())
        
        with col3:
            avg_rating = df['rating'].mean()
            st.metric('Avg Rating', f"{avg_rating:.1f}/100")
        
        with col4:
            avg_price = df['price'].mean()
            st.metric('Avg Price', utils.format_price(avg_price))
        
        # Show top rated phones
        st.markdown('---')
        st.subheader('⭐ Top Rated Smartphones')
        
        top_phones = df.nlargest(10, 'rating')
        
        for idx, (_, phone) in enumerate(top_phones.iterrows(), 1):
            display_phone_card(phone, idx, show_match=False)

if __name__ == '__main__':
    main()