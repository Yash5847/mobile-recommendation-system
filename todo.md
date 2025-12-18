# Mobile Recommendation System - Development Plan

## Project Overview
Build a production-ready Mobile Recommendation System with ML models, recommendation engine, and interactive Streamlit dashboard.

## Dataset Features
- **Brand & Model**: brand_name, model
- **Price & Rating**: price, rating (target variable)
- **Connectivity**: has_5g, has_nfc, has_ir_blaster
- **Processor**: processor_brand, num_cores, processor_speed
- **Battery**: battery_capacity, fast_charging_available, fast_charging
- **Memory**: ram_capacity, internal_memory, extended_memory_available, extended_upto
- **Display**: screen_size, refresh_rate, resolution_width, resolution_height
- **Camera**: num_rear_cameras, num_front_cameras, primary_camera_rear, primary_camera_front
- **OS**: os

## Development Tasks

### Phase 1: Data Science & ML Pipeline
1. **data_preprocessing.py** - Data cleaning, handling missing values, feature engineering
2. **eda_analysis.py** - Exploratory data analysis with visualizations
3. **model_training.py** - Train 4 ML models (Linear Regression, KNN, Random Forest, Gradient Boosting)
4. **model_evaluation.py** - Evaluate models using MAE, RMSE, R²

### Phase 2: Recommendation System
5. **recommendation_engine.py** - Content-based recommendation using KNN/cosine similarity

### Phase 3: Streamlit Dashboard
6. **app.py** - Main Streamlit application with:
   - Sidebar for user inputs
   - KPI cards (predicted rating, budget range, recommendation score)
   - Interactive charts (Price vs Rating, Brand distribution, Battery vs Rating)
   - Recommendation display (table + cards)
   - Clean, modern UI with icons

### Phase 4: Utilities & Configuration
7. **utils.py** - Helper functions for loading models, preprocessing user inputs
8. **config.py** - Configuration constants and paths

### Phase 5: Documentation
9. **README.md** - Complete project documentation
10. **requirements.txt** - All dependencies

## File Structure
```
streamlit_template/
├── app.py                      # Main Streamlit dashboard
├── data_preprocessing.py       # Data cleaning & feature engineering
├── eda_analysis.py            # Exploratory data analysis
├── model_training.py          # ML model training
├── model_evaluation.py        # Model evaluation
├── recommendation_engine.py   # Recommendation system
├── utils.py                   # Helper functions
├── config.py                  # Configuration
├── requirements.txt           # Dependencies
├── README.md                  # Documentation
├── data/
│   └── Smartphones_cleaned_dataset.csv
├── models/                    # Saved models
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── recommendation_model.pkl
├── outputs/                   # EDA visualizations
│   └── eda_plots/
└── notebooks/                 # Optional Jupyter notebooks
```

## Tech Stack
- **ML/DS**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Dashboard**: Streamlit
- **Model Persistence**: joblib

## Success Criteria
✓ Clean, modular code with best practices
✓ 4 trained ML models with evaluation metrics
✓ Working recommendation system
✓ Interactive, attractive dashboard
✓ Complete documentation
✓ Production-ready deployment instructions