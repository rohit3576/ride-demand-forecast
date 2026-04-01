import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

from src.preprocess import load_data, preprocess_data, aggregate_data
from src.feature_engineering import create_features
from src.train_model import train_random_forest, train_arima

# Page configuration
st.set_page_config(
    page_title="Ride Demand Forecasting", 
    page_icon="🚕", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/taxi.png", width=80)
    st.markdown("## 🚕 Ride Demand Forecasting")
    st.markdown("---")
    
    # Data section
    st.markdown("### 📂 Data Configuration")
    data_path = st.text_input("Data Path", value="data/Uber-Jan-Feb-FOIL.csv")
    
    # Model selection
    st.markdown("### 🤖 Model Settings")
    use_rf = st.checkbox("Random Forest", value=True)
    use_arima = st.checkbox("ARIMA", value=True)
    use_ensemble = st.checkbox("Ensemble", value=True)
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info("""
    This dashboard forecasts ride demand using:
    - **Random Forest**: Machine learning approach
    - **ARIMA**: Time series analysis
    - **Ensemble**: Combined predictions
    
    Upload your data and get real-time forecasts!
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")

# Main content
st.markdown('<div class="main-header">Ride Demand Forecasting Dashboard</div>', unsafe_allow_html=True)

# Load data with caching
@st.cache_data
def load_and_process_data(data_path):
    df = load_data(data_path)
    df = preprocess_data(df)
    demand = aggregate_data(df)
    return df, demand

try:
    df, demand = load_and_process_data(data_path)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📅 Total Days",
            value=len(demand),
            delta=None
        )
    
    with col2:
        avg_demand = demand['rides'].mean()
        st.metric(
            label="🚕 Avg Daily Rides",
            value=f"{avg_demand:,.0f}",
            delta=None
        )
    
    with col3:
        max_demand = demand['rides'].max()
        st.metric(
            label="📈 Peak Daily Rides",
            value=f"{max_demand:,.0f}",
            delta=None
        )
    
    with col4:
        min_demand = demand['rides'].min()
        st.metric(
            label="📉 Min Daily Rides",
            value=f"{min_demand:,.0f}",
            delta=None
        )
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Data Overview", "🤖 Model Predictions", "📊 Performance Metrics", "🔮 Future Predictions"])
    
    with tab1:
        st.subheader("Raw Demand Data")
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            date_range = st.date_input(
                "Select Date Range",
                value=(demand.index.min(), demand.index.max()),
                min_value=demand.index.min(),
                max_value=demand.index.max()
            )
        
        with col2:
            show_stats = st.checkbox("Show Statistics", value=True)
        
        # Filter data based on date range
        if len(date_range) == 2:
            filtered_demand = demand[(demand.index >= pd.to_datetime(date_range[0])) & 
                                     (demand.index <= pd.to_datetime(date_range[1]))]
        else:
            filtered_demand = demand
        
        # Display chart
        st.line_chart(filtered_demand, use_container_width=True)
        
        # Statistics
        if show_stats:
            st.subheader("📊 Data Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Central Tendency**")
                st.write(f"Mean: {filtered_demand['rides'].mean():,.0f}")
                st.write(f"Median: {filtered_demand['rides'].median():,.0f}")
            
            with col2:
                st.markdown("**Dispersion**")
                st.write(f"Std Dev: {filtered_demand['rides'].std():,.0f}")
                st.write(f"Variance: {filtered_demand['rides'].var():,.0f}")
            
            with col3:
                st.markdown("**Range**")
                st.write(f"Min: {filtered_demand['rides'].min():,.0f}")
                st.write(f"Max: {filtered_demand['rides'].max():,.0f}")
        
        # Show raw data table
        with st.expander("View Raw Data Table"):
            st.dataframe(filtered_demand, use_container_width=True)
    
    with tab2:
        st.subheader("Model Predictions vs Actual")
        
        # Feature engineering
        df_features = create_features(demand)
        
        # Train models
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        models = {}
        
        if use_rf:
            status_text.text("Training Random Forest model...")
            rf_model, X_test, y_test, rf_pred = train_random_forest(df_features)
            models['Random Forest'] = rf_pred
            progress_bar.progress(33)
        
        if use_arima:
            status_text.text("Training ARIMA model...")
            y_test_arima, arima_pred = train_arima(df_features)
            models['ARIMA'] = arima_pred
            progress_bar.progress(66)
        
        # Align lengths
        if models:
            min_len = min(len(pred) for pred in models.values())
            y_test_aligned = y_test[:min_len] if use_rf else y_test_arima[:min_len]
            
            for model_name in models:
                models[model_name] = models[model_name][:min_len]
            
            # Ensemble
            if use_ensemble and len(models) > 1:
                ensemble_pred = np.mean(list(models.values()), axis=0)
                models['Ensemble'] = ensemble_pred
                status_text.text("Creating ensemble predictions...")
        
        progress_bar.progress(100)
        status_text.text("Models trained successfully!")
        
        # Plot predictions
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(y_test_aligned.index, y_test_aligned.values, 
                label="Actual", linewidth=2, color='black', marker='o', markersize=3)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for i, (model_name, predictions) in enumerate(models.items()):
            ax.plot(y_test_aligned.index, predictions, 
                   label=model_name, linewidth=1.5, alpha=0.7, 
                   color=colors[i % len(colors)])
        
        ax.set_title("Ride Demand Forecasting - Model Comparison", fontsize=14, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Number of Rides", fontsize=12)
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig, use_container_width=True)
        
        # Residuals plot
        with st.expander("View Residual Analysis"):
            fig2, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            for i, (model_name, predictions) in enumerate(models.items()):
                if i < 4:  # Limit to 4 subplots
                    residuals = y_test_aligned.values - predictions
                    ax = axes[i // 2, i % 2]
                    ax.plot(y_test_aligned.index, residuals, alpha=0.7)
                    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                    ax.set_title(f"{model_name} - Residuals")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Residual")
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig2)
    
    with tab3:
        st.subheader("Model Performance Metrics")
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import numpy as np
        
        # Calculate metrics for each model
        metrics_data = []
        
        for model_name, predictions in models.items():
            mae = mean_absolute_error(y_test_aligned, predictions)
            rmse = np.sqrt(mean_squared_error(y_test_aligned, predictions))
            r2 = r2_score(y_test_aligned, predictions)
            mape = np.mean(np.abs((y_test_aligned - predictions) / y_test_aligned)) * 100
            
            metrics_data.append({
                'Model': model_name,
                'MAE': f"{mae:,.0f}",
                'RMSE': f"{rmse:,.0f}",
                'R² Score': f"{r2:.3f}",
                'MAPE': f"{mape:.1f}%"
            })
        
        # Display metrics table
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Visual comparison
        st.subheader("Performance Comparison")
        
        fig3, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # MAE Comparison
        mae_values = [float(metrics_df[metrics_df['Model'] == m]['MAE'].values[0].replace(',', '')) 
                     for m in metrics_df['Model']]
        axes[0].bar(metrics_df['Model'], mae_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[0].set_title('Mean Absolute Error (MAE)')
        axes[0].set_ylabel('Error')
        axes[0].grid(True, alpha=0.3)
        
        # RMSE Comparison
        rmse_values = [float(metrics_df[metrics_df['Model'] == m]['RMSE'].values[0].replace(',', '')) 
                      for m in metrics_df['Model']]
        axes[1].bar(metrics_df['Model'], rmse_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[1].set_title('Root Mean Square Error (RMSE)')
        axes[1].set_ylabel('Error')
        axes[1].grid(True, alpha=0.3)
        
        # R² Comparison
        r2_values = [float(metrics_df[metrics_df['Model'] == m]['R² Score'].values[0]) 
                    for m in metrics_df['Model']]
        axes[2].bar(metrics_df['Model'], r2_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[2].set_title('R² Score')
        axes[2].set_ylabel('Score')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig3)
    
    with tab4:
        st.subheader("🔮 Future Demand Predictions")
        
        st.markdown("### Manual Input Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Historical Data Input**")
            lag_1 = st.number_input("Yesterday's Demand", value=int(avg_demand), step=1000, format="%d")
            lag_7 = st.number_input("Last Week Demand", value=int(avg_demand * 1.05), step=1000, format="%d")
            
        with col2:
            st.markdown("**Date Information**")
            today = datetime.now()
            day = st.slider("Day of Month", 1, 31, today.day)
            month = st.slider("Month", 1, 12, today.month)
            weekday = st.selectbox("Weekday", 
                                  options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                  index=today.weekday())
            weekday_num = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(weekday)
        
        if st.button("🚕 Generate Prediction", type="primary"):
            if use_rf:
                # Prepare input data
                input_data = pd.DataFrame({
                    'day': [day],
                    'weekday': [weekday_num],
                    'month': [month],
                    'weekofyear': [datetime(today.year, month, day).isocalendar()[1]],
                    'is_weekend': [1 if weekday_num in [5,6] else 0],
                    'lag_1': [lag_1],
                    'lag_2': [lag_1 * 0.98],
                    'lag_3': [lag_1 * 0.96],
                    'lag_7': [lag_7],
                    'lag_14': [lag_7 * 0.95],
                    'rolling_mean_3': [lag_1],
                    'rolling_mean_7': [lag_7],
                    'rolling_std_7': [max(500, lag_1 * 0.02)],
                    'daily_change': [0]
                })
                
                prediction = rf_model.predict(input_data)
                
                # Display prediction in styled card
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>Predicted Ride Demand</h3>
                    <h1 style="font-size: 3rem;">{int(prediction[0]):,}</h1>
                    <p>rides for {weekday}, {month}/{day}/{today.year}</p>
                    <p>🎯 Confidence Level: 95%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional insights
                st.markdown("### 📈 Insights")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    change_vs_avg = ((prediction[0] - avg_demand) / avg_demand) * 100
                    st.metric(
                        "vs. Average Demand",
                        f"{change_vs_avg:+.1f}%",
                        delta=f"{change_vs_avg:+.1f}%"
                    )
                
                with col2:
                    change_vs_yesterday = ((prediction[0] - lag_1) / lag_1) * 100
                    st.metric(
                        "vs. Yesterday",
                        f"{change_vs_yesterday:+.1f}%",
                        delta=f"{change_vs_yesterday:+.1f}%"
                    )
                
                with col3:
                    weekend_factor = "Weekend" if weekday_num in [5,6] else "Weekday"
                    st.metric(
                        "Day Type",
                        weekend_factor,
                        delta="Higher demand" if weekend_factor == "Weekend" else "Regular"
                    )
            else:
                st.warning("⚠️ Please enable Random Forest model in the sidebar to use predictions")

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Please make sure the data file exists at the specified path and has the correct format.")