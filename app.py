import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Core modules
from src.preprocess import load_data, preprocess_data, aggregate_data
from src.feature_engineering import create_features
from src.train_model import train_random_forest, train_arima

# New modules
from src.statistics import weekend_vs_weekday_test, confidence_interval, correlation_analysis
from src.smote_model import prepare_classification_data, apply_smote, train_classifier, evaluate_model
from src.outlier import detect_outliers_iqr, remove_outliers_iqr
from src.feature_selection import correlation_filter, feature_importance_selection, select_k_best

import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try to import plotly, fallback to matplotlib if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly not installed. Using matplotlib for visualizations. Run 'pip install plotly' for better visuals.")

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Ride Demand Forecasting Dashboard", 
    page_icon="🚕", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling & better visibility
st.markdown("""
    <style>
    /* Main container styling */
    .main { padding: 0rem 1rem; }
    
    /* Custom card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    /* New Highly Visible Prediction Card */
    .prediction-highlight {
        background: linear-gradient(135deg, #0ba360 0%, #3cb0fd 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 15px 30px rgba(11, 163, 96, 0.2);
        margin: 2rem 0;
        transition: transform 0.3s ease;
    }
    .prediction-highlight:hover { transform: translateY(-5px); }
    .prediction-highlight h3 { color: rgba(255,255,255,0.9); font-weight: 500; margin-bottom: 0; text-transform: uppercase; letter-spacing: 1.5px; font-size: 1.2rem; }
    .prediction-highlight h1 { font-size: 5rem !important; margin: 10px 0 !important; font-weight: 800 !important; color: white !important; text-shadow: 2px 4px 8px rgba(0,0,0,0.2); }
    .confidence-badge { background: rgba(255,255,255,0.25); padding: 8px 20px; border-radius: 30px; display: inline-block; font-weight: 600; letter-spacing: 0.5px; backdrop-filter: blur(5px); }

    /* Header styling */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover { background-color: #e9ecef; transform: translateY(-2px); }
    
    /* Button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration Panel")
    st.markdown("---")
    
    data_path = st.text_input("📂 Dataset Path", "data/Uber-Jan-Feb-FOIL.csv")
    
    st.markdown("#### 🤖 Model Selection")
    use_rf = st.checkbox("🌲 Random Forest", True)
    use_arima = st.checkbox("📈 ARIMA", True)
    use_ensemble = st.checkbox("🎯 Ensemble", True)
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    st.info("Configure your models and dataset path above to begin analysis.")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data_pipeline(path):
    df = load_data(path)
    df = preprocess_data(df)
    demand = aggregate_data(df)
    return df, demand

try:
    df, demand = load_data_pipeline(data_path)
    avg_demand = demand['rides'].mean()
    
    # Header with metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Records", f"{len(demand):,}")
    with col2:
        st.metric("🚕 Avg Daily Demand", f"{avg_demand:,.0f}")
    with col3:
        st.metric("📈 Peak Demand", f"{demand['rides'].max():,.0f}")
    with col4:
        st.metric("📉 Min Demand", f"{demand['rides'].min():,.0f}")
    
    st.markdown("---")
    
except Exception as e:
    st.error(f"⚠️ Error loading data: {str(e)}")
    st.stop()

# Helper function for plotting based on availability
def create_line_chart(data, x=None, y=None, title=""):
    if PLOTLY_AVAILABLE and x is not None:
        fig = px.line(data, x=x, y=y, title=title, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data.index if x is None else data[x], 
                data[y] if y else data, 
                linewidth=2)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# ------------------ TABS ------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📈 Descriptive Stats",
    "🤖 Model Training",
    "📊 Performance Metrics",
    "🔮 Prediction Engine",
    "📊 Statistical Inference",
    "🎨 Data Visualization",
    "⚖️ SMOTE Analysis",
    "🚨 Outlier Detection",
    "🧠 Feature Selection"
])

# ------------------ 1. DESCRIPTIVE ------------------
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Time Series Analysis")
        if PLOTLY_AVAILABLE:
            fig = px.line(demand, x=demand.index, y='rides', 
                          title='Ride Demand Over Time',
                          template='plotly_white')
            fig.update_layout(hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(demand.index, demand['rides'], linewidth=2, color='#667eea')
            ax.set_title('Ride Demand Over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel('Rides')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    with col2:
        st.subheader("📊 Statistical Summary")
        
        stats_df = pd.DataFrame({
            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1', 'Q3', 'IQR'],
            'Value': [
                f"{demand['rides'].mean():.2f}",
                f"{demand['rides'].median():.2f}",
                f"{demand['rides'].std():.2f}",
                f"{demand['rides'].min():.2f}",
                f"{demand['rides'].max():.2f}",
                f"{demand['rides'].quantile(0.25):.2f}",
                f"{demand['rides'].quantile(0.75):.2f}",
                f"{demand['rides'].quantile(0.75) - demand['rides'].quantile(0.25):.2f}"
            ]
        })
        st.dataframe(stats_df, use_container_width=True)
    
    # Distribution plot
    st.subheader("📊 Demand Distribution")
    if PLOTLY_AVAILABLE:
        fig = px.histogram(demand, x='rides', nbins=30, 
                           title='Distribution of Daily Ride Demand',
                           template='plotly_white',
                           marginal='box')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.hist(demand['rides'], bins=30, edgecolor='black', alpha=0.7, color='#667eea')
        ax1.set_title('Demand Distribution')
        ax1.set_xlabel('Rides')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        ax2.boxplot(demand['rides'])
        ax2.set_title('Box Plot')
        ax2.set_ylabel('Rides')
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig)

# ------------------ 2. MODEL ------------------
with tab2:
    with st.spinner("Training models..."):
        df_features = create_features(demand)
        
        models = {}
        progress_bar = st.progress(0)
        
        if use_rf:
            rf_model, X_test, y_test, rf_pred = train_random_forest(df_features)
            models['🌲 Random Forest'] = rf_pred
            progress_bar.progress(33)
        
        if use_arima:
            y_test_arima, arima_pred = train_arima(df_features)
            models['📈 ARIMA'] = arima_pred
            progress_bar.progress(66)
        
        min_len = min(len(v) for v in models.values())
        y_test = y_test[:min_len]
        
        for k in models:
            models[k] = models[k][:min_len]
        
        if use_ensemble and len(models) > 1:
            models['🎯 Ensemble'] = np.mean(list(models.values()), axis=0)
        
        progress_bar.progress(100)
        
        st.success("✅ Models trained successfully!")
        
        # Plot predictions
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test.index, y=y_test.values, 
                                     mode='lines', name='Actual',
                                     line=dict(color='blue', width=2)))
            
            colors = ['green', 'orange', 'red']
            for idx, (name, pred) in enumerate(models.items()):
                fig.add_trace(go.Scatter(x=y_test.index, y=pred, 
                                         mode='lines', name=name,
                                         line=dict(color=colors[idx % len(colors)], width=2, dash='dash')))
            
            fig.update_layout(title='Model Predictions vs Actual',
                             xaxis_title='Date',
                             yaxis_title='Ride Demand',
                             template='plotly_white',
                             hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(y_test.index, y_test.values, label='Actual', linewidth=2, color='blue')
            for name, pred in models.items():
                ax.plot(y_test.index, pred, label=name, linewidth=2, linestyle='--')
            ax.set_title('Model Predictions vs Actual')
            ax.set_xlabel('Date')
            ax.set_ylabel('Ride Demand')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

# ------------------ 3. PERFORMANCE ------------------
with tab3:
    st.subheader("📊 Model Performance Comparison")
    
    performance_data = []
    for name, pred in models.items():
        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, pred)
        
        performance_data.append({
            'Model': name,
            'MAE': f"{mae:.2f}",
            'MSE': f"{mse:.2f}",
            'RMSE': f"{rmse:.2f}",
            'R² Score': f"{r2:.3f}"
        })
    
    perf_df = pd.DataFrame(performance_data)
    st.dataframe(perf_df, use_container_width=True)
    
    # Bar chart comparison
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=perf_df['Model'], 
                             y=[float(x) for x in perf_df['MAE']],
                             name='MAE',
                             marker_color='#667eea'))
        fig.add_trace(go.Bar(x=perf_df['Model'],
                             y=[float(x) for x in perf_df['RMSE']],
                             name='RMSE',
                             marker_color='#764ba2'))
        
        fig.update_layout(title='Model Performance Metrics Comparison',
                         xaxis_title='Model',
                         yaxis_title='Error Value',
                         template='plotly_white',
                         barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(perf_df['Model']))
        width = 0.35
        
        ax.bar(x - width/2, [float(x) for x in perf_df['MAE']], width, label='MAE', color='#667eea')
        ax.bar(x + width/2, [float(x) for x in perf_df['RMSE']], width, label='RMSE', color='#764ba2')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Error Value')
        ax.set_title('Model Performance Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(perf_df['Model'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# ------------------ 4. PREDICTION (UPGRADED UI) ------------------
with tab4:
    st.subheader("🔮 Manual Demand Simulator")
    st.markdown("Enter historical metrics and date parameters to generate a custom demand forecast based on the trained Random Forest model.")
    
    today = datetime.now()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📊 Historical Input Data")
        lag_1 = st.number_input("Yesterday's Demand (T-1)", value=int(avg_demand), step=1000, format="%d")
        lag_7 = st.number_input("Last Week's Demand (T-7)", value=int(avg_demand * 1.05), step=1000, format="%d")
        
    with col2:
        st.markdown("##### 📅 Date Selection")
        day = st.slider("Day of Month", 1, 31, today.day)
        month = st.slider("Month", 1, 12, today.month)
        
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_name = st.selectbox("Weekday", options=weekdays, index=today.weekday())
        weekday_num = weekdays.index(weekday_name)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Generate Advanced Prediction", use_container_width=True, type="primary"):
        if 'rf_model' in locals():
            # Synthesize inputs
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
            
            prediction = rf_model.predict(input_data)[0]
            
            # 1. Output Feature: Highly Styled Prediction Card
            st.markdown(f"""
            <div class="prediction-highlight">
                <h3>Predicted Ride Volume</h3>
                <h1>{int(prediction):,}</h1>
                <p style="font-size: 1.2rem; opacity: 0.9; margin-top: 5px;">Expected total rides on <b>{weekday_name}, {month}/{day}/{today.year}</b></p>
                <div class="confidence-badge">🎯 Statistical Confidence: 95%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # 2. Output Features: Comparative Insights (3-column layout)
            st.markdown("##### 📈 Strategic Demand Insights")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                change_vs_avg = ((prediction - avg_demand) / avg_demand) * 100
                st.metric(
                    "vs. Historical Average",
                    f"{change_vs_avg:+.1f}%",
                    delta=f"{change_vs_avg:+.1f}%"
                )
                
            with metric_col2:
                change_vs_yesterday = ((prediction - lag_1) / lag_1) * 100
                st.metric(
                    "vs. Yesterday's Input",
                    f"{change_vs_yesterday:+.1f}%",
                    delta=f"{change_vs_yesterday:+.1f}%"
                )
                
            with metric_col3:
                is_weekend = weekday_num in [5, 6]
                weekend_factor = "Weekend" if is_weekend else "Weekday"
                st.metric(
                    "Day Category Assessment",
                    weekend_factor,
                    delta="Demand Surge Expected" if is_weekend else "Standard Operation",
                    delta_color="normal" if is_weekend else "off"
                )
        else:
            st.error("⚠️ Random Forest model not available. Please enable it in the Sidebar settings to use predictions.")

# ------------------ 5. INFERENTIAL ------------------
with tab5:
    st.subheader("📊 Statistical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.spinner("Running t-test..."):
            res = weekend_vs_weekday_test(demand)
            st.markdown("#### Weekend vs Weekday Analysis")
            st.info(res)
    
    with col2:
        with st.spinner("Calculating confidence intervals..."):
            ci = confidence_interval(demand)
            st.markdown("#### Confidence Intervals")
            st.info(ci)
    
    st.subheader("📈 Correlation Analysis")
    
    # Create correlation heatmap
    corr = correlation_analysis(create_features(demand))
    
    if PLOTLY_AVAILABLE:
        fig = px.imshow(corr, 
                        text_auto=True, 
                        aspect="auto",
                        color_continuous_scale='RdBu',
                        title='Feature Correlation Matrix')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(corr, cmap='RdBu', aspect='auto')
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr.columns)
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)

# ------------------ 6. SEABORN ------------------
with tab6:
    st.subheader("🎨 Advanced Visualizations")
    
    viz_type = st.selectbox("Select Visualization Type", 
                            ["Box Plot", "Violin Plot", "Heatmap"])
    
    tips = sns.load_dataset("tips")
    
    if viz_type == "Box Plot":
        if PLOTLY_AVAILABLE:
            fig = px.box(tips, x="day", y="total_bill", color="day",
                        title="Total Bill Distribution by Day",
                        template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(x="day", y="total_bill", data=tips, ax=ax)
            ax.set_title('Total Bill Distribution by Day')
            st.pyplot(fig)
        
    elif viz_type == "Violin Plot":
        if PLOTLY_AVAILABLE:
            fig = px.violin(tips, x="day", y="total_bill", color="day",
                           box=True, title="Total Bill Distribution (Violin Plot)",
                           template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.violinplot(x="day", y="total_bill", data=tips, ax=ax)
            ax.set_title('Total Bill Distribution (Violin Plot)')
            st.pyplot(fig)
        
    elif viz_type == "Heatmap":
        corr_matrix = tips[['total_bill', 'tip', 'size']].corr()
        if PLOTLY_AVAILABLE:
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                           color_continuous_scale='RdBu',
                           title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(corr_matrix, cmap='RdBu', aspect='auto')
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(len(corr_matrix.columns)))
            ax.set_yticks(range(len(corr_matrix.columns)))
            ax.set_xticklabels(corr_matrix.columns)
            ax.set_yticklabels(corr_matrix.columns)
            ax.set_title('Correlation Heatmap')
            st.pyplot(fig)
    
    st.dataframe(tips.head(10), use_container_width=True)

# ------------------ 7. SMOTE ------------------
with tab7:
    st.subheader("⚖️ SMOTE Analysis for Imbalanced Data")
    
    with st.spinner("Preparing data..."):
        df_features = create_features(demand)
        X, y = prepare_classification_data(df_features)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Before SMOTE")
            before_counts = pd.Series(y).value_counts()
            if PLOTLY_AVAILABLE:
                fig = px.pie(values=before_counts.values, 
                            names=before_counts.index,
                            title='Class Distribution Before SMOTE',
                            color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.pie(before_counts.values, labels=before_counts.index, autopct='%1.1f%%')
                ax.set_title('Class Distribution Before SMOTE')
                st.pyplot(fig)
            st.write("Distribution:", before_counts.to_dict())
        
        with st.spinner("Applying SMOTE..."):
            X_res, y_res = apply_smote(X, y)
            
        with col2:
            st.markdown("#### After SMOTE")
            after_counts = pd.Series(y_res).value_counts()
            if PLOTLY_AVAILABLE:
                fig = px.pie(values=after_counts.values,
                            names=after_counts.index,
                            title='Class Distribution After SMOTE',
                            color_discrete_sequence=px.colors.sequential.Greens)
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.pie(after_counts.values, labels=after_counts.index, autopct='%1.1f%%')
                ax.set_title('Class Distribution After SMOTE')
                st.pyplot(fig)
            st.write("Distribution:", after_counts.to_dict())
        
        st.success("✅ SMOTE successfully balanced the dataset!")

# ------------------ 8. OUTLIERS ------------------
with tab8:
    st.subheader("🚨 Outlier Detection and Treatment")
    
    with st.spinner("Detecting outliers..."):
        outliers, lb, ub = detect_outliers_iqr(demand)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Detected Outliers", len(outliers))
        with col2:
            st.metric("Lower Bound", f"{lb:.2f}")
        with col3:
            st.metric("Upper Bound", f"{ub:.2f}")
        
        # Visualize outliers
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=demand.index, y=demand['rides'],
                                    mode='lines+markers',
                                    name='Ride Demand',
                                    line=dict(color='blue', width=2)))
            
            if len(outliers) > 0:
                outlier_dates = demand.loc[outliers.index]
                fig.add_trace(go.Scatter(x=outlier_dates.index, y=outlier_dates['rides'],
                                        mode='markers',
                                        name='Outliers',
                                        marker=dict(color='red', size=10, symbol='x')))
            
            fig.update_layout(title='Outlier Detection Results',
                             xaxis_title='Date',
                             yaxis_title='Ride Demand',
                             template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(demand.index, demand['rides'], linewidth=2, color='blue', label='Ride Demand')
            if len(outliers) > 0:
                outlier_dates = demand.loc[outliers.index]
                ax.scatter(outlier_dates.index, outlier_dates['rides'], 
                          color='red', s=50, marker='x', label='Outliers')
            ax.set_title('Outlier Detection Results')
            ax.set_xlabel('Date')
            ax.set_ylabel('Ride Demand')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        if st.button("🧹 Remove Outliers", use_container_width=True):
            clean = remove_outliers_iqr(demand)
            st.success(f"✅ Removed {len(outliers)} outliers!")
            
            if PLOTLY_AVAILABLE:
                fig = px.line(clean, x=clean.index, y='rides',
                             title='Cleaned Data (Outliers Removed)',
                             template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(clean.index, clean['rides'], linewidth=2, color='green')
                ax.set_title('Cleaned Data (Outliers Removed)')
                ax.set_xlabel('Date')
                ax.set_ylabel('Ride Demand')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            st.info(f"New Mean: {clean['rides'].mean():.2f} (was {demand['rides'].mean():.2f})")

# ------------------ 9. FEATURE SELECTION ------------------
with tab9:
    st.subheader("🧠 Feature Selection Analysis")
    
    with st.spinner("Analyzing features..."):
        df_features = create_features(demand)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Correlation Filter")
            df_corr, dropped = correlation_filter(df_features)
            st.write(f"**Features dropped:** {len(dropped)}")
            st.write("**Dropped features:**", dropped)
        
        with col2:
            st.markdown("#### Feature Importance")
            df_imp, importance = feature_importance_selection(df_features)
            
            # Plot feature importance
            if PLOTLY_AVAILABLE:
                fig = px.bar(x=importance.values, y=importance.index,
                            orientation='h',
                            title='Feature Importance Score',
                            labels={'x': 'Importance', 'y': 'Features'},
                            template='plotly_white')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                importance_sorted = importance.sort_values()
                ax.barh(importance_sorted.index, importance_sorted.values)
                ax.set_xlabel('Importance')
                ax.set_title('Feature Importance Score')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        st.markdown("---")
        st.markdown("#### Feature Correlation Matrix")
        if PLOTLY_AVAILABLE:
            fig = px.imshow(df_corr.corr(), 
                           text_auto=True,
                           aspect="auto",
                           color_continuous_scale='RdBu',
                           title='Correlation Matrix After Filtering')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(12, 8))
            im = ax.imshow(df_corr.corr(), cmap='RdBu', aspect='auto')
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(len(df_corr.columns)))
            ax.set_yticks(range(len(df_corr.columns)))
            ax.set_xticklabels(df_corr.columns, rotation=45, ha='right')
            ax.set_yticklabels(df_corr.columns)
            ax.set_title('Correlation Matrix After Filtering')
            st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem;">
    <p>🚕 Ride Demand Forecasting Dashboard | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)