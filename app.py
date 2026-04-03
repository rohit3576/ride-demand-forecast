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

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Ride Demand Forecasting", page_icon="🚕", layout="wide")

st.title("🚕 Ride Demand Forecasting Dashboard")

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("Settings")

    data_path = st.text_input("Dataset Path", "data/Uber-Jan-Feb-FOIL.csv")

    use_rf = st.checkbox("Random Forest", True)
    use_arima = st.checkbox("ARIMA", True)
    use_ensemble = st.checkbox("Ensemble", True)

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data_pipeline(path):
    df = load_data(path)
    df = preprocess_data(df)
    demand = aggregate_data(df)
    return df, demand

df, demand = load_data_pipeline(data_path)

avg_demand = demand['rides'].mean()

# ------------------ TABS ------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📈 Descriptive Stats",
    "🤖 Model",
    "📊 Performance",
    "🔮 Prediction",
    "📊 Inferential",
    "🍽️ Seaborn",
    "⚖️ SMOTE",
    "🚨 Outliers",
    "🧠 Feature Selection"
])

# ------------------ 1. DESCRIPTIVE ------------------
with tab1:
    st.line_chart(demand)

    st.write("Mean:", demand['rides'].mean())
    st.write("Median:", demand['rides'].median())
    st.write("Std:", demand['rides'].std())
    st.write("Min:", demand['rides'].min())
    st.write("Max:", demand['rides'].max())

# ------------------ 2. MODEL ------------------
with tab2:
    df_features = create_features(demand)

    models = {}

    if use_rf:
        rf_model, X_test, y_test, rf_pred = train_random_forest(df_features)
        models['RF'] = rf_pred

    if use_arima:
        y_test_arima, arima_pred = train_arima(df_features)
        models['ARIMA'] = arima_pred

    min_len = min(len(v) for v in models.values())
    y_test = y_test[:min_len]

    for k in models:
        models[k] = models[k][:min_len]

    if use_ensemble and len(models) > 1:
        models['Ensemble'] = np.mean(list(models.values()), axis=0)

    fig, ax = plt.subplots()

    ax.plot(y_test.index, y_test.values, label="Actual")

    for name, pred in models.items():
        ax.plot(y_test.index, pred, label=name)

    ax.legend()
    st.pyplot(fig)

# ------------------ 3. PERFORMANCE ------------------
with tab3:
    from sklearn.metrics import mean_absolute_error

    for name, pred in models.items():
        mae = mean_absolute_error(y_test, pred)
        st.write(f"{name} MAE:", mae)

# ------------------ 4. PREDICTION ------------------
with tab4:
    lag_1 = st.number_input("Yesterday Demand", value=int(avg_demand))
    lag_7 = st.number_input("Last Week Demand", value=int(avg_demand))

    if st.button("Predict"):
        input_data = pd.DataFrame({
            'day':[1],
            'weekday':[1],
            'month':[1],
            'weekofyear':[1],
            'is_weekend':[0],
            'lag_1':[lag_1],
            'lag_2':[lag_1],
            'lag_3':[lag_1],
            'lag_7':[lag_7],
            'lag_14':[lag_7],
            'rolling_mean_3':[lag_1],
            'rolling_mean_7':[lag_7],
            'rolling_std_7':[100],
            'daily_change':[0]
        })

        pred = rf_model.predict(input_data)
        st.success(f"Predicted Demand: {int(pred[0])}")

# ------------------ 5. INFERENTIAL ------------------
with tab5:
    res = weekend_vs_weekday_test(demand)
    st.write(res)

    ci = confidence_interval(demand)
    st.write(ci)

    corr = correlation_analysis(create_features(demand))
    sns.heatmap(corr)
    st.pyplot()

# ------------------ 6. SEABORN ------------------
with tab6:
    tips = sns.load_dataset("tips")

    st.dataframe(tips.head())

    sns.boxplot(x="day", y="total_bill", data=tips)
    st.pyplot()

# ------------------ 7. SMOTE ------------------
with tab7:
    df_features = create_features(demand)
    X, y = prepare_classification_data(df_features)

    st.write("Before:", y.value_counts())

    X_res, y_res = apply_smote(X, y)
    st.write("After:", pd.Series(y_res).value_counts())

# ------------------ 8. OUTLIERS ------------------
with tab8:
    outliers, lb, ub = detect_outliers_iqr(demand)
    st.write("Outliers:", len(outliers))

    clean = remove_outliers_iqr(demand)
    st.line_chart(clean)

# ------------------ 9. FEATURE SELECTION ------------------
with tab9:
    df_features = create_features(demand)

    df_corr, dropped = correlation_filter(df_features)
    st.write("Dropped:", dropped)

    df_imp, importance = feature_importance_selection(df_features)
    st.bar_chart(importance)