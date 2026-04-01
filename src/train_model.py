import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA


def train_random_forest(df):
    """
    Train Random Forest model
    """
    X = df.drop('rides', axis=1)
    y = df['rides']

    split = int(len(df) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    rf_pred = model.predict(X_test)

    return model, X_test, y_test, rf_pred


def train_arima(df):
    """
    Train ARIMA model (on target only)
    """
    y = df['rides']

    split = int(len(y) * 0.8)

    train, test = y[:split], y[split:]

    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()

    arima_pred = model_fit.forecast(steps=len(test))

    return test, arima_pred


def evaluate(y_true, pred, model_name="Model"):
    mae = mean_absolute_error(y_true, pred)
    r2 = r2_score(y_true, pred)

    print(f"\n📊 {model_name} Results:")
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")

    return mae, r2


# ✅ TEST BLOCK
if __name__ == "__main__":
    from preprocess import load_data, preprocess_data, aggregate_data
    from feature_engineering import create_features

    # Load pipeline
    df = load_data("data/Uber-Jan-Feb-FOIL.csv")
    df = preprocess_data(df)
    demand = aggregate_data(df)
    df_features = create_features(demand)

    print("\n🚀 Training Models...\n")

    # 🌳 Random Forest
    rf_model, X_test, y_test, rf_pred = train_random_forest(df_features)
    evaluate(y_test, rf_pred, "Random Forest")

    # 📈 ARIMA
    y_test_arima, arima_pred = train_arima(df_features)
    evaluate(y_test_arima, arima_pred, "ARIMA")

    # 🔥 Ensemble (Average)
    min_len = min(len(rf_pred), len(arima_pred))
    ensemble_pred = (rf_pred[:min_len] + arima_pred[:min_len]) / 2

    evaluate(y_test[:min_len], ensemble_pred, "Ensemble (RF + ARIMA)")

    print("\n✅ All Models Trained Successfully!")