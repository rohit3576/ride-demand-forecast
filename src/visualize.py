import matplotlib.pyplot as plt


def plot_predictions(y_test, rf_pred, arima_pred, ensemble_pred):
    """
    Plot Actual vs Predictions with correct time axis
    """

    plt.figure(figsize=(12, 6))

    # ✅ Use datetime index
    time_index = y_test.index

    # Actual values
    plt.plot(time_index, y_test.values, label="Actual", linewidth=2)

    # Predictions
    plt.plot(time_index, rf_pred, label="Random Forest")
    plt.plot(time_index, arima_pred, label="ARIMA")
    plt.plot(time_index, ensemble_pred, label="Ensemble")

    plt.title("🚕 Ride Demand Forecasting")
    plt.xlabel("Date")
    plt.ylabel("Number of Rides")
    plt.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ✅ TEST BLOCK
if __name__ == "__main__":
    from preprocess import load_data, preprocess_data, aggregate_data
    from feature_engineering import create_features
    from train_model import train_random_forest, train_arima

    # Load pipeline
    df = load_data("data/Uber-Jan-Feb-FOIL.csv")
    df = preprocess_data(df)
    demand = aggregate_data(df)
    df_features = create_features(demand)

    # Train models
    rf_model, X_test, y_test, rf_pred = train_random_forest(df_features)
    y_test_arima, arima_pred = train_arima(df_features)

    # Ensure same length
    min_len = min(len(rf_pred), len(arima_pred))
    y_test = y_test[:min_len]
    rf_pred = rf_pred[:min_len]
    arima_pred = arima_pred[:min_len]

    # Ensemble
    ensemble_pred = (rf_pred + arima_pred) / 2

    # Plot
    plot_predictions(y_test, rf_pred, arima_pred, ensemble_pred)