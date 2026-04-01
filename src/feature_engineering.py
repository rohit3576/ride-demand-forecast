def create_features(df):
    """
    Create advanced time-based and lag features
    """
    df = df.copy()

    # 📅 Time-based features
    df['day'] = df.index.day
    df['weekday'] = df.index.weekday
    df['month'] = df.index.month
    df['weekofyear'] = df.index.isocalendar().week.astype(int)

    # Weekend flag
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

    # 🔁 Lag features (IMPORTANT 🔥)
    df['lag_1'] = df['rides'].shift(1)
    df['lag_2'] = df['rides'].shift(2)
    df['lag_3'] = df['rides'].shift(3)
    df['lag_7'] = df['rides'].shift(7)
    df['lag_14'] = df['rides'].shift(14)

    # 📊 Rolling statistics (trend + variation)
    df['rolling_mean_3'] = df['rides'].rolling(3).mean()
    df['rolling_mean_7'] = df['rides'].rolling(7).mean()
    df['rolling_std_7'] = df['rides'].rolling(7).std()

    # 🔥 Optional: growth feature
    df['daily_change'] = df['rides'].diff()

    # Drop null values (caused by lag/rolling)
    df.dropna(inplace=True)

    return df