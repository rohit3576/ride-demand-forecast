import pandas as pd
import numpy as np


def load_data(file_path):
    """
    Load dataset
    """
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df):
    """
    Complete data cleaning pipeline
    """

    df = df.copy()

    # ✅ 1. Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    print("📊 Columns:", df.columns)

    # ✅ 2. Remove duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"✅ Removed {before - len(df)} duplicate rows")

    # ✅ 3. Handle missing values
    missing_before = df.isnull().sum().sum()

    # Forward fill for time-series
    df.fillna(method='ffill', inplace=True)

    # If still missing → fill with median
    df.fillna(df.median(numeric_only=True), inplace=True)

    print(f"✅ Missing values handled: {missing_before}")

    # ✅ 4. Detect datetime column
    datetime_col = None
    for col in df.columns:
        if 'date' in col or 'time' in col:
            datetime_col = col
            break

    if datetime_col is None:
        raise Exception("❌ No datetime column found")

    print(f"✅ Using datetime column: {datetime_col}")

    # Convert to datetime
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')

    # Drop invalid datetime rows
    df = df.dropna(subset=[datetime_col])

    # Sort & index
    df = df.sort_values(datetime_col)
    df.set_index(datetime_col, inplace=True)

    # ✅ 5. Handle inconsistencies
    # Example: negative trips (invalid)
    if 'trips' in df.columns:
        negative_count = (df['trips'] < 0).sum()
        df = df[df['trips'] >= 0]
        print(f"✅ Removed {negative_count} negative values")

    # ✅ 6. Standardization (scaling optional)
    # Normalize trips if needed
    if 'trips' in df.columns:
        df['trips'] = df['trips'].astype(float)

    return df


def aggregate_data(df):
    """
    Convert data → final time series demand
    """

    if 'trips' in df.columns:
        demand = df.groupby(df.index)['trips'].sum().to_frame(name='rides')
    else:
        demand = df.resample('h').size().to_frame(name='rides')

    # ✅ 7. Outlier handling (light cleaning)
    Q1 = demand['rides'].quantile(0.25)
    Q3 = demand['rides'].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    demand['rides'] = demand['rides'].clip(lower, upper)

    return demand