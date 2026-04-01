import pandas as pd


def load_data(file_path):
    """
    Load dataset
    """
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df):
    """
    Clean and prepare datetime column
    """
    # Remove extra spaces
    df.columns = df.columns.str.strip()

    print("📊 Available Columns:", df.columns)

    # Auto-detect datetime column
    datetime_col = None
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            datetime_col = col
            break

    if datetime_col is None:
        raise Exception("❌ No datetime column found")

    print(f"✅ Using datetime column: {datetime_col}")

    # Convert to datetime
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Sort
    df = df.sort_values(datetime_col)

    # Set index
    df.set_index(datetime_col, inplace=True)

    return df


def aggregate_data(df):
    """
    Convert data → final time series demand
    """
    if 'trips' in df.columns:
        # 🔥 IMPORTANT: group by date (multiple base stations)
        demand = df.groupby(df.index)['trips'].sum().to_frame(name='rides')
    else:
        # fallback for raw datasets
        demand = df.resample('h').size().to_frame(name='rides')

    return demand


# ✅ TEST BLOCK
if __name__ == "__main__":
    df = load_data("data/Uber-Jan-Feb-FOIL.csv")

    df = preprocess_data(df)

    demand = aggregate_data(df)

    print("\n✅ Data Processed Successfully!\n")
    print(demand.head())