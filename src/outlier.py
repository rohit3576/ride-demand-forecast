import pandas as pd


def detect_outliers_iqr(df):
    """
    Detect outliers using IQR
    """

    Q1 = df['rides'].quantile(0.25)
    Q3 = df['rides'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df['rides'] < lower_bound) | (df['rides'] > upper_bound)]

    return outliers, lower_bound, upper_bound


def remove_outliers_iqr(df):
    """
    Remove outliers
    """

    Q1 = df['rides'].quantile(0.25)
    Q3 = df['rides'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_clean = df[(df['rides'] >= lower_bound) & (df['rides'] <= upper_bound)]

    return df_clean


def cap_outliers(df):
    """
    Cap outliers instead of removing
    """

    Q1 = df['rides'].quantile(0.25)
    Q3 = df['rides'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df['rides'] = df['rides'].clip(lower_bound, upper_bound)

    return df