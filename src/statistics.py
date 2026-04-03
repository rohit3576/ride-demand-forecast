import pandas as pd
import numpy as np
from scipy import stats


def weekend_vs_weekday_test(df):
    """
    Perform t-test between weekday and weekend demand
    """

    df = df.copy()

    # Create weekday/weekend
    df['weekday'] = df.index.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6])

    weekend = df[df['is_weekend']]['rides']
    weekday = df[~df['is_weekend']]['rides']

    # T-test
    t_stat, p_value = stats.ttest_ind(weekend, weekday)

    result = {
        "Weekend Mean": weekend.mean(),
        "Weekday Mean": weekday.mean(),
        "T-Statistic": t_stat,
        "P-Value": p_value
    }

    return result


def confidence_interval(df, confidence=0.95):
    """
    Calculate confidence interval for ride demand
    """

    data = df['rides']
    mean = np.mean(data)
    std_err = stats.sem(data)

    h = std_err * stats.t.ppf((1 + confidence) / 2., len(data)-1)

    return {
        "Mean": mean,
        "Lower Bound": mean - h,
        "Upper Bound": mean + h
    }


def correlation_analysis(df):
    """
    Correlation matrix
    """

    return df.corr()