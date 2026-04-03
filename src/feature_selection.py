import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression


def correlation_filter(df, threshold=0.9):
    """
    Remove highly correlated features
    """

    corr_matrix = df.corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    df_reduced = df.drop(columns=to_drop)

    return df_reduced, to_drop


def feature_importance_selection(df, top_n=8):
    """
    Select top features using Random Forest importance
    """

    X = df.drop('rides', axis=1)
    y = df['rides']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    importance = pd.Series(model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)

    selected_features = importance.head(top_n).index.tolist()

    df_selected = df[selected_features + ['rides']]

    return df_selected, importance


def select_k_best(df, k=8):
    """
    Select top k features using statistical test
    """

    X = df.drop('rides', axis=1)
    y = df['rides']

    selector = SelectKBest(score_func=f_regression, k=k)
    X_new = selector.fit_transform(X, y)

    selected_cols = X.columns[selector.get_support()].tolist()

    df_selected = df[selected_cols + ['rides']]

    return df_selected, selected_cols