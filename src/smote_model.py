import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def prepare_classification_data(df):
    """
    Convert regression → classification
    """

    df = df.copy()

    # Create demand categories
    df['demand_level'] = pd.qcut(
        df['rides'], 
        q=3, 
        labels=['Low', 'Medium', 'High']
    )

    # Features & target
    X = df.drop(['rides', 'demand_level'], axis=1)
    y = df['demand_level']

    return X, y


def apply_smote(X, y):
    """
    Apply SMOTE
    """

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled


def train_classifier(X, y):
    """
    Train Random Forest classifier
    """

    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return model, y_test, preds


def evaluate_model(y_test, preds):
    """
    Evaluate classification performance
    """

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    return acc, report