"""Data loading and preprocessing for the orders dataset."""

import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data")
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "orders.csv")

TARGET = "status"
POSITIVE_LABEL = "delivered"
DROP_COLS = ["order_id", "customer_id", "shipping_address", "delivery_date", TARGET]
NUMERIC_COLS = ["num_items", "subtotal", "tax", "shipping", "total"]
CATEGORICAL_COLS = ["payment_method"]
PAYMENT_METHOD_VALUES = ["apple_pay", "credit_card", "debit_card", "paypal"]


def load_data(path: str = None) -> pd.DataFrame:
    """
    Load the dataset CSV and perform minimal cleaning.
    """
    if path is None:
        path = DEFAULT_DATA_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    
    df = pd.read_csv(path)
    return df


def preprocess(
    df: pd.DataFrame,
    fit_scaler: bool = True,
) -> tuple:
    """
    Encode categoricals and scale numeric features.
    Returns: (X, y, feature_names)
    """
    y = (df[TARGET].astype(str) == POSITIVE_LABEL).astype(int)

    X = df.copy()

    for col in NUMERIC_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    for col in CATEGORICAL_COLS:
        X[col] = X[col].fillna("unknown").astype(str)

    X["order_date"] = pd.to_datetime(X["order_date"], errors="coerce")
    X["order_year"] = X["order_date"].dt.year.fillna(0).astype(int)
    X["order_month"] = X["order_date"].dt.month.fillna(0).astype(int)
    X["order_dayofweek"] = X["order_date"].dt.dayofweek.fillna(0).astype(int)
    X["order_hour"] = X["order_date"].dt.hour.fillna(0).astype(int)
    X = X.drop(columns=DROP_COLS + ["order_date"], errors="ignore")

    X = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=False)

    # Keep a stable training/inference schema regardless of which categories
    # are present in a particular split or prediction slice.
    expected_dummy_cols = [f"payment_method_{v}" for v in PAYMENT_METHOD_VALUES]
    for col in expected_dummy_cols:
        if col not in X.columns:
            X[col] = 0.0

    base_cols = NUMERIC_COLS + ["order_year", "order_month", "order_dayofweek", "order_hour"] + expected_dummy_cols
    remaining_cols = [c for c in X.columns if c not in base_cols]
    X = X[base_cols + remaining_cols]

    scaler = StandardScaler()

    if fit_scaler:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    X_final = pd.DataFrame(X_scaled, columns=X.columns, index=df.index)

    return X_final, y, list(X_final.columns)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Stratified train/test split."""
    # Handle cases where stratification is not possible (e.g., too few samples)
    stratify = y if len(y.unique()) > 1 and y.value_counts().min() > 1 else None
    
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
