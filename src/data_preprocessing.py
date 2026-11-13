"""
Data preprocessing and feature engineering for the Telco Customer Churn project.

Pipeline: load -> clean -> engineer features -> encode -> split -> cap outliers -> scale.
Outlier bounds and scaler are fit on the TRAINING set only to avoid data leakage.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET = "Churn"
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend", "num_addons"]
ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]
BINARY_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "MultipleLines"] + ADDON_COLS


def load_data(path: str) -> pd.DataFrame:
    """Load the raw Telco churn CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types, handle missing values, drop identifiers."""
    df = df.copy()
    df = df.drop(columns=["customerID"], errors="ignore")
    # TotalCharges is text with blanks for brand-new customers (tenure == 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)  # new customers have paid nothing yet
    df = df.drop_duplicates().reset_index(drop=True)
    # Collapse "No internet service" / "No phone service" into "No"
    df = df.replace({"No internet service": "No", "No phone service": "No"})
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing columns."""
    df = df.copy()
    df["num_addons"] = (df[ADDON_COLS] == "Yes").sum(axis=1)
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 72],
        labels=["0-12m", "13-24m", "25-48m", "49-72m"]
    ).astype(str)
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode target, binary columns and one-hot encode remaining categoricals."""
    df = df.copy()
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    for col in BINARY_COLS:
        df[col] = (df[col] == "Yes").astype(int)
    df["gender"] = (df["gender"] == "Male").astype(int)
    cat_cols = list(df.select_dtypes(include=["object", "string"]).columns)
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    return df


def fit_outlier_bounds(df: pd.DataFrame, cols: list, k: float = 1.5) -> dict:
    """Compute IQR bounds on the training data."""
    bounds = {}
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        bounds[c] = (q1 - k * iqr, q3 + k * iqr)
    return bounds


def apply_outlier_bounds(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """Winsorize (cap) values to the given bounds."""
    df = df.copy()
    for c, (lo, hi) in bounds.items():
        df[c] = df[c].clip(lo, hi)
    return df


def preprocess(path: str, test_size: float = 0.2, random_state: int = 42):
    """Full pipeline. Returns X_train, X_test, y_train, y_test and the fitted scaler."""
    df = clean_data(load_data(path))
    df = engineer_features(df)
    df = encode_features(df)

    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    bounds = fit_outlier_bounds(X_train, NUMERIC_COLS)
    X_train = apply_outlier_bounds(X_train, bounds)
    X_test = apply_outlier_bounds(X_test, bounds)

    scaler = StandardScaler()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    Xtr, Xte, ytr, yte, _ = preprocess(p)
    print("Train:", Xtr.shape, "Test:", Xte.shape, "Churn rate:", round(ytr.mean(), 3))
