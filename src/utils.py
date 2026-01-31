from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_data(path: str) -> pd.DataFrame:
    """
    Load the assignment dataset.
    Supports .xlsx and .csv.
    """
    path_l = path.lower()
    if path_l.endswith(".xlsx"):
        return pd.read_excel(path, engine="openpyxl")
    elif path_l.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError("Unsupported file type. Use .xlsx or .csv")


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light cleaning:
    - normalize column names (strip spaces)
    - strip whitespace from string cells
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    return df


def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a couple of simple engineered features.
    Keep it lightweight (small dataset).
    """
    df = df.copy()

    # admissions_per_day = prior admissions normalized by LOS (avoid divide-by-zero)
    if "num_previous_admissions" in df.columns and "length_of_stay" in df.columns:
        los = df["length_of_stay"].replace(0, np.nan)
        df["admissions_per_day"] = (df["num_previous_admissions"] / los).fillna(0.0)
    else:
        df["admissions_per_day"] = 0.0

    # Age bucket for a simple non-linear age effect
    if "age" in df.columns:
        bins = [0, 29, 44, 59, 74, 120]
        labels = ["0-29", "30-44", "45-59", "60-74", "75+"]
        df["age_bucket"] = pd.cut(df["age"], bins=bins, labels=labels, include_lowest=True)
        df["age_bucket"] = df["age_bucket"].astype(str)
    else:
        df["age_bucket"] = "unknown"

    return df


def encode_binary_target(y: pd.Series) -> Tuple[pd.Series, Dict]:
    """
    Robustly encode binary target.

    Handles:
    - already 0/1 ints
    - strings like "0"/"1"
    - booleans
    - any 2-class labels (maps first label -> 0, second -> 1)
    """
    y = y.copy()

    # If already numeric 0/1
    if pd.api.types.is_numeric_dtype(y):
        uniq = sorted(pd.Series(y.dropna().unique()).tolist())
        if set(uniq).issubset({0, 1}):
            return y.astype(int), {"0": 0, "1": 1}

    # Convert common string/boolean forms
    y_str = y.astype(str).str.strip().str.lower()
    if set(y_str.unique()).issubset({"0", "1"}):
        return y_str.astype(int), {"0": 0, "1": 1}

    if set(y_str.unique()).issubset({"true", "false"}):
        mapped = y_str.map({"false": 0, "true": 1}).astype(int)
        return mapped, {"false": 0, "true": 1}

    # Generic 2-class mapping
    uniq = pd.Series(y_str.unique()).dropna().tolist()
    if len(uniq) != 2:
        raise ValueError(f"Target is not binary. Unique values: {uniq}")

    mapping = {uniq[0]: 0, uniq[1]: 1}
    return y_str.map(mapping).astype(int), mapping


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> SplitData:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
