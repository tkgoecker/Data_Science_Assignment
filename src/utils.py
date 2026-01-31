from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load assignment data from Excel.
    """
    # engine=openpyxl is safest for xlsx
    return pd.read_excel(data_path, engine="openpyxl")


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning that is safe for unknown schemas.
    - strips column whitespace
    - standardizes column names a bit (keeps original meaning)
    - drops fully-empty rows
    """
    df = df.copy()

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # Drop completely empty rows
    df = df.dropna(how="all")

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional feature creation that is safe even if columns don't exist.
    We keep this conservative so it won't break on unknown datasets.
    """
    df = df.copy()

    # Example: if there is a date column, we can create year/month/dayofweek
    # We'll look for common date-like columns.
    date_candidates = [c for c in df.columns if "date" in c.lower() or "dt" == c.lower()]
    for c in date_candidates[:1]:
        try:
            d = pd.to_datetime(df[c], errors="coerce")
            if d.notna().any():
                df[f"{c}_year"] = d.dt.year
                df[f"{c}_month"] = d.dt.month
                df[f"{c}_dayofweek"] = d.dt.dayofweek
        except Exception:
            pass

    return df


def infer_target_column(df: pd.DataFrame) -> str:
    """
    Try to infer the binary target column.
    We look for common target names first; otherwise choose a column with 2 unique values.
    """
    preferred = [
        "target", "label", "outcome", "y",
        "readmitted", "readmission", "churn",
        "fraud", "is_fraud", "default", "is_default",
        "approved", "is_approved"
    ]

    lower_map = {c.lower(): c for c in df.columns}

    for name in preferred:
        if name in lower_map:
            return lower_map[name]

    # If not found, pick a column with 2 distinct non-null values (binary-ish)
    for c in df.columns:
        vals = df[c].dropna().unique()
        if len(vals) == 2:
            return c

    raise ValueError(
        "Could not infer target column. "
        "Please rename your target to one of: target/label/outcome/y or ensure it has 2 unique values."
    )


def infer_text_column(df: pd.DataFrame) -> Optional[str]:
    """
    Try to infer a free-text column for NLP.
    We look for common names; otherwise return the longest average string column.
    """
    preferred = [
        "text", "note", "notes", "comment", "comments",
        "description", "summary", "narrative", "report"
    ]

    lower_map = {c.lower(): c for c in df.columns}
    for name in preferred:
        if name in lower_map:
            return lower_map[name]

    # Try to guess: object columns with longer strings
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not object_cols:
        return None

    best_col = None
    best_score = 0.0

    for c in object_cols:
        s = df[c].dropna().astype(str)
        if s.empty:
            continue
        # average length as a simple heuristic
        score = s.str.len().mean()
        if score > best_score:
            best_score = score
            best_col = c

    # Only consider it "text" if strings are reasonably long on average
    if best_col is not None and best_score >= 20:
        return best_col

    return None


def train_test_split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> SplitData:
    """
    Stratified split for classification tasks.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
