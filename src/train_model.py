from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt

from src.utils import (
    load_data,
    basic_cleaning,
    add_derived_features,
    infer_text_column,
    infer_target_column,
    train_test_split_stratified,
)


def _print(msg: str) -> None:
    print(msg, flush=True)


def build_preprocessor(X: pd.DataFrame, text_col: str | None):
    X = X.copy()

    if text_col and text_col in X.columns:
        X = X.drop(columns=[text_col])

    for id_col in ["patient_id", "id", "member_id", "mrn"]:
        if id_col in X.columns:
            X = X.drop(columns=[id_col])

    numeric_features = X.select_dtypes(include=["number", "int", "float", "Int64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_features, categorical_features


def get_feature_names(preprocessor: ColumnTransformer, numeric_features: list[str], categorical_features: list[str]) -> list[str]:
    if len(categorical_features) > 0:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = ohe.get_feature_names_out(categorical_features).tolist()
    else:
        cat_names = []
    return numeric_features + cat_names


def evaluate_model(model_name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, reports_dir: Path) -> dict:
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    roc = roc_auc_score(y_test, y_score)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    _print("\n" + "=" * 90)
    _print(f"{model_name} Evaluation")
    _print("=" * 90)
    _print(f"ROC AUC: {roc:.4f}")
    _print(f"F1:      {f1:.4f}")
    _print("\nConfusion Matrix (rows=true, cols=pred):")
    _print(str(cm))
    _print("\nClassification Report:")
    _print(classification_report(y_test, y_pred, digits=4))

    RocCurveDisplay.from_predictions(y_test, y_score)
    plt.title(f"ROC Curve - {model_name}")

    roc_path = reports_dir / f"roc_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(roc_path, bbox_inches="tight")
    _print(f"Saved ROC curve to: {roc_path}")

    try:
        plt.show()
    except Exception:
        pass

    plt.close()

    return {
        "model": model_name,
        "roc_auc": float(roc),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }


def print_logreg_top_features(model: Pipeline, feature_names: list[str], top_n: int = 15) -> list[dict]:
    clf = model.named_steps["clf"]
    coefs = clf.coef_.ravel()
    order = np.argsort(np.abs(coefs))[::-1][:top_n]

    _print("\nTop influential Logistic Regression features (by |coef|):")
    out = []
    for idx in order:
        row = {"feature": feature_names[idx], "coef": float(coefs[idx])}
        out.append(row)
        _print(f"  {row['feature'][:45]:45s}  coef={row['coef']: .4f}")
    return out


def permutation_importance_top(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, feature_names: list[str], top_n: int = 15) -> list[dict]:
    preprocessor = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]
    X_test_t = preprocessor.transform(X_test)

    r = permutation_importance(
        clf,
        X_test_t,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="f1",
    )

    importances = r.importances_mean
    order = np.argsort(importances)[::-1][:top_n]

    _print("\nTop features by permutation importance (scoring=F1):")
    out = []
    for idx in order:
        row = {"feature": feature_names[idx], "importance": float(importances[idx])}
        out.append(row)
        _print(f"  {row['feature'][:45]:45s}  importance={row['importance']: .4f}")
    return out


def main(data_path: str) -> None:
    _print("\n>>> STARTING src.train_model main() <<<")
    _print(f"CWD: {os.getcwd()}")
    _print(f"Data path: {data_path}")

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    df = load_data(data_path)
    df = basic_cleaning(df)
    df = add_derived_features(df)

    target_col = infer_target_column(df)
    text_col = infer_text_column(df)

    _print(f"\nLoaded data shape: {df.shape}")
    _print(f"Target column: {target_col}")
    _print(f"Text column: {text_col if text_col else 'None found'}")

    _print("\nTarget distribution:")
    _print(str(df[target_col].value_counts(dropna=False)))

    _print("\nMissing values (top 10):")
    _print(str(df.isna().sum().sort_values(ascending=False).head(10)))

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    preprocessor, numeric_features, categorical_features = build_preprocessor(X, text_col=text_col)
    split = train_test_split_stratified(X, y, test_size=0.2, random_state=42)

    logreg = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])

    rf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
        )),
    ])

    _print("\nFitting models...")
    logreg.fit(split.X_train, split.y_train)
    rf.fit(split.X_train, split.y_train)
    _print("Done fitting models.")

    results = []
    results.append(evaluate_model("Logistic Regression", logreg, split.X_test, split.y_test, reports_dir))
    results.append(evaluate_model("Random Forest", rf, split.X_test, split.y_test, reports_dir))

    feature_names = get_feature_names(preprocessor, numeric_features, categorical_features)
    logreg_top = print_logreg_top_features(logreg, feature_names, top_n=15)
    rf_top = permutation_importance_top(rf, split.X_test, split.y_test, feature_names, top_n=15)

    metrics_out = {
        "results": results,
        "logreg_top_features": logreg_top,
        "rf_permutation_importance": rf_top,
        "data_shape": list(df.shape),
        "target_col": target_col,
        "text_col": text_col,
    }

    metrics_path = reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_out, indent=2))
    _print(f"\nSaved metrics JSON to: {metrics_path}")

    _print("\n>>> FINISHED src.train_model main() <<<\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/Assignment_Data.xlsx")
    args = parser.parse_args()
    main(args.data_path)
