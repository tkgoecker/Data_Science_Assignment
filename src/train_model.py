from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.utils import (
    load_data,
    basic_cleaning,
    validate_columns,
    add_feature_engineering,
    encode_binary_target,
    stratified_split,
)


def _print(msg: str) -> None:
    print(msg, flush=True)


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return []


def plot_roc(y_true: np.ndarray, y_proba: np.ndarray, title: str, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
    plt.figure()
    disp.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def top_features_from_logreg(model: LogisticRegression, feature_names: List[str], top_k: int = 12) -> List[Dict]:
    if not feature_names or not hasattr(model, "coef_"):
        return []

    coefs = model.coef_.ravel()
    idx = np.argsort(np.abs(coefs))[::-1][:top_k]
    return [{"feature": feature_names[i], "coef": float(coefs[i])} for i in idx]


def top_features_from_rf(model: RandomForestClassifier, feature_names: List[str], top_k: int = 12) -> List[Dict]:
    if not feature_names or not hasattr(model, "feature_importances_"):
        return []

    imps = model.feature_importances_
    idx = np.argsort(imps)[::-1][:top_k]
    return [{"feature": feature_names[i], "importance": float(imps[i])} for i in idx]


def main(data_path: str) -> None:
    _print("\n>>> STARTING src.train_model main() <<<")
    _print(f"CWD: {Path.cwd()}")
    _print(f"Data path: {data_path}\n")

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    required_cols = [
        "patient_id",
        "age",
        "gender",
        "diagnosis_code",
        "num_previous_admissions",
        "medication_type",
        "length_of_stay",
        "readmitted_30_days",
        "discharge_note",
    ]

    df = load_data(data_path)
    df = basic_cleaning(df)
    validate_columns(df, required_cols)

    _print(f"Loaded data shape: {df.shape}")

    target_col = "readmitted_30_days"
    text_col = "discharge_note"

    _print(f"Target column: {target_col}")
    _print(f"Text column: {text_col}\n")

    _print("Target distribution:")
    _print(str(df[target_col].value_counts()))
    _print("\nMissing values (top 10):")
    _print(str(df.isna().sum().sort_values(ascending=False).head(10)))
    _print("")

    # Feature engineering
    df = add_feature_engineering(df)

    # Encode target to 0/1 safely
    y, target_mapping = encode_binary_target(df[target_col])
    _print(f"Target mapping: {target_mapping}\n")

    # Build X (drop id + target + free text)
    X = df.drop(columns=["patient_id", target_col, text_col])

    numeric_features = ["age", "num_previous_admissions", "length_of_stay", "admissions_per_day"]
    categorical_features = ["gender", "diagnosis_code", "medication_type", "age_bucket"]

    split = stratified_split(X, y, test_size=0.2, random_state=42)

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),
    }

    metrics: Dict[str, Dict] = {}
    top_feats: Dict[str, List[Dict]] = {}

    for name, clf in models.items():
        _print(f"--- Training: {name} ---")

        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])
        pipe.fit(split.X_train, split.y_train)

        y_proba = pipe.predict_proba(split.X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        auc = roc_auc_score(split.y_test, y_proba)
        f1 = f1_score(split.y_test, y_pred)
        cm = confusion_matrix(split.y_test, y_pred).tolist()

        metrics[name] = {
            "roc_auc": float(auc),
            "f1": float(f1),
            "confusion_matrix": cm,
        }

        roc_path = reports_dir / f"roc_{name}.png"
        cm_path = reports_dir / f"confusion_{name}.png"
        plot_roc(split.y_test.values, y_proba, f"ROC Curve - {name}", roc_path)
        plot_confusion(split.y_test.values, y_pred, f"Confusion Matrix - {name}", cm_path)

        fitted_preprocessor = pipe.named_steps["preprocess"]
        feature_names = get_feature_names(fitted_preprocessor)

        fitted_model = pipe.named_steps["model"]
        if name == "logistic_regression":
            top_feats[name] = top_features_from_logreg(fitted_model, feature_names)
        else:
            top_feats[name] = top_features_from_rf(fitted_model, feature_names)

        _print(f"ROC AUC: {auc:.4f} | F1: {f1:.4f}")
        _print(f"Saved ROC -> {roc_path}")
        _print(f"Saved Confusion Matrix -> {cm_path}\n")

    out = {
        "target": target_col,
        "text_column": text_col,
        "rows": int(df.shape[0]),
        "models": metrics,
        "top_features": top_feats,
    }

    metrics_path = reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    _print(f"Saved metrics JSON to: {metrics_path}")
    _print("\n>>> FINISHED src.train_model main() <<<\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="Data/Assignment_Data.xlsx")
    args = parser.parse_args()
    main(args.data_path)
