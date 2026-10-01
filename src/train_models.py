"""Train and compare machine-learning models for hit-song prediction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_xy(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("Feature CSV must contain a 'label' column with 0/1 values.")

    drop_cols = [c for c in ["label", "track_id", "song_name", "artist"] if c in df.columns]
    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number]).fillna(0)
    y = pd.to_numeric(df["label"], errors="raise").astype(int)

    if X.empty:
        raise ValueError("No numeric feature columns were found.")
    if set(y.unique()) != {0, 1}:
        raise ValueError("The label column must contain both classes: 0 and 1.")

    return X, y


def _scaled_pca_pipeline(estimator, random_state: int) -> Pipeline:
    """Create fresh preprocessing objects for one model pipeline."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=random_state)),
            ("model", estimator),
        ]
    )


def build_models(random_state: int) -> Dict[str, Pipeline]:
    return {
        "logistic_regression": _scaled_pca_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced"), random_state
        ),
        "lda": _scaled_pca_pipeline(
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), random_state
        ),
        "linear_svm": _scaled_pca_pipeline(
            SVC(kernel="linear", class_weight="balanced"), random_state
        ),
        "rbf_svm": _scaled_pca_pipeline(
            SVC(kernel="rbf", class_weight="balanced"), random_state
        ),
        "poly_svm": _scaled_pca_pipeline(
            SVC(kernel="poly", degree=3, class_weight="balanced"), random_state
        ),
        "random_forest": Pipeline(
            [("model", RandomForestClassifier(n_estimators=250, random_state=random_state, class_weight="balanced"))]
        ),
        "gradient_boosting": Pipeline(
            [("model", GradientBoostingClassifier(random_state=random_state))]
        ),
        "neural_network": _scaled_pca_pipeline(
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                alpha=0.001,
                max_iter=600,
                random_state=random_state,
            ),
            random_state,
        ),
    }


def evaluate(csv_path: Path, out_dir: Path, random_state: int = 42) -> None:
    X, y = load_xy(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=random_state,
    )

    if y_train.value_counts().min() < 5:
        raise ValueError("At least 5 training samples per class are required for 5-fold cross-validation.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "f1": make_scorer(f1_score, zero_division=0),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
    }

    rows = []
    trained_models: Dict[str, Pipeline] = {}

    for name, model in build_models(random_state).items():
        cv_scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        model.fit(X_train, y_train)
        trained_models[name] = model
        y_pred = model.predict(X_test)

        rows.append(
            {
                "model": name,
                "cv_accuracy": float(np.mean(cv_scores["test_accuracy"])),
                "cv_f1": float(np.mean(cv_scores["test_f1"])),
                "cv_precision": float(np.mean(cv_scores["test_precision"])),
                "cv_recall": float(np.mean(cv_scores["test_recall"])),
                "test_accuracy": float(accuracy_score(y_test, y_pred)),
                "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
            }
        )

    results = pd.DataFrame(rows).sort_values("cv_f1", ascending=False).reset_index(drop=True)
    best_row = results.iloc[0]
    best_model_name = str(best_row["model"])
    best_pipeline = trained_models[best_model_name]

    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "model_metrics.csv", index=False)
    joblib.dump(best_pipeline, out_dir / "best_model.joblib")

    summary = {
        "selection_metric": "cv_f1",
        "best_model": best_model_name,
        "best_cv_f1": float(best_row["cv_f1"]),
        "best_test_accuracy": float(best_row["test_accuracy"]),
        "best_test_f1": float(best_row["test_f1"]),
        "best_test_precision": float(best_row["test_precision"]),
        "best_test_recall": float(best_row["test_recall"]),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(results.to_string(index=False))
    print(f"\nBest model by cross-validated F1: {best_model_name}")
    print(f"Saved model: {out_dir / 'best_model.joblib'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hit-song prediction models")
    parser.add_argument("--features", type=Path, required=True, help="Feature CSV with numeric columns and label")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    evaluate(args.features, args.out, args.seed)
