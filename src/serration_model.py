"""
serration_model.py
==================
Baseline models for characterising mechanical instability from
nanoindentation serration features.

Two models are provided:

1. **Random Forest Classifier** — classifies samples as
   "high serration" vs "low serration" based on extracted burst features.
   The label is derived from a configurable burst-count threshold.

2. **Random Forest Regressor** — predicts a continuous "instability
   score" (normalised burst activity metric) from serration features.

These serve as baselines before the integrated fusion model.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)

from src.serration_features import (
    extract_serration_features,
    build_serration_feature_matrix,
)


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def compute_instability_score(sample: dict) -> float:
    """Compute a continuous instability score ∈ [0, 1] for a serration sample.

    Combines burst count, total burst amplitude, and displacement-rate
    statistics into a single normalised score.

    The score is heuristic and designed for synthetic validation;
    real-world calibration will be needed later.
    """
    feats = extract_serration_features(sample)
    # Weighted combination (weights are heuristic placeholders)
    raw = (
        0.30 * min(feats["burst_count"] / 50.0, 1.0)
        + 0.25 * min(feats["burst_disp_fraction"], 1.0)
        + 0.20 * min(feats["amp_max_nm"] / 30.0, 1.0)
        + 0.15 * min(feats["dh_dP_skewness"] / 5.0, 1.0)
        + 0.10 * min(feats["burst_frequency_per_mN"] / 2.0, 1.0)
    )
    return float(np.clip(raw, 0.0, 1.0))


def label_high_serration(samples: List[dict],
                         threshold_burst_count: int = 10) -> np.ndarray:
    """Binary label: 1 if burst_count >= threshold, else 0."""
    labels = []
    for s in samples:
        bursts = s.get("detected_bursts", s.get("bursts", []))
        labels.append(1 if len(bursts) >= threshold_burst_count else 0)
    return np.array(labels, dtype=int)


# ---------------------------------------------------------------------------
# Build X, y
# ---------------------------------------------------------------------------

def build_serration_Xy_classification(
    samples: List[dict],
    threshold_burst_count: int = 10,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Feature matrix + binary high-serration labels."""
    X, _ = build_serration_feature_matrix(samples)
    y = label_high_serration(samples, threshold_burst_count)
    return X, y


def build_serration_Xy_regression(
    samples: List[dict],
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Feature matrix + continuous instability scores."""
    X, _ = build_serration_feature_matrix(samples)
    y = np.array([compute_instability_score(s) for s in samples])
    return X, y


# ---------------------------------------------------------------------------
# Train / evaluate — Classification
# ---------------------------------------------------------------------------

def train_serration_classifier(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    n_estimators: int = 200,
    random_state: int = 42,
    save_path: Optional[str] = None,
) -> Dict:
    """Train a Random Forest classifier for high vs low serration.

    Returns
    -------
    dict with model, metrics, feature_importances, y_test, y_pred.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    cv_scores = cross_val_score(
        model, X, y, cv=min(5, len(y)), scoring="f1"
    )
    metrics["cv_f1_mean"] = float(cv_scores.mean())
    metrics["cv_f1_std"] = float(cv_scores.std())

    feat_imp = dict(zip(X.columns, model.feature_importances_))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        metrics_path = save_path.replace(".pkl", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"💾 Serration classifier saved to {save_path}")

    return {
        "model": model,
        "metrics": metrics,
        "feature_importances": feat_imp,
        "y_test": y_test,
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# Train / evaluate — Regression (instability score)
# ---------------------------------------------------------------------------

def train_serration_regressor(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    n_estimators: int = 200,
    random_state: int = 42,
    save_path: Optional[str] = None,
) -> Dict:
    """Train a Random Forest regressor for instability score.

    Returns
    -------
    dict with model, metrics, feature_importances, y_test, y_pred.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
    }

    cv_scores = cross_val_score(
        model, X, y, cv=min(5, len(y)), scoring="neg_mean_absolute_error"
    )
    metrics["cv_mae_mean"] = float(-cv_scores.mean())
    metrics["cv_mae_std"] = float(cv_scores.std())

    feat_imp = dict(zip(X.columns, model.feature_importances_))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        metrics_path = save_path.replace(".pkl", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"💾 Serration regressor saved to {save_path}")

    return {
        "model": model,
        "metrics": metrics,
        "feature_importances": feat_imp,
        "y_test": y_test,
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.synthetic_serration_data import generate_serration_dataset
    from src.serration_preprocessing import preprocess_serration_all

    print("Generating synthetic serration data …")
    samples = generate_serration_dataset(n_samples=100, seed=42)
    samples = preprocess_serration_all(samples)

    print("\n── Classification (high vs low serration) ──")
    X_cls, y_cls = build_serration_Xy_classification(samples, threshold_burst_count=10)
    print(f"  X: {X_cls.shape}  |  Positive: {y_cls.sum()}  Negative: {(1 - y_cls).sum()}")
    result_cls = train_serration_classifier(X_cls, y_cls,
                                            save_path="models/serration_clf.pkl")
    for k, v in result_cls["metrics"].items():
        print(f"    {k:20s} = {v:.4f}")

    print("\n── Regression (instability score) ──")
    X_reg, y_reg = build_serration_Xy_regression(samples)
    print(f"  X: {X_reg.shape}  |  y range: [{y_reg.min():.3f}, {y_reg.max():.3f}]")
    result_reg = train_serration_regressor(X_reg, y_reg,
                                           save_path="models/serration_reg.pkl")
    for k, v in result_reg["metrics"].items():
        print(f"    {k:20s} = {v:.4f}")
