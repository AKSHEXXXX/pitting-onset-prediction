"""
baseline_model.py
=================
Random Forest regression baseline for predicting the pitting onset
potential (V) from hand-crafted features extracted from polarization curves.

Feature engineering
-------------------
For each curve the following scalar features are extracted:
  • V_range, I_range          – overall sweep extents
  • I_mean_passive            – mean current in the first 30 % of the curve
  • dI_dV_max                 – peak first derivative
  • d2I_dV2_max               – peak second derivative
  • I_std                     – current std dev (noisiness proxy)
  • log_I_mean, log_I_std     – statistics of log10(|I|)
  • passive_slope             – linear slope in the passive region
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocessing import (
    smooth_savgol,
    compute_derivative,
    compute_second_derivative,
    log_current,
)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(sample: dict) -> dict:
    """Compute hand-crafted scalar features from a single sample.

    Parameters
    ----------
    sample : dict
        Must contain ``potential_V`` and ``current_A``.

    Returns
    -------
    dict of {feature_name: float}
    """
    V = sample["potential_V"]
    I = sample["current_A"]
    I_smooth = smooth_savgol(I)

    dI_dV = compute_derivative(I_smooth, V)
    d2I_dV2 = compute_second_derivative(I_smooth, V)
    logI = log_current(I_smooth)

    passive_end = max(int(len(V) * 0.3), 5)
    I_passive = I_smooth[:passive_end]
    V_passive = V[:passive_end]

    # Passive-region linear slope via simple least-squares
    if len(V_passive) > 1:
        coeffs = np.polyfit(V_passive, I_passive, 1)
        passive_slope = coeffs[0]
    else:
        passive_slope = 0.0

    return {
        "V_range": float(V.max() - V.min()),
        "I_range": float(I.max() - I.min()),
        "I_mean_passive": float(np.mean(I_passive)),
        "I_std": float(np.std(I)),
        "dI_dV_max": float(np.max(dI_dV)),
        "dI_dV_mean": float(np.mean(dI_dV)),
        "d2I_dV2_max": float(np.max(d2I_dV2)),
        "log_I_mean": float(np.mean(logI)),
        "log_I_std": float(np.std(logI)),
        "passive_slope": float(passive_slope),
    }


def build_feature_matrix(samples: List[dict]) -> Tuple[pd.DataFrame, np.ndarray]:
    """Build (X, y) from a list of samples.

    Samples without a pitting onset label are **excluded**.

    Returns
    -------
    X : pd.DataFrame  (n_samples, n_features)
    y : np.ndarray     (n_samples,)  – pitting onset potential (V).
    """
    rows = []
    targets = []
    for s in samples:
        if s.get("pitting_onset_potential_V") is None:
            continue  # skip no-pitting samples for regression
        feats = extract_features(s)
        feats["sample_id"] = s["sample_id"]
        rows.append(feats)
        targets.append(s["pitting_onset_potential_V"])

    df = pd.DataFrame(rows).set_index("sample_id")
    y = np.array(targets, dtype=np.float64)
    return df, y


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def train_baseline(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    n_estimators: int = 200,
    random_state: int = 42,
    save_path: Optional[str] = None,
) -> dict:
    """Train a Random Forest and return metrics.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Target array (onset potential in V).
    test_size : float
        Fraction held out for testing.
    n_estimators : int
        Number of trees.
    random_state : int
        Seed.
    save_path : str | None
        If given, persist the trained model as a pickle file.

    Returns
    -------
    dict with keys ``model``, ``metrics``, ``feature_importances``,
    ``y_test``, ``y_pred``.
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

    # Cross-validation on the full set for robustness
    cv_scores = cross_val_score(
        model, X, y, cv=min(5, len(y)), scoring="neg_mean_absolute_error"
    )
    metrics["cv_mae_mean"] = float(-cv_scores.mean())
    metrics["cv_mae_std"] = float(cv_scores.std())

    # Feature importances
    feat_imp = dict(zip(X.columns, model.feature_importances_))

    # Save model
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        print(f"💾 Model saved to {save_path}")

    # Save metrics alongside model
    if save_path:
        metrics_path = save_path.replace(".pkl", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    return {
        "model": model,
        "metrics": metrics,
        "feature_importances": feat_imp,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def load_model(path: str) -> RandomForestRegressor:
    """Load a previously saved model."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.synthetic_data import generate_dataset

    print("Generating synthetic data …")
    samples = generate_dataset(n_samples=100, seed=42)

    print("Building feature matrix …")
    X, y = build_feature_matrix(samples)
    print(f"  Features shape : {X.shape}")
    print(f"  Targets shape  : {y.shape}")

    print("Training Random Forest baseline …")
    result = train_baseline(X, y, save_path="models/rf_baseline.pkl")

    print("\n📊 Baseline metrics:")
    for k, v in result["metrics"].items():
        print(f"   {k:20s} = {v:.4f}")

    print("\n🌲 Feature importances:")
    for feat, imp in sorted(
        result["feature_importances"].items(), key=lambda x: -x[1]
    ):
        print(f"   {feat:25s}  {imp:.4f}")
