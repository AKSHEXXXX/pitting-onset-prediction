"""
evaluation.py
=============
Evaluation metrics and scoring utilities for both the baseline
(Random Forest) and LSTM models.
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray,
                       y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics.

    Returns
    -------
    dict with MAE, RMSE, R², MAPE.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    # Mean Absolute Percentage Error (avoid division by zero)
    nonzero = np.abs(y_true) > 1e-12
    if nonzero.any():
        mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero])
                                     / y_true[nonzero])) * 100)
    else:
        mape = float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape_pct": mape,
    }


def onset_index_error(true_indices: np.ndarray,
                      pred_indices: np.ndarray,
                      total_points: int) -> Dict[str, float]:
    """Compute errors expressed in array-index units.

    Useful for measuring how many data-points off the prediction is.
    """
    abs_err = np.abs(true_indices - pred_indices).astype(float)
    return {
        "mean_index_error": float(abs_err.mean()),
        "median_index_error": float(np.median(abs_err)),
        "max_index_error": float(abs_err.max()),
        "pct_within_5": float(np.mean(abs_err <= 5) * 100),
        "pct_within_10": float(np.mean(abs_err <= 10) * 100),
        "pct_within_1pct": float(
            np.mean(abs_err / total_points <= 0.01) * 100
        ),
    }


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def classification_metrics(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           threshold: float = 0.5) -> Dict[str, float]:
    """Compute binary classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels (0 or 1).
    y_pred : np.ndarray
        Predicted probabilities or hard labels.
    threshold : float
        Decision boundary (applied if y_pred is probabilistic).
    """
    y_hard = (y_pred >= threshold).astype(int)
    y_true_int = y_true.astype(int)

    return {
        "accuracy": float(accuracy_score(y_true_int, y_hard)),
        "precision": float(precision_score(y_true_int, y_hard, zero_division=0)),
        "recall": float(recall_score(y_true_int, y_hard, zero_division=0)),
        "f1": float(f1_score(y_true_int, y_hard, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_int, y_hard).tolist(),
    }


def find_optimal_threshold(y_true: np.ndarray,
                           y_prob: np.ndarray,
                           metric: str = "f1",
                           n_steps: int = 200) -> Dict[str, float]:
    """Sweep decision thresholds and find the one that maximises a metric.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities.
    metric : str
        Which metric to optimise: ``"f1"`` (default), ``"precision"``, or ``"recall"``.
    n_steps : int
        Number of threshold values to evaluate between 0 and 1.

    Returns
    -------
    dict with ``best_threshold``, ``best_<metric>``, and the full metrics at
    that threshold.
    """
    best_score = -1.0
    best_thresh = 0.5
    best_metrics = {}

    for thresh in np.linspace(0.01, 0.99, n_steps):
        m = classification_metrics(y_true, y_prob, threshold=thresh)
        score = m[metric]
        if score > best_score:
            best_score = score
            best_thresh = float(thresh)
            best_metrics = m

    best_metrics["best_threshold"] = best_thresh
    best_metrics[f"best_{metric}"] = best_score
    return best_metrics


# ---------------------------------------------------------------------------
# Heuristic detection evaluation
# ---------------------------------------------------------------------------

def evaluate_detection(samples: List[dict]) -> Dict[str, float]:
    """Evaluate onset detection accuracy across a batch of samples.

    Each sample should have both ground-truth and detected onset fields:
        - ``pitting_onset_index`` (ground truth)
        - ``detected_onset_index`` (from ``onset_detection.detect_pitting_onset``)
    """
    true_idxs, pred_idxs = [], []
    n_points = 0

    for s in samples:
        gt = s.get("pitting_onset_index")
        dt = s.get("detected_onset_index", s.get("pitting_onset_index"))
        if gt is None or dt is None:
            continue
        true_idxs.append(gt)
        pred_idxs.append(dt)
        n_points = max(n_points, len(s["potential_V"]))

    if not true_idxs:
        return {"error": "No valid samples for evaluation"}

    true_arr = np.array(true_idxs)
    pred_arr = np.array(pred_idxs)

    return onset_index_error(true_arr, pred_arr, n_points)


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_metrics(metrics: dict, title: str = "Metrics") -> None:
    """Pretty-print a metrics dictionary."""
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:25s} = {v:.4f}")
        else:
            print(f"  {k:25s} = {v}")
    print(f"{'─' * 50}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick demo with random data
    rng = np.random.default_rng(0)
    y_true = rng.uniform(0.2, 0.8, 50)
    y_pred = y_true + rng.normal(0, 0.05, 50)

    metrics = regression_metrics(y_true, y_pred)
    print_metrics(metrics, title="Regression Demo")

    y_true_cls = rng.integers(0, 2, 100).astype(float)
    y_pred_cls = rng.random(100)
    cls_metrics = classification_metrics(y_true_cls, y_pred_cls)
    print_metrics(cls_metrics, title="Classification Demo")
