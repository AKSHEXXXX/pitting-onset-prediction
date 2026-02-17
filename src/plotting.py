"""
plotting.py
============
Visualisation utilities for polarization curves, onset detection results,
model predictions, and training history.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for scripts
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Tuple
from sklearn.metrics import roc_curve, precision_recall_curve


# Consistent styling
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

FIGURES_DIR = "figures"


def _save(fig, name: str, dpi: int = 150) -> str:
    """Save figure and return path."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Saved {path}")
    return path


# ---------------------------------------------------------------------------
# 1. Single polarization curve with onset
# ---------------------------------------------------------------------------

def plot_curve(
    potential: np.ndarray,
    current: np.ndarray,
    onset_index: Optional[int] = None,
    predicted_index: Optional[int] = None,
    title: str = "Polarization Curve",
    log_scale: bool = False,
    save_name: Optional[str] = None,
) -> None:
    """Plot a single V vs I curve, optionally marking true/predicted onset."""
    fig, ax = plt.subplots()

    y_data = np.log10(np.abs(current) + 1e-12) if log_scale else current
    ylabel = "log₁₀|I| (A/cm²)" if log_scale else "Current (A/cm²)"

    ax.plot(potential, y_data, "b-", linewidth=1.0, label="I(V)")

    if onset_index is not None:
        ax.axvline(potential[onset_index], color="red", linestyle="--",
                   linewidth=1.5, label=f"True onset (V={potential[onset_index]:.3f})")
        ax.plot(potential[onset_index], y_data[onset_index], "ro",
                markersize=8, zorder=5)

    if predicted_index is not None:
        ax.axvline(potential[predicted_index], color="green", linestyle=":",
                   linewidth=1.5, label=f"Predicted onset (V={potential[predicted_index]:.3f})")
        ax.plot(potential[predicted_index], y_data[predicted_index], "g^",
                markersize=8, zorder=5)

    ax.set_xlabel("Potential (V vs. Ref)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)

    if save_name:
        _save(fig, save_name)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 2. Derivative analysis
# ---------------------------------------------------------------------------

def plot_derivative_analysis(
    potential: np.ndarray,
    current: np.ndarray,
    dI_dV: np.ndarray,
    onset_index: Optional[int] = None,
    threshold: Optional[float] = None,
    save_name: Optional[str] = None,
) -> None:
    """Two-panel plot: I(V) on top, dI/dV on bottom."""
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    ax1.plot(potential, current, "b-", linewidth=1.0)
    ax1.set_ylabel("Current (A/cm²)")
    ax1.set_title("Polarization Curve & Derivative Analysis")

    ax2.plot(potential, dI_dV, "m-", linewidth=1.0, label="dI/dV")
    ax2.set_ylabel("dI/dV")
    ax2.set_xlabel("Potential (V vs. Ref)")

    if threshold is not None:
        ax2.axhline(threshold, color="orange", linestyle="--",
                    label=f"Threshold = {threshold:.2e}")

    if onset_index is not None:
        for ax in (ax1, ax2):
            ax.axvline(potential[onset_index], color="red", linestyle="--",
                       alpha=0.7, label="Onset")

    ax2.legend(fontsize=9)
    fig.tight_layout()

    if save_name:
        _save(fig, save_name)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 3. Batch comparison (true vs predicted)
# ---------------------------------------------------------------------------

def plot_true_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "True vs Predicted Onset Potential",
    save_name: Optional[str] = None,
) -> None:
    """Scatter plot of true vs predicted onset potentials."""
    fig, ax = plt.subplots()

    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5, s=40)

    lo = min(y_true.min(), y_pred.min()) - 0.05
    hi = max(y_true.max(), y_pred.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Ideal (y=x)")

    ax.set_xlabel("True Onset Potential (V)")
    ax.set_ylabel("Predicted Onset Potential (V)")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    if save_name:
        _save(fig, save_name)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 4. Training history
# ---------------------------------------------------------------------------

def plot_training_history(
    history: Dict[str, list],
    title: str = "Training History",
    save_name: Optional[str] = None,
) -> None:
    """Plot train/val loss curves."""
    fig, ax = plt.subplots()

    if "train_loss" in history:
        ax.plot(history["train_loss"], label="Train loss")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()

    if save_name:
        _save(fig, save_name)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 5. Feature importances (bar chart)
# ---------------------------------------------------------------------------

def plot_feature_importances(
    importances: Dict[str, float],
    title: str = "Feature Importances (Random Forest)",
    save_name: Optional[str] = None,
) -> None:
    """Horizontal bar chart of feature importances."""
    sorted_items = sorted(importances.items(), key=lambda x: x[1])
    names = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.4)))
    ax.barh(names, values, color="steelblue", edgecolor="k", linewidth=0.5)
    ax.set_xlabel("Importance")
    ax.set_title(title)

    if save_name:
        _save(fig, save_name)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 6. Multi-curve overlay
# ---------------------------------------------------------------------------

def plot_multiple_curves(
    samples: List[dict],
    max_curves: int = 10,
    log_scale: bool = True,
    save_name: Optional[str] = None,
) -> None:
    """Overlay multiple polarization curves for visual comparison."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for s in samples[:max_curves]:
        V = s["potential_V"]
        I = s["current_A"]
        y = np.log10(np.abs(I) + 1e-12) if log_scale else I
        label = s.get("sample_id", "")
        ax.plot(V, y, linewidth=0.8, alpha=0.7, label=label)

        idx = s.get("pitting_onset_index")
        if idx is not None:
            ax.plot(V[idx], y[idx], "o", markersize=6)

    ylabel = "log₁₀|I|" if log_scale else "Current"
    ax.set_xlabel("Potential (V vs. Ref)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Polarization Curves (n={min(len(samples), max_curves)})")
    if len(samples) <= 15:
        ax.legend(fontsize=7, ncol=2)

    if save_name:
        _save(fig, save_name)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 7. Serration curve plot
# ---------------------------------------------------------------------------

def plot_serration_curve(
    load: np.ndarray,
    displacement: np.ndarray,
    bursts: Optional[List[dict]] = None,
    title: str = "Nanoindentation Load–Displacement",
    save_name: Optional[str] = None,
) -> None:
    """Plot a single nanoindentation load-displacement curve with burst markers."""
    fig, ax = plt.subplots()
    ax.plot(displacement, load, "b-", linewidth=0.8, label="Load–Displacement")

    if bursts:
        for b in bursts:
            idx = b["index"]
            if idx < len(load):
                ax.plot(displacement[idx], load[idx], "r^", markersize=6, zorder=5)
        ax.plot([], [], "r^", markersize=6, label=f"Bursts (n={len(bursts)})")

    ax.set_xlabel("Displacement (nm)")
    ax.set_ylabel("Load (mN)")
    ax.set_title(title)
    ax.legend(fontsize=9)

    if save_name:
        _save(fig, save_name)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 8. Serration burst histogram
# ---------------------------------------------------------------------------

def plot_burst_amplitude_histogram(
    samples: List[dict],
    title: str = "Burst Amplitude Distribution",
    save_name: Optional[str] = None,
) -> None:
    """Histogram of burst amplitudes across multiple serration samples."""
    all_amps = []
    for s in samples:
        bursts = s.get("detected_bursts", s.get("bursts", []))
        all_amps.extend([b["amplitude_nm"] for b in bursts])

    if not all_amps:
        print("  ⚠️  No bursts found for histogram")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_amps, bins=30, edgecolor="k", alpha=0.7, color="coral")
    ax.set_xlabel("Burst Amplitude (nm)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.axvline(np.mean(all_amps), color="navy", linestyle="--",
               label=f"Mean = {np.mean(all_amps):.1f} nm")
    ax.legend()

    if save_name:
        _save(fig, save_name)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 9. Correlation heatmap (serration vs corrosion)
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    corr_df: "pd.DataFrame",
    top_n: int = 15,
    title: str = "Serration–Corrosion Feature Correlations",
    save_name: Optional[str] = None,
) -> None:
    """Bar chart of top correlations between serration features and corrosion target."""
    import pandas as pd
    df = corr_df.head(top_n).copy()
    df = df.sort_values("pearson_r", key=abs)

    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.45)))
    colors = ["steelblue" if r >= 0 else "coral" for r in df["pearson_r"]]
    ax.barh(df["feature"], df["pearson_r"], color=colors, edgecolor="k", linewidth=0.5)
    ax.set_xlabel("Pearson r")
    ax.set_title(title)
    ax.axvline(0, color="black", linewidth=0.5)

    if save_name:
        _save(fig, save_name)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 10. ROC & Precision-Recall curves (fusion model)
# ---------------------------------------------------------------------------

def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title_prefix: str = "Fusion Model",
    save_name: Optional[str] = None,
) -> None:
    """Side-by-side ROC and Precision-Recall curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    ax1.plot(fpr, tpr, "b-", linewidth=2, label=f"AUC = {auc:.3f}")
    ax1.plot([0, 1], [0, 1], "r--", alpha=0.5, label="Random")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(f"{title_prefix} — ROC Curve")
    ax1.legend()

    # Precision-Recall
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    from sklearn.metrics import average_precision_score
    ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    ax2.plot(rec, prec, "g-", linewidth=2, label=f"AP = {ap:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"{title_prefix} — Precision-Recall")
    ax2.legend()

    fig.tight_layout()

    if save_name:
        _save(fig, save_name)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 11. Fracture risk distribution
# ---------------------------------------------------------------------------

def plot_risk_distribution(
    risk_scores: np.ndarray,
    risk_binary: np.ndarray,
    title: str = "Fracture Risk Score Distribution",
    save_name: Optional[str] = None,
) -> None:
    """Histogram of fracture risk scores coloured by binary label."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(risk_scores[risk_binary == 0], bins=20, alpha=0.6, color="steelblue",
            edgecolor="k", label="Low risk")
    ax.hist(risk_scores[risk_binary == 1], bins=20, alpha=0.6, color="coral",
            edgecolor="k", label="High risk")
    ax.set_xlabel("Fracture Risk Score")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

    if save_name:
        _save(fig, save_name)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 12. Multi-curve serration overlay
# ---------------------------------------------------------------------------

def plot_multiple_serration_curves(
    samples: List[dict],
    max_curves: int = 8,
    save_name: Optional[str] = None,
) -> None:
    """Overlay multiple nanoindentation curves."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for s in samples[:max_curves]:
        disp = s["displacement_nm"]
        load = s["load_mN"]
        ax.plot(disp, load, linewidth=0.8, alpha=0.7, label=s.get("sample_id", ""))
        bursts = s.get("detected_bursts", s.get("bursts", []))
        for b in bursts[:20]:  # limit markers
            idx = b["index"]
            if idx < len(load):
                ax.plot(disp[idx], load[idx], "r^", markersize=4, alpha=0.5)

    ax.set_xlabel("Displacement (nm)")
    ax.set_ylabel("Load (mN)")
    ax.set_title(f"Nanoindentation Curves (n={min(len(samples), max_curves)})")
    if len(samples) <= 12:
        ax.legend(fontsize=7, ncol=2)

    if save_name:
        _save(fig, save_name)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.synthetic_data import generate_dataset

    samples = generate_dataset(n_samples=8, seed=0)
    plot_multiple_curves(samples, log_scale=True, save_name="demo_curves.png")
    plot_curve(
        samples[0]["potential_V"], samples[0]["current_A"],
        onset_index=samples[0]["pitting_onset_index"],
        title=f"Sample {samples[0]['sample_id']}",
        save_name="demo_single_curve.png",
    )
    print("✅ Demo plots generated in figures/")
