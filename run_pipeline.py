"""
run_pipeline.py
===============
End-to-end pipeline covering all three project phases:

  **Phase 1** — Corrosion-only LSTM model
    1. Generate (or load) polarization curve data
    2. Preprocess (smooth, normalise, derivatives)
    3. Heuristic onset detection
    4. Train baseline Random Forest
    5. Train LSTM classifier

  **Phase 2** — Shear band / serration signals
    6. Generate (or load) nanoindentation serration data
    7. Preprocess & detect bursts
    8. Extract serration features & train baseline models
    9. Align serration features with corrosion predictions

  **Phase 3** — Integrated fracture risk prediction
    10. Build fused feature set
    11. Train Gradient Boosting fusion classifier
    12. Train dual-branch neural fusion model
    13. Evaluate and visualise

Usage
-----
    python run_pipeline.py                   # all phases, synthetic data
    python run_pipeline.py --phase 1         # Phase 1 only
    python run_pipeline.py --phase 2         # Phases 1 & 2
    python run_pipeline.py data_raw/         # Phase 1 with real corrosion data
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Phase 1 modules
from src.synthetic_data import generate_dataset, save_samples_csv
from src.data_loader import load_directory
from src.preprocessing import preprocess_all, create_windows, min_max_normalize
from src.onset_detection import detect_all
from src.baseline_model import build_feature_matrix, train_baseline
from src.lstm_model import (
    PittingLSTM,
    prepare_lstm_data,
    PolarizationWindowDataset,
    train_lstm,
    predict,
)

# Phase 2 modules
from src.synthetic_serration_data import generate_serration_dataset, save_serration_csv
from src.serration_preprocessing import preprocess_serration_all
from src.serration_features import build_serration_feature_matrix
from src.serration_model import (
    build_serration_Xy_classification,
    build_serration_Xy_regression,
    train_serration_classifier,
    train_serration_regressor,
)
from src.alignment import (
    pair_by_index,
    build_aligned_features,
    compute_correlations,
    print_alignment_summary,
)

# Phase 3 modules
from src.fusion_model import (
    build_fusion_dataset,
    train_rf_fusion,
    train_neural_fusion,
    build_fracture_risk_labels,
)

# Evaluation & plotting
from src.evaluation import (
    regression_metrics,
    classification_metrics,
    evaluate_detection,
    print_metrics,
)
from src.plotting import (
    plot_curve,
    plot_multiple_curves,
    plot_derivative_analysis,
    plot_true_vs_predicted,
    plot_training_history,
    plot_feature_importances,
    plot_serration_curve,
    plot_burst_amplitude_histogram,
    plot_multiple_serration_curves,
    plot_correlation_heatmap,
    plot_roc_pr_curves,
    plot_risk_distribution,
)

import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Phase 1 — Corrosion LSTM
# ---------------------------------------------------------------------------

def run_phase1(data_dir=None, n_synthetic=100, seed=42):
    """Phase 1: Corrosion-only modelling."""
    print("\n" + "=" * 60)
    print("  PHASE 1 — Corrosion-Only LSTM Model")
    print("=" * 60)

    # ── 1. Data ──────────────────────────────────────────────
    if data_dir and os.path.isdir(data_dir):
        print(f"\n📂 Loading real data from {data_dir} …")
        samples = load_directory(data_dir, file_ext="csv")
    else:
        print(f"\n🧪 Generating {n_synthetic} synthetic polarization curves …")
        samples = generate_dataset(n_samples=n_synthetic, seed=seed)
        save_samples_csv(samples, "data_raw/synthetic")

    # ── 2. Preprocess ────────────────────────────────────────
    print("\n⚙️  Preprocessing …")
    samples = preprocess_all(samples, smooth=True, normalize=True, compute_deriv=True)

    # ── 3. Heuristic onset detection ─────────────────────────
    print("\n🔍 Running heuristic onset detection …")
    samples = detect_all(samples)
    det_metrics = evaluate_detection(samples)
    print_metrics(det_metrics, title="Heuristic Detection")

    # ── 4. Plots ─────────────────────────────────────────────
    print("\n🎨 Generating Phase 1 plots …")
    plot_multiple_curves(samples[:8], log_scale=True,
                         save_name="p1_exploration_curves.png")
    s0 = samples[0]
    plot_curve(s0["potential_V"], s0["current_A"],
               onset_index=s0.get("pitting_onset_index"),
               title=f"Sample {s0['sample_id']}",
               save_name="p1_sample_curve.png")
    if "dI_dV" in s0 and "onset_info" in s0:
        plot_derivative_analysis(
            s0["potential_V"], s0["current_A"], s0["dI_dV"],
            onset_index=s0.get("pitting_onset_index"),
            threshold=s0["onset_info"].get("threshold"),
            save_name="p1_derivative_analysis.png",
        )

    # ── 5. Baseline Random Forest ────────────────────────────
    print("\n🌲 Training Random Forest baseline …")
    X_rf, y_rf = build_feature_matrix(samples)
    rf_result = None
    if len(X_rf) > 5:
        rf_result = train_baseline(X_rf, y_rf, save_path="models/rf_baseline.pkl")
        print_metrics(rf_result["metrics"], title="Random Forest Baseline")
        plot_true_vs_predicted(rf_result["y_test"], rf_result["y_pred"],
                               save_name="p1_rf_true_vs_pred.png")
        plot_feature_importances(rf_result["feature_importances"],
                                  save_name="p1_rf_feature_importances.png")
    else:
        print("  ⚠️  Not enough labelled samples for RF training")

    # ── 6. LSTM Classifier ───────────────────────────────────
    print("\n🧠 Training LSTM classifier …")
    X_lstm, y_lstm = prepare_lstm_data(samples, window_size=50, stride=5,
                                        mode="classification")
    print(f"   Windows: {X_lstm.shape[0]}  "
          f"(positive={int(y_lstm.sum())}, negative={int(len(y_lstm) - y_lstm.sum())})")

    split = int(0.8 * len(X_lstm))
    train_ds = PolarizationWindowDataset(X_lstm[:split], y_lstm[:split])
    val_ds = PolarizationWindowDataset(X_lstm[split:], y_lstm[split:])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PittingLSTM(input_size=2, hidden_size=64, num_layers=2,
                        dropout=0.2, task="classification")
    history = train_lstm(model, train_loader, val_loader,
                         epochs=30, lr=1e-3, device=device,
                         save_path="models/lstm_classifier.pt")

    preds = predict(model, X_lstm[split:], device=device)
    cls_metrics = classification_metrics(y_lstm[split:], preds)
    print_metrics(cls_metrics, title="LSTM Classifier (Validation)")
    plot_training_history(history, save_name="p1_lstm_training_history.png")

    return {
        "samples": samples,
        "det_metrics": det_metrics,
        "rf_result": rf_result,
        "cls_metrics": cls_metrics,
    }


# ---------------------------------------------------------------------------
# Phase 2 — Serration signals
# ---------------------------------------------------------------------------

def run_phase2(corr_samples, n_synthetic=100, seed=42):
    """Phase 2: Shear band serration modelling & alignment."""
    print("\n" + "=" * 60)
    print("  PHASE 2 — Shear Band Serration Signals")
    print("=" * 60)

    # ── 1. Serration data ────────────────────────────────────
    print(f"\n🧪 Generating {n_synthetic} synthetic serration curves …")
    serr_samples = generate_serration_dataset(n_samples=n_synthetic, seed=seed)
    save_serration_csv(serr_samples, "data_raw/serration_synthetic")

    # ── 2. Preprocess serration ──────────────────────────────
    print("\n⚙️  Preprocessing serration data …")
    serr_samples = preprocess_serration_all(serr_samples)

    # ── 3. Serration exploration plots ───────────────────────
    print("\n🎨 Generating Phase 2 plots …")
    plot_multiple_serration_curves(serr_samples[:8],
                                   save_name="p2_serration_curves.png")
    s0 = serr_samples[0]
    plot_serration_curve(s0["load_mN"], s0["displacement_nm"],
                         bursts=s0.get("detected_bursts", s0.get("bursts")),
                         title=f"Serration: {s0['sample_id']}",
                         save_name="p2_single_serration.png")
    plot_burst_amplitude_histogram(serr_samples,
                                    save_name="p2_burst_histogram.png")

    # ── 4. Feature extraction ────────────────────────────────
    print("\n📊 Extracting serration features …")
    X_serr, serr_ids = build_serration_feature_matrix(serr_samples)
    print(f"   Feature matrix: {X_serr.shape}")

    # ── 5. Serration classification baseline ─────────────────
    print("\n🌲 Training serration classifier (high vs low serration) …")
    X_cls, y_cls = build_serration_Xy_classification(serr_samples, threshold_burst_count=10)
    serr_cls_result = None
    if len(np.unique(y_cls)) > 1 and len(y_cls) > 10:
        serr_cls_result = train_serration_classifier(
            X_cls, y_cls, save_path="models/serration_clf.pkl"
        )
        print_metrics(serr_cls_result["metrics"], title="Serration Classifier")
        plot_feature_importances(serr_cls_result["feature_importances"],
                                  title="Serration Classifier — Feature Importances",
                                  save_name="p2_serr_clf_importances.png")
    else:
        print("  ⚠️  Insufficient class balance for serration classifier")

    # ── 6. Serration regression baseline ─────────────────────
    print("\n🌲 Training serration regressor (instability score) …")
    X_reg, y_reg = build_serration_Xy_regression(serr_samples)
    serr_reg_result = train_serration_regressor(
        X_reg, y_reg, save_path="models/serration_reg.pkl"
    )
    print_metrics(serr_reg_result["metrics"], title="Serration Regressor")
    plot_true_vs_predicted(serr_reg_result["y_test"], serr_reg_result["y_pred"],
                           title="Serration Regressor — True vs Predicted",
                           save_name="p2_serr_reg_scatter.png")

    # ── 7. Alignment with corrosion data ─────────────────────
    print("\n🔗 Aligning serration & corrosion features …")
    pairs = pair_by_index(corr_samples, serr_samples)
    aligned_df = build_aligned_features(pairs)
    print_alignment_summary(pairs, aligned_df)

    # Correlation analysis
    corr_table = compute_correlations(aligned_df)
    print("Top serration–corrosion correlations:")
    print(corr_table.head(10).to_string(index=False))
    plot_correlation_heatmap(corr_table, save_name="p2_correlation_chart.png")

    return {
        "serr_samples": serr_samples,
        "pairs": pairs,
        "aligned_df": aligned_df,
        "corr_table": corr_table,
        "serr_cls_result": serr_cls_result,
        "serr_reg_result": serr_reg_result,
    }


# ---------------------------------------------------------------------------
# Phase 3 — Integrated fracture risk prediction
# ---------------------------------------------------------------------------

def run_phase3(pairs):
    """Phase 3: Combined corrosion + serration fusion model."""
    print("\n" + "=" * 60)
    print("  PHASE 3 — Integrated Fracture Risk Prediction")
    print("=" * 60)

    # ── 1. Build fused dataset ───────────────────────────────
    print("\n📊 Building fusion dataset …")
    X, y_score, y_binary = build_fusion_dataset(pairs)
    print(f"   Features: {X.shape}")
    print(f"   Risk+ (high): {y_binary.sum()}  Risk- (low): {(1 - y_binary).sum()}")

    plot_risk_distribution(y_score, y_binary,
                           save_name="p3_risk_distribution.png")

    # ── 2. Gradient Boosting fusion ──────────────────────────
    print("\n🌳 Training Gradient Boosting fusion classifier …")
    gb_result = None
    if len(np.unique(y_binary)) > 1:
        gb_result = train_rf_fusion(X, y_binary,
                                    save_path="models/fusion_gb.pkl")
        print_metrics(gb_result["metrics"], title="Gradient Boosting Fusion")
        plot_roc_pr_curves(gb_result["y_test"], gb_result["y_prob"],
                           title_prefix="GB Fusion",
                           save_name="p3_gb_roc_pr.png")
        plot_feature_importances(gb_result["feature_importances"],
                                  title="Fusion — Feature Importances",
                                  save_name="p3_fusion_importances.png")
    else:
        print("  ⚠️  Single class in labels — skipping GB fusion")

    # ── 3. Neural dual-branch fusion ─────────────────────────
    print("\n🧠 Training dual-branch neural fusion model …")
    nn_result = None
    if len(np.unique(y_binary)) > 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        nn_result = train_neural_fusion(
            X, y_binary,
            epochs=80, lr=1e-3, batch_size=32,
            device=device,
            save_path="models/fusion_nn.pt",
        )
        print_metrics(nn_result["metrics"], title="Neural Fusion")
        plot_roc_pr_curves(nn_result["y_test"], nn_result["y_prob"],
                           title_prefix="Neural Fusion",
                           save_name="p3_nn_roc_pr.png")
        plot_training_history(nn_result["history"],
                              title="Neural Fusion Training",
                              save_name="p3_nn_training_history.png")
    else:
        print("  ⚠️  Single class in labels — skipping neural fusion")

    return {
        "gb_result": gb_result,
        "nn_result": nn_result,
        "y_score": y_score,
        "y_binary": y_binary,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run(data_dir: str | None = None,
        n_synthetic: int = 100,
        seed: int = 42,
        max_phase: int = 3) -> None:
    """Execute the full multi-phase pipeline."""

    print("=" * 60)
    print("  Pitting Onset & Fracture Risk Prediction Pipeline")
    print("=" * 60)

    summary = {}

    # ── Phase 1 ──────────────────────────────────────────────
    p1 = run_phase1(data_dir=data_dir, n_synthetic=n_synthetic, seed=seed)
    summary["phase1"] = {
        "n_samples": len(p1["samples"]),
        "heuristic_detection": p1["det_metrics"],
        "random_forest": p1["rf_result"]["metrics"] if p1["rf_result"] else None,
        "lstm_classifier": p1["cls_metrics"],
    }

    if max_phase < 2:
        _save_summary(summary)
        return

    # ── Phase 2 ──────────────────────────────────────────────
    p2 = run_phase2(p1["samples"], n_synthetic=n_synthetic, seed=seed)
    summary["phase2"] = {
        "n_serration_samples": len(p2["serr_samples"]),
        "n_aligned_pairs": len(p2["pairs"]),
        "serration_classifier": (
            p2["serr_cls_result"]["metrics"] if p2["serr_cls_result"] else None
        ),
        "serration_regressor": p2["serr_reg_result"]["metrics"],
    }

    if max_phase < 3:
        _save_summary(summary)
        return

    # ── Phase 3 ──────────────────────────────────────────────
    p3 = run_phase3(p2["pairs"])
    summary["phase3"] = {
        "gb_fusion": p3["gb_result"]["metrics"] if p3["gb_result"] else None,
        "nn_fusion": p3["nn_result"]["metrics"] if p3["nn_result"] else None,
    }

    _save_summary(summary)

    # ── Final summary ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅  Full Pipeline Complete (Phases 1–3)")
    print("=" * 60)
    print("  Outputs:")
    print("    models/   → rf_baseline.pkl, lstm_classifier.pt,")
    print("                serration_clf.pkl, serration_reg.pkl,")
    print("                fusion_gb.pkl, fusion_nn.pt")
    print("    figures/  → p1_*, p2_*, p3_* plots")
    print("    results/  → pipeline_summary.json")
    print("=" * 60)


def _save_summary(summary):
    os.makedirs("results", exist_ok=True)
    with open("results/pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n💾 Saved results/pipeline_summary.json")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pitting Onset & Fracture Risk Prediction Pipeline"
    )
    parser.add_argument("data_dir", nargs="?", default=None,
                        help="Directory containing real CSV data (optional)")
    parser.add_argument("--n-synthetic", type=int, default=100,
                        help="Number of synthetic curves if no data_dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", type=int, default=3, choices=[1, 2, 3],
                        help="Run up to this phase (1, 2, or 3)")

    args = parser.parse_args()
    run(data_dir=args.data_dir, n_synthetic=args.n_synthetic,
        seed=args.seed, max_phase=args.phase)
