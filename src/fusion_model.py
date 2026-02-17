"""
fusion_model.py
===============
Phase 3 — Integrated corrosion + serration model for predicting
corrosion-assisted fracture risk.

Architecture
------------
Two fusion approaches are provided:

1. **Feature-level fusion (Random Forest)**
   Concatenate corrosion features and serration features into a single
   vector → Random Forest → fracture risk probability.

2. **Neural feature fusion (MLP / dual-branch)**
   - Branch A: corrosion features → FC layers → embedding
   - Branch B: serration features → FC layers → embedding
   - Fusion: concatenate embeddings → FC head → risk score ∈ [0, 1]

The target variable ``fracture_risk`` is a synthetic label ∈ [0, 1]
combining pitting onset proximity and serration instability.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, brier_score_loss,
)

from src.alignment import build_aligned_features, pair_by_index
from src.serration_model import compute_instability_score


# ===================================================================
# Fracture risk label generation
# ===================================================================

def compute_fracture_risk(
    corr_sample: dict,
    serr_sample: dict,
    w_corrosion: float = 0.5,
    w_serration: float = 0.5,
) -> float:
    """Compute a synthetic fracture risk score ∈ [0, 1].

    Combines:
      - Corrosion risk: how early pitting onset occurs (lower onset V → higher risk)
      - Serration risk: instability score from nanoindentation

    Parameters
    ----------
    corr_sample, serr_sample : dict
    w_corrosion, w_serration : float
        Relative weights (must sum to 1.0).

    Returns
    -------
    float ∈ [0, 1]
    """
    # Corrosion component: early onset = high risk
    onset_V = corr_sample.get("pitting_onset_potential_V")
    if onset_V is not None:
        # Normalise: onset in [-0.4, 1.5] → risk in [1, 0]
        corr_risk = 1.0 - np.clip((onset_V + 0.4) / 1.9, 0, 1)
    else:
        corr_risk = 0.1  # no pitting → low risk

    # Serration component
    serr_risk = compute_instability_score(serr_sample)

    risk = w_corrosion * corr_risk + w_serration * serr_risk
    return float(np.clip(risk, 0, 1))


def build_fracture_risk_labels(
    pairs: List[Tuple[dict, dict]],
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute continuous risk scores and binary labels for paired samples.

    Returns
    -------
    risk_scores : np.ndarray  (n,)   — continuous ∈ [0, 1]
    risk_binary : np.ndarray  (n,)   — 1 if score >= threshold
    """
    scores = np.array([compute_fracture_risk(c, s) for c, s in pairs])
    binary = (scores >= threshold).astype(int)
    return scores, binary


# ===================================================================
# Build X, y for fusion models
# ===================================================================

def build_fusion_dataset(
    pairs: List[Tuple[dict, dict]],
    risk_threshold: float = 0.5,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build feature matrix with aligned corrosion + serration features
    and fracture risk labels.

    Returns
    -------
    X : pd.DataFrame  (n_pairs, n_features)
    y_score : np.ndarray  — continuous risk score
    y_binary : np.ndarray — binary risk label
    """
    aligned = build_aligned_features(pairs)

    # Compute labels
    risk_scores, risk_binary = build_fracture_risk_labels(pairs, risk_threshold)
    aligned["fracture_risk_score"] = risk_scores
    aligned["fracture_risk_binary"] = risk_binary

    # Feature columns (all numeric, excluding IDs and labels)
    feature_cols = [
        c for c in aligned.columns
        if c.startswith("corr_") or c.startswith("serr_")
    ]
    X = aligned[feature_cols].copy()
    # Fill any NaN with 0
    X = X.fillna(0.0)

    return X, risk_scores, risk_binary


# ===================================================================
# Model 1: Random Forest fusion
# ===================================================================

def train_rf_fusion(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    n_estimators: int = 300,
    random_state: int = 42,
    save_path: Optional[str] = None,
) -> Dict:
    """Train a Random Forest classifier on fused features for fracture risk.

    Returns dict with model, metrics, feature_importances, y_test, y_pred, y_prob.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y,
    )

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "avg_precision": float(average_precision_score(y_test, y_prob)),
        "brier_score": float(brier_score_loss(y_test, y_prob)),
    }

    cv_scores = cross_val_score(
        model, X, y, cv=min(5, len(y)), scoring="roc_auc"
    )
    metrics["cv_auc_mean"] = float(cv_scores.mean())
    metrics["cv_auc_std"] = float(cv_scores.std())

    feat_imp = dict(zip(X.columns, model.feature_importances_))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        metrics_path = save_path.replace(".pkl", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"💾 Fusion RF saved to {save_path}")

    return {
        "model": model,
        "metrics": metrics,
        "feature_importances": feat_imp,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


# ===================================================================
# Model 2: Dual-branch neural fusion
# ===================================================================

class FusionDataset(Dataset):
    """Dataset for the dual-branch fusion model."""

    def __init__(self, X_corr: np.ndarray, X_serr: np.ndarray, y: np.ndarray):
        self.X_corr = torch.tensor(X_corr, dtype=torch.float32)
        self.X_serr = torch.tensor(X_serr, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_corr[idx], self.X_serr[idx], self.y[idx]


class DualBranchFusionNet(nn.Module):
    """Dual-branch MLP that fuses corrosion and serration features.

    Architecture:
        corrosion_features → FC(d_corr, 32) → ReLU → Dropout
        serration_features → FC(d_serr, 32) → ReLU → Dropout
        [concat] → FC(64, 32) → ReLU → Dropout → FC(32, 1) → Sigmoid

    Output: probability of corrosion-assisted fracture risk ∈ [0, 1].
    """

    def __init__(
        self,
        n_corr_features: int,
        n_serr_features: int,
        hidden_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.corr_branch = nn.Sequential(
            nn.Linear(n_corr_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
        )
        self.serr_branch = nn.Sequential(
            nn.Linear(n_serr_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_corr: torch.Tensor, x_serr: torch.Tensor) -> torch.Tensor:
        emb_corr = self.corr_branch(x_corr)
        emb_serr = self.serr_branch(x_serr)
        fused = torch.cat([emb_corr, emb_serr], dim=1)
        return self.fusion_head(fused).squeeze(-1)


def split_features(X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Split a combined feature DataFrame into corrosion and serration arrays.

    Returns
    -------
    X_corr, X_serr : np.ndarray
    corr_cols, serr_cols : list[str]
    """
    corr_cols = [c for c in X.columns if c.startswith("corr_")]
    serr_cols = [c for c in X.columns if c.startswith("serr_")]
    return (
        X[corr_cols].values.astype(np.float32),
        X[serr_cols].values.astype(np.float32),
        corr_cols,
        serr_cols,
    )


def train_neural_fusion(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    hidden_dim: int = 32,
    dropout: float = 0.3,
    device: str = "cpu",
    save_path: Optional[str] = None,
) -> Dict:
    """Train the dual-branch neural fusion model.

    Returns dict with model, history, metrics, y_test, y_pred, y_prob.
    """
    X_corr, X_serr, corr_cols, serr_cols = split_features(X)
    y_arr = y.astype(np.float32)

    # Train/val split
    n = len(y_arr)
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    split = int(n * (1 - test_size))

    train_idx, test_idx = idx[:split], idx[split:]

    train_ds = FusionDataset(X_corr[train_idx], X_serr[train_idx], y_arr[train_idx])
    test_ds = FusionDataset(X_corr[test_idx], X_serr[test_idx], y_arr[test_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = DualBranchFusionNet(
        n_corr_features=len(corr_cols),
        n_serr_features=len(serr_cols),
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        epoch_loss = 0.0
        for xc, xs, yb in train_loader:
            xc, xs, yb = xc.to(device), xs.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xc, xs)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(yb)
        epoch_loss /= len(train_loader.dataset)
        history["train_loss"].append(epoch_loss)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xc, xs, yb in test_loader:
                xc, xs, yb = xc.to(device), xs.to(device), yb.to(device)
                pred = model(xc, xs)
                val_loss += criterion(pred, yb).item() * len(yb)
        val_loss /= len(test_loader.dataset)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  train={epoch_loss:.4f}  val={val_loss:.4f}")

    # Final evaluation
    model.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for xc, xs, yb in test_loader:
            xc, xs = xc.to(device), xs.to(device)
            probs = model(xc, xs).cpu().numpy()
            all_probs.extend(probs)
            all_true.extend(yb.numpy())

    y_prob = np.array(all_probs)
    y_true = np.array(all_true)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true.astype(int), y_pred)),
        "precision": float(precision_score(y_true.astype(int), y_pred, zero_division=0)),
        "recall": float(recall_score(y_true.astype(int), y_pred, zero_division=0)),
        "f1": float(f1_score(y_true.astype(int), y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        meta = {
            "n_corr_features": len(corr_cols),
            "n_serr_features": len(serr_cols),
            "hidden_dim": hidden_dim,
            "metrics": metrics,
        }
        with open(save_path.replace(".pt", "_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"💾 Fusion neural model saved to {save_path}")

    return {
        "model": model,
        "history": history,
        "metrics": metrics,
        "y_test": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


# ===================================================================
# CLI
# ===================================================================
if __name__ == "__main__":
    from src.synthetic_data import generate_dataset
    from src.synthetic_serration_data import generate_serration_dataset
    from src.preprocessing import preprocess_all
    from src.serration_preprocessing import preprocess_serration_all

    print("=== Phase 3: Fusion Model ===\n")
    corr = generate_dataset(n_samples=100, seed=42)
    corr = preprocess_all(corr)
    serr = generate_serration_dataset(n_samples=100, seed=42)
    serr = preprocess_serration_all(serr)

    pairs = pair_by_index(corr, serr)
    X, y_score, y_binary = build_fusion_dataset(pairs)

    print(f"Features: {X.shape}  |  Risk+ : {y_binary.sum()}  Risk- : {(1-y_binary).sum()}\n")

    print("── Gradient Boosting Fusion ──")
    rf_res = train_rf_fusion(X, y_binary, save_path="models/fusion_gb.pkl")
    for k, v in rf_res["metrics"].items():
        print(f"  {k:20s} = {v:.4f}")

    print("\n── Neural Dual-Branch Fusion ──")
    nn_res = train_neural_fusion(X, y_binary, epochs=80, save_path="models/fusion_nn.pt")
    for k, v in nn_res["metrics"].items():
        print(f"  {k:20s} = {v:.4f}")
