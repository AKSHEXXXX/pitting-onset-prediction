"""
lstm_model.py
=============
LSTM-based sequence model for predicting pitting onset from
windowed polarization-curve segments.

Architecture
------------
Input  → (batch, seq_len, 2)       [V, I] per time-step
       → LSTM layer(s)
       → Fully-connected head
Output → scalar onset potential  **or**  binary onset flag per window

Two modes are supported:
  1. **Regression** – predict the onset potential directly.
  2. **Classification** – predict whether a given window contains the
     onset point (binary).
"""

import os
import json
import numpy as np
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.preprocessing import (
    preprocess_sample,
    create_windows,
    min_max_normalize,
)


# ===================================================================
# Dataset
# ===================================================================

class PolarizationWindowDataset(Dataset):
    """PyTorch dataset of sliding-window segments."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Parameters
        ----------
        X : np.ndarray (n_windows, seq_len, 2)
        y : np.ndarray (n_windows,)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_lstm_data(
    samples: List[dict],
    window_size: int = 50,
    stride: int = 5,
    mode: str = "classification",
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a list of samples into windowed X, y arrays.

    Parameters
    ----------
    samples : list[dict]
    window_size : int
    stride : int
    mode : str
        ``"classification"`` → binary labels (window contains onset).
        ``"regression"``     → onset potential as label for positive windows.
    """
    all_X, all_y = [], []

    for s in samples:
        V, _ = min_max_normalize(s["potential_V"])
        I, _ = min_max_normalize(s["current_A"])
        target_idx = s.get("pitting_onset_index")

        X_win, y_win = create_windows(V, I, window_size, stride, target_idx)

        if target_idx is None:
            # Sample has no pitting – all windows are negative
            y_win = np.zeros(len(y_win), dtype=np.float64)

        if mode == "regression" and target_idx is not None:
            onset_V = s["pitting_onset_potential_V"]
            # For regression, only keep positive windows, label = onset V
            mask = y_win == 1.0
            X_win = X_win[mask]
            y_win = np.full(mask.sum(), onset_V, dtype=np.float64)

        all_X.append(X_win)
        all_y.append(y_win)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y


# ===================================================================
# Model
# ===================================================================

class PittingLSTM(nn.Module):
    """LSTM model for pitting onset prediction.

    Parameters
    ----------
    input_size : int
        Number of features per time-step (default 2: V and I).
    hidden_size : int
        LSTM hidden dimension.
    num_layers : int
        Stacked LSTM layers.
    dropout : float
        Dropout between LSTM layers (only when num_layers > 1).
    bidirectional : bool
        Use bidirectional LSTM.
    output_size : int
        1 for regression or binary classification.
    task : str
        ``"classification"`` applies sigmoid; ``"regression"`` is raw.
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        output_size: int = 1,
        task: str = "classification",
    ):
        super().__init__()
        self.task = task
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        fc_input = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_input, fc_input // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_input // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        out : (batch, output_size)
        """
        # LSTM output: (batch, seq_len, hidden * directions)
        lstm_out, _ = self.lstm(x)
        # Take the last time-step's output
        last = lstm_out[:, -1, :]
        out = self.fc(last)

        if self.task == "classification":
            out = torch.sigmoid(out)

        return out.squeeze(-1)


# ===================================================================
# Focal Loss (handles class imbalance)
# ===================================================================

class FocalLoss(nn.Module):
    """Focal Loss for binary classification with class imbalance.

    Reduces loss contribution from easy negatives, focusing training
    on hard-to-classify onset windows.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    alpha : float
        Weighting factor for the positive class (default 0.75).
    gamma : float
        Focusing parameter; higher = more focus on hard examples (default 2.0).
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(1e-7, 1 - 1e-7)
        # alpha weighting: alpha for positives, (1-alpha) for negatives
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        p_t = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - p_t) ** self.gamma
        loss = -alpha_t * focal_weight * torch.log(p_t)
        return loss.mean()


# ===================================================================
# Temporal Attention
# ===================================================================

class TemporalAttention(nn.Module):
    """Additive attention over LSTM hidden states.

    Instead of only using the last hidden state, learns a weighted
    combination of all time-step outputs — letting the model focus on
    the region of the window that matters most for onset detection.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False),
        )

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        lstm_output : (batch, seq_len, hidden_size)

        Returns
        -------
        context : (batch, hidden_size)  — weighted sum of hidden states
        weights : (batch, seq_len)      — attention weights (for visualisation)
        """
        scores = self.attn(lstm_output).squeeze(-1)       # (batch, seq_len)
        weights = torch.softmax(scores, dim=1)             # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1),
                            lstm_output).squeeze(1)        # (batch, hidden)
        return context, weights


class PittingLSTMWithAttention(nn.Module):
    """LSTM + Temporal Attention for pitting onset detection.

    Same interface as ``PittingLSTM`` but replaces last-hidden-state
    pooling with a learned attention mechanism.
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        output_size: int = 1,
        task: str = "classification",
    ):
        super().__init__()
        self.task = task
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        lstm_out_size = hidden_size * self.num_directions

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.attention = TemporalAttention(lstm_out_size)

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_size, lstm_out_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)                # (batch, seq, hidden*dirs)
        context, self._attn_weights = self.attention(lstm_out)  # (batch, hidden*dirs)
        out = self.fc(context)
        if self.task == "classification":
            out = torch.sigmoid(out)
        return out.squeeze(-1)

    @property
    def attn_weights(self) -> Optional[torch.Tensor]:
        """Last forward pass attention weights (for interpretability)."""
        return getattr(self, "_attn_weights", None)


# ===================================================================
# Oversampling / Weighted Sampler
# ===================================================================

def make_weighted_sampler(y: np.ndarray) -> torch.utils.data.WeightedRandomSampler:
    """Create a WeightedRandomSampler that balances classes.

    Positive (onset) windows get higher sampling probability so each
    mini-batch has roughly equal representation.
    """
    counts = np.bincount(y.astype(int))
    class_weights = 1.0 / counts
    sample_weights = class_weights[y.astype(int)]
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(y),
        replacement=True,
    )


# ===================================================================
# Training loop
# ===================================================================

def train_lstm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
    save_path: Optional[str] = None,
    criterion: Optional[nn.Module] = None,
) -> dict:
    """Train the LSTM model.

    Parameters
    ----------
    criterion : nn.Module, optional
        Loss function. If *None*, defaults to BCELoss (classification)
        or MSELoss (regression).

    Returns
    -------
    dict with ``train_losses``, ``val_losses``, ``best_val_loss``.
    """
    model = model.to(device)

    if criterion is None:
        if getattr(model, 'task', 'classification') == "classification":
            criterion = nn.BCELoss()
        else:
            criterion = nn.MSELoss()
    criterion = criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(y_batch)

        epoch_loss /= len(train_loader.dataset)
        history["train_loss"].append(epoch_loss)

        # --- Validate ---
        val_loss = None
        if val_loader is not None:
            model.eval()
            vloss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    preds = model(X_batch)
                    vloss += criterion(preds, y_batch).item() * len(y_batch)
            val_loss = vloss / len(val_loader.dataset)
            history["val_loss"].append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(model.state_dict(), save_path)

        if epoch % 10 == 0 or epoch == 1:
            msg = f"Epoch {epoch:3d}/{epochs}  train_loss={epoch_loss:.6f}"
            if val_loss is not None:
                msg += f"  val_loss={val_loss:.6f}"
            print(msg)

    history["best_val_loss"] = best_val

    # Save final metrics
    if save_path:
        metrics_path = save_path.replace(".pt", "_history.json")
        with open(metrics_path, "w") as f:
            json.dump(history, f, indent=2)

    return history


# ===================================================================
# Inference helper
# ===================================================================

def predict(model: PittingLSTM,
            X: np.ndarray,
            device: str = "cpu") -> np.ndarray:
    """Run inference on windowed data.

    Parameters
    ----------
    X : np.ndarray (n_windows, seq_len, 2)

    Returns
    -------
    np.ndarray (n_windows,)
    """
    model.eval()
    tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(tensor).cpu().numpy()
    return preds


# ===================================================================
# CLI
# ===================================================================
if __name__ == "__main__":
    from src.synthetic_data import generate_dataset

    print("Generating synthetic data …")
    samples = generate_dataset(n_samples=60, seed=0)

    print("Preparing windowed LSTM data (classification mode) …")
    X, y = prepare_lstm_data(samples, window_size=50, stride=5,
                             mode="classification")
    print(f"  X shape: {X.shape}   y shape: {y.shape}")
    print(f"  Positive windows: {int(y.sum())}  Negative: {int(len(y) - y.sum())}")

    # Train/val split
    split = int(0.8 * len(X))
    train_ds = PolarizationWindowDataset(X[:split], y[:split])
    val_ds = PolarizationWindowDataset(X[split:], y[split:])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = PittingLSTM(
        input_size=2, hidden_size=64, num_layers=2,
        dropout=0.2, task="classification",
    )
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(model)

    print("\nTraining …")
    history = train_lstm(
        model, train_loader, val_loader,
        epochs=30, lr=1e-3,
        save_path="models/lstm_classifier.pt",
    )

    print(f"\n✅ Best validation loss: {history['best_val_loss']:.6f}")
