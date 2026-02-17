"""
preprocessing.py
================
Signal preprocessing utilities for polarization curves:
  • Min-max / standard normalisation
  • Savitzky-Golay smoothing
  • Numerical derivatives (dI/dV, d²I/dV²)
  • Log-current transformation
  • Sliding-window sequence creation for LSTM input
"""

import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple, List


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def min_max_normalize(arr: np.ndarray,
                      feature_range: Tuple[float, float] = (0.0, 1.0)
                      ) -> Tuple[np.ndarray, dict]:
    """Min-max normalisation to a target range.

    Returns
    -------
    normalised : np.ndarray
    params : dict   – ``{"min": …, "max": …, "range": …}`` for inverse transform.
    """
    lo, hi = feature_range
    arr_min, arr_max = arr.min(), arr.max()
    denom = arr_max - arr_min
    if denom == 0:
        denom = 1.0  # constant signal edge case
    normalised = lo + (arr - arr_min) / denom * (hi - lo)
    params = {"min": arr_min, "max": arr_max, "range": feature_range}
    return normalised, params


def standard_normalize(arr: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Zero-mean, unit-variance normalisation.

    Returns
    -------
    normalised : np.ndarray
    params : dict   – ``{"mean": …, "std": …}`` for inverse transform.
    """
    mu, sigma = arr.mean(), arr.std()
    if sigma == 0:
        sigma = 1.0
    normalised = (arr - mu) / sigma
    params = {"mean": mu, "std": sigma}
    return normalised, params


def inverse_min_max(arr: np.ndarray, params: dict) -> np.ndarray:
    """Invert min-max normalisation."""
    lo, hi = params["range"]
    return (arr - lo) / (hi - lo) * (params["max"] - params["min"]) + params["min"]


def inverse_standard(arr: np.ndarray, params: dict) -> np.ndarray:
    """Invert standard normalisation."""
    return arr * params["std"] + params["mean"]


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smooth_savgol(signal: np.ndarray,
                  window_length: int = 11,
                  polyorder: int = 3) -> np.ndarray:
    """Apply Savitzky-Golay smoothing.

    Parameters
    ----------
    signal : np.ndarray
        1-D input signal.
    window_length : int
        Length of the filter window (must be odd and > polyorder).
    polyorder : int
        Polynomial order for the local fit.
    """
    if window_length % 2 == 0:
        window_length += 1  # enforce odd
    if window_length > len(signal):
        window_length = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
    return savgol_filter(signal, window_length, polyorder)


def moving_average(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving-average smoothing."""
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")


# ---------------------------------------------------------------------------
# Derivative computation
# ---------------------------------------------------------------------------

def compute_derivative(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """First derivative dy/dx using central differences (``np.gradient``)."""
    return np.gradient(y, x)


def compute_second_derivative(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Second derivative d²y/dx² via double ``np.gradient``."""
    dy = np.gradient(y, x)
    return np.gradient(dy, x)


# ---------------------------------------------------------------------------
# Log-current transform
# ---------------------------------------------------------------------------

def log_current(current: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute log10(|I|), with a small epsilon to avoid log(0)."""
    return np.log10(np.abs(current) + eps)


# ---------------------------------------------------------------------------
# Windowing / sequence creation (for LSTM)
# ---------------------------------------------------------------------------

def create_windows(potential: np.ndarray,
                   current: np.ndarray,
                   window_size: int = 50,
                   stride: int = 1,
                   target_index: int | None = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Slide a fixed-size window over the V/I curve.

    Parameters
    ----------
    potential : np.ndarray  (N,)
    current   : np.ndarray  (N,)
    window_size : int
        Number of time-steps per window.
    stride : int
        Step between consecutive windows.
    target_index : int | None
        If given, each window's label is 1 when the window *contains*
        the onset index, else 0.  If None, labels are NaN.

    Returns
    -------
    X : np.ndarray  (n_windows, window_size, 2)
        Stacked [V, I] features.
    y : np.ndarray  (n_windows,)
        Binary labels (or NaN).
    """
    N = len(potential)
    assert N == len(current), "V and I must have the same length"

    starts = list(range(0, N - window_size + 1, stride))
    X = np.empty((len(starts), window_size, 2), dtype=np.float64)
    y = np.full(len(starts), np.nan, dtype=np.float64)

    for i, s in enumerate(starts):
        X[i, :, 0] = potential[s: s + window_size]
        X[i, :, 1] = current[s: s + window_size]
        if target_index is not None:
            y[i] = 1.0 if s <= target_index < s + window_size else 0.0

    return X, y


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_sample(sample: dict,
                      smooth: bool = True,
                      normalize: bool = True,
                      compute_deriv: bool = True,
                      ) -> dict:
    """Apply the full preprocessing chain to a single sample *in place*.

    Adds keys:
        ``current_smoothed``, ``current_norm``, ``potential_norm``,
        ``dI_dV``, ``d2I_dV2``, ``log_I``

    Parameters
    ----------
    sample : dict
        Must contain ``potential_V`` and ``current_A``.
    """
    V = sample["potential_V"].copy()
    I = sample["current_A"].copy()

    if smooth:
        I = smooth_savgol(I)
        sample["current_smoothed"] = I.copy()

    if normalize:
        V_norm, V_params = min_max_normalize(V)
        I_norm, I_params = min_max_normalize(I)
        sample["potential_norm"] = V_norm
        sample["current_norm"] = I_norm
        sample["norm_params"] = {"V": V_params, "I": I_params}

    if compute_deriv:
        sample["dI_dV"] = compute_derivative(I, V)
        sample["d2I_dV2"] = compute_second_derivative(I, V)

    sample["log_I"] = log_current(I)

    return sample


def preprocess_all(samples: List[dict], **kwargs) -> List[dict]:
    """Apply ``preprocess_sample`` to every sample in a list."""
    return [preprocess_sample(s, **kwargs) for s in samples]


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick sanity check with random data
    V = np.linspace(-0.5, 1.5, 500)
    I = np.random.randn(500) * 1e-5
    sample = {
        "sample_id": "test",
        "potential_V": V,
        "current_A": I,
        "pitting_onset_potential_V": None,
        "pitting_onset_index": None,
    }
    out = preprocess_sample(sample)
    print("✅ Preprocessing test passed")
    for key in ["current_smoothed", "potential_norm", "current_norm",
                "dI_dV", "d2I_dV2", "log_I"]:
        print(f"   {key:25s}  shape={out[key].shape}")
