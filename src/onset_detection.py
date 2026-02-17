"""
onset_detection.py
==================
Heuristic pitting-onset detection from polarization curves.

Strategy
--------
1.  Smooth the current signal (Savitzky-Golay).
2.  Compute the first derivative dI/dV.
3.  Identify the region where dI/dV exceeds a threshold **after** the
    passive region, signalling a sudden current rise.
4.  Refine the detected index by looking for the inflection point
    (peak in d²I/dV²) in a neighbourhood around the first detection.
"""

import numpy as np
from typing import Tuple, Optional, List

from src.preprocessing import (
    smooth_savgol,
    compute_derivative,
    compute_second_derivative,
)


# ---------------------------------------------------------------------------
# Core detection function
# ---------------------------------------------------------------------------

def detect_pitting_onset(
    potential: np.ndarray,
    current: np.ndarray,
    smooth_window: int = 15,
    derivative_threshold_factor: float = 3.0,
    min_passive_frac: float = 0.2,
    refine_window: int = 20,
) -> Tuple[Optional[int], Optional[float], dict]:
    """Detect the pitting onset index and potential.

    Parameters
    ----------
    potential : np.ndarray (N,)
        Potential array (V vs. reference).
    current : np.ndarray (N,)
        Current array (A or A/cm²).
    smooth_window : int
        Savitzky-Golay window for initial smoothing.
    derivative_threshold_factor : float
        Onset is flagged where dI/dV > median + factor * MAD
        (median absolute deviation) of the passive region.
    min_passive_frac : float
        Fraction of the curve assumed to be passive (from the start).
    refine_window : int
        Half-window for inflection-point refinement.

    Returns
    -------
    onset_index : int | None
        Index in the original arrays.  None if no pitting detected.
    onset_potential : float | None
        Potential at the onset point.
    info : dict
        Diagnostic data (``dI_dV``, ``threshold``, ``smoothed_I``, …).
    """
    N = len(potential)
    assert N == len(current), "V and I must have the same length"

    # Step 1 – smooth
    I_smooth = smooth_savgol(current, window_length=smooth_window)

    # Step 2 – first derivative
    dI_dV = compute_derivative(I_smooth, potential)

    # Step 3 – threshold from passive region
    passive_end = max(int(N * min_passive_frac), 10)
    passive_dI = dI_dV[:passive_end]
    med = np.median(passive_dI)
    mad = np.median(np.abs(passive_dI - med)) + 1e-30  # robust scale
    threshold = med + derivative_threshold_factor * mad

    # Search after passive region
    onset_index = None
    for i in range(passive_end, N):
        if dI_dV[i] > threshold:
            onset_index = i
            break

    # Step 4 – refine via d²I/dV² peak
    if onset_index is not None:
        lo = max(0, onset_index - refine_window)
        hi = min(N, onset_index + refine_window)
        d2I = compute_second_derivative(I_smooth[lo:hi], potential[lo:hi])
        refined_local = int(np.argmax(d2I))
        onset_index = lo + refined_local

    onset_potential = float(potential[onset_index]) if onset_index is not None else None

    info = {
        "smoothed_I": I_smooth,
        "dI_dV": dI_dV,
        "threshold": threshold,
        "passive_end_index": passive_end,
    }

    return onset_index, onset_potential, info


# ---------------------------------------------------------------------------
# Batch detection
# ---------------------------------------------------------------------------

def detect_all(samples: List[dict], **kwargs) -> List[dict]:
    """Run onset detection on a list of samples.

    Adds / overwrites ``pitting_onset_potential_V``, ``pitting_onset_index``,
    and ``onset_info`` on each sample dict.
    """
    for s in samples:
        idx, pot, info = detect_pitting_onset(
            s["potential_V"], s["current_A"], **kwargs
        )
        s["pitting_onset_index"] = idx
        s["pitting_onset_potential_V"] = pot
        s["onset_info"] = info
    return samples


# ---------------------------------------------------------------------------
# Alternative simple method (fallback)
# ---------------------------------------------------------------------------

def detect_onset_simple(
    potential: np.ndarray,
    current: np.ndarray,
    current_jump_factor: float = 5.0,
) -> Tuple[Optional[int], Optional[float]]:
    """Simple fallback: flag the first point where current exceeds
    ``current_jump_factor × median(|I|)`` in the anodic region.
    """
    abs_I = np.abs(current)
    threshold = current_jump_factor * np.median(abs_I)
    # search anodic half only
    mid = len(potential) // 2
    for i in range(mid, len(potential)):
        if abs_I[i] > threshold:
            return i, float(potential[i])
    return None, None


# ---------------------------------------------------------------------------
# CLI sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.synthetic_data import generate_single_curve

    V, I, true_idx = generate_single_curve(n_points=500, noise_level=1e-6)
    det_idx, det_V, info = detect_pitting_onset(V, I)

    print(f"True onset index : {true_idx}  (V = {V[true_idx]:.4f})")
    if det_idx is not None:
        print(f"Detected index   : {det_idx}  (V = {det_V:.4f})")
        print(f"Error (indices)  : {abs(det_idx - true_idx)}")
    else:
        print("⚠️  No pitting onset detected")
