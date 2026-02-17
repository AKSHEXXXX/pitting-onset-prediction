"""
serration_preprocessing.py
==========================
Preprocessing utilities for nanoindentation serration curves:
  • Smoothing and noise reduction
  • Elastic/plastic region segmentation
  • Burst (serration) detection via derivative thresholding
  • Displacement-rate computation
"""

import numpy as np
from scipy.signal import savgol_filter, find_peaks
from typing import Tuple, List, Optional, Dict

from src.preprocessing import min_max_normalize, standard_normalize


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smooth_displacement(displacement: np.ndarray,
                        window_length: int = 21,
                        polyorder: int = 3) -> np.ndarray:
    """Apply Savitzky-Golay smoothing to a displacement signal."""
    if window_length % 2 == 0:
        window_length += 1
    if window_length > len(displacement):
        window_length = len(displacement) if len(displacement) % 2 == 1 else len(displacement) - 1
    return savgol_filter(displacement, window_length, polyorder)


# ---------------------------------------------------------------------------
# Displacement rate (dh/dP or dh/dt proxy)
# ---------------------------------------------------------------------------

def compute_displacement_rate(displacement: np.ndarray,
                              load: np.ndarray) -> np.ndarray:
    """Compute dh/dP (displacement rate w.r.t. load) using central differences."""
    return np.gradient(displacement, load)


def compute_displacement_acceleration(displacement: np.ndarray,
                                      load: np.ndarray) -> np.ndarray:
    """Compute d²h/dP² (second derivative of displacement w.r.t. load)."""
    dh = np.gradient(displacement, load)
    return np.gradient(dh, load)


# ---------------------------------------------------------------------------
# Burst / serration detection
# ---------------------------------------------------------------------------

def detect_bursts(
    load: np.ndarray,
    displacement: np.ndarray,
    smooth_window: int = 21,
    threshold_factor: float = 3.0,
    min_burst_separation: int = 10,
    min_amplitude_nm: float = 1.0,
) -> List[Dict]:
    """Detect serration bursts from a load-displacement curve.

    Strategy:
    1. Smooth the displacement signal.
    2. Compute the displacement rate dh/dP.
    3. Identify peaks in dh/dP that exceed median + factor × MAD.
    4. For each peak, measure the burst amplitude from the
       displacement step around the peak.

    Parameters
    ----------
    load, displacement : np.ndarray
        Load (mN) and displacement (nm) arrays.
    smooth_window : int
        Savitzky-Golay window for initial smoothing.
    threshold_factor : float
        Threshold = median(dh/dP) + factor × MAD(dh/dP).
    min_burst_separation : int
        Minimum number of points between consecutive burst detections.
    min_amplitude_nm : float
        Minimum burst amplitude to retain (nm).

    Returns
    -------
    list[dict]
        Each dict: {index, load_mN, amplitude_nm, dh_dP_peak}.
    """
    h_smooth = smooth_displacement(displacement, window_length=smooth_window)
    dh_dP = compute_displacement_rate(h_smooth, load)

    # Robust threshold from the first 30% (mostly elastic)
    n_elastic = max(int(len(load) * 0.3), 20)
    elastic_dh = dh_dP[:n_elastic]
    med = np.median(elastic_dh)
    mad = np.median(np.abs(elastic_dh - med)) + 1e-30
    threshold = med + threshold_factor * mad

    # Find peaks in displacement rate above threshold
    peaks, properties = find_peaks(
        dh_dP,
        height=threshold,
        distance=min_burst_separation,
    )

    bursts: List[Dict] = []
    for pk in peaks:
        # Measure amplitude: displacement jump around the peak
        lo = max(0, pk - 5)
        hi = min(len(displacement) - 1, pk + 5)
        amp = float(displacement[hi] - displacement[lo])

        if amp < min_amplitude_nm:
            continue

        bursts.append({
            "index": int(pk),
            "load_mN": float(load[pk]),
            "amplitude_nm": abs(amp),
            "dh_dP_peak": float(dh_dP[pk]),
        })

    return bursts


# ---------------------------------------------------------------------------
# Preprocess a single serration sample
# ---------------------------------------------------------------------------

def preprocess_serration_sample(
    sample: dict,
    smooth: bool = True,
    detect: bool = True,
    normalize: bool = True,
) -> dict:
    """Apply the full preprocessing pipeline to a serration sample in-place.

    Adds keys:
        displacement_smoothed, dh_dP, d2h_dP2,
        load_norm, displacement_norm,
        detected_bursts (if detect=True)
    """
    load = sample["load_mN"].copy()
    disp = sample["displacement_nm"].copy()

    if smooth:
        disp_smooth = smooth_displacement(disp)
        sample["displacement_smoothed"] = disp_smooth
    else:
        disp_smooth = disp

    # Derivatives
    sample["dh_dP"] = compute_displacement_rate(disp_smooth, load)
    sample["d2h_dP2"] = compute_displacement_acceleration(disp_smooth, load)

    if normalize:
        load_norm, load_params = min_max_normalize(load)
        disp_norm, disp_params = min_max_normalize(disp_smooth)
        sample["load_norm"] = load_norm
        sample["displacement_norm"] = disp_norm
        sample["serration_norm_params"] = {"load": load_params, "disp": disp_params}

    if detect:
        sample["detected_bursts"] = detect_bursts(load, disp)

    return sample


def preprocess_serration_all(samples: List[dict], **kwargs) -> List[dict]:
    """Apply ``preprocess_serration_sample`` to every sample."""
    return [preprocess_serration_sample(s, **kwargs) for s in samples]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.synthetic_serration_data import generate_serration_dataset

    samples = generate_serration_dataset(n_samples=5, seed=0)
    samples = preprocess_serration_all(samples)
    for s in samples:
        n_det = len(s.get("detected_bursts", []))
        n_true = s["n_bursts"]
        print(f"  {s['sample_id']}  true_bursts={n_true:3d}  detected={n_det:3d}")
