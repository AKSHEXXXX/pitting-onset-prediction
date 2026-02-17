"""
serration_features.py
=====================
Feature extraction from nanoindentation serration curves.

Extracted features characterise mechanical instability through:
  • Burst statistics: count, amplitude distribution, frequency
  • Displacement-rate statistics: peak, mean, variance
  • Energy proxies: cumulative burst displacement, burst energy fraction
  • Temporal patterns: inter-burst intervals, burst clustering
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional

from src.serration_preprocessing import (
    preprocess_serration_sample,
    detect_bursts,
    compute_displacement_rate,
)


# ---------------------------------------------------------------------------
# Single-sample feature extraction
# ---------------------------------------------------------------------------

def extract_serration_features(sample: dict) -> Dict[str, float]:
    """Extract scalar features from a preprocessed serration sample.

    Parameters
    ----------
    sample : dict
        Must contain ``load_mN``, ``displacement_nm``.
        If ``detected_bursts`` is present it is used directly;
        otherwise bursts are detected on the fly.

    Returns
    -------
    dict of {feature_name: float}
    """
    load = sample["load_mN"]
    disp = sample["displacement_nm"]

    # Get bursts (prefer pre-detected)
    bursts = sample.get("detected_bursts")
    if bursts is None:
        bursts = detect_bursts(load, disp)

    n_points = len(load)

    # --- Burst count & frequency ---
    burst_count = len(bursts)
    # Frequency: bursts per mN of load range
    load_range = float(load.max() - load.min()) if load.max() > load.min() else 1.0
    burst_frequency = burst_count / load_range

    # --- Amplitude statistics ---
    if burst_count > 0:
        amps = np.array([b["amplitude_nm"] for b in bursts])
        amp_mean = float(np.mean(amps))
        amp_std = float(np.std(amps))
        amp_max = float(np.max(amps))
        amp_median = float(np.median(amps))
        amp_total = float(np.sum(amps))
    else:
        amp_mean = amp_std = amp_max = amp_median = amp_total = 0.0

    # --- Inter-burst intervals ---
    if burst_count > 1:
        indices = np.array([b["index"] for b in bursts])
        intervals = np.diff(indices).astype(float)
        ibi_mean = float(np.mean(intervals))
        ibi_std = float(np.std(intervals))
        ibi_min = float(np.min(intervals))
        # Coefficient of variation of intervals (regularity measure)
        ibi_cv = float(ibi_std / (ibi_mean + 1e-12))
    else:
        ibi_mean = ibi_std = ibi_cv = 0.0
        ibi_min = float(n_points)

    # --- Displacement rate statistics ---
    dh_dP = sample.get("dh_dP")
    if dh_dP is None:
        dh_dP = compute_displacement_rate(disp, load)
    dh_max = float(np.max(dh_dP))
    dh_mean = float(np.mean(dh_dP))
    dh_std = float(np.std(dh_dP))
    # Skewness of displacement rate (positive skew → more burst-like)
    dh_skew = float(np.mean(((dh_dP - dh_mean) / (dh_std + 1e-12)) ** 3))
    # Kurtosis (high → heavy-tailed, more extreme bursts)
    dh_kurtosis = float(np.mean(((dh_dP - dh_mean) / (dh_std + 1e-12)) ** 4) - 3.0)

    # --- Energy proxy ---
    # Fraction of total displacement attributable to bursts
    total_disp = float(disp.max() - disp.min()) if disp.max() > disp.min() else 1.0
    burst_disp_fraction = amp_total / total_disp

    # --- Yield-related ---
    yield_load = sample.get("yield_load_mN", load.max() * 0.15)
    max_load = float(load.max())
    hardening_ratio = max_load / (yield_load + 1e-12)

    return {
        # Burst statistics
        "burst_count": float(burst_count),
        "burst_frequency_per_mN": burst_frequency,
        "amp_mean_nm": amp_mean,
        "amp_std_nm": amp_std,
        "amp_max_nm": amp_max,
        "amp_median_nm": amp_median,
        "amp_total_nm": amp_total,
        # Inter-burst intervals
        "ibi_mean": ibi_mean,
        "ibi_std": ibi_std,
        "ibi_min": ibi_min,
        "ibi_cv": ibi_cv,
        # Displacement rate
        "dh_dP_max": dh_max,
        "dh_dP_mean": dh_mean,
        "dh_dP_std": dh_std,
        "dh_dP_skewness": dh_skew,
        "dh_dP_kurtosis": dh_kurtosis,
        # Energy / displacement
        "burst_disp_fraction": burst_disp_fraction,
        "total_displacement_nm": total_disp,
        # Mechanical
        "max_load_mN": max_load,
        "yield_load_mN": float(yield_load),
        "hardening_ratio": hardening_ratio,
    }


# ---------------------------------------------------------------------------
# Batch feature matrix
# ---------------------------------------------------------------------------

def build_serration_feature_matrix(
    samples: List[dict],
) -> Tuple[pd.DataFrame, List[str]]:
    """Build a feature matrix from a list of serration samples.

    Returns
    -------
    X : pd.DataFrame  (n_samples, n_features)
    sample_ids : list[str]
    """
    rows = []
    ids = []
    for s in samples:
        feats = extract_serration_features(s)
        feats["sample_id"] = s["sample_id"]
        rows.append(feats)
        ids.append(s["sample_id"])

    df = pd.DataFrame(rows).set_index("sample_id")
    return df, ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.synthetic_serration_data import generate_serration_dataset
    from src.serration_preprocessing import preprocess_serration_all

    samples = generate_serration_dataset(n_samples=20, seed=42)
    samples = preprocess_serration_all(samples)

    X, ids = build_serration_feature_matrix(samples)
    print(f"Feature matrix: {X.shape}")
    print(f"\nFeature columns:\n  {list(X.columns)}")
    print(f"\nSample statistics:\n{X.describe().T[['mean', 'std', 'min', 'max']]}")
