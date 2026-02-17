"""
alignment.py
============
Alignment module for pairing corrosion (polarization) samples with
serration (nanoindentation) samples by material identifier.

In real experiments, each specimen yields both a polarization curve and
a nanoindentation curve.  This module:

1. Pairs samples by ``material`` field (or ``sample_id`` prefix).
2. Merges corrosion-side features (onset potential, LSTM predictions)
   with serration-side features (burst statistics, instability score).
3. Computes correlation statistics between serration activity and
   early corrosion behaviour.
4. Produces a joint feature table ready for the Phase 3 fusion model.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from scipy.stats import pearsonr, spearmanr

from src.baseline_model import extract_features as extract_corrosion_features
from src.serration_features import extract_serration_features
from src.serration_model import compute_instability_score


# ---------------------------------------------------------------------------
# Pairing strategies
# ---------------------------------------------------------------------------

def pair_by_material(
    corrosion_samples: List[dict],
    serration_samples: List[dict],
) -> List[Tuple[dict, dict]]:
    """Pair corrosion and serration samples that share the same material.

    For synthetic data where there is no 1-to-1 specimen mapping, we
    pair by material type (many-to-many, then zip within each material).

    Returns
    -------
    list of (corrosion_sample, serration_sample) tuples.
    """
    from collections import defaultdict

    corr_by_mat: Dict[str, List[dict]] = defaultdict(list)
    serr_by_mat: Dict[str, List[dict]] = defaultdict(list)

    for s in corrosion_samples:
        corr_by_mat[s.get("material", "unknown")].append(s)
    for s in serration_samples:
        serr_by_mat[s.get("material", "unknown")].append(s)

    pairs = []
    for mat in corr_by_mat:
        c_list = corr_by_mat[mat]
        s_list = serr_by_mat.get(mat, [])
        n = min(len(c_list), len(s_list))
        for i in range(n):
            pairs.append((c_list[i], s_list[i]))

    return pairs


def pair_by_index(
    corrosion_samples: List[dict],
    serration_samples: List[dict],
) -> List[Tuple[dict, dict]]:
    """Simple 1-to-1 index-based pairing (for synthetic benchmarks)."""
    n = min(len(corrosion_samples), len(serration_samples))
    return [(corrosion_samples[i], serration_samples[i]) for i in range(n)]


# ---------------------------------------------------------------------------
# Build aligned feature table
# ---------------------------------------------------------------------------

def build_aligned_features(
    pairs: List[Tuple[dict, dict]],
) -> pd.DataFrame:
    """Build a combined feature table from paired (corrosion, serration) samples.

    Each row contains:
      - Corrosion features (prefixed ``corr_``)
      - Serration features (prefixed ``serr_``)
      - Corrosion label: ``onset_potential_V``
      - Serration label: ``instability_score``

    Returns
    -------
    pd.DataFrame  (n_pairs, n_corr_feats + n_serr_feats + labels)
    """
    rows = []
    for corr, serr in pairs:
        c_feats = extract_corrosion_features(corr)
        s_feats = extract_serration_features(serr)

        row = {}
        # Prefix corrosion features
        for k, v in c_feats.items():
            row[f"corr_{k}"] = v
        # Prefix serration features
        for k, v in s_feats.items():
            row[f"serr_{k}"] = v

        # Labels
        row["onset_potential_V"] = corr.get("pitting_onset_potential_V")
        row["instability_score"] = compute_instability_score(serr)

        # Identifiers
        row["corr_sample_id"] = corr.get("sample_id", "")
        row["serr_sample_id"] = serr.get("sample_id", "")
        row["material"] = corr.get("material", serr.get("material", ""))

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def compute_correlations(
    aligned_df: pd.DataFrame,
    corrosion_target: str = "onset_potential_V",
    serration_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute Pearson & Spearman correlations between serration features
    and a corrosion target variable.

    Parameters
    ----------
    aligned_df : pd.DataFrame
        Output of ``build_aligned_features``.
    corrosion_target : str
        Column name for the corrosion variable to correlate against.
    serration_cols : list[str] | None
        Serration feature columns.  If None, auto-detected via ``serr_`` prefix.

    Returns
    -------
    pd.DataFrame with columns: feature, pearson_r, pearson_p, spearman_r, spearman_p.
    """
    if serration_cols is None:
        serration_cols = [c for c in aligned_df.columns if c.startswith("serr_")]

    # Drop rows where corrosion target is missing
    df = aligned_df.dropna(subset=[corrosion_target])
    target = df[corrosion_target].values

    results = []
    for col in serration_cols:
        vals = df[col].values
        if np.std(vals) < 1e-12 or np.std(target) < 1e-12:
            pr, pp, sr, sp = 0.0, 1.0, 0.0, 1.0
        else:
            pr, pp = pearsonr(vals, target)
            sr, sp = spearmanr(vals, target)
        results.append({
            "feature": col,
            "pearson_r": float(pr),
            "pearson_p": float(pp),
            "spearman_r": float(sr),
            "spearman_p": float(sp),
        })

    corr_df = pd.DataFrame(results)
    corr_df = corr_df.sort_values("pearson_r", key=abs, ascending=False)
    return corr_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_alignment_summary(
    pairs: List[Tuple[dict, dict]],
    aligned_df: pd.DataFrame,
) -> None:
    """Print a summary of the alignment and correlation analysis."""
    print(f"\n{'─' * 60}")
    print(f"  Alignment Summary")
    print(f"{'─' * 60}")
    print(f"  Paired samples          : {len(pairs)}")
    print(f"  Corrosion features      : {sum(1 for c in aligned_df.columns if c.startswith('corr_'))}")
    print(f"  Serration features      : {sum(1 for c in aligned_df.columns if c.startswith('serr_'))}")

    # Materials breakdown
    if "material" in aligned_df.columns:
        mat_counts = aligned_df["material"].value_counts()
        print(f"  Materials represented   : {len(mat_counts)}")
        for mat, cnt in mat_counts.items():
            print(f"    {mat:20s}: {cnt} pairs")

    print(f"{'─' * 60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.synthetic_data import generate_dataset
    from src.synthetic_serration_data import generate_serration_dataset
    from src.preprocessing import preprocess_all
    from src.serration_preprocessing import preprocess_serration_all

    print("Generating corrosion data …")
    corr_samples = generate_dataset(n_samples=60, seed=42)
    corr_samples = preprocess_all(corr_samples)

    print("Generating serration data …")
    serr_samples = generate_serration_dataset(n_samples=60, seed=42)
    serr_samples = preprocess_serration_all(serr_samples)

    print("Pairing by index …")
    pairs = pair_by_index(corr_samples, serr_samples)
    aligned = build_aligned_features(pairs)

    print_alignment_summary(pairs, aligned)

    print("Correlation analysis (serration vs onset potential):")
    corr_table = compute_correlations(aligned)
    print(corr_table.to_string(index=False))
