"""
synthetic_data.py
=================
Generate synthetic (dummy) polarization curves that mimic real
electrochemical corrosion data for end-to-end pipeline testing.

Curve anatomy
-------------
1. **Cathodic region** (V < E_corr):  exponential decay of current.
2. **Passive region** (E_corr < V < E_pit):  near-constant low current.
3. **Pitting region** (V > E_pit):  rapid exponential current rise.

Each region is governed by simple parametric models with optional
Gaussian noise.  The *pitting onset index* is recorded as the ground-
truth label.
"""

import os
import json
import numpy as np
from typing import Tuple, List

from src.dataset_template import validate_sample


# ---------------------------------------------------------------------------
# Single curve generator
# ---------------------------------------------------------------------------

def generate_single_curve(
    n_points: int = 500,
    V_start: float = -0.8,
    V_end: float = 1.5,
    E_corr: float | None = None,
    E_pit: float | None = None,
    I_passive: float = 1e-6,
    noise_level: float = 5e-7,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Generate one synthetic polarization curve.

    Parameters
    ----------
    n_points : int
        Number of data points.
    V_start, V_end : float
        Potential sweep range (V).
    E_corr : float | None
        Corrosion potential.  Randomised if None.
    E_pit : float | None
        Pitting onset potential.  Randomised if None.
    I_passive : float
        Baseline passive current (A/cm²).
    noise_level : float
        Gaussian noise standard deviation.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    V : np.ndarray (n_points,)
    I : np.ndarray (n_points,)
    onset_index : int
        Ground-truth pitting onset index.
    """
    rng = np.random.default_rng(seed)

    # Random corrosion / pitting potentials if not specified
    if E_corr is None:
        E_corr = rng.uniform(-0.4, -0.1)
    if E_pit is None:
        E_pit = rng.uniform(E_corr + 0.3, E_corr + 0.8)

    V = np.linspace(V_start, V_end, n_points)
    I = np.zeros_like(V)

    for i, v in enumerate(V):
        if v < E_corr:
            # Cathodic region – exponential decay
            I[i] = -I_passive * np.exp(-8.0 * (v - E_corr))
        elif v < E_pit:
            # Passive region – low constant current + small slope
            I[i] = I_passive * (1.0 + 0.3 * (v - E_corr))
        else:
            # Pitting region – rapid rise
            I[i] = I_passive * np.exp(6.0 * (v - E_pit)) * (1.0 + 0.3 * (E_pit - E_corr))

    # Add noise
    I += rng.normal(0, noise_level, size=n_points)

    # Determine onset index (first V >= E_pit)
    onset_index = int(np.searchsorted(V, E_pit))
    onset_index = min(onset_index, n_points - 1)

    return V, I, onset_index


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_dataset(
    n_samples: int = 50,
    n_points: int = 500,
    noise_range: Tuple[float, float] = (1e-7, 1e-6),
    seed: int = 42,
    include_no_pitting: float = 0.1,
) -> List[dict]:
    """Generate a dataset of synthetic polarization curves.

    Parameters
    ----------
    n_samples : int
        Number of curves to generate.
    n_points : int
        Points per curve.
    noise_range : tuple
        (min, max) noise standard deviation; sampled uniformly per curve.
    seed : int
        Base random seed.
    include_no_pitting : float
        Fraction of curves with no pitting (onset beyond V range).

    Returns
    -------
    list[dict]
        List of sample dicts conforming to the canonical schema.
    """
    rng = np.random.default_rng(seed)
    materials = ["SS304", "SS316", "Al7075", "Al2024", "Ti6Al4V"]
    electrolytes = ["3.5% NaCl", "0.5M H2SO4", "1M NaOH", "Seawater"]

    samples: List[dict] = []

    for i in range(n_samples):
        noise = rng.uniform(*noise_range)
        no_pit = rng.random() < include_no_pitting

        # If no pitting, push E_pit beyond the scan range
        E_pit_override = 2.0 if no_pit else None

        V, I, onset_idx = generate_single_curve(
            n_points=n_points,
            noise_level=noise,
            E_pit=E_pit_override,
            seed=seed + i,
        )

        sample = {
            "sample_id": f"SYN-{i:04d}",
            "potential_V": V,
            "current_A": I,
            "pitting_onset_potential_V": (
                None if no_pit else float(V[onset_idx])
            ),
            "pitting_onset_index": None if no_pit else onset_idx,
            "material": rng.choice(materials),
            "electrolyte": rng.choice(electrolytes),
            "scan_rate_mV_s": float(rng.choice([0.5, 1.0, 2.0, 5.0])),
            "metadata": {
                "synthetic": True,
                "noise_level": noise,
                "no_pitting": no_pit,
            },
        }
        validate_sample(sample)
        samples.append(sample)

    print(f"✅ Generated {len(samples)} synthetic polarization curves")
    return samples


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_samples_csv(samples: List[dict], output_dir: str) -> None:
    """Save each sample as an individual CSV in *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    for s in samples:
        fname = os.path.join(output_dir, f"{s['sample_id']}.csv")
        import pandas as pd
        df = pd.DataFrame({
            "potential_V": s["potential_V"],
            "current_A": s["current_A"],
        })
        # Store scalar labels in the first row (for round-trip convenience)
        df["pitting_onset_potential_V"] = s["pitting_onset_potential_V"]
        df["pitting_onset_index"] = s["pitting_onset_index"]
        df.to_csv(fname, index=False)
    print(f"💾 Saved {len(samples)} CSVs to {output_dir}")


def save_samples_json(samples: List[dict], output_dir: str) -> None:
    """Save each sample as a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    for s in samples:
        fname = os.path.join(output_dir, f"{s['sample_id']}.json")
        out = {k: v for k, v in s.items()}
        out["potential_V"] = s["potential_V"].tolist()
        out["current_A"] = s["current_A"].tolist()
        with open(fname, "w") as f:
            json.dump(out, f, indent=2, default=str)
    print(f"💾 Saved {len(samples)} JSONs to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    samples = generate_dataset(n_samples=10, seed=0)
    save_samples_csv(samples, "data_raw/synthetic")
    for s in samples[:3]:
        tag = "⚡ pitting" if s["pitting_onset_index"] is not None else "  passive"
        print(
            f"  {s['sample_id']}  {tag}  "
            f"onset_V={s['pitting_onset_potential_V']}"
        )
