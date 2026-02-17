"""
synthetic_serration_data.py
===========================
Generate synthetic nanoindentation load-displacement curves that exhibit
serrated flow (shear-band bursts) typical of metallic glasses and
nanocrystalline alloys.

Curve anatomy
-------------
1. **Elastic region** — linear load increase (Hertzian-like).
2. **Plastic region** — smooth elasto-plastic flow with superimposed
   discrete displacement bursts ("pop-ins" / serrations).
3. Each serration is modelled as a rapid step increase in displacement
   at roughly constant load, governed by a stochastic point process.

The burst catalogue (amplitude, duration, inter-event time) is recorded
as ground-truth labels and can be used for feature extraction validation.
"""

import os
import json
import numpy as np
from typing import Tuple, List, Optional


# ---------------------------------------------------------------------------
# Single serration curve generator
# ---------------------------------------------------------------------------

def generate_single_serration_curve(
    n_points: int = 2000,
    max_load_mN: float = 50.0,
    elastic_modulus: float = 200.0,
    yield_load_frac: float = 0.15,
    burst_rate: float = 0.03,
    burst_amp_mean: float = 5.0,
    burst_amp_std: float = 2.5,
    noise_level: float = 0.3,
    seed: Optional[int] = None,
) -> dict:
    """Generate one synthetic nanoindentation curve with serrations.

    Parameters
    ----------
    n_points : int
        Number of data points in the load-displacement curve.
    max_load_mN : float
        Maximum applied load (mN).
    elastic_modulus : float
        Effective modulus for the elastic region (arbitrary units).
    yield_load_frac : float
        Fraction of max_load at which plastic yielding (and serrations) begin.
    burst_rate : float
        Probability per point of a burst event occurring in the plastic region.
    burst_amp_mean, burst_amp_std : float
        Log-normal parameters for burst displacement amplitudes (nm).
    noise_level : float
        Gaussian noise on displacement (nm).
    seed : int | None
        Random seed.

    Returns
    -------
    dict with keys:
        load_mN, displacement_nm, bursts (list of dicts),
        n_bursts, material, metadata
    """
    rng = np.random.default_rng(seed)

    load = np.linspace(0, max_load_mN, n_points)
    yield_load = max_load_mN * yield_load_frac
    yield_idx = int(n_points * yield_load_frac)

    displacement = np.zeros(n_points)
    bursts: List[dict] = []

    # Elastic region: displacement ~ load / E  (simplified Hertzian)
    for i in range(yield_idx):
        displacement[i] = load[i] / elastic_modulus * 1000  # nm

    # Plastic region: smooth plastic flow + stochastic bursts
    base_disp = displacement[yield_idx - 1] if yield_idx > 0 else 0.0
    for i in range(yield_idx, n_points):
        # Smooth plastic flow (power-law hardening)
        smooth_inc = (load[i] - yield_load) / elastic_modulus * 500
        smooth_inc += 0.05 * ((i - yield_idx) / (n_points - yield_idx)) ** 0.5 * 1000
        displacement[i] = base_disp + smooth_inc

        # Stochastic burst
        if rng.random() < burst_rate:
            amp = max(0.5, rng.lognormal(
                np.log(burst_amp_mean), burst_amp_std / burst_amp_mean
            ))
            displacement[i:] += amp
            bursts.append({
                "index": i,
                "load_mN": float(load[i]),
                "amplitude_nm": float(amp),
                "displacement_at_burst_nm": float(displacement[i]),
            })

    # Add measurement noise
    displacement += rng.normal(0, noise_level, size=n_points)
    displacement = np.maximum(displacement, 0)  # physical constraint

    return {
        "load_mN": load,
        "displacement_nm": displacement,
        "bursts": bursts,
        "n_bursts": len(bursts),
        "yield_load_mN": float(yield_load),
        "yield_index": yield_idx,
        "max_load_mN": float(max_load_mN),
    }


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_serration_dataset(
    n_samples: int = 50,
    n_points: int = 2000,
    seed: int = 42,
    burst_rate_range: Tuple[float, float] = (0.01, 0.06),
    burst_amp_range: Tuple[float, float] = (2.0, 10.0),
) -> List[dict]:
    """Generate a dataset of synthetic nanoindentation serration curves.

    Parameters
    ----------
    n_samples : int
        Number of curves.
    n_points : int
        Points per curve.
    seed : int
        Base random seed.
    burst_rate_range : tuple
        (min, max) burst probability per point.
    burst_amp_range : tuple
        (min, max) mean burst amplitude (nm).

    Returns
    -------
    list[dict]  — each dict contains load/displacement arrays, burst catalogue,
                  and metadata.
    """
    rng = np.random.default_rng(seed)
    materials = ["BMG-Zr", "BMG-Cu", "BMG-Pd", "NC-Al", "NC-Ni", "HEA-CoCrNi"]

    samples: List[dict] = []
    for i in range(n_samples):
        burst_rate = rng.uniform(*burst_rate_range)
        burst_amp = rng.uniform(*burst_amp_range)
        max_load = rng.uniform(30.0, 80.0)
        noise = rng.uniform(0.1, 0.5)

        curve = generate_single_serration_curve(
            n_points=n_points,
            max_load_mN=max_load,
            burst_rate=burst_rate,
            burst_amp_mean=burst_amp,
            noise_level=noise,
            seed=seed + i,
        )

        material = rng.choice(materials)

        sample = {
            "sample_id": f"SER-{i:04d}",
            "load_mN": curve["load_mN"],
            "displacement_nm": curve["displacement_nm"],
            "bursts": curve["bursts"],
            "n_bursts": curve["n_bursts"],
            "yield_load_mN": curve["yield_load_mN"],
            "yield_index": curve["yield_index"],
            "max_load_mN": curve["max_load_mN"],
            "material": material,
            "metadata": {
                "synthetic": True,
                "burst_rate": burst_rate,
                "burst_amp_mean": burst_amp,
                "noise_level": noise,
            },
        }
        samples.append(sample)

    print(f"✅ Generated {len(samples)} synthetic serration curves")
    return samples


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_serration_csv(samples: List[dict], output_dir: str) -> None:
    """Save each serration sample as a CSV."""
    import pandas as pd
    os.makedirs(output_dir, exist_ok=True)
    for s in samples:
        fname = os.path.join(output_dir, f"{s['sample_id']}.csv")
        df = pd.DataFrame({
            "load_mN": s["load_mN"],
            "displacement_nm": s["displacement_nm"],
        })
        df.to_csv(fname, index=False)
    print(f"💾 Saved {len(samples)} serration CSVs to {output_dir}")


def save_serration_json(samples: List[dict], output_dir: str) -> None:
    """Save each serration sample as a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    for s in samples:
        fname = os.path.join(output_dir, f"{s['sample_id']}.json")
        out = {k: v for k, v in s.items()}
        out["load_mN"] = s["load_mN"].tolist()
        out["displacement_nm"] = s["displacement_nm"].tolist()
        with open(fname, "w") as f:
            json.dump(out, f, indent=2, default=str)
    print(f"💾 Saved {len(samples)} serration JSONs to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    samples = generate_serration_dataset(n_samples=10, seed=0)
    save_serration_csv(samples, "data_raw/serration_synthetic")
    for s in samples[:5]:
        print(f"  {s['sample_id']}  bursts={s['n_bursts']:3d}  "
              f"material={s['material']}")
