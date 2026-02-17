"""
data_loader.py
==============
Utilities for loading polarization-curve data from various file formats
(CSV, JSON, NumPy, Excel) and converting them into the canonical sample
schema defined in ``dataset_template.py``.
"""

import json
import os
import glob
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.dataset_template import validate_sample


# ---------------------------------------------------------------------------
# Single-file loaders
# ---------------------------------------------------------------------------

def load_csv(filepath: str,
             potential_col: str = "potential_V",
             current_col: str = "current_A",
             **kwargs) -> dict:
    """Load a single polarization curve from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    potential_col : str
        Column name for potential values.
    current_col : str
        Column name for current values.

    Returns
    -------
    dict
        Sample dictionary conforming to the canonical schema.
    """
    df = pd.read_csv(filepath, **kwargs)
    if potential_col not in df.columns or current_col not in df.columns:
        raise KeyError(
            f"CSV must contain '{potential_col}' and '{current_col}' columns. "
            f"Found: {list(df.columns)}"
        )

    sample = {
        "sample_id": Path(filepath).stem,
        "potential_V": df[potential_col].to_numpy(dtype=np.float64),
        "current_A": df[current_col].to_numpy(dtype=np.float64),
        "pitting_onset_potential_V": None,
        "pitting_onset_index": None,
        "material": "",
        "electrolyte": "",
        "scan_rate_mV_s": 0.0,
        "metadata": {"source_file": filepath},
    }

    # If the CSV contains label columns, pull them in
    if "pitting_onset_potential_V" in df.columns:
        sample["pitting_onset_potential_V"] = float(
            df["pitting_onset_potential_V"].iloc[0]
        )
    if "pitting_onset_index" in df.columns:
        sample["pitting_onset_index"] = int(
            df["pitting_onset_index"].iloc[0]
        )

    return sample


def load_json(filepath: str) -> dict:
    """Load a single polarization curve from a JSON file.

    Expected JSON structure::

        {
            "sample_id": "...",
            "potential_V": [...],
            "current_A": [...],
            "pitting_onset_potential_V": 0.35,
            "pitting_onset_index": 212,
            ...
        }
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    data["potential_V"] = np.asarray(data["potential_V"], dtype=np.float64)
    data["current_A"] = np.asarray(data["current_A"], dtype=np.float64)

    # Fill optional keys with defaults
    data.setdefault("pitting_onset_potential_V", None)
    data.setdefault("pitting_onset_index", None)
    data.setdefault("material", "")
    data.setdefault("electrolyte", "")
    data.setdefault("scan_rate_mV_s", 0.0)
    data.setdefault("metadata", {"source_file": filepath})

    return data


def load_numpy(filepath: str) -> dict:
    """Load a polarization curve stored as a ``.npz`` archive.

    Expected keys inside the archive: ``potential_V``, ``current_A``.
    Optional keys: ``pitting_onset_potential_V``, ``pitting_onset_index``.
    """
    npz = np.load(filepath, allow_pickle=True)
    sample = {
        "sample_id": Path(filepath).stem,
        "potential_V": npz["potential_V"].astype(np.float64),
        "current_A": npz["current_A"].astype(np.float64),
        "pitting_onset_potential_V": (
            float(npz["pitting_onset_potential_V"])
            if "pitting_onset_potential_V" in npz
            else None
        ),
        "pitting_onset_index": (
            int(npz["pitting_onset_index"])
            if "pitting_onset_index" in npz
            else None
        ),
        "material": "",
        "electrolyte": "",
        "scan_rate_mV_s": 0.0,
        "metadata": {"source_file": filepath},
    }
    return sample


# ---------------------------------------------------------------------------
# Batch / directory loaders
# ---------------------------------------------------------------------------

def load_directory(directory: str,
                   file_ext: str = "csv",
                   **kwargs) -> List[dict]:
    """Load all polarization curves from a directory.

    Parameters
    ----------
    directory : str
        Path to the folder containing data files.
    file_ext : str
        File extension to look for (``csv``, ``json``, or ``npz``).

    Returns
    -------
    list[dict]
        List of sample dictionaries.
    """
    loaders = {
        "csv": load_csv,
        "json": load_json,
        "npz": load_numpy,
    }
    if file_ext not in loaders:
        raise ValueError(f"Unsupported extension '{file_ext}'. Use one of {list(loaders)}")

    pattern = os.path.join(directory, f"*.{file_ext}")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"⚠️  No .{file_ext} files found in {directory}")
        return []

    samples = []
    for fp in files:
        try:
            sample = loaders[file_ext](fp, **kwargs)
            validate_sample(sample)
            samples.append(sample)
        except Exception as e:
            print(f"⚠️  Skipping {fp}: {e}")

    print(f"✅ Loaded {len(samples)} samples from {directory}")
    return samples


def samples_to_dataframe(samples: List[dict]) -> pd.DataFrame:
    """Convert a list of sample dicts into a summary DataFrame.

    Each row contains scalar metadata; the V/I arrays are stored as objects.
    """
    rows = []
    for s in samples:
        rows.append({
            "sample_id": s["sample_id"],
            "n_points": len(s["potential_V"]),
            "V_min": s["potential_V"].min(),
            "V_max": s["potential_V"].max(),
            "I_min": s["current_A"].min(),
            "I_max": s["current_A"].max(),
            "pitting_onset_V": s["pitting_onset_potential_V"],
            "pitting_onset_idx": s["pitting_onset_index"],
            "material": s.get("material", ""),
            "electrolyte": s.get("electrolyte", ""),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.data_loader <directory> [csv|json|npz]")
        sys.exit(1)

    directory = sys.argv[1]
    ext = sys.argv[2] if len(sys.argv) > 2 else "csv"
    samples = load_directory(directory, file_ext=ext)
    if samples:
        df = samples_to_dataframe(samples)
        print(df.to_string(index=False))
