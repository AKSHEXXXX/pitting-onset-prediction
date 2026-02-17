"""
dataset_template.py
====================
Defines the canonical data schema for polarization curve samples.
Each sample is a dictionary containing potential (V), current (I),
and a label for pitting onset.

This template can be used to validate incoming data or to generate
placeholder structures before the real dataset arrives.
"""

import numpy as np
from typing import TypedDict, Optional


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------
class PolarizationSample(TypedDict):
    """Schema for a single polarization-curve sample.

    Attributes
    ----------
    sample_id : str
        Unique identifier for the sample.
    potential_V : np.ndarray
        1-D array of applied potential values (volts vs. reference).
    current_A : np.ndarray
        1-D array of measured current values (amperes or A/cm²).
    pitting_onset_potential_V : float | None
        The potential value at which pitting onset occurs.
        None if the curve does not exhibit pitting.
    pitting_onset_index : int | None
        Index into *potential_V* / *current_A* arrays corresponding
        to the pitting onset point.  None if no pitting.
    material : str
        Material identifier (e.g. "SS304", "Al7075").
    electrolyte : str
        Electrolyte description (e.g. "3.5% NaCl").
    scan_rate_mV_s : float
        Scan rate in mV/s used during the experiment.
    metadata : dict
        Any additional experiment metadata.
    """
    sample_id: str
    potential_V: np.ndarray
    current_A: np.ndarray
    pitting_onset_potential_V: Optional[float]
    pitting_onset_index: Optional[int]
    material: str
    electrolyte: str
    scan_rate_mV_s: float
    metadata: dict


def create_empty_sample() -> dict:
    """Return a blank sample that conforms to the schema."""
    return {
        "sample_id": "",
        "potential_V": np.array([], dtype=np.float64),
        "current_A": np.array([], dtype=np.float64),
        "pitting_onset_potential_V": None,
        "pitting_onset_index": None,
        "material": "",
        "electrolyte": "",
        "scan_rate_mV_s": 0.0,
        "metadata": {},
    }


def validate_sample(sample: dict) -> bool:
    """Check that *sample* conforms to the expected schema.

    Returns True if valid, raises ValueError otherwise.
    """
    required_keys = {
        "sample_id", "potential_V", "current_A",
        "pitting_onset_potential_V", "pitting_onset_index",
    }
    missing = required_keys - set(sample.keys())
    if missing:
        raise ValueError(f"Sample is missing keys: {missing}")

    if len(sample["potential_V"]) != len(sample["current_A"]):
        raise ValueError(
            "potential_V and current_A must have the same length "
            f"({len(sample['potential_V'])} vs {len(sample['current_A'])})"
        )

    if len(sample["potential_V"]) == 0:
        raise ValueError("potential_V / current_A arrays must not be empty")

    return True


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = create_empty_sample()
    sample["sample_id"] = "DEMO-001"
    sample["potential_V"] = np.linspace(-0.5, 1.5, 500)
    sample["current_A"] = np.random.randn(500) * 1e-6
    sample["pitting_onset_potential_V"] = 0.35
    sample["pitting_onset_index"] = 212

    validate_sample(sample)
    print("✅ Sample is valid")
    print(f"   ID           : {sample['sample_id']}")
    print(f"   Points       : {len(sample['potential_V'])}")
    print(f"   Onset at V   : {sample['pitting_onset_potential_V']}")
    print(f"   Onset index  : {sample['pitting_onset_index']}")
