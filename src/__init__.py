"""
Pitting Onset Prediction Pipeline
==================================
ML pipeline for predicting pitting corrosion onset from polarization curve data.

Modules:
    - data_loader:      Load polarization curves from CSV / JSON / raw files
    - preprocessing:    Normalise, smooth, compute derivatives, create windows
    - onset_detection:  Heuristic pitting-onset detection (sudden current rise)
    - synthetic_data:   Generate dummy corrosion curves for testing
    - baseline_model:   Random Forest regression baseline
    - lstm_model:       LSTM sequence model for onset prediction
    - evaluation:       Metrics and scoring utilities
    - plotting:         Visualisation helpers
"""

__version__ = "0.1.0"
