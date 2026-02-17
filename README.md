# Pitting Onset Prediction — ML Pipeline

> **Predict pitting corrosion onset potential from electrochemical polarization curves** using heuristic detection, Random Forest regression, and LSTM sequence classification.

---

## Project Structure

```
pitting-onset-prediction/
├── data_raw/               # Original polarization curve files (CSV / JSON / NPZ)
├── data_processed/         # Cleaned, windowed sequences ready for models
├── src/                    # Core Python modules
│   ├── __init__.py
│   ├── dataset_template.py # Canonical sample schema (V, I, labels)
│   ├── data_loader.py      # Load CSV / JSON / NPZ polarization data
│   ├── preprocessing.py    # Normalisation, smoothing, derivatives, windowing
│   ├── onset_detection.py  # Heuristic pitting onset detection (dI/dV threshold)
│   ├── synthetic_data.py   # Generate synthetic dummy corrosion curves
│   ├── baseline_model.py   # Random Forest regression baseline
│   ├── lstm_model.py       # LSTM sequence classifier / regressor (PyTorch)
│   ├── evaluation.py       # Metrics (MAE, RMSE, R², F1, confusion matrix, …)
│   └── plotting.py         # Visualisation utilities
├── models/                 # Saved model artefacts (.pkl, .pt)
├── results/                # Prediction outputs & pipeline summary JSON
├── figures/                # Generated plots
├── notebooks/
│   └── 01_experiment.ipynb # Interactive walkthrough notebook
├── run_pipeline.py         # End-to-end CLI pipeline runner
├── requirements.txt        # Python dependencies
└── README.md               # ← You are here
```

---

## Quick Start

### 1. Install dependencies

```bash
cd pitting-onset-prediction
pip install -r requirements.txt
```

### 2. Run the full pipeline (synthetic data)

```bash
python run_pipeline.py
```

This will:
1. Generate 100 synthetic polarization curves
2. Preprocess (smooth, normalise, compute derivatives)
3. Run heuristic onset detection
4. Train a Random Forest baseline
5. Train an LSTM binary classifier
6. Evaluate all models and save results to `results/`
7. Generate plots in `figures/`

### 3. Run with real data

```bash
python run_pipeline.py data_raw/my_experiment_folder/
```

Place your CSV files (with columns `potential_V` and `current_A`) in a subfolder of `data_raw/`.

### 4. Interactive notebook

Open `notebooks/01_experiment.ipynb` in Jupyter / VS Code for a step-by-step walkthrough.

---

## Dataset Schema

Each sample is a dictionary with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `sample_id` | `str` | Unique identifier |
| `potential_V` | `np.ndarray` | Applied potential (V vs. reference) |
| `current_A` | `np.ndarray` | Measured current (A or A/cm²) |
| `pitting_onset_potential_V` | `float \| None` | Onset potential (V) — `None` if no pitting |
| `pitting_onset_index` | `int \| None` | Array index of onset point |
| `material` | `str` | Material ID (e.g. "SS304") |
| `electrolyte` | `str` | Electrolyte description |
| `scan_rate_mV_s` | `float` | Scan rate in mV/s |
| `metadata` | `dict` | Any additional info |

---

## What Has Been Implemented ✅

All of the following work end-to-end on **synthetic dummy data**:

### Data & Preprocessing
- **Synthetic curve generator** — configurable cathodic / passive / pitting regions with noise
- **Multi-format loader** — CSV, JSON, NumPy `.npz`
- **Preprocessing pipeline** — Savitzky-Golay smoothing, min-max & standard normalisation, dI/dV & d²I/dV² derivatives, log-current transform
- **Sliding-window creator** — fixed-size overlapping windows for LSTM input

### Pitting Onset Detection
- **Derivative-threshold heuristic** — identifies sudden current rise via dI/dV exceeding a robust threshold (median + k × MAD), refined by d²I/dV² inflection point
- **Simple fallback method** — absolute current threshold

### Machine Learning Models
- **Random Forest regression** — hand-crafted features (passive slope, dI/dV statistics, log-I stats) → predicts onset potential directly
- **LSTM classifier (PyTorch)** — bidirectional option, configurable layers/hidden size → binary classification (does this window contain onset?)
- **LSTM regression mode** — scaffolded for predicting onset potential from positive windows

### Evaluation & Visualisation
- Regression metrics: MAE, RMSE, R², MAPE
- Classification metrics: accuracy, precision, recall, F1, confusion matrix
- Onset-index error: mean/median/max index error, % within 5/10 indices
- Plots: curves with onset markers, derivative analysis, true-vs-predicted scatter, training history, feature importances, multi-curve overlay

---

## What Remains To Be Done 🔲

The following tasks require the **real experimental dataset**:

### Phase 1 — Data Integration
- [ ] Collect real polarization curve files and place in `data_raw/`
- [ ] Annotate ground-truth pitting onset (expert labels)
- [ ] Validate data format against `dataset_template.py` schema
- [ ] Adjust preprocessing parameters (smoothing window, normalisation) per material

### Phase 2 — Model Training & Tuning
- [ ] Train Random Forest on real features → tune `n_estimators`, `max_depth`
- [ ] Train LSTM on real windowed data → tune `hidden_size`, `num_layers`, `window_size`, `lr`
- [ ] Implement k-fold cross-validation for robust metrics
- [ ] Class balancing (oversampling / weighted loss) for onset windows
- [ ] Hyperparameter search (grid / Bayesian)

### Phase 3 — Advanced Extensions
- [ ] **Shear band feature integration** — add microstructural features (grain size, shear band density, crystallographic orientation) as auxiliary inputs to the LSTM
- [ ] **Fracture risk estimation** — extend the model to output a combined risk score incorporating pitting onset + mechanical properties (yield strength, fracture toughness)
- [ ] **Attention mechanism** — add temporal attention to the LSTM for interpretability (which part of the curve matters most)
- [ ] **Transformer / 1D-CNN alternatives** — benchmark against LSTM
- [ ] **Multi-material model** — single model that generalises across alloy families

### Phase 4 — Deployment
- [ ] ONNX model export
- [ ] REST API for inference (FastAPI / Flask)
- [ ] Batch prediction script for new experimental data
- [ ] Automated report generation (PDF with curves + predictions)

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy`, `pandas` | Data handling |
| `scipy` | Signal processing (Savitzky-Golay) |
| `scikit-learn` | Random Forest, metrics, train/test split |
| `torch` | LSTM model |
| `matplotlib` | Plotting |

Install everything with:
```bash
pip install -r requirements.txt
```

---

## License

This project is licensed under the [MIT License](LICENSE).
