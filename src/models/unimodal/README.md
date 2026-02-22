# Unimodal Downstream Classifiers

Binary classifiers for sepsis prediction using frozen foundation-model
embeddings. Classifiers are organised by type, with per-modality training
scripts inside each:

```
src/models/unimodal/
  logistic_regression/      # Logistic Regression (linear probe)
    train_ehr_lr.py
    train_ecg_lr.py
  xgboost/                  # XGBoost (gradient-boosted trees)
    train_ehr_xgboost.py
    train_ecg_xgboost.py
  mlp/                      # MLP (dynamic, Optuna-tuned hyperparameters)
    tune_mlp.py              # Optuna hyperparameter tuning
    train_unimodal_mlp.py    # Unified MLP training (uses Optuna results)
```

## Pipeline

```
.pt embeddings  →  train_{modality}_{classifier}.py  →  model artifact
                                                     →  results/ (metrics JSON + predictions CSV)
                                                     →  TensorBoard logs
```

## Prerequisites

Embeddings must exist in `$PROCESSED_DATA_DIR/{modality}/embeddings/`:
```
train_embeddings.pt   # dict {embeddings, labels, sample_ids}
valid_embeddings.pt
test_embeddings.pt
```


Generate them with:
```bash
# EHR (MOTOR)
python -m src.scripts.extract_embeddings.ehr_embeddings

# ECG (ECG-FM)
python -m src.scripts.extract_embeddings.ecg_embeddings
```

## Classifier Details

### Logistic Regression (`lr/`)
Linear probe — the standard baseline for evaluating embedding quality.
Features are standardised with `StandardScaler` (fit on train only).

### XGBoost (`xgboost/`)
Gradient-boosted trees with GPU acceleration (`device=cuda`).
Uses early stopping on validation AUROC.


### MLP (`mlp/`)
Dynamic Multilayer Perceptron (MLP) with architecture and training hyperparameters selected via Optuna tuning. Supports input normalization, BatchNorm/LayerNorm, ReLU/GELU activations, dropout, and early stopping. Trained with PyTorch + `BCEWithLogitsLoss`.

**Workflow:**
- Run `tune_mlp.py` to search for optimal hyperparameters for each modality.
- Run `train_unimodal_mlp.py` to train the MLP using the tuned parameters.


## Running


All scripts must be run as modules from the project root. Replace `ehr` with `ecg` in the script name to run for the ECG modality.

```bash

# EHR classifiers
python -m src.models.unimodal.logistic_regression.train_ehr_lr
python -m src.models.unimodal.xgboost.train_ehr_xgboost
# MLP (Optuna-tuned)
python -m src.models.unimodal.mlp.tune_mlp --modality ehr --n_trials 50
python -m src.models.unimodal.mlp.train_unimodal_mlp --modality ehr

# ECG classifiers
python -m src.models.unimodal.logistic_regression.train_ecg_lr
python -m src.models.unimodal.xgboost.train_ecg_xgboost
# MLP (Optuna-tuned)
python -m src.models.unimodal.mlp.tune_mlp --modality ecg --n_trials 50
python -m src.models.unimodal.mlp.train_unimodal_mlp --modality ecg
```


### Batch Running: `run_classifiers.py`

To run all available unimodal classifier scripts sequentially, or only those for a specific modality or algorithm, use:

```bash
# Run all classifiers
python -m src.scripts.training.unimodal.run_classifiers

# Run only classifiers for a specific modality (e.g., 'ehr' or 'ecg') or algorithm ('lr', 'xgboost', 'mlp')
python -m src.scripts.training.unimodal.run_classifiers --modality ehr
python -m src.scripts.training.unimodal.run_classifiers --algorithm mlp
```

This script will automatically discover and execute all `train_*.py` scripts.

## Metrics

All classifiers share evaluation utilities from `src/utils/evaluation.py`:

| Metric | Description |
|--------|-------------|
| AUROC | Area Under ROC Curve |
| AUPRC | Area Under Precision-Recall Curve |
| F1 | Harmonic mean of Precision and Recall |
| Precision | Positive Predictive Value (PPV) |
| Recall | Sensitivity (True Positive Rate) |
| Specificity | True Negative Rate |

## Output Locations

| Artifact | Path |
|----------|------|
| Model weights | `$MODELS_DATA_DIR/{modality}/{classifier}/` |
| Metrics JSON | `$RESULTS_DATA_DIR/{modality}/{classifier}/` |
| Predictions CSV | `$RESULTS_DATA_DIR/{modality}/{classifier}/` |
| TensorBoard | `$RESULTS_DATA_DIR/tensorboard/{modality}/{classifier}/` |

## Tests

```bash
pytest tests/test_classifiers.py -v
```

## Current Modalities

| Modality | Foundation Model | Status |
|----------|-----------------|--------|
| EHR | MOTOR (`motor-t-base`) | ✅ Complete |
| ECG | ECG-FM (`ecg-fm`) |  ✅ Complete |
| CXR images | TBD | 🔲 Planned |
| CXR reports | TBD | 🔲 Planned |
