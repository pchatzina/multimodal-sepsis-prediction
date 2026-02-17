# Unimodal Downstream Classifiers

Binary classifiers for sepsis prediction using frozen foundation-model
embeddings. Classifiers are organised by type, with per-modality training
scripts inside each:

```
src/models/unimodal/
  logistic_regression/      # Logistic Regression (linear probe)
    train_ehr_lr.py
  xgboost/                  # XGBoost (gradient-boosted trees)
    train_ehr_xgboost.py
  mlp/                      # MLP (two-hidden-layer feed-forward network)
    train_ehr_mlp.py
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
python -m src.scripts.extract_embeddings.extract_ehr_embeddings
```

## Classifier Details

### Logistic Regression (`lr/`)
Linear probe — the standard baseline for evaluating embedding quality.
Features are standardised with `StandardScaler` (fit on train only).

### XGBoost (`xgboost/`)
Gradient-boosted trees with GPU acceleration (`device=cuda`).
Uses early stopping on validation AUROC.

### MLP (`mlp/`)
Two-hidden-layer feed-forward network (256 → 64 → 1) with BatchNorm,
dropout, and early stopping. Trained with PyTorch + `BCEWithLogitsLoss`.

## Running

All scripts must be run as modules from the project root:

```bash
# EHR classifiers
python -m src.models.unimodal.logistic_regression.train_ehr_lr
python -m src.models.unimodal.xgboost.train_ehr_xgboost
python -m src.models.unimodal.mlp.train_ehr_mlp
```

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
| ECG | ECG-FM (`ecg-fm`) | 🔲 Planned |
| CXR images | TBD | 🔲 Planned |
| CXR reports | TBD | 🔲 Planned |
