# ECG - Unimodal Classifier Comparison

Test set: N = 1426 (+832 / −594), threshold = 0.5

| Metric | LR | XGBoost | MLP |
|---|:---:|:---:|:---:|
| AUROC | **0.6043** | 0.5800 | 0.5683 |
| AUPRC | **0.6761** | 0.6369 | 0.6276 |
| Accuracy | 0.5898 | **0.6038** | 0.5820 |
| F1 | 0.6715 | **0.7168** | 0.6902 |
| Precision | **0.6301** | 0.6148 | 0.6081 |
| Recall | 0.7188 | **0.8594** | 0.7981 |
| Specificity | **0.4091** | 0.2458 | 0.2795 |

### Confusion Matrix

| | LR | XGBoost | MLP |
|---|:---:|:---:|:---:|
| TP | 598 | 715 | 664 |
| FP | 351 | 448 | 428 |
| FN | 234 | 117 | 168 |
| TN | 243 | 146 | 166 |

### Generalisation Gap (Val → Test)

| Metric | LR | XGBoost | MLP |
|---|:---:|:---:|:---:|
| AUROC | +0.0243 | +0.0139 | -0.0118 |
| AUPRC | +0.0302 | +0.0004 | -0.0315 |
