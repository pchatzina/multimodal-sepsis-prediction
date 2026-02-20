# ECG - Unimodal Classifier Comparison

Test set: N = 1426 (+832 / −594), threshold = 0.5

| Metric | LR | XGBoost | MLP |
|---|:---:|:---:|:---:|
| AUROC | **0.6043** | 0.5800 | 0.5804 |
| AUPRC | **0.6761** | 0.6369 | 0.6391 |
| Accuracy | 0.5898 | **0.6038** | 0.5933 |
| F1 | 0.6715 | **0.7168** | 0.7004 |
| Precision | **0.6301** | 0.6148 | 0.6141 |
| Recall | 0.7188 | **0.8594** | 0.8149 |
| Specificity | **0.4091** | 0.2458 | 0.2828 |

### Confusion Matrix

| | LR | XGBoost | MLP |
|---|:---:|:---:|:---:|
| TP | 598 | 715 | 678 |
| FP | 351 | 448 | 426 |
| FN | 234 | 117 | 154 |
| TN | 243 | 146 | 168 |

### Generalisation Gap (Val → Test)

| Metric | LR | XGBoost | MLP |
|---|:---:|:---:|:---:|
| AUROC | +0.0243 | +0.0139 | +0.0078 |
| AUPRC | +0.0302 | +0.0004 | -0.0053 |
