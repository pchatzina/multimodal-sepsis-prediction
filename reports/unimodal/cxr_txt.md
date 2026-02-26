# CXR_TXT - Unimodal Classifier Comparison

Test set: N = 408 (+278 / −130), threshold = 0.5

| Metric | LR | XGBoost | MLP |
|---|:---:|:---:|:---:|
| AUROC | 0.5740 | **0.5853** | 0.5685 |
| AUPRC | 0.7177 | **0.7706** | 0.7576 |
| Accuracy | 0.6078 | **0.6789** | 0.6691 |
| F1 | 0.7163 | **0.8006** | 0.7894 |
| Precision | **0.7063** | 0.6939 | 0.6970 |
| Recall | 0.7266 | **0.9460** | 0.9101 |
| Specificity | **0.3538** | 0.1077 | 0.1538 |

### Confusion Matrix

| | LR | XGBoost | MLP |
|---|:---:|:---:|:---:|
| TP | 202 | 263 | 253 |
| FP | 84 | 116 | 110 |
| FN | 76 | 15 | 25 |
| TN | 46 | 14 | 20 |

### Generalisation Gap (Val → Test)

| Metric | LR | XGBoost | MLP |
|---|:---:|:---:|:---:|
| AUROC | +0.0259 | -0.0695 | -0.0969 |
| AUPRC | +0.0095 | -0.0239 | -0.0275 |
