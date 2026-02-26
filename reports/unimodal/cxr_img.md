# CXR_IMG - Unimodal Classifier Comparison

Test set: N = 408 (+278 / −130), threshold = 0.5

| Metric | LR | XGBoost | MLP |
|---|:---:|:---:|:---:|
| AUROC | 0.5399 | **0.5890** | 0.5886 |
| AUPRC | 0.7243 | **0.7450** | 0.7345 |
| Accuracy | 0.5686 | 0.6618 | **0.6863** |
| F1 | 0.6777 | 0.7915 | **0.8019** |
| Precision | 0.6903 | 0.6823 | **0.7038** |
| Recall | 0.6655 | **0.9424** | 0.9317 |
| Specificity | **0.3615** | 0.0615 | 0.1615 |

### Confusion Matrix

| | LR | XGBoost | MLP |
|---|:---:|:---:|:---:|
| TP | 185 | 262 | 259 |
| FP | 83 | 122 | 109 |
| FN | 93 | 16 | 19 |
| TN | 47 | 8 | 21 |

### Generalisation Gap (Val → Test)

| Metric | LR | XGBoost | MLP |
|---|:---:|:---:|:---:|
| AUROC | -0.0114 | -0.0439 | -0.0777 |
| AUPRC | +0.0230 | -0.0237 | -0.0530 |
