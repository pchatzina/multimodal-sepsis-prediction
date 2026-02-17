# EHR - Unimodal Classifier Comparison

Test set: N = 2332 (+1268 / −1064), threshold = 0.5

| Metric | LR | XGBoost | MLP |
|---|:---:|:---:|:---:|
| AUROC | 0.8493 | 0.8670 | **0.8678** |
| AUPRC | 0.8846 | 0.8974 | **0.8984** |
| Accuracy | 0.7633 | 0.7702 | **0.7792** |
| F1 | 0.7799 | 0.7901 | **0.7914** |
| Precision | 0.7887 | 0.7846 | **0.8135** |
| Recall | 0.7713 | **0.7957** | 0.7705 |
| Specificity | 0.7538 | 0.7397 | **0.7895** |

### Confusion Matrix

| | LR | XGBoost | MLP |
|---|:---:|:---:|:---:|
| TP | 978 | 1009 | 977 |
| FP | 262 | 277 | 224 |
| FN | 290 | 259 | 291 |
| TN | 802 | 787 | 840 |

### Generalisation Gap (Val → Test)

| Metric | LR | XGBoost | MLP |
|---|:---:|:---:|:---:|
| AUROC | -0.0249 | -0.0136 | -0.0259 |
| AUPRC | -0.0181 | -0.0115 | -0.0184 |
