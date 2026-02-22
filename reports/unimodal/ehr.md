# EHR - Unimodal Classifier Comparison

Test set: N = 2332 (+1268 / −1064), threshold = 0.5

| Metric | LR | XGBoost | MLP |
|---|:---:|:---:|:---:|
| AUROC | 0.8494 | 0.8635 | **0.8687** |
| AUPRC | 0.8847 | 0.8952 | **0.8995** |
| Accuracy | 0.7629 | 0.7749 | **0.7779** |
| F1 | 0.7794 | 0.7936 | **0.7983** |
| Precision | 0.7885 | **0.7914** | 0.7885 |
| Recall | 0.7705 | 0.7957 | **0.8084** |
| Specificity | **0.7538** | 0.7500 | 0.7415 |

### Confusion Matrix

| | LR | XGBoost | MLP |
|---|:---:|:---:|:---:|
| TP | 977 | 1009 | 1025 |
| FP | 262 | 266 | 275 |
| FN | 291 | 259 | 243 |
| TN | 802 | 798 | 789 |

### Generalisation Gap (Val → Test)

| Metric | LR | XGBoost | MLP |
|---|:---:|:---:|:---:|
| AUROC | -0.0248 | -0.0206 | -0.0229 |
| AUPRC | -0.0181 | -0.0168 | -0.0189 |
