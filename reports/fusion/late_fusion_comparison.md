# Late-Fusion Sepsis Model - Experiment Comparison

**Test set:** N = 2332 (+1268 / −1064), threshold = 0.5

| Metric | Option A (No Dropout) | Option A (EHR Dropout) | Option B (No Dropout) | Option B (EHR Dropout) |
|---|:---:|:---:|:---:|:---:|
| AUROC | 0.8597 | 0.8613 | 0.8624 | **0.8696** |
| AUPRC | 0.8902 | 0.8904 | 0.8947 | **0.8970** |
| Accuracy | 0.7637 | 0.7599 | 0.7650 | **0.7822** |
| F1 | 0.7891 | 0.7885 | 0.7697 | **0.8048** |
| Precision | 0.7665 | 0.7565 | **0.8237** | 0.7849 |
| Recall | 0.8131 | 0.8233 | 0.7224 | **0.8257** |
| Specificity | 0.7049 | 0.6842 | **0.8158** | 0.7303 |

### Confusion Matrix

| | Option A (No Dropout) | Option A (EHR Dropout) | Option B (No Dropout) | Option B (EHR Dropout) |
|---|:---:|:---:|:---:|:---:|
| TP | 1031 | 1044 | 916 | 1047 |
| FP | 314 | 336 | 196 | 287 |
| FN | 237 | 224 | 352 | 221 |
| TN | 750 | 728 | 868 | 777 |

### Generalisation Gap (Val → Test)

| Metric | Option A (No Dropout) | Option A (EHR Dropout) | Option B (No Dropout) | Option B (EHR Dropout) |
|---|:---:|:---:|:---:|:---:|
| AUROC | -0.0281 | -0.0268 | -0.0265 | -0.0200 |
| AUPRC | -0.0241 | -0.0237 | -0.0206 | -0.0179 |
