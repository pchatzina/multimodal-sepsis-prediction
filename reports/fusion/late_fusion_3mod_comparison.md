# Late-Fusion Sepsis Model (3 Modalities) - Experiment Comparison

**Test set:** N = 2332 (+1268 / −1064), threshold = 0.5

| Metric | Option A (No Dropout) | Option A (EHR Dropout) | Option B (No Dropout) | Option B (EHR Dropout) |
|---|:---:|:---:|:---:|:---:|
| AUROC | 0.8668 | 0.8664 | **0.8691** | 0.8686 |
| AUPRC | 0.8978 | 0.8974 | **0.8987** | 0.8980 |
| Accuracy | 0.7732 | 0.7706 | 0.7813 | **0.7826** |
| F1 | **0.7965** | 0.7934 | 0.7923 | 0.7937 |
| Precision | 0.7776 | 0.7774 | 0.8190 | **0.8200** |
| Recall | **0.8162** | 0.8099 | 0.7674 | 0.7689 |
| Specificity | 0.7218 | 0.7237 | 0.7979 | **0.7989** |

### Confusion Matrix

| Metric | Option A (No Dropout) | Option A (EHR Dropout) | Option B (No Dropout) | Option B (EHR Dropout) |
|---|:---:|:---:|:---:|:---:|
| TP | 1035 | 1027 | 973 | 975 |
| FP | 296 | 294 | 215 | 214 |
| FN | 233 | 241 | 295 | 293 |
| TN | 768 | 770 | 849 | 850 |

### Generalisation Gap (Val → Test)

| Metric | Option A (No Dropout) | Option A (EHR Dropout) | Option B (No Dropout) | Option B (EHR Dropout) |
|---|:---:|:---:|:---:|:---:|
| AUROC | -0.0279 | -0.0277 | -0.0270 | -0.0270 |
| AUPRC | -0.0235 | -0.0234 | -0.0223 | -0.0227 |
