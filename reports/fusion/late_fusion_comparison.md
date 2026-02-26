# Late-Fusion Sepsis Model - Experiment Comparison

**Test set:** N = 2332 (+1268 / −1064), threshold = 0.5

| Metric | Option A (No Dropout) | Option A (EHR Dropout) | Option B (No Dropout) | Option B (EHR Dropout) |
|---|:---:|:---:|:---:|:---:|
| AUROC | **0.8710** | 0.8695 | 0.8673 | 0.8669 |
| AUPRC | **0.8968** | 0.8947 | 0.8929 | 0.8923 |
| Accuracy | 0.7783 | 0.7783 | **0.7796** | 0.7796 |
| F1 | 0.7889 | **0.7944** | 0.7936 | 0.7922 |
| Precision | **0.8180** | 0.8011 | 0.8085 | 0.8126 |
| Recall | 0.7618 | **0.7879** | 0.7792 | 0.7729 |
| Specificity | **0.7979** | 0.7669 | 0.7801 | 0.7876 |

### Confusion Matrix

| | Option A (No Dropout) | Option A (EHR Dropout) | Option B (No Dropout) | Option B (EHR Dropout) |
|---|:---:|:---:|:---:|:---:|
| TP | 966 | 999 | 988 | 980 |
| FP | 215 | 248 | 234 | 226 |
| FN | 302 | 269 | 280 | 288 |
| TN | 849 | 816 | 830 | 838 |

### Generalisation Gap (Val → Test)

| Metric | Option A (No Dropout) | Option A (EHR Dropout) | Option B (No Dropout) | Option B (EHR Dropout) |
|---|:---:|:---:|:---:|:---:|
| AUROC | -0.0202 | -0.0225 | -0.0269 | -0.0270 |
| AUPRC | -0.0196 | -0.0228 | -0.0245 | -0.0249 |
