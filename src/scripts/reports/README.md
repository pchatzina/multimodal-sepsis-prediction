# Report Generation Scripts (Unimodal & Fusion)

This directory contains scripts for generating comparison reports of classifier performance for both unimodal and fusion (late-fusion) models.

## Unimodal Classifier Reports

Scripts for generating markdown reports comparing the performance of unimodal classifiers (e.g., Logistic Regression, XGBoost, MLP) for each modality.

### Script: `compare_classifiers.py`

Generates markdown reports comparing the performance of different unimodal classifiers for a specified modality.

**Usage:**

```bash
python -m src.scripts.reports.unimodal.compare_classifiers --modality <modality>
```

- `<modality>`: Specify the modality to compare (e.g., `ehr`, `ecg`).

**Output:**
- Markdown report summarizing metrics (AUROC, AUPRC, F1, etc.) for each classifier.
- Results are saved in the relevant reports directory.

**Requirements:**
- Classifier results and metrics must be available for the specified modality.
- See the main README and unimodal classifier README for details on generating classifier outputs.

---

## Fusion (Late-Fusion) Reports

Scripts for generating markdown reports comparing the performance of late-fusion experiments.

### Script: `compare_fusion.py`

Compares late-fusion sepsis prediction results for the 4 main experiment types (Option A/B × No Dropout/EHR Dropout) and generates a markdown report.

**Usage:**

```bash
python -m src.scripts.reports.fusion.compare_fusion
```

**Output:**
- Markdown report with comparison tables for all fusion experiments, including metrics (AUROC, AUPRC, F1, etc.), confusion matrices, and generalization gap (Val → Test).
- Results are saved in the fusion reports directory.

**Requirements:**
- Fusion experiment results and metrics must be available in the expected results directory.

See the script for further details and customization options.