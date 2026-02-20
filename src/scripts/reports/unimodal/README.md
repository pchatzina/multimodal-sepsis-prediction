# Unimodal Classifier Reports

This folder contains scripts for generating comparison reports of unimodal classifier performance.

## Script: `compare_classifiers.py`

Generates markdown reports comparing the performance of different unimodal classifiers (e.g., Logistic Regression, XGBoost, MLP) for each modality.

### Usage

Run as a module from the project root:

```bash
python -m src.scripts.reports.unimodal.compare_classifiers --modality <modality>
```

- `<modality>`: Specify the modality to compare (e.g., `ehr`, `ecg`).

### Output

- Markdown report summarizing metrics (AUROC, AUPRC, F1, etc.) for each classifier.
- Results are saved in the relevant reports directory.

### Requirements

- Classifier results and metrics must be available for the specified modality.
- See the main README and unimodal classifier README for details on generating classifier outputs.

