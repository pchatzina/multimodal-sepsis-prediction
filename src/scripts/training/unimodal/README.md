# Unimodal Training Scripts

This directory contains scripts for tuning and training all unimodal classifiers (EHR, ECG, etc.) in the project.

## Workflow

1. **Tune all MLP hyperparameters (Optuna):**

Run the following command to tune MLP hyperparameters for all modalities using Optuna. This will sequentially run tuning jobs for each supported modality and save the best hyperparameters to the appropriate results folders.

```bash
python -m src.utils.bash_wrapper tune-mlp
```

2. **Train and evaluate all unimodal classifiers:**

After tuning, run all unimodal classifiers (Logistic Regression, XGBoost, MLP) for all modalities:

```bash
python -m src.scripts.training.unimodal.run_classifiers
```

This will automatically discover and execute all classifier training scripts for each modality, using the tuned MLP hyperparameters where applicable.
