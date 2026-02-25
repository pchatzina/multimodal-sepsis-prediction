# Training Scripts (Unimodal & Fusion)

This directory contains scripts for tuning and training all unimodal and fusion classifiers (EHR, ECG, CXR, multimodal fusion) in the project.

## Unimodal Workflow

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

---

## Fusion Workflow

Batch execution for late-fusion sepsis prediction is handled by:

```bash
python -m src.scripts.training.fusion.run_fusion_experiments --tune_trials 30 --dropout_rate 0.3
```

This script performs:

1. **(Optional) Hyperparameter tuning** for both Option A and Option B fusion models using Optuna, if `--tune_trials` is set > 0.
2. **Runs 4 distinct fusion training experiments:**
   - Option A (Scratch) + No Dropout
   - Option A (Scratch) + EHR Dropout
   - Option B (Pre-trained) + No Dropout
   - Option B (Pre-trained) + EHR Dropout

**Option A (Scratch):** Initializes brand new unimodal MLPs inside the fusion model. The MLPs for each modality are trained from scratch as part of the fusion model, with no prior knowledge from unimodal training.

**Option B (Pre-trained):** Loads the exact pre-trained weights from the unimodal MLPs for each modality into the fusion model. The fusion model starts from these pre-trained unimodal MLPs and continues training them jointly with the fusion layers.

All experiments use the specified `--dropout_rate` for the EHR dropout runs. Results and logs are saved to the appropriate results folders.

See the script for further details and customization options.