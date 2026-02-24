"""
Unified Optuna hyperparameter tuning script for unimodal MLPs.

Dynamically tests combinations of learning rates, hidden layer sizes,
normalization techniques, activations, and dropout rates across modalities.

Usage:
    python -m src.models.unimodal.mlp.tune_mlp --modality ehr --n_trials 50
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.utils.config import Config
from src.utils.evaluation import compute_metrics, load_embeddings

logger = logging.getLogger(__name__)

# ==========================================
# CONFIG & PATH MAPPING
# ==========================================
EPOCHS = 50
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_modality_dir(modality: str) -> Path:
    if modality == "ehr":
        return Config.EHR_EMBEDDINGS_DIR
    elif modality == "ecg":
        return Config.ECG_EMBEDDINGS_DIR
    elif modality == "cxr_img":
        return Config.CXR_IMG_EMBEDDINGS_DIR
    elif modality == "cxr_txt":
        return Config.CXR_TXT_EMBEDDINGS_DIR
    else:
        raise ValueError(f"Unknown modality: {modality}")


# ==========================================
# DYNAMIC MODEL
# ==========================================
class TunableMLP(nn.Module):
    def __init__(self, trial: optuna.Trial, input_dim: int):
        super().__init__()

        # Suggest hyperparameters
        use_input_norm = trial.suggest_categorical("use_input_norm", [True, False])
        norm_type = trial.suggest_categorical("norm_type", ["batch", "layer"])
        activation_name = trial.suggest_categorical("activation", ["ReLU", "GELU"])
        use_dropout = trial.suggest_categorical("use_dropout", [True, False])
        dropout_rate = (
            trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
            if use_dropout
            else 0.0
        )
        hidden_dim_1 = trial.suggest_categorical("hidden_dim_1", [128, 256, 512])
        hidden_dim_2 = trial.suggest_categorical("hidden_dim_2", [32, 64, 128])

        activation_layer = nn.ReLU() if activation_name == "ReLU" else nn.GELU()

        # ── DYNAMIC PROJECTION TO COMMON DIMENSION (768) ──
        self.projection = None
        current_dim = input_dim

        if input_dim != 768:
            self.projection = nn.Linear(input_dim, 768)
            current_dim = 768

        layers = []

        # Note: We now use current_dim instead of input_dim
        if use_input_norm:
            layers.append(nn.LayerNorm(current_dim))

        layers.append(nn.Linear(current_dim, hidden_dim_1))
        layers.append(
            nn.BatchNorm1d(hidden_dim_1)
            if norm_type == "batch"
            else nn.LayerNorm(hidden_dim_1)
        )
        layers.append(activation_layer)
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_dim_1, hidden_dim_2))
        layers.append(
            nn.BatchNorm1d(hidden_dim_2)
            if norm_type == "batch"
            else nn.LayerNorm(hidden_dim_2)
        )
        layers.append(activation_layer)
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_dim_2, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if self.projection is not None:
            x = self.projection(x)
        return self.network(x)


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def create_dataloader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_model(model: nn.Module, dataloader: DataLoader) -> float:
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(DEVICE)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(targets.numpy())

    metrics = compute_metrics(
        np.array(all_targets).flatten(), np.array(all_probs).flatten()
    )
    return metrics["auroc"]


# ==========================================
# OPTUNA OBJECTIVE
# ==========================================
def objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
) -> float:

    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    train_loader = create_dataloader(X_train, y_train, batch_size, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size, shuffle=False)

    model = TunableMLP(trial, input_dim).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_auroc = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

        val_auroc = evaluate_model(model, val_loader)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

        trial.report(val_auroc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val_auroc


# ==========================================
# MAIN
# ==========================================
def main():
    Config.setup_logging()
    Config.set_seed(42)

    parser = argparse.ArgumentParser(description="Tune MLP hyperparameters.")
    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        choices=["ehr", "ecg", "cxr_img", "cxr_txt"],
    )
    parser.add_argument("--n_trials", type=int, default=50)
    args = parser.parse_args()

    data_dir = get_modality_dir(args.modality)
    output_dir = Config.RESULTS_DIR / args.modality / "tuning"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"--- Loading Data for {args.modality.upper()} ---")
    X_train, y_train, _ = load_embeddings(data_dir / "train_embeddings.pt")
    X_val, y_val, _ = load_embeddings(data_dir / "valid_embeddings.pt")

    input_dim = X_train.shape[1]

    study = optuna.create_study(direction="maximize")

    # --- INJECT BASELINES ---
    if args.modality == "ehr":
        study.enqueue_trial(
            {
                "batch_size": 256,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "use_input_norm": False,
                "norm_type": "batch",
                "activation": "ReLU",
                "use_dropout": True,
                "dropout_rate": 0.3,
                "hidden_dim_1": 256,
                "hidden_dim_2": 64,
            }
        )
    elif args.modality == "ecg":
        study.enqueue_trial(
            {
                "batch_size": 128,
                "lr": 1e-4,
                "weight_decay": 1e-4,
                "use_input_norm": True,
                "norm_type": "batch",
                "activation": "GELU",
                "use_dropout": True,
                "dropout_rate": 0.3,
                "hidden_dim_1": 256,
                "hidden_dim_2": 64,
            }
        )
    elif args.modality == "cxr_txt":
        study.enqueue_trial(
            {
                "batch_size": 128,
                "lr": 1e-4,
                "weight_decay": 1e-3,
                "use_input_norm": True,
                "norm_type": "layer",
                "activation": "GELU",
                "use_dropout": True,
                "dropout_rate": 0.4,
                "hidden_dim_1": 256,
                "hidden_dim_2": 64,
            }
        )

    logger.info(
        f"Starting Optuna study for {args.modality.upper()} with {args.n_trials} trials..."
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, input_dim),
        n_trials=args.n_trials,
    )

    logger.info(f"\n=== BEST TRIAL FOR {args.modality.upper()} ===")
    logger.info(f"Best Val AUROC: {study.best_value:.4f}")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")

    results_path = output_dir / "best_hyperparameters.json"
    with open(results_path, "w") as f:
        json.dump(
            {"best_val_auroc": study.best_value, "params": study.best_params},
            f,
            indent=4,
        )

    logger.info(f"Saved optimal parameters to {results_path}")


if __name__ == "__main__":
    main()
