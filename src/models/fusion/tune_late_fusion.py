"""
Optuna hyperparameter tuning script for the Late-Fusion Sepsis Prediction Model.

Dynamically tests combinations of learning rates, hidden layer sizes, dropout,
and the crucial `lambda_weight` for the composite loss function.

Usage:
    # Tune Option A (Scratch)
    python -m src.models.fusion.tune_late_fusion --n_trials 30

    # Tune Option B (Pre-trained)
    python -m src.models.fusion.tune_late_fusion --n_trials 30 --load_pretrained
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils.config import Config
from src.utils.evaluation import compute_metrics
from src.data.multimodal_dataset import MultimodalSepsisDataset
from src.models.fusion.late_fusion_model import LateFusionSepsisModel
from src.models.fusion.loss import composite_sepsis_loss
from src.models.fusion.train_late_fusion import (
    load_pretrained_unimodal_weights,
)

logger = logging.getLogger(__name__)

# ==========================================
# CONFIG & PATH MAPPING
# ==========================================
EPOCHS = 50
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(
    model: torch.nn.Module, dataloader: DataLoader, lambda_weight: float
) -> float:
    """Evaluates the model and returns AUROC for Optuna tuning."""
    model.eval()
    all_probs, all_targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            embeddings = {k: v.to(DEVICE) for k, v in batch["embeddings"].items()}
            masks = {k: v.to(DEVICE) for k, v in batch["masks"].items()}
            targets = batch["label"].to(DEVICE)

            outputs = model(embeddings, masks)
            # We don't strictly need to compute loss here, just get the final probabilities
            probs = outputs["p_final"].cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(targets.cpu().numpy())

    metrics = compute_metrics(
        np.array(all_targets).flatten(), np.array(all_probs).flatten()
    )
    return metrics["auroc"]


# ==========================================
# OPTUNA OBJECTIVE
# ==========================================
def objective(
    trial: optuna.Trial,
    train_dataset: MultimodalSepsisDataset,
    val_dataset: MultimodalSepsisDataset,
    is_pretrained: bool,
    unimodal_configs: dict = None,
) -> float:

    # 1. Suggest Hyperparameters
    config = {
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "lambda_weight": trial.suggest_float("lambda_weight", 0.1, 1.0, step=0.1),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1),
        # Unimodal MLP dimensions (Only used if NOT pretrained, i.e., Option A)
        "uni_hidden_1": trial.suggest_categorical("uni_hidden_1", [128, 256, 512]),
        "uni_hidden_2": trial.suggest_categorical("uni_hidden_2", [32, 64, 128]),
        # Gating Network dimensions
        "gate_hidden_1": trial.suggest_categorical("gate_hidden_1", [128, 256, 512]),
        "gate_hidden_2": trial.suggest_categorical("gate_hidden_2", [32, 64, 128]),
        # Synergy Head dimensions
        "syn_hidden_1": trial.suggest_categorical("syn_hidden_1", [128, 256, 512]),
        "syn_hidden_2": trial.suggest_categorical("syn_hidden_2", [32, 64, 128]),
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    # 2. Initialize Model
    input_dims = {"ehr": 768, "ecg": 768, "img": 1024, "txt": 768}
    model = LateFusionSepsisModel(
        input_dims=input_dims,
        config=config,
        unimodal_configs=unimodal_configs,  # <-- Pass it to the model!
        common_dim=768,
    ).to(DEVICE)

    if is_pretrained:
        # Load weights if running tuning for Option B
        load_pretrained_unimodal_weights(model, DEVICE)

    optimizer = optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    best_val_auroc = 0.0
    epochs_without_improvement = 0

    # 3. Training Loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in train_loader:
            embeddings = {k: v.to(DEVICE) for k, v in batch["embeddings"].items()}
            masks = {k: v.to(DEVICE) for k, v in batch["masks"].items()}
            targets = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(embeddings, masks)

            loss, _, _ = composite_sepsis_loss(
                outputs["p_final"],
                outputs["p_unimodal"],
                masks,
                targets,
                config["lambda_weight"],
            )

            loss.backward()
            optimizer.step()

        val_auroc = evaluate_model(model, val_loader, config["lambda_weight"])

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

        # Report to Optuna for pruning
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

    parser = argparse.ArgumentParser(
        description="Tune Late Fusion Model hyperparameters."
    )
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument(
        "--load_pretrained",
        action="store_true",
        help="Tune for Option B (Pre-trained weights)",
    )
    args = parser.parse_args()

    # Determine paths and suffixes based on the mode
    suffix = "_pretrained" if args.load_pretrained else ""
    output_dir = Config.RESULTS_DIR / "fusion" / "tuning"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"best_hyperparameters{suffix}.json"

    # --- 1. LOAD UNIMODAL CONFIGS FOR OPTION B ---
    unimodal_configs = None
    if args.load_pretrained:
        unimodal_configs = {}
        mod_map = {"ehr": "ehr", "ecg": "ecg", "img": "cxr_img", "txt": "cxr_txt"}
        for mod, folder_name in mod_map.items():
            json_path = (
                Config.RESULTS_DIR
                / folder_name
                / "tuning"
                / "best_hyperparameters.json"
            )
            if json_path.exists():
                with open(json_path, "r") as f:
                    unimodal_configs[mod] = json.load(f)["params"]
            else:
                logger.warning(
                    f"Could not find unimodal config for {mod} at {json_path}"
                )
        logger.info("Loaded exact unimodal architectures for Option B Tuning.")

    logger.info("--- Loading Multimodal Dataset ---")
    train_dataset = MultimodalSepsisDataset(split="train")
    val_dataset = MultimodalSepsisDataset(split="valid")

    study = optuna.create_study(direction="maximize")

    # --- INJECT BASELINE ---
    study.enqueue_trial(
        {
            "batch_size": 512,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "lambda_weight": 0.4,
            "dropout_rate": 0.1,
            "uni_hidden_1": 256,
            "uni_hidden_2": 128,
            "gate_hidden_1": 512,
            "gate_hidden_2": 128,
            "syn_hidden_1": 512,
            "syn_hidden_2": 128,
        }
    )

    mode_name = (
        "OPTION B (Pre-trained)" if args.load_pretrained else "OPTION A (Scratch)"
    )
    logger.info(
        f"Starting Optuna study for LATE FUSION [{mode_name}] with {args.n_trials} trials..."
    )

    study.optimize(
        lambda trial: objective(
            trial, train_dataset, val_dataset, args.load_pretrained, unimodal_configs
        ),
        n_trials=args.n_trials,
    )

    logger.info("\n=== BEST TRIAL FOR LATE FUSION ===")
    logger.info(f"Best Val AUROC: {study.best_value:.4f}")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")

    # Save to JSON
    with open(results_path, "w") as f:
        json.dump(
            {"best_val_auroc": study.best_value, "params": study.best_params},
            f,
            indent=4,
        )

    logger.info(f"Saved optimal parameters to {results_path}")


if __name__ == "__main__":
    main()
