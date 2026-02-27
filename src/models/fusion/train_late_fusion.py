"""
Training script for the Late-Fusion Sepsis Prediction Model.
Supports both Option A (Joint Training from scratch) and Option B (Fine-tuning pre-trained MLPs).

Usage:
    # Option A: Train from scratch
    python -m src.models.fusion.train_late_fusion --lambda_weight 0.4 --modalities ehr img txt

    # Option B: Load pre-trained unimodal weights
    python -m src.models.fusion.train_late_fusion --lambda_weight 0.4 --load_pretrained --modalities ehr img txt
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.utils.config import Config
from src.utils.evaluation import (
    compute_metrics,
    log_metrics_to_tensorboard,
    print_metrics,
    save_metrics,
    save_predictions,
)
from src.data.multimodal_dataset import MultimodalSepsisDataset
from src.models.fusion.late_fusion_model import LateFusionSepsisModel
from src.models.fusion.loss import composite_sepsis_loss

logger = logging.getLogger(__name__)

NUM_EPOCHS = 100
PATIENCE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_fusion_paths(
    is_pretrained: bool = False, ehr_dropout_rate: float = 0.0, num_mods: int = 4
):
    """Defines and creates directories for fusion model artifacts."""
    base_results = Config.RESULTS_DIR / "fusion"
    model_dir = Config.FUSION_MODEL_DIR

    # 1. Build a descriptive suffix (Added num_mods to prevent overwriting)
    mode_str = "pretrained" if is_pretrained else "scratch"
    dropout_str = "ehr_dropout" if ehr_dropout_rate > 0.0 else "no_dropout"
    suffix = f"_{num_mods}mod_{mode_str}_{dropout_str}"
    tuning_suffix = f"_{num_mods}mod_pretrained" if is_pretrained else f"_{num_mods}mod"

    # 2. Define isolated paths for this specific run
    paths = {
        "model_save_path": model_dir / f"best_late_fusion_model{suffix}.pt",
        "metrics_save_path": base_results / f"test_metrics_fusion{suffix}.json",
        "val_metrics_save_path": base_results / f"val_metrics_fusion{suffix}.json",
        "predictions_save_path": base_results / f"test_predictions_fusion{suffix}.csv",
        "tb_dir": Config.TENSORBOARD_LOG_DIR / f"fusion{suffix}",
        "tuning_file": base_results
        / "tuning"
        / f"best_hyperparameters{tuning_suffix}.json",
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    base_results.mkdir(parents=True, exist_ok=True)
    paths["tb_dir"].mkdir(parents=True, exist_ok=True)

    return paths


def load_pretrained_unimodal_weights(
    fusion_model: torch.nn.Module, device: torch.device
):
    """Loads pre-trained weights from the unimodal MLPs into the fusion model."""
    logger.info("--- Loading Pre-trained Unimodal Weights (Option B) ---")

    # Dynamically read the modalities from the model itself
    modalities = fusion_model.modalities

    model_paths = {
        "ehr": Config.EHR_MLP_MODEL_DIR / "best_ehr_mlp.pt",
        "ecg": Config.ECG_MLP_MODEL_DIR / "best_ecg_mlp.pt",
        "img": Config.CXR_IMG_MLP_MODEL_DIR / "best_cxr_img_mlp.pt",
        "txt": Config.CXR_TXT_MLP_MODEL_DIR / "best_cxr_txt_mlp.pt",
    }

    for mod in modalities:
        path = model_paths[mod]
        if not path.exists():
            logger.warning(
                f"Pre-trained weights not found for {mod} at {path}. Starting from scratch."
            )
            continue

        state_dict = torch.load(path, map_location=device, weights_only=True)

        if "projection.weight" in state_dict:
            fusion_model.projectors[mod].weight.data = state_dict["projection.weight"]
            fusion_model.projectors[mod].bias.data = state_dict["projection.bias"]
            logger.info(f"  -> Loaded projection weights for {mod.upper()}.")
        elif (
            fusion_model.projectors[mod].in_features
            == fusion_model.projectors[mod].out_features
        ):
            torch.nn.init.eye_(fusion_model.projectors[mod].weight)
            torch.nn.init.zeros_(fusion_model.projectors[mod].bias)
            logger.info(
                f"  -> Initialized identity projection for {mod.upper()} (no projection in unimodal)."
            )

        network_state_dict = {
            k.replace("network.", ""): v
            for k, v in state_dict.items()
            if k.startswith("network.")
        }

        try:
            fusion_model.unimodal_mlps[mod].network.load_state_dict(
                network_state_dict, strict=True
            )
            logger.info(f"  -> Loaded MLP network weights for {mod.upper()}.\n")
        except Exception as e:
            logger.error(
                f"  -> Failed to load MLP weights for {mod.upper()}. Architecture mismatch? Error: {e}\n"
            )


def evaluate_model(
    model: torch.nn.Module, dataloader: DataLoader, lambda_weight: float
) -> tuple:
    model.eval()
    total_loss = 0.0
    all_probs, all_targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            embeddings = {k: v.to(DEVICE) for k, v in batch["embeddings"].items()}
            masks = {k: v.to(DEVICE) for k, v in batch["masks"].items()}
            targets = batch["label"].to(DEVICE)

            outputs = model(embeddings, masks)
            loss, _, _ = composite_sepsis_loss(
                outputs["p_final"], outputs["p_unimodal"], masks, targets, lambda_weight
            )

            total_loss += loss.item() * targets.size(0)
            all_probs.extend(outputs["p_final"].cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    return np.array(all_targets).flatten(), np.array(all_probs).flatten(), avg_loss


def main():
    Config.setup_logging()
    Config.set_seed(42)

    parser = argparse.ArgumentParser(description="Train Late Fusion Model.")
    parser.add_argument("--lambda_weight", type=float, default=0.4)
    parser.add_argument("--load_pretrained", action="store_true")
    parser.add_argument("--ehr_dropout_rate", type=float, default=0.0)
    parser.add_argument("--modalities", nargs="+", default=["ehr", "ecg", "img", "txt"])
    args = parser.parse_args()

    paths = get_fusion_paths(
        is_pretrained=args.load_pretrained,
        ehr_dropout_rate=args.ehr_dropout_rate,
        num_mods=len(args.modalities),
    )

    run_type = (
        "OPTION B (Pre-trained)" if args.load_pretrained else "OPTION A (Scratch)"
    )
    dropout_type = (
        f"WITH EHR Dropout ({args.ehr_dropout_rate})"
        if args.ehr_dropout_rate > 0.0
        else "NO Dropout"
    )
    logger.info(
        f"=== Starting Training Pipeline for {len(args.modalities)}-MOD LATE FUSION - {run_type} | {dropout_type} ==="
    )

    if paths["tuning_file"].exists():
        with open(paths["tuning_file"], "r") as f:
            best_params = json.load(f)["params"]
        logger.info(f"Loaded dynamic hyperparameters: {best_params}")
    else:
        logger.info("No tuning file found. Using default hyperparameters.")
        best_params = {
            "batch_size": 128,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "uni_hidden_1": 256,
            "uni_hidden_2": 128,
            "gate_hidden_1": 512,
            "gate_hidden_2": 128,
            "syn_hidden_1": 512,
            "syn_hidden_2": 128,
            "dropout_rate": 0.1,
        }

    unimodal_configs = None
    if args.load_pretrained:
        unimodal_configs = {}
        # Only load configs for ACTIVE modalities
        mod_map = {"ehr": "ehr", "ecg": "ecg", "img": "cxr_img", "txt": "cxr_txt"}
        for mod in args.modalities:
            folder_name = mod_map[mod]
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
        logger.info("Loaded exact unimodal architectures for Option B.")

    batch_size = best_params.get("batch_size", 128)
    writer = SummaryWriter(log_dir=str(paths["tb_dir"]))

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        MultimodalSepsisDataset(
            "train",
            ehr_dropout_rate=args.ehr_dropout_rate,
            active_modalities=args.modalities,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        MultimodalSepsisDataset(
            "valid", ehr_dropout_rate=0.0, active_modalities=args.modalities
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    test_loader = DataLoader(
        MultimodalSepsisDataset(
            "test", ehr_dropout_rate=0.0, active_modalities=args.modalities
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    base_input_dims = {"ehr": 768, "ecg": 768, "img": 1024, "txt": 768}
    input_dims = {k: base_input_dims[k] for k in args.modalities}
    model = LateFusionSepsisModel(
        input_dims=input_dims,
        config=best_params,
        unimodal_configs=unimodal_configs,
        common_dim=768,
        active_modalities=args.modalities,
    ).to(DEVICE)

    if args.load_pretrained:
        load_pretrained_unimodal_weights(model, DEVICE)
    else:
        logger.info("--- Initializing all weights from scratch (Option A) ---")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=best_params.get("lr", 1e-4),
        weight_decay=best_params.get("weight_decay", 1e-5),
    )

    best_val_auroc = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0

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
                args.lambda_weight,
            )

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.size(0)

        train_loss /= len(train_loader.dataset)

        val_true, val_prob, val_loss = evaluate_model(
            model, val_loader, args.lambda_weight
        )
        val_metrics = compute_metrics(val_true, val_prob)

        logger.info(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUROC: {val_metrics['auroc']:.4f}"
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        log_metrics_to_tensorboard(writer, val_metrics, epoch, prefix="Val")

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), paths["model_save_path"])
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                logger.warning(f"Early stopping triggered at epoch {epoch}")
                break

    logger.info("--- Evaluating Best Model on Test Set ---")
    model.load_state_dict(torch.load(paths["model_save_path"]))

    val_true_final, val_prob_final, _ = evaluate_model(
        model, val_loader, args.lambda_weight
    )
    save_metrics(
        paths["val_metrics_save_path"], compute_metrics(val_true_final, val_prob_final)
    )

    test_true, test_prob, _ = evaluate_model(model, test_loader, args.lambda_weight)
    test_pred = (test_prob >= 0.5).astype(int)
    test_metrics = compute_metrics(test_true, test_prob)
    print_metrics(
        test_metrics, name=f"LATE FUSION TEST SET ({run_type} | {dropout_type})"
    )

    log_metrics_to_tensorboard(writer, test_metrics, global_step=0, prefix="Test")
    writer.close()

    save_metrics(paths["metrics_save_path"], test_metrics)
    test_ids = test_loader.dataset.subject_ids
    save_predictions(
        paths["predictions_save_path"], test_ids, test_true, test_prob, test_pred
    )
    logger.info("=== FUSION Training Pipeline Complete! ===")


if __name__ == "__main__":
    main()
