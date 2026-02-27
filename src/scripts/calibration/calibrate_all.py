"""
Temperature Scaling Calibration for Multiple Fusion Model Variants + EHR Baseline.

Optimizes individual temperature parameters (T) across different model configurations.
Includes the standalone unimodal EHR MLP for a fair baseline comparison.

Usage:
    # Calibrate 4-Modality baseline
    python -m src.scripts.calibration.calibrate_all --modalities ehr ecg img txt

    # Calibrate 3-Modality Champion
    python -m src.scripts.calibration.calibrate_all --modalities ehr img txt
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.utils.config import Config
from src.data.multimodal_dataset import MultimodalSepsisDataset
from src.models.fusion.late_fusion_model import LateFusionSepsisModel
from src.models.unimodal.mlp.train_unimodal_mlp import DynamicModalityMLP
from src.utils.evaluation import load_embeddings

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemperatureScaler(nn.Module):
    """Holds and optimizes temperature parameters for active branches."""

    def __init__(self, keys: list[str]):
        super().__init__()
        self.temperatures = nn.ParameterDict(
            {key: nn.Parameter(torch.ones(1) * 1.5) for key in keys}
        )

    def forward(self, logits_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        calibrated_probs = {}
        for key, logit in logits_dict.items():
            t = self.temperatures[key]
            calibrated_probs[key] = torch.sigmoid(logit / t)
        return calibrated_probs


def gather_fusion_validation_logits(
    model: nn.Module, dataloader: DataLoader
) -> tuple[dict, torch.Tensor]:
    """Dynamically collects logits for the LateFusionSepsisModel."""
    model.eval()
    all_logits = {}
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            embeddings = {k: v.to(DEVICE) for k, v in batch["embeddings"].items()}
            masks = {k: v.to(DEVICE) for k, v in batch["masks"].items()}
            targets = batch["label"].to(DEVICE).float()

            outputs = model(embeddings, masks)

            if not all_logits:
                for key in outputs["logits"].keys():
                    all_logits[key] = []
                all_logits["add"] = []
                all_logits["final"] = []

            for key in outputs["logits"].keys():
                all_logits[key].append(outputs["logits"][key])

            p_add = torch.zeros_like(outputs["p_final"])
            for i, mod in enumerate(model.modalities):
                p_add += outputs["weights"][:, i : i + 1] * outputs["p_unimodal"][mod]

            all_logits["add"].append(torch.logit(p_add, eps=1e-7))
            all_logits["final"].append(torch.logit(outputs["p_final"], eps=1e-7))
            all_targets.append(targets)

    for key in all_logits.keys():
        all_logits[key] = torch.cat(all_logits[key], dim=0)
    return all_logits, torch.cat(all_targets, dim=0)


def gather_unimodal_validation_logits(
    model: nn.Module, data_path: Path, batch_size: int = 256
) -> tuple[dict, torch.Tensor]:
    """Collects raw logits from a Unimodal MLP."""
    X_val, y_val, _ = load_embeddings(data_path)
    X_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    val_loader = DataLoader(
        TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=False
    )

    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(DEVICE)
            logits = model(inputs)
            all_logits.extend(logits.cpu())
            all_targets.extend(targets)

    logits_dict = {"final": torch.stack(all_logits).squeeze()}
    targets_tensor = torch.stack(all_targets).squeeze()

    return logits_dict, targets_tensor


def optimize_temperatures(logits_dict: dict, targets: torch.Tensor) -> dict:
    """Uses LBFGS to find the optimal temperature for each active branch."""
    scaler = TemperatureScaler(list(logits_dict.keys())).to(DEVICE)
    bce_loss = nn.BCELoss()

    logits_dict = {k: v.to(DEVICE) for k, v in logits_dict.items()}
    targets = targets.to(DEVICE).float()

    optimizer = optim.LBFGS(scaler.parameters(), lr=0.01, max_iter=50)

    def eval_closure():
        optimizer.zero_grad()
        calibrated_probs = scaler(logits_dict)
        loss = 0.0
        for key in calibrated_probs.keys():
            loss += bce_loss(calibrated_probs[key].squeeze(), targets)
        loss.backward()
        return loss

    optimizer.step(eval_closure)

    optimal_temps = {}
    for key, param in scaler.temperatures.items():
        optimal_temps[key] = param.item()
        logger.info(f"Optimal T_{key}: {optimal_temps[key]:.4f}")

    return optimal_temps


def main():
    Config.setup_logging()
    Config.set_seed(42)

    parser = argparse.ArgumentParser(description="Calibrate Fusion Models.")
    parser.add_argument("--modalities", nargs="+", default=["ehr", "ecg", "img", "txt"])
    args = parser.parse_args()
    num_mods = len(args.modalities)

    logger.info(
        f"=== Starting Calibration Pipeline for {num_mods}-Modality Architecture ==="
    )

    # 1. Dynamically Construct Experiment Paths
    tuning_suffix_scratch = f"_{num_mods}mod"
    tuning_suffix_pre = f"_{num_mods}mod_pretrained"

    experiments = {
        "ehr_mlp_baseline": {
            "weights": Config.EHR_MLP_MODEL_DIR / "best_ehr_mlp.pt",
            "config": Config.RESULTS_DIR
            / "ehr"
            / "tuning"
            / "best_hyperparameters.json",
            "model_type": "unimodal",
        },
        "scratch_no_dropout": {
            "weights": Config.FUSION_MODEL_DIR
            / f"best_late_fusion_model_{num_mods}mod_scratch_no_dropout.pt",
            "config": Config.RESULTS_DIR
            / "fusion"
            / "tuning"
            / f"best_hyperparameters{tuning_suffix_scratch}.json",
            "model_type": "fusion",
            "is_pretrained": False,
        },
        "scratch_ehr_dropout": {
            "weights": Config.FUSION_MODEL_DIR
            / f"best_late_fusion_model_{num_mods}mod_scratch_ehr_dropout.pt",
            "config": Config.RESULTS_DIR
            / "fusion"
            / "tuning"
            / f"best_hyperparameters{tuning_suffix_scratch}.json",
            "model_type": "fusion",
            "is_pretrained": False,
        },
        "pretrained_no_dropout": {
            "weights": Config.FUSION_MODEL_DIR
            / f"best_late_fusion_model_{num_mods}mod_pretrained_no_dropout.pt",
            "config": Config.RESULTS_DIR
            / "fusion"
            / "tuning"
            / f"best_hyperparameters{tuning_suffix_pre}.json",
            "model_type": "fusion",
            "is_pretrained": True,
        },
        "pretrained_ehr_dropout": {
            "weights": Config.FUSION_MODEL_DIR
            / f"best_late_fusion_model_{num_mods}mod_pretrained_ehr_dropout.pt",
            "config": Config.RESULTS_DIR
            / "fusion"
            / "tuning"
            / f"best_hyperparameters{tuning_suffix_pre}.json",
            "model_type": "fusion",
            "is_pretrained": True,
        },
    }

    fusion_val_loader = DataLoader(
        MultimodalSepsisDataset(
            "valid", ehr_dropout_rate=0.0, active_modalities=args.modalities
        ),
        batch_size=256,
        shuffle=False,
    )

    all_experiments_results = {}

    unimodal_configs = {}
    mod_map = {"ehr": "ehr", "ecg": "ecg", "img": "cxr_img", "txt": "cxr_txt"}
    for mod in args.modalities:
        folder_name = mod_map[mod]
        json_path = (
            Config.RESULTS_DIR / folder_name / "tuning" / "best_hyperparameters.json"
        )
        if json_path.exists():
            with open(json_path, "r") as f:
                unimodal_configs[mod] = json.load(f)["params"]

    for exp_name, paths in experiments.items():
        logger.info(f"--- Starting Calibration for {exp_name} ---")

        weights_path = Path(paths["weights"])
        config_path = Path(paths["config"])

        if not weights_path.exists() or not config_path.exists():
            logger.warning(f"Missing files for {exp_name}. Skipping.")
            continue

        with open(config_path, "r") as f:
            config = json.load(f)["params"]

        logger.info("Gathering validation logits...")

        # --- DYNAMIC INFERENCE ---
        if paths.get("model_type") == "unimodal":
            model = DynamicModalityMLP(input_dim=768, config=config).to(DEVICE)
            model.load_state_dict(
                torch.load(weights_path, map_location=DEVICE, weights_only=True)
            )

            data_path = Config.EHR_EMBEDDINGS_DIR / "valid_embeddings.pt"
            logits_dict, targets = gather_unimodal_validation_logits(model, data_path)

        else:  # fusion
            base_input_dims = {"ehr": 768, "ecg": 768, "img": 1024, "txt": 768}
            input_dims = {k: base_input_dims[k] for k in args.modalities}

            current_unimodal_configs = (
                unimodal_configs if paths.get("is_pretrained") else None
            )

            model = LateFusionSepsisModel(
                input_dims=input_dims,
                config=config,
                unimodal_configs=current_unimodal_configs,
                active_modalities=args.modalities,
            ).to(DEVICE)
            model.load_state_dict(
                torch.load(weights_path, map_location=DEVICE, weights_only=True)
            )

            logits_dict, targets = gather_fusion_validation_logits(
                model, fusion_val_loader
            )

        logger.info("Optimizing temperatures via LBFGS...")
        optimal_temps = optimize_temperatures(logits_dict, targets)
        all_experiments_results[exp_name] = optimal_temps

    output_path = (
        Config.RESULTS_DIR
        / "fusion"
        / f"master_calibration_temperatures_{num_mods}mod.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_experiments_results, f, indent=4)

    logger.info(f"Saved master calibration parameters to {output_path}")


if __name__ == "__main__":
    main()
