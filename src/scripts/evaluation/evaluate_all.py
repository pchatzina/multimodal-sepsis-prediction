"""
Master Evaluation Script for All Fusion Model Variants + EHR Baseline.

Evaluates performance on the test set for the specified modality combinations,
plus the standalone EHR MLP. Applies optimized temperatures for calibration,
computes standard metrics, saves predictions, and plots comparative reliability diagrams.

Usage:
    # Evaluate 4-Modality baseline
    python -m src.scripts.evaluation.evaluate_all --modalities ehr ecg img txt

    # Evaluate 3-Modality Champion
    python -m src.scripts.evaluation.evaluate_all --modalities ehr img txt
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.utils.config import Config
from src.data.multimodal_dataset import MultimodalSepsisDataset
from src.models.fusion.late_fusion_model import LateFusionSepsisModel
from src.models.unimodal.mlp.train_unimodal_mlp import DynamicModalityMLP
from src.utils.evaluation import (
    compute_metrics,
    print_metrics,
    save_predictions,
    save_metrics,
    plot_reliability_diagrams,
    load_embeddings,
)

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_fusion_test_predictions(
    model: torch.nn.Module, dataloader: DataLoader
) -> tuple:
    """Runs inference on the test set for the LateFusionSepsisModel."""
    model.eval()
    all_p_final = []
    all_targets = []
    all_sample_ids = []

    with torch.no_grad():
        for batch in dataloader:
            embeddings = {k: v.to(DEVICE) for k, v in batch["embeddings"].items()}
            masks = {k: v.to(DEVICE) for k, v in batch["masks"].items()}
            targets = batch["label"].to(DEVICE).float()
            sample_ids = batch.get("sample_id", np.arange(len(targets)))

            outputs = model(embeddings, masks)
            all_p_final.append(outputs["p_final"].cpu())
            all_targets.append(targets.cpu())
            all_sample_ids.extend(sample_ids)

    return (
        torch.cat(all_targets, dim=0).numpy().squeeze(),
        torch.cat(all_p_final, dim=0).numpy().squeeze(),
        all_sample_ids,
    )


def get_unimodal_test_predictions(
    model: torch.nn.Module, data_path: Path, batch_size: int = 256
) -> tuple:
    """Runs inference on the test set for a unimodal DynamicModalityMLP."""
    X_test, y_test, test_ids = load_embeddings(data_path)
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    test_loader = DataLoader(
        TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=False
    )

    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(targets.numpy())

    return np.array(all_targets).flatten(), np.array(all_probs).flatten(), test_ids


def main():
    Config.setup_logging()
    Config.set_seed(42)

    parser = argparse.ArgumentParser(description="Evaluate Fusion Models.")
    parser.add_argument("--modalities", nargs="+", default=["ehr", "ecg", "img", "txt"])
    args = parser.parse_args()

    num_mods = len(args.modalities)
    logger.info(f"=== Starting Evaluation for {num_mods}-Modality Architecture ===")

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
            "modality": "ehr",
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

    # 2. Load Fusion Dataloader
    fusion_test_loader = DataLoader(
        MultimodalSepsisDataset(
            "test", ehr_dropout_rate=0.0, active_modalities=args.modalities
        ),
        batch_size=256,
        shuffle=False,
    )

    # 3. Load Calibration Temps (Use a dynamically named JSON file to avoid overwriting 4-mod temps)
    temps_path = (
        Config.RESULTS_DIR
        / "fusion"
        / f"master_calibration_temperatures_{num_mods}mod.json"
    )
    if temps_path.exists():
        with open(temps_path, "r") as f:
            calibration_temps = json.load(f)
        logger.info(f"Loaded {num_mods}-mod calibration temperatures.")
    else:
        logger.warning(
            f"Calibration temps not found at {temps_path}. Falling back to T=1.0"
        )
        calibration_temps = {}

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

    uncalibrated_plot_data = {}
    calibrated_plot_data = {}

    # Define base output directory for this specific modality count
    eval_out_dir = (
        Config.RESULTS_DIR / "fusion" / "evaluation" / f"{num_mods}mod_architecture"
    )

    # 4. Iterate through all variants
    for exp_name, paths in experiments.items():
        logger.info(f"\n{'=' * 60}\nEvaluating: {exp_name}\n{'=' * 60}")

        weights_path = Path(paths["weights"])
        config_path = Path(paths["config"])

        if not weights_path.exists() or not config_path.exists():
            logger.warning(f"Missing files for {exp_name}. Skipping.")
            continue

        with open(config_path, "r") as f:
            config = json.load(f)["params"]

        # --- DYNAMIC MODEL INITIALIZATION AND INFERENCE ---
        if paths["model_type"] == "fusion":
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
            y_true, p_uncalibrated, sample_ids = get_fusion_test_predictions(
                model, fusion_test_loader
            )

            t_val = calibration_temps.get(exp_name, {}).get("final", 1.0)

        elif paths["model_type"] == "unimodal":
            data_path = Config.EHR_EMBEDDINGS_DIR / "test_embeddings.pt"
            model = DynamicModalityMLP(input_dim=768, config=config).to(DEVICE)
            model.load_state_dict(
                torch.load(weights_path, map_location=DEVICE, weights_only=True)
            )

            y_true, p_uncalibrated, sample_ids = get_unimodal_test_predictions(
                model, data_path
            )
            t_val = calibration_temps.get(exp_name, {}).get("final", 1.0)

        # --- APPLY CALIBRATION AND EVALUATE ---
        p_uncalibrated_clipped = np.clip(p_uncalibrated, 1e-7, 1 - 1e-7)
        logit_final = np.log(p_uncalibrated_clipped / (1 - p_uncalibrated_clipped))
        p_calibrated = 1 / (1 + np.exp(-logit_final / t_val))

        metrics_uncal = compute_metrics(y_true, p_uncalibrated)
        print_metrics(metrics_uncal, name=f"{exp_name} (Uncalibrated)")

        metrics_cal = compute_metrics(y_true, p_calibrated)
        print_metrics(metrics_cal, name=f"{exp_name} (Calibrated - T={t_val:.3f})")

        exp_results_dir = eval_out_dir / exp_name
        exp_results_dir.mkdir(parents=True, exist_ok=True)

        save_predictions(
            exp_results_dir / "preds_uncalibrated.csv",
            sample_ids,
            y_true,
            p_uncalibrated,
        )
        save_metrics(exp_results_dir / "metrics_uncalibrated.json", metrics_uncal)
        save_predictions(
            exp_results_dir / "preds_calibrated.csv", sample_ids, y_true, p_calibrated
        )
        save_metrics(exp_results_dir / "metrics_calibrated.json", metrics_cal)

        uncalibrated_plot_data[exp_name] = (y_true, p_uncalibrated)
        calibrated_plot_data[exp_name] = (y_true, p_calibrated)

    # 5. Generate Comparative Reliability Diagrams
    plot_dir = eval_out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_reliability_diagrams(
        uncalibrated_plot_data,
        save_path=plot_dir / "reliability_diagrams_uncalibrated.png",
        title=f"Reliability Diagrams Before Calibration ({num_mods} Modalities)",
    )
    plot_reliability_diagrams(
        calibrated_plot_data,
        save_path=plot_dir / "reliability_diagrams_calibrated.png",
        title=f"Reliability Diagrams After Temperature Scaling ({num_mods} Modalities)",
    )


if __name__ == "__main__":
    main()
