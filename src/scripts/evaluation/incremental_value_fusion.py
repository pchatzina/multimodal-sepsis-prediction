"""
Incremental Value Analysis via Inference Masking.

Takes the best calibrated Late-Fusion model and systematically evaluates it
on the test set by artificially masking out combinations of modalities.
Outputs predictions and metrics for each combination to be used in downstream
statistical significance testing.

Usage:
    python -m src.scripts.evaluation.incremental_value_fusion
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.config import Config
from src.data.multimodal_dataset import MultimodalSepsisDataset
from src.models.fusion.late_fusion_model import LateFusionSepsisModel
from src.utils.evaluation import (
    compute_metrics,
    print_metrics,
    save_predictions,
    save_metrics,
)

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The Champion Model Configuration
CHAMPION_NAME = "scratch_ehr_dropout"
WEIGHTS_FILE = "best_late_fusion_model_scratch_ehr_dropout.pt"
CONFIG_FILE = "best_hyperparameters.json"

# Define the 8 logical combinations assuming EHR is the foundational baseline
MODALITY_COMBINATIONS = {
    "1_EHR_Only": {"ehr": 1, "ecg": 0, "img": 0, "txt": 0},
    "2_EHR_ECG": {"ehr": 1, "ecg": 1, "img": 0, "txt": 0},
    "3_EHR_IMG": {"ehr": 1, "ecg": 0, "img": 1, "txt": 0},
    # "4_EHR_TXT": {"ehr": 1, "ecg": 0, "img": 0, "txt": 1},
    "5_EHR_ECG_IMG": {"ehr": 1, "ecg": 1, "img": 1, "txt": 0},
    # "6_EHR_ECG_TXT": {"ehr": 1, "ecg": 1, "img": 0, "txt": 1},
    "7_EHR_IMG_TXT": {"ehr": 1, "ecg": 0, "img": 1, "txt": 1},
    "8_All_Modalities": {"ehr": 1, "ecg": 1, "img": 1, "txt": 1},
}


def get_masked_predictions(
    model: torch.nn.Module, dataloader: DataLoader, active_mods: dict[str, int]
) -> tuple:
    """Runs inference while systematically forcing specific modality masks to 0."""
    model.eval()
    all_p_final = []
    all_targets = []
    all_sample_ids = []

    with torch.no_grad():
        for batch in dataloader:
            embeddings = {k: v.to(DEVICE) for k, v in batch["embeddings"].items()}
            targets = batch["label"].to(DEVICE).float()
            sample_ids = batch.get("sample_id", np.arange(len(targets)))

            # Apply the artificial ablation mask
            # We multiply so that if a patient is genuinely missing an ECG (batch mask = 0),
            # it stays 0 even if active_mods says it should be 1.
            masks = {}
            for mod in ["ehr", "ecg", "img", "txt"]:
                masks[mod] = (batch["masks"][mod] * active_mods[mod]).to(DEVICE)

            outputs = model(embeddings, masks)

            all_p_final.append(outputs["p_final"].cpu())
            all_targets.append(targets.cpu())
            all_sample_ids.extend(sample_ids)

    y_true = torch.cat(all_targets, dim=0).numpy().squeeze()
    p_uncalibrated = torch.cat(all_p_final, dim=0).numpy().squeeze()

    return y_true, p_uncalibrated, all_sample_ids


def main():
    Config.setup_logging()
    Config.set_seed(42)

    logger.info(f"--- Starting Incremental Value Analysis for {CHAMPION_NAME} ---")

    # 1. Load Data
    test_loader = DataLoader(
        MultimodalSepsisDataset("test", ehr_dropout_rate=0.0),
        batch_size=256,
        shuffle=False,
    )

    # 2. Load Model
    model_path = Config.FUSION_MODEL_DIR / WEIGHTS_FILE
    tuning_file = Config.RESULTS_DIR / "fusion" / "tuning" / CONFIG_FILE

    if not model_path.exists() or not tuning_file.exists():
        logger.error("Champion model weights or config missing. Aborting.")
        return

    with open(tuning_file, "r") as f:
        config = json.load(f)["params"]

    input_dims = {"ehr": 768, "ecg": 768, "img": 1024, "txt": 768}
    model = LateFusionSepsisModel(
        input_dims=input_dims, config=config, unimodal_configs=None
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE, weights_only=True)
    )

    # 3. Load Calibration Temperature
    temps_path = Config.RESULTS_DIR / "fusion" / "master_calibration_temperatures.json"
    t_val = 1.0
    if temps_path.exists():
        with open(temps_path, "r") as f:
            temps = json.load(f)
            t_val = temps.get(CHAMPION_NAME, {}).get("final", 1.0)
    logger.info(f"Loaded calibration temperature T={t_val:.4f}")

    # 4. Evaluate Combinations
    base_out_dir = Config.RESULTS_DIR / "fusion" / "incremental_value"
    base_out_dir.mkdir(parents=True, exist_ok=True)

    for combo_name, active_mods in MODALITY_COMBINATIONS.items():
        logger.info(f"\nEvaluating Masked Combination: {combo_name}")
        logger.info(f"Active Modalities: {active_mods}")

        y_true, p_uncalibrated, sample_ids = get_masked_predictions(
            model, test_loader, active_mods
        )

        # Apply Temperature Scaling
        p_uncal_clipped = np.clip(p_uncalibrated, 1e-7, 1 - 1e-7)
        logit_final = np.log(p_uncal_clipped / (1 - p_uncal_clipped))
        p_calibrated = 1 / (1 + np.exp(-logit_final / t_val))

        # Compute and Print Metrics
        metrics = compute_metrics(y_true, p_calibrated)
        print_metrics(metrics, name=f"{combo_name} (Calibrated)")

        # Save Artifacts for Statistical Testing
        combo_dir = base_out_dir / combo_name
        save_predictions(
            combo_dir / "preds_calibrated.csv", sample_ids, y_true, p_calibrated
        )
        save_metrics(combo_dir / "metrics.json", metrics)

    logger.info(f"\nIncremental value artifacts saved to {base_out_dir}")
    logger.info("Ready for Statistical Significance Testing.")


if __name__ == "__main__":
    main()
