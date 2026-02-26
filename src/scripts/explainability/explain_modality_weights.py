"""
Extracts and visualizes the modality gating weights and synergy coefficient
from the champion Late-Fusion Sepsis model across the test set.

Usage:
    python -m src.scripts.explainability.explain_modality_weights
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src.utils.config import Config
from src.data.multimodal_dataset import MultimodalSepsisDataset
from src.models.fusion.late_fusion_model import LateFusionSepsisModel

logger = logging.getLogger(__name__)

# Champion model settings based on thesis findings
CHAMPION_SUFFIX = "_scratch_ehr_dropout"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODALITIES = ["ehr", "ecg", "img", "txt"]


def setup_explainability_dirs() -> dict:
    """Sets up and returns directories for saving explainability artifacts."""
    base_exp_dir = Config.RESULTS_DIR / "explainability" / "modality_weights"
    base_exp_dir.mkdir(parents=True, exist_ok=True)

    return {
        "output_dir": base_exp_dir,
        "csv_path": base_exp_dir / "test_set_modality_weights.csv",
        "plot_weights_path": base_exp_dir / "modality_weights_distribution.pdf",
        "plot_beta_path": base_exp_dir / "synergy_beta_distribution.pdf",
    }


def load_champion_model() -> LateFusionSepsisModel:
    """Loads the champion Late-Fusion model and its exact hyperparameters."""
    fusion_dir = Config.RESULTS_DIR / "fusion"
    tuning_file = fusion_dir / "tuning" / "best_hyperparameters.json"
    model_path = Config.FUSION_MODEL_DIR / f"best_late_fusion_model{CHAMPION_SUFFIX}.pt"

    if not tuning_file.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Missing model artifacts. Check {tuning_file} and {model_path}"
        )

    with open(tuning_file, "r") as f:
        best_params = json.load(f)["params"]

    # Initialize model (Scratch mode, so unimodal_configs=None)
    input_dims = {"ehr": 768, "ecg": 768, "img": 1024, "txt": 768}
    model = LateFusionSepsisModel(
        input_dims=input_dims,
        config=best_params,
        unimodal_configs=None,
        common_dim=768,
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    logger.info(f"Successfully loaded champion model from {model_path}")

    return model


def extract_weights(model: torch.nn.Module, dataloader: DataLoader) -> pd.DataFrame:
    """Runs inference on the dataloader to extract gating weights and beta."""
    logger.info("Extracting modality weights across the dataset...")

    results = {
        "subject_id": [],
        "true_label": [],
        "p_final": [],
        "beta": [],
        "w_ehr": [],
        "w_ecg": [],
        "w_img": [],
        "w_txt": [],
    }

    with torch.no_grad():
        for batch in dataloader:
            embeddings = {k: v.to(DEVICE) for k, v in batch["embeddings"].items()}
            masks = {k: v.to(DEVICE) for k, v in batch["masks"].items()}
            targets = batch["label"].numpy()
            subject_ids = batch.get(
                "subject_id", np.zeros(len(targets))
            )  # Fallback if IDs aren't batched

            # Forward pass
            outputs = model(embeddings, masks)

            # Extract weights (Shape: [B, 4]) and beta (Shape: [B, 1])
            w_batch = outputs["weights"].cpu().numpy()
            beta_batch = outputs["beta"].cpu().numpy().flatten()
            p_final_batch = outputs["p_final"].cpu().numpy().flatten()

            # Append to results
            results["subject_id"].extend(subject_ids)
            results["true_label"].extend(targets)
            results["p_final"].extend(p_final_batch)
            results["beta"].extend(beta_batch)

            for i, mod in enumerate(MODALITIES):
                results[f"w_{mod}"].extend(w_batch[:, i])

    df = pd.DataFrame(results)
    logger.info(f"Extraction complete. Processed {len(df)} patients.")
    return df


def plot_distributions(df: pd.DataFrame, paths: dict):
    """Generates thesis-ready visualizations of the gating logic."""
    logger.info("Generating visualizations...")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # 1. Modality Weights Plot (Melt DataFrame for Seaborn)
    weight_cols = [f"w_{mod}" for mod in MODALITIES]
    df_melted = df.melt(
        id_vars=["subject_id", "true_label"],
        value_vars=weight_cols,
        var_name="Modality",
        value_name="Attention Weight",
    )
    # Clean up labels for the plot
    df_melted["Modality"] = df_melted["Modality"].str.replace("w_", "").str.upper()
    df_melted["Diagnosis"] = df_melted["true_label"].map({0: "Control", 1: "Sepsis"})

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_melted,
        x="Modality",
        y="Attention Weight",
        hue="Diagnosis",
        palette="muted",
        showfliers=False,  # Hide extreme outliers for a cleaner thesis plot
    )
    plt.title("Distribution of Modality Importance Weights ($w_i$)")
    plt.ylabel("Weight ($w_i$)")
    plt.xlabel("Modality")
    plt.tight_layout()
    plt.savefig(paths["plot_weights_path"], dpi=300)
    plt.close()

    # 2. Synergy Beta Plot
    plt.figure(figsize=(8, 5))
    df["Diagnosis"] = df["true_label"].map({0: "Control", 1: "Sepsis"})
    sns.kdeplot(
        data=df,
        x="beta",
        hue="Diagnosis",
        fill=True,
        common_norm=False,
        palette="muted",
    )
    plt.title("Distribution of Synergy Coefficient ($\\beta$)")
    plt.xlabel("Synergy Contribution ($\\beta$)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(paths["plot_beta_path"], dpi=300)
    plt.close()

    logger.info(f"Saved plots to {paths['output_dir']}")


def main():
    Config.setup_logging()
    Config.set_seed(42)

    paths = setup_explainability_dirs()
    model = load_champion_model()

    # We use the test dataset for final thesis numbers
    g = torch.Generator()
    g.manual_seed(42)
    test_loader = DataLoader(
        MultimodalSepsisDataset("test", ehr_dropout_rate=0.0),
        batch_size=128,
        shuffle=False,
        num_workers=4,
        generator=g,
    )

    # Run Extraction & Plotting
    weights_df = extract_weights(model, test_loader)
    weights_df.to_csv(paths["csv_path"], index=False)
    logger.info(f"Saved tabular weight data to {paths['csv_path']}")

    plot_distributions(weights_df, paths)
    logger.info("=== Step 3.1: Modality-Level Explainability Complete ===")


if __name__ == "__main__":
    main()
