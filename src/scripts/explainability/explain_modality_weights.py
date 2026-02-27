"""
Extracts and visualizes the modality gating weights and synergy coefficient
from the champion Late-Fusion Sepsis model across the test set.

Usage:
    # OPTION 1: 4-Modality baseline weights (Champion: scratch_ehr_dropout)
    python -m src.scripts.explainability.explain_modality_weights \
        --modalities ehr ecg img txt \
        --model_variant scratch_ehr_dropout

    # OPTION 2: 3-Modality champion weights (Champion: pretrained_ehr_dropout)
    python -m src.scripts.explainability.explain_modality_weights \
        --modalities ehr img txt \
        --model_variant pretrained_ehr_dropout
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_explainability_dirs(num_mods: int) -> dict:
    """Sets up and returns directories for saving explainability artifacts isolated by modality count."""
    base_exp_dir = (
        Config.RESULTS_DIR
        / "explainability"
        / "modality_weights"
        / f"{num_mods}mod_architecture"
    )
    base_exp_dir.mkdir(parents=True, exist_ok=True)

    return {
        "output_dir": base_exp_dir,
        "csv_path": base_exp_dir / "test_set_modality_weights.csv",
        "plot_weights_path": base_exp_dir / "modality_weights_distribution.pdf",
        "plot_beta_path": base_exp_dir / "synergy_beta_distribution.pdf",
    }


def load_champion_model(
    active_modalities: list, model_variant: str
) -> LateFusionSepsisModel:
    """Loads the champion Late-Fusion model and its exact hyperparameters."""
    num_mods = len(active_modalities)
    fusion_dir = Config.RESULTS_DIR / "fusion"

    # Determine correct tuning file based on whether variant uses pretraining
    is_pretrained = "pretrained" in model_variant
    tuning_suffix = f"_{num_mods}mod_pretrained" if is_pretrained else f"_{num_mods}mod"
    tuning_file = fusion_dir / "tuning" / f"best_hyperparameters{tuning_suffix}.json"

    model_path = (
        Config.FUSION_MODEL_DIR
        / f"best_late_fusion_model_{num_mods}mod_{model_variant}.pt"
    )

    if not tuning_file.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Missing model artifacts.\nChecked Tuning: {tuning_file}\nChecked Weights: {model_path}"
        )

    with open(tuning_file, "r") as f:
        best_params = json.load(f)["params"]

    # Load unimodal configs if model was pretrained
    unimodal_configs = None
    if is_pretrained:
        unimodal_configs = {}
        mod_map = {"ehr": "ehr", "ecg": "ecg", "img": "cxr_img", "txt": "cxr_txt"}
        for mod in active_modalities:
            json_path = (
                Config.RESULTS_DIR
                / mod_map[mod]
                / "tuning"
                / "best_hyperparameters.json"
            )
            if json_path.exists():
                with open(json_path, "r") as f:
                    unimodal_configs[mod] = json.load(f)["params"]

    input_dims = {"ehr": 768, "ecg": 768, "img": 1024, "txt": 768}
    current_dims = {k: input_dims[k] for k in active_modalities}

    model = LateFusionSepsisModel(
        input_dims=current_dims,
        config=best_params,
        unimodal_configs=unimodal_configs,
        common_dim=768,
        active_modalities=active_modalities,
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    logger.info(
        f"Successfully loaded champion model ({model_variant}) from {model_path}"
    )

    return model


def extract_weights(
    model: torch.nn.Module, dataloader: DataLoader, active_modalities: list
) -> pd.DataFrame:
    """Runs inference on the dataloader to extract gating weights and beta."""
    logger.info(
        f"Extracting modality weights across the dataset for {active_modalities}..."
    )

    results = {
        "subject_id": [],
        "true_label": [],
        "p_final": [],
        "beta": [],
    }
    for mod in active_modalities:
        results[f"w_{mod}"] = []

    with torch.no_grad():
        for batch in dataloader:
            embeddings = {k: v.to(DEVICE) for k, v in batch["embeddings"].items()}
            masks = {k: v.to(DEVICE) for k, v in batch["masks"].items()}
            targets = batch["label"].numpy()
            subject_ids = batch.get("subject_id", np.zeros(len(targets)))

            outputs = model(embeddings, masks)

            w_batch = outputs["weights"].cpu().numpy()
            beta_batch = outputs["beta"].cpu().numpy().flatten()
            p_final_batch = outputs["p_final"].cpu().numpy().flatten()

            results["subject_id"].extend(subject_ids)
            results["true_label"].extend(targets)
            results["p_final"].extend(p_final_batch)
            results["beta"].extend(beta_batch)

            for i, mod in enumerate(active_modalities):
                results[f"w_{mod}"].extend(w_batch[:, i])

    df = pd.DataFrame(results)
    logger.info(f"Extraction complete. Processed {len(df)} patients.")
    return df


def plot_distributions(df: pd.DataFrame, paths: dict, active_modalities: list):
    """Generates thesis-ready visualizations of the gating logic."""
    logger.info("Generating visualizations...")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    weight_cols = [f"w_{mod}" for mod in active_modalities]
    df_melted = df.melt(
        id_vars=["subject_id", "true_label"],
        value_vars=weight_cols,
        var_name="Modality",
        value_name="Attention Weight",
    )
    df_melted["Modality"] = df_melted["Modality"].str.replace("w_", "").str.upper()
    df_melted["Diagnosis"] = df_melted["true_label"].map({0: "Control", 1: "Sepsis"})

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_melted,
        x="Modality",
        y="Attention Weight",
        hue="Diagnosis",
        palette="muted",
        showfliers=False,
    )
    plt.title(
        f"Distribution of Modality Importance Weights ($w_i$) - {len(active_modalities)} Modalities"
    )
    plt.ylabel("Weight ($w_i$)")
    plt.xlabel("Modality")
    plt.tight_layout()
    plt.savefig(paths["plot_weights_path"], dpi=300)
    plt.close()

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

    parser = argparse.ArgumentParser(description="Extract and plot modality weights.")
    parser.add_argument("--modalities", nargs="+", default=["ehr", "ecg", "img", "txt"])
    parser.add_argument(
        "--model_variant",
        type=str,
        required=True,
        help="The winning experiment variant (e.g., 'scratch_ehr_dropout' or 'pretrained_ehr_dropout')",
    )
    args = parser.parse_args()

    paths = setup_explainability_dirs(len(args.modalities))
    model = load_champion_model(args.modalities, args.model_variant)

    g = torch.Generator()
    g.manual_seed(42)
    test_loader = DataLoader(
        MultimodalSepsisDataset(
            "test", ehr_dropout_rate=0.0, active_modalities=args.modalities
        ),
        batch_size=128,
        shuffle=False,
        num_workers=4,
        generator=g,
    )

    weights_df = extract_weights(model, test_loader, args.modalities)
    weights_df.to_csv(paths["csv_path"], index=False)
    logger.info(f"Saved tabular weight data to {paths['csv_path']}")

    plot_distributions(weights_df, paths, args.modalities)
    logger.info("=== Modality-Level Explainability Complete ===")


if __name__ == "__main__":
    main()
