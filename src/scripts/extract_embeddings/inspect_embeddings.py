"""
Inspect the statistical distributions of extracted embeddings for any modality.

Prints out Mean, Std, Min, Max, and Label Distributions to verify the
health of the latent space before training classifiers or performing late-fusion.

Usage:
    python -m src.scripts.extract_embeddings.inspect_embeddings --modality ehr
    python -m src.scripts.extract_embeddings.inspect_embeddings --modality ecg
"""

import argparse
import logging
import torch
from pathlib import Path

from src.utils.config import Config

logger = logging.getLogger(__name__)

# ==========================================
# REGISTRY
# ==========================================

# Map modalities to their respective embedding directories
MODALITY_DIRS = {
    "ehr": Config.EHR_EMBEDDINGS_DIR,
    "ecg": Config.ECG_EMBEDDINGS_DIR,
    # You can uncomment and add these as we build the CXR pipelines!
    # "cxr_img": Config.CXR_IMG_EMBEDDINGS_DIR,
    # "cxr_txt": Config.CXR_TXT_EMBEDDINGS_DIR,
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================


def inspect_split(split_name: str, file_path: Path):
    """Loads a .pt embedding file and prints its statistical summary."""
    print(f"\n{'-' * 60}")
    print(f"Inspecting: {file_path.name}")
    print(f"{'-' * 60}")

    if not file_path.exists():
        print(f"[!] Warning: {file_path.name} not found.")
        return

    data = torch.load(file_path, map_location="cpu", weights_only=False)

    embeddings = data["embeddings"]
    labels = data["labels"]
    sample_ids = data["sample_ids"]

    # --- Basic Info ---
    print(f"Embeddings shape    : {embeddings.shape}")
    print(f"Embedding dtype     : {embeddings.dtype}")
    print(f"Number of samples   : {len(sample_ids)}")
    print(f"Number of labels    : {len(labels)}")
    print(f"Embedding dimension : {embeddings.shape[1]}")

    # --- Statistics ---
    print(f"\nEmbedding Statistics:")
    print(f"  Mean : {embeddings.mean().item():.6f}")
    print(f"  Std  : {embeddings.std().item():.6f}")
    print(f"  Min  : {embeddings.min().item():.6f}")
    print(f"  Max  : {embeddings.max().item():.6f}")

    has_nan = torch.isnan(embeddings).any().item()
    has_inf = torch.isinf(embeddings).any().item()
    print(f"  Contains NaN : {has_nan}")
    print(f"  Contains Inf : {has_inf}")

    # --- Label Distribution ---
    if labels is not None and len(labels) > 0:
        # Cast to int to handle both string ("1") and int (1) cleanly
        int_labels = [int(l) for l in labels]
        unique_labels = set(int_labels)
        print(f"\nLabel Distribution:")
        for label in sorted(unique_labels):
            count = int_labels.count(label)
            percentage = (count / len(int_labels)) * 100
            print(f"  Class {label}: {count} ({percentage:.1f}%)")

    # --- Sample IDs ---
    print(f"\nFirst 5 sample IDs : {sample_ids[:5]}")
    print(f"Last 5 sample IDs  : {sample_ids[-5:]}")


# ==========================================
# MAIN EXECUTION
# ==========================================


def main():
    Config.setup_logging(level=logging.WARNING)  # Suppress debug logs for clean output

    parser = argparse.ArgumentParser(
        description="Inspect embedding statistics for a specific modality."
    )
    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        choices=list(MODALITY_DIRS.keys()),
        help="The modality to inspect (e.g., ehr, ecg).",
    )
    args = parser.parse_args()

    embedding_dir = MODALITY_DIRS[args.modality]
    splits = ["train", "valid", "test"]

    print("=" * 60)
    print(f"{args.modality.upper()} EMBEDDING INSPECTION")
    print("=" * 60)

    for split in splits:
        file_path = embedding_dir / f"{split}_embeddings.pt"
        inspect_split(split, file_path)

    print("\n" + "=" * 60)
    print("Inspection Complete.")


if __name__ == "__main__":
    main()
