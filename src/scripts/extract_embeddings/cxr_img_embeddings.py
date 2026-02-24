"""Extract CXR Image embeddings using the frozen torchxrayvision Foundation Model.

Loads metadata and labels from the database, pre-processes JPG images on the fly,
computes DenseNet121 features, and saves per-split embedding .pt files to
Config.CXR_IMG_EMBEDDINGS_DIR.

Usage:
    python -m src.scripts.extract_embeddings.cxr_img_embeddings
"""

import logging
import os
from pathlib import Path

import numpy as np
import skimage.io
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchxrayvision as xrv
from tqdm import tqdm

from src.utils.config import Config
from src.utils.database import query_to_df

logger = logging.getLogger(__name__)


def process_image(img_path: Path, transform: transforms.Compose) -> torch.Tensor:
    """Loads a JPG image, normalizes it for xrv, and applies transforms."""
    # 1. Read as grayscale (returns 0-255 for JPGs)
    img = skimage.io.imread(img_path, as_gray=True)

    # 2. Normalize to [-1024, 1024] range expected by torchxrayvision
    img = xrv.datasets.normalize(img, 255)

    # 3. Add channel dimension (1, H, W) for grayscale
    img = img[None, ...]

    # 4. Apply CenterCrop and Resize(224)
    img = transform(img)

    return torch.from_numpy(img).float()


def main():
    Config.setup_logging()
    Config.check_dirs()
    Config.set_seed()  # Ensure reproducibility

    output_dir = Config.CXR_IMG_EMBEDDINGS_DIR
    raw_img_dir = Config.RAW_CXR_IMG_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ── 1. LOAD MODEL ──────────────────────────────────────────────────
    logger.info("Loading densenet121-res224-all foundation model...")
    # Using the pre-trained weights covering MIMIC-CXR
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model = model.to(device)
    model.eval()

    # Define standard xrv transforms
    transform = transforms.Compose(
        [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
    )

    # ── 2. QUERY DATABASE ──────────────────────────────────────────────
    logger.info("Querying cohort and splits from database...")
    query = """
        SELECT 
            c.subject_id, 
            cxr.study_path, 
            c.sepsis_label, 
            s.dataset_split
        FROM mimiciv_ext.cohort c
        JOIN mimiciv_ext.cohort_cxr cxr ON c.subject_id = cxr.subject_id
        JOIN mimiciv_ext.dataset_splits s ON c.subject_id = s.subject_id
        WHERE s.modality_signature LIKE '%CXR%'
    """
    df = query_to_df(query)
    logger.info("Loaded %d CXR records for processing.", len(df))

    # Map DB split names to file split names (matching ehr_embeddings.py)
    split_map = {"train": "train", "validate": "valid", "test": "test"}

    # ── 3. EXTRACT EMBEDDINGS PER SPLIT ────────────────────────────────
    with torch.no_grad():
        for db_split, file_split in split_map.items():
            split_df = df[df["dataset_split"] == db_split].reset_index(drop=True)

            if split_df.empty:
                logger.warning("No records found for split: %s", db_split)
                continue

            logger.info(
                "Processing '%s' split (%d images)...", file_split, len(split_df)
            )

            embeddings_list = []
            labels_list = []
            sample_ids_list = []

            for _, row in tqdm(
                split_df.iterrows(), total=len(split_df), desc=f"{file_split} split"
            ):
                subject_id = int(row["subject_id"])
                label = int(row["sepsis_label"])

                # Construct full path: RAW_CXR_IMG_DIR / study_path
                # (Assuming study_path in DB looks like "files/p10/p10001884/...")
                img_path = raw_img_dir / row["study_path"]

                if not img_path.exists():
                    logger.error("Image not found: %s", img_path)
                    continue

                try:
                    # Preprocess and move to device: shape [1, 1, 224, 224]
                    img_tensor = (
                        process_image(img_path, transform).unsqueeze(0).to(device)
                    )

                    # Extract dense features (before classifier head)
                    # model.features returns shape [1, 1024, 7, 7]
                    features = model.features(img_tensor)
                    features = F.relu(features, inplace=True)

                    # Global Average Pooling reduces spatial dims to [1, 1024, 1, 1]
                    pooled = F.adaptive_avg_pool2d(features, (1, 1))

                    # Flatten the tensor perfectly to a 1D vector of size 1024
                    pooled_1d = pooled.view(1024)

                    embeddings_list.append(pooled_1d.cpu().numpy())
                    labels_list.append(label)
                    sample_ids_list.append(subject_id)

                except Exception as e:
                    logger.error(
                        "Failed to process subject %d at %s: %s",
                        subject_id,
                        img_path,
                        e,
                    )

            # ── 4. SAVE SPLIT TENSORS ──────────────────────────────────────
            if embeddings_list:
                embeddings_tensor = torch.from_numpy(
                    np.array(embeddings_list, dtype=np.float32)
                )
                out_path = output_dir / f"{file_split}_embeddings.pt"

                torch.save(
                    {
                        "embeddings": embeddings_tensor,
                        "labels": labels_list,
                        "sample_ids": sample_ids_list,
                    },
                    out_path,
                )
                logger.info(
                    "Saved %s: %d embeddings, dim=%d → %s",
                    file_split,
                    embeddings_tensor.shape[0],
                    embeddings_tensor.shape[1],
                    out_path,
                )


if __name__ == "__main__":
    main()
