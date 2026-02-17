"""Extract EHR embeddings using the frozen MOTOR foundation model.

Loads labels and ontology, computes transformer features via femr,
and saves per-split embedding .pt files to Config.EHR_EMBEDDINGS_DIR.

Usage:
    python -m src.scripts.extract_embeddings.extract_ehr_embeddings
"""

import logging
import pickle

import meds_reader
import numpy as np
import pandas as pd
import torch

import femr.models.transformer

from src.utils.config import Config
from src.utils.database import query_to_df

logger = logging.getLogger(__name__)


def main():
    Config.setup_logging()
    Config.check_dirs()

    prep_dir = Config.MOTOR_PRETRAINING_FILES_DIR
    output_dir = Config.EHR_EMBEDDINGS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_path = Config.EHR_LABELS_DIR / "labels.parquet"
    if not labels_path.exists():
        logger.error("Labels file not found: %s", labels_path)
        return

    labels_df = pd.read_parquet(labels_path)
    logger.info("Loaded %d labels from %s", len(labels_df), labels_path)

    # Required conversion for femr without generating pandas warnings
    labels_df["prediction_time"] = pd.to_datetime(labels_df["prediction_time"]).apply(
        lambda x: x.to_pydatetime()
    )

    ontology_path = prep_dir / "ontology.pkl"
    if not ontology_path.exists():
        logger.error("Ontology file not found: %s", ontology_path)
        return

    with open(ontology_path, "rb") as f:
        ontology = pickle.load(f)

    logger.info("Computing MOTOR features (this may take a while)...")
    with meds_reader.SubjectDatabase(str(Config.COHORT_MEDS_READER_DIR)) as database:
        features = femr.models.transformer.compute_features(
            db=database,
            model_path=str(Config.MOTOR_MODEL_DIR),
            labels=labels_df.to_dict("records"),
            ontology=ontology,
            device=torch.device("cuda"),
            tokens_per_batch=8192,
            num_proc=8,
        )

    feature_array = features["features"]
    subject_ids = features["subject_ids"]
    logger.info("Extracted features for %d subjects, dim=%d", *feature_array.shape)

    splits_query = (
        "SELECT subject_id, dataset_split, sepsis_label FROM mimiciv_ext.dataset_splits"
    )
    splits_df = query_to_df(splits_query)
    sid_to_idx = {int(sid): idx for idx, sid in enumerate(subject_ids)}

    split_map = {"train": "train", "validate": "valid", "test": "test"}

    for db_split, file_split in split_map.items():
        split_df = splits_df[splits_df["dataset_split"] == db_split]
        all_sids = split_df["subject_id"].tolist()
        split_sids = [s for s in all_sids if s in sid_to_idx]

        n_skipped = len(all_sids) - len(split_sids)
        if n_skipped > 0:
            logger.warning(
                "Split '%s': %d / %d subjects missing from extracted features",
                db_split, n_skipped, len(all_sids),
            )

        indices = [sid_to_idx[s] for s in split_sids]
        embeddings = torch.from_numpy(feature_array[indices].astype(np.float32))

        label_lookup = split_df.set_index("subject_id")["sepsis_label"]
        labels = [str(int(label_lookup[s])) for s in split_sids]

        out_path = output_dir / f"{file_split}_embeddings.pt"
        torch.save(
            {
                "embeddings": embeddings,
                "labels": labels,
                "sample_ids": [str(s) for s in split_sids],
            },
            out_path,
        )
        logger.info(
            "Saved %s: %d embeddings, dim=%d → %s",
            file_split, embeddings.shape[0], embeddings.shape[1], out_path,
        )


if __name__ == "__main__":
    main()
