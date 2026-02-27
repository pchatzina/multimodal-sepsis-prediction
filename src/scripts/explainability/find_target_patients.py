"""
Queries the database to find specific EHR-only and Multimodal patients
in the test set, mapping them directly to their batched memory index
for targeted Captum explainability.

Usage:
    python -m src.scripts.explainability.find_target_patients
"""

import logging
import datasets
from src.utils.config import Config
from src.utils.database import query_to_df

logger = logging.getLogger(__name__)


def main():
    Config.setup_logging()

    # 1. Map every subject_id to its specific batch index
    test_batches_path = Config.RESULTS_DIR / "explainability" / "test_batches"
    if not test_batches_path.exists():
        logger.error(f"test_batches not found at {test_batches_path}")
        return

    test_batches = datasets.Dataset.load_from_disk(test_batches_path)

    sid_to_batch = {}
    for batch_idx, batch in enumerate(test_batches):
        for sid in batch["subject_ids"]:
            sid_to_batch[int(sid)] = batch_idx

    logger.info("Querying database for test set patients...")
    query = """
        SELECT subject_id, modality_signature, sepsis_label 
        FROM mimiciv_ext.dataset_splits 
        WHERE dataset_split = 'test'
    """
    df = query_to_df(query)

    ehr_only = []
    multimodal = []

    # 2. Categorize patients that actually exist in the compiled memory batches
    for _, row in df.iterrows():
        sid = int(row["subject_id"])
        if sid not in sid_to_batch:
            continue

        signature = str(row["modality_signature"])
        label = int(row["sepsis_label"])
        patient_info = {
            "subject_id": sid,
            "batch_index": sid_to_batch[sid],
            "label": label,
        }

        if "CXR" in signature:
            multimodal.append(patient_info)
        else:
            ehr_only.append(patient_info)

    # We prioritize positive Sepsis cases for interesting explainability vignettes
    ehr_positive = [p for p in ehr_only if p["label"] == 1][:3]
    multi_positive = [p for p in multimodal if p["label"] == 1][:3]

    print("\n" + "=" * 70)
    print("🎯 TARGET SUBJECT IDs FOR EXPLAINABILITY")
    print("=" * 70)

    print("\n[ EHR-ONLY CASES ]")
    for p in ehr_positive:
        print(f"Subject ID: {p['subject_id']} (Found in Batch {p['batch_index']})")

    print("\n[ MULTIMODAL (EHR + IMG + TXT) CASES ]")
    for p in multi_positive:
        print(f"Subject ID: {p['subject_id']} (Found in Batch {p['batch_index']})")

    print("\nPass these exact Subject IDs to the Captum explainer using --subject_ids")


if __name__ == "__main__":
    main()
