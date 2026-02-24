"""Extract CXR Text embeddings using the frozen Bio_ClinicalBERT Foundation Model.

Loads metadata and labels from the database, pre-processes text reports on the fly,
computes Bio_ClinicalBERT features, and saves per-split embedding .pt files to
Config.CXR_TXT_EMBEDDINGS_DIR.

Usage:
    python -m src.scripts.extract_embeddings.cxr_txt_embeddings
"""

import os
import re
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from src.utils.config import Config
from src.utils.database import query_to_df


def clean_report_text(text: str) -> str:
    """
    Cleans MIMIC-CXR report text by removing underscore artifacts
    and normalizing whitespace.
    """
    # Replace continuous underscores with a space
    text = re.sub(r"_+", " ", text)
    # Replace multiple spaces, newlines, or tabs with a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def build_file_path(subject_id: int, study_id: int) -> Path:
    """Constructs the MIMIC-CXR file path based on subject and study IDs."""
    subject_str = str(subject_id)
    folder_prefix = f"p{subject_str[:2]}"
    return (
        Config.RAW_CXR_TXT_DIR
        / "mimic-cxr-reports"
        / "files"
        / folder_prefix
        / f"p{subject_str}"
        / f"s{study_id}.txt"
    )


def extract_embeddings():
    """Extracts Bio_ClinicalBERT embeddings for CXR reports and saves them per split."""

    # 1. Ensure output directories exist & set seed for reproducibility
    Config.check_dirs()
    Config.set_seed()

    output_dir = Config.CXR_TXT_EMBEDDINGS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Model and Tokenizer
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    model_dir = Config.CXR_TXT_PRETRAINED_MODEL_DIR

    print(f"Loading foundation model: {model_name}...")

    # We tell Hugging Face to use your specific directory as the cache
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=model_dir)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 3. Query the Database
    print("Fetching cohort data from database...")
    query = """
        SELECT 
            c.subject_id, 
            c.study_id, 
            s.sepsis_label, 
            s.dataset_split
        FROM mimiciv_ext.cohort_cxr c
        JOIN mimiciv_ext.dataset_splits s 
          ON c.subject_id = s.subject_id
        WHERE s.dataset_split IS NOT NULL;
    """
    df = query_to_df(query)

    # Dictionary to hold the split data
    # Format: {"train": {"embeddings": [], "labels": [], "sample_ids": []}, ...}
    splits_data = {
        "train": {"embeddings": [], "labels": [], "sample_ids": []},
        "valid": {"embeddings": [], "labels": [], "sample_ids": []},
        "test": {"embeddings": [], "labels": [], "sample_ids": []},
    }

    print(f"Processing {len(df)} reports...")

    # 4. Process each report
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Embeddings"):
            subject_id = int(row["subject_id"])
            study_id = int(row["study_id"])
            label = int(row["sepsis_label"])
            split = str(row["dataset_split"]).lower()  # train, val, or test

            if split in ["validate"]:
                split = "valid"

            file_path = build_file_path(subject_id, study_id)

            if not file_path.exists():
                continue

            # Read and clean text
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            clean_text = clean_report_text(raw_text)

            # Tokenize and move to device
            inputs = tokenizer(
                clean_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False,
            ).to(device)

            # Forward pass
            outputs = model(**inputs)

            # Extract the [CLS] token embedding (first token of the last hidden state)
            # Shape: [1, 768] -> flatten to [768]
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()

            # Append to appropriate split
            if split in splits_data:
                splits_data[split]["embeddings"].append(cls_embedding)
                splits_data[split]["labels"].append(label)
                splits_data[split]["sample_ids"].append(subject_id)

    # 5. Compile and Save Tensors per Architectural Rule 3
    print("\nSaving embedding tensors...")
    for split_name, data in splits_data.items():
        if len(data["sample_ids"]) == 0:
            print(f"Warning: No data found for split '{split_name}'. Skipping.")
            continue

        # Stack list of 1D tensors into a 2D tensor of shape [N, 768]
        stacked_embeddings = torch.stack(data["embeddings"])

        # Prepare final dictionary
        final_dict = {
            "embeddings": stacked_embeddings,
            "labels": data["labels"],
            "sample_ids": data["sample_ids"],
        }

        save_path = output_dir / f"{split_name}_embeddings.pt"
        torch.save(final_dict, save_path)
        print(
            f"Saved {split_name} split: {stacked_embeddings.shape[0]} samples to {save_path.name}"
        )

    print("✅ CXR Text Embedding extraction complete!")


if __name__ == "__main__":
    extract_embeddings()
