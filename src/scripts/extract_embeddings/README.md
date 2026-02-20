# Embedding Extraction Scripts

Extract per-subject embedding vectors from frozen foundation models.
Each modality has a dedicated script that reads preprocessed data, runs
inference with the pretrained model, and saves standardised `.pt` files.

## Output Format

All scripts produce the same `.pt` dict per split:

```python
{
    "embeddings": Tensor[N, D],   # float32, one row per subject/sample
    "labels":     List[int],      # 0 or 1
    "sample_ids": List[int],      # subject_id
}
```

Files: `train_embeddings.pt`, `valid_embeddings.pt`, `test_embeddings.pt`

## Scripts

### EHR — `extract_ehr_embeddings.py`

Runs the frozen MOTOR transformer over the MEDS cohort database.

```bash
python -m src.scripts.extract_embeddings.extract_ehr_embeddings
```

- **Input:** Cohort MEDS database (`Config.COHORT_MEDS_READER_DIR`),
  pretrained MOTOR model (`Config.MOTOR_MODEL_DIR`)
- **Output:** `Config.EHR_EMBEDDINGS_DIR/{split}_embeddings.pt`
- **Labels & splits:** Pulled from PostgreSQL (`mimiciv_ext.dataset_splits`)


### ECG — `extract_ecg_embeddings.py`

Runs the frozen ECG-FM model over preprocessed ECG `.mat` files.

```bash
python -m src.scripts.extract_embeddings.extract_ecg_embeddings
```

- **Input:** ECG manifests (`Config.ECG_MANIFEST_DIR`), pretrained ECG-FM model (`Config.ECG_PRETRAINED_MODEL_DIR`)
- **Output:** `Config.ECG_EMBEDDINGS_DIR/{split}_embeddings.pt`

### Inspect Embeddings — `inspect_embeddings.py`

Utility script to inspect, summarize, or debug the generated embedding files for any modality.

```bash
python -m src.scripts.extract_embeddings.inspect_embeddings --file <path_to_embeddings.pt>
```

- **Input:** Path to a `.pt` embeddings file (EHR or ECG)
- **Functionality:** Prints shape, label distribution, and sample IDs; can be extended for further diagnostics.

