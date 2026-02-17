# Embedding Extraction Scripts

Extract per-subject embedding vectors from frozen foundation models.
Each modality has a dedicated script that reads preprocessed data, runs
inference with the pretrained model, and saves standardised `.pt` files.

## Output Format

All scripts produce the same `.pt` dict per split:

```python
{
    "embeddings": Tensor[N, D],   # float32, one row per subject/sample
    "labels":     List[str],      # "0" or "1"
    "sample_ids": List[str],      # subject_id or study_id as string
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
