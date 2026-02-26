"""Tests for the Late-Fusion Sepsis Prediction pipeline.

Validates:
- Optuna tuning JSON artifacts exist and are well-formed.
- Fusion model (.pt) files exist for all 4 experiment variants.
- Result files (test/val metrics, predictions) are consistent.

Run:
    pytest tests/test_fusion.py -v
"""

import json
import pandas as pd
import pytest

from src.utils.config import Config

# The 4 experimental variants we ran
VARIANTS = [
    "scratch_no_dropout",
    "scratch_ehr_dropout",
    "pretrained_no_dropout",
    "pretrained_ehr_dropout",
]

# The 2 tuning modes
TUNING_MODES = ["scratch", "pretrained"]


# ==========================================
# TUNING TESTS
# ==========================================


@pytest.mark.parametrize("mode", TUNING_MODES)
def test_fusion_tuning_files(mode):
    tuning_dir = Config.RESULTS_DIR / "fusion" / "tuning"
    if not tuning_dir.exists():
        pytest.skip("Fusion tuning directory not found.")

    # FIX: Use an empty string for the scratch mode, or "_pretrained" for pretrained
    suffix = "_pretrained" if mode == "pretrained" else ""
    path = tuning_dir / f"best_hyperparameters{suffix}.json"

    assert path.exists(), f"Missing tuning file: {path}"

    with open(path) as f:
        data = json.load(f)

    assert "best_val_auroc" in data
    assert "params" in data
    assert len(data["params"]) > 0


# ==========================================
# MODEL ARTIFACT TESTS
# ==========================================
@pytest.mark.parametrize("variant", VARIANTS)
def test_fusion_model_artifacts(variant):
    model_dir = Config.FUSION_MODEL_DIR
    if not model_dir.exists():
        pytest.skip("Fusion model directory not found.")

    path = model_dir / f"best_late_fusion_model_{variant}.pt"
    assert path.exists(), f"Missing model artifact: {path}"
    assert path.stat().st_size > 0


# ==========================================
# METRICS & PREDICTIONS TESTS
# ==========================================
@pytest.mark.parametrize("variant", VARIANTS)
def test_fusion_metrics_json(variant):
    results_dir = Config.RESULTS_DIR / "fusion"
    if not results_dir.exists():
        pytest.skip("Fusion results directory not found.")

    # 1. Test validation metrics
    val_path = results_dir / f"val_metrics_fusion_{variant}.json"
    assert val_path.exists(), f"Missing val metrics: {val_path}"

    # 2. Test test metrics
    test_path = results_dir / f"test_metrics_fusion_{variant}.json"
    assert test_path.exists(), f"Missing test metrics: {test_path}"

    with open(test_path) as f:
        m = json.load(f)

    assert "auroc" in m, "AUROC missing from metrics"
    assert "auprc" in m, "AUPRC missing from metrics"
    assert 0.0 <= m["auroc"] <= 1.0, "AUROC out of bounds"


@pytest.mark.parametrize("variant", VARIANTS)
def test_fusion_predictions_csv(variant):
    results_dir = Config.RESULTS_DIR / "fusion"
    if not results_dir.exists():
        pytest.skip("Fusion results directory not found.")

    path = results_dir / f"test_predictions_fusion_{variant}.csv"
    assert path.exists(), f"Missing predictions CSV: {path}"

    df = pd.read_csv(path)
    assert "sample_id" in df.columns
    assert "label" in df.columns
    assert "probability" in df.columns
    assert len(df) > 0, "Prediction CSV is empty"
