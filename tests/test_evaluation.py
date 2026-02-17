"""Tests for the shared evaluation utilities.

Validates ``src.utils.evaluation`` helpers with synthetic data —
no external dependencies (DB, GPU, .pt files) required.

Run:
    pytest tests/test_evaluation.py -v
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils.evaluation import (
    compute_metrics,
    load_embeddings,
    print_metrics,
    save_metrics,
    save_predictions,
)


# ==========================================
# FIXTURES
# ==========================================


@pytest.fixture()
def perfect_predictions():
    """Predictions where the model is perfect."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    return y_true, y_prob


@pytest.fixture()
def random_predictions(seed=42):
    """Realistic-sized random predictions."""
    rng = np.random.RandomState(seed)
    n = 500
    y_true = rng.randint(0, 2, size=n)
    y_prob = rng.rand(n)
    return y_true, y_prob


@pytest.fixture()
def sample_pt_file(tmp_path):
    """Create a temporary .pt embedding file."""
    n, d = 50, 128
    data = {
        "embeddings": torch.randn(n, d),
        "labels": [str(i % 2) for i in range(n)],
        "sample_ids": [str(1000 + i) for i in range(n)],
    }
    path = tmp_path / "test_embeddings.pt"
    torch.save(data, path)
    return path, n, d


# ==========================================
# compute_metrics
# ==========================================


class TestComputeMetrics:
    """Verify compute_metrics returns correct structure and values."""

    EXPECTED_KEYS = {
        "auroc",
        "auprc",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "specificity",
        "tp",
        "fp",
        "tn",
        "fn",
        "threshold",
        "n_samples",
        "n_positive",
        "n_negative",
    }

    def test_returns_all_keys(self, perfect_predictions):
        y_true, y_prob = perfect_predictions
        m = compute_metrics(y_true, y_prob)
        assert set(m.keys()) == self.EXPECTED_KEYS

    def test_perfect_predictions(self, perfect_predictions):
        y_true, y_prob = perfect_predictions
        m = compute_metrics(y_true, y_prob)
        assert m["auroc"] == pytest.approx(1.0)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["specificity"] == pytest.approx(1.0)

    def test_sample_counts(self, perfect_predictions):
        y_true, y_prob = perfect_predictions
        m = compute_metrics(y_true, y_prob)
        assert m["n_samples"] == 6
        assert m["n_positive"] == 3
        assert m["n_negative"] == 3
        assert m["tp"] + m["fn"] == m["n_positive"]
        assert m["tn"] + m["fp"] == m["n_negative"]

    def test_metrics_are_bounded(self, random_predictions):
        y_true, y_prob = random_predictions
        m = compute_metrics(y_true, y_prob)
        for key in (
            "auroc",
            "auprc",
            "accuracy",
            "f1",
            "precision",
            "recall",
            "specificity",
        ):
            assert 0.0 <= m[key] <= 1.0, f"{key} out of [0, 1]"

    def test_custom_threshold(self, random_predictions):
        y_true, y_prob = random_predictions
        m_low = compute_metrics(y_true, y_prob, threshold=0.3)
        m_high = compute_metrics(y_true, y_prob, threshold=0.7)
        assert m_low["threshold"] == pytest.approx(0.3)
        assert m_high["threshold"] == pytest.approx(0.7)
        # Lower threshold → more positives → higher recall
        assert m_low["recall"] >= m_high["recall"]

    def test_all_values_json_serialisable(self, random_predictions):
        y_true, y_prob = random_predictions
        m = compute_metrics(y_true, y_prob)
        # Should not raise
        json.dumps(m)


# ==========================================
# load_embeddings
# ==========================================


class TestLoadEmbeddings:
    """Verify load_embeddings reads .pt files correctly."""

    def test_shapes_and_types(self, sample_pt_file):
        path, n, d = sample_pt_file
        X, y, ids = load_embeddings(path)
        assert X.shape == (n, d)
        assert X.dtype == np.float32
        assert y.shape == (n,)
        assert y.dtype == int
        assert len(ids) == n

    def test_labels_are_binary(self, sample_pt_file):
        path, _, _ = sample_pt_file
        _, y, _ = load_embeddings(path)
        assert set(np.unique(y)) <= {0, 1}

    def test_sample_ids_are_strings(self, sample_pt_file):
        path, _, _ = sample_pt_file
        _, _, ids = load_embeddings(path)
        assert all(isinstance(s, str) for s in ids)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_embeddings(tmp_path / "nonexistent.pt")


# ==========================================
# save_predictions
# ==========================================


class TestSavePredictions:
    """Verify predictions CSV is written correctly."""

    def test_csv_roundtrip(self, tmp_path, random_predictions):
        y_true, y_prob = random_predictions
        ids = [str(i) for i in range(len(y_true))]
        path = tmp_path / "preds.csv"
        save_predictions(path, ids, y_true, y_prob)

        import pandas as pd

        df = pd.read_csv(path)
        assert len(df) == len(y_true)
        assert set(df.columns) == {"sample_id", "label", "probability", "prediction"}
        np.testing.assert_array_equal(df["label"].values, y_true)

    def test_creates_parent_dirs(self, tmp_path, random_predictions):
        y_true, y_prob = random_predictions
        ids = [str(i) for i in range(len(y_true))]
        path = tmp_path / "deep" / "nested" / "preds.csv"
        save_predictions(path, ids, y_true, y_prob)
        assert path.exists()


# ==========================================
# save_metrics
# ==========================================


class TestSaveMetrics:
    """Verify metrics JSON is written correctly."""

    def test_json_roundtrip(self, tmp_path, random_predictions):
        y_true, y_prob = random_predictions
        m = compute_metrics(y_true, y_prob)
        path = tmp_path / "metrics.json"
        save_metrics(path, m)

        with open(path) as f:
            loaded = json.load(f)
        assert loaded == pytest.approx(m)


# ==========================================
# print_metrics (smoke test)
# ==========================================


class TestPrintMetrics:
    """Verify print_metrics runs without error."""

    def test_smoke(self, capsys, random_predictions):
        y_true, y_prob = random_predictions
        m = compute_metrics(y_true, y_prob)
        print_metrics(m, "Smoke Test")
        captured = capsys.readouterr()
        assert "AUROC" in captured.out
        assert "Smoke Test" in captured.out
