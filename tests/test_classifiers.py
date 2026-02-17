"""Tests for downstream classifiers across all modalities.

Validates:
- Output artifacts (model files, scalers) exist.
- Result files (metrics, predictions) are consistent.

These are **integration tests** against real pipeline outputs. Individual
tests skip when the relevant artifacts have not yet been generated.

Run:
    pytest tests/test_classifiers.py -v
"""

import json
import pandas as pd
import pytest

from src.utils.config import Config


# ==========================================
# BASE ARTIFACT TESTS
# ==========================================


class _BaseLRArtifacts:
    """Verify Logistic Regression artifacts (model + scaler)."""

    MODEL_DIR = None

    def test_model_file(self):
        if self.MODEL_DIR is None or not self.MODEL_DIR.exists():
            pytest.skip(f"LR directory not found for {self.__class__.__name__}")
        path = self.MODEL_DIR / "model.pkl"
        assert path.exists(), f"Missing: {path}"
        assert path.stat().st_size > 0

    def test_scaler_file(self):
        if self.MODEL_DIR is None or not self.MODEL_DIR.exists():
            pytest.skip(f"LR directory not found for {self.__class__.__name__}")
        path = self.MODEL_DIR / "scaler.pkl"
        assert path.exists(), f"Missing: {path}"


class _BaseXGBoostArtifacts:
    """Verify XGBoost artifacts (JSON model)."""

    MODEL_DIR = None

    def test_model_file(self):
        if self.MODEL_DIR is None or not self.MODEL_DIR.exists():
            pytest.skip(f"XGBoost directory not found for {self.__class__.__name__}")
        path = self.MODEL_DIR / "model.json"
        assert path.exists(), f"Missing: {path}"
        assert path.stat().st_size > 0


class _BaseMLPArtifacts:
    """Verify PyTorch MLP artifacts (.pt model)."""

    MODEL_DIR = None

    def test_model_file(self):
        if self.MODEL_DIR is None or not self.MODEL_DIR.exists():
            pytest.skip(f"MLP directory not found for {self.__class__.__name__}")
        path = self.MODEL_DIR / "model.pt"
        assert path.exists(), f"Missing: {path}"
        assert path.stat().st_size > 0


# ==========================================
# BASE RESULTS TESTS
# ==========================================


class _BaseResults:
    """Validate saved result files (JSON metrics, CSV predictions)."""

    RESULTS_DIR = None

    def test_test_metrics_json(self):
        if self.RESULTS_DIR is None or not self.RESULTS_DIR.exists():
            pytest.skip(f"Results directory not found for {self.__class__.__name__}")
        path = self.RESULTS_DIR / "test_metrics.json"
        assert path.exists(), f"Missing: {path}"
        with open(path) as f:
            m = json.load(f)
        assert "auroc" in m
        assert "auprc" in m
        assert 0.0 <= m["auroc"] <= 1.0

    def test_val_metrics_json(self):
        if self.RESULTS_DIR is None or not self.RESULTS_DIR.exists():
            pytest.skip(f"Results directory not found for {self.__class__.__name__}")
        path = self.RESULTS_DIR / "val_metrics.json"
        assert path.exists(), f"Missing: {path}"

    def test_predictions_csv(self):
        if self.RESULTS_DIR is None or not self.RESULTS_DIR.exists():
            pytest.skip(f"Results directory not found for {self.__class__.__name__}")
        path = self.RESULTS_DIR / "test_predictions.csv"
        assert path.exists(), f"Missing: {path}"

        df = pd.read_csv(path)
        assert "sample_id" in df.columns
        assert "label" in df.columns
        assert "probability" in df.columns
        assert len(df) > 0


# ==========================================
# EHR CLASSIFIERS
# ==========================================


class TestEHRLRArtifacts(_BaseLRArtifacts):
    MODEL_DIR = Config.EHR_LR_MODEL_DIR


class TestEHRXGBoostArtifacts(_BaseXGBoostArtifacts):
    MODEL_DIR = Config.EHR_XGBOOST_MODEL_DIR


class TestEHRMLPArtifacts(_BaseMLPArtifacts):
    MODEL_DIR = Config.EHR_MLP_MODEL_DIR


class TestEHRLRResults(_BaseResults):
    RESULTS_DIR = Config.RESULTS_DIR / "ehr" / "lr"


class TestEHRXGBoostResults(_BaseResults):
    RESULTS_DIR = Config.RESULTS_DIR / "ehr" / "xgboost"


class TestEHRMLPResults(_BaseResults):
    RESULTS_DIR = Config.RESULTS_DIR / "ehr" / "mlp"


# ==========================================
# ECG CLASSIFIERS (Placeholders)
# ==========================================


class TestECGLRArtifacts(_BaseLRArtifacts):
    MODEL_DIR = getattr(Config, "ECG_LR_MODEL_DIR", None)


class TestECGXGBoostArtifacts(_BaseXGBoostArtifacts):
    MODEL_DIR = getattr(Config, "ECG_XGBOOST_MODEL_DIR", None)


# Add TestECGMLPArtifacts here if you decide to train an MLP for ECG!


class TestECGLRResults(_BaseResults):
    RESULTS_DIR = (
        Config.RESULTS_DIR / "ecg" / "lr" if hasattr(Config, "RESULTS_DIR") else None
    )


class TestECGXGBoostResults(_BaseResults):
    RESULTS_DIR = (
        Config.RESULTS_DIR / "ecg" / "xgboost"
        if hasattr(Config, "RESULTS_DIR")
        else None
    )
