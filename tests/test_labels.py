"""Tests for label files across all modalities.

Validates that the label artifacts produced by the labeler scripts
exist, have correct structure, and are internally consistent.

These are **integration tests** against real pipeline outputs — individual
tests skip when the relevant artifacts have not yet been generated.

Run:
    pytest tests/test_labels.py -v
"""

import pandas as pd
import pytest

from src.utils.config import Config

# ==========================================
# PATHS
# ==========================================

EHR_LABELS_DIR = Config.EHR_LABELS_DIR


# ==========================================
# EHR LABEL TESTS
# ==========================================


class TestEHRLabels:
    """Validate labels.parquet produced by ehr_labels.py."""

    @pytest.fixture(scope="class")
    def labels_df(self):
        path = EHR_LABELS_DIR / "labels.parquet"
        if not path.exists():
            pytest.skip("EHR labels.parquet not found — run ehr_labels first")
        return pd.read_parquet(path)

    def test_required_columns(self, labels_df):
        required = {"subject_id", "prediction_time", "boolean_value"}
        assert required.issubset(labels_df.columns), (
            f"Missing columns: {required - set(labels_df.columns)}"
        )

    def test_not_empty(self, labels_df):
        assert len(labels_df) > 0, "Labels file is empty"

    def test_boolean_value_is_bool(self, labels_df):
        assert labels_df["boolean_value"].dtype == bool, (
            f"Expected bool dtype, got {labels_df['boolean_value'].dtype}"
        )

    def test_subject_ids_are_integers(self, labels_df):
        assert pd.api.types.is_integer_dtype(labels_df["subject_id"]), (
            f"Expected integer subject_id, got {labels_df['subject_id'].dtype}"
        )

    def test_prediction_time_is_datetime(self, labels_df):
        assert pd.api.types.is_datetime64_any_dtype(labels_df["prediction_time"]), (
            f"Expected datetime, got {labels_df['prediction_time'].dtype}"
        )

    def test_no_null_values(self, labels_df):
        nulls = (
            labels_df[["subject_id", "prediction_time", "boolean_value"]].isnull().sum()
        )
        assert nulls.sum() == 0, f"Null values found:\n{nulls[nulls > 0]}"

    def test_has_both_classes(self, labels_df):
        values = labels_df["boolean_value"].unique()
        assert True in values and False in values, (
            f"Expected both classes, got: {values}"
        )
