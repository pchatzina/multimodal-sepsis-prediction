import pytest
import numpy as np
import scipy.io as sio
from pathlib import Path
from src.utils.config import Config

PREPROCESSED_DIR = Config.ECG_PROCESSED_ROOT_DIR / "preprocessed"


@pytest.fixture(scope="module")
def mat_files():
    """Collect all .mat files from the preprocessed ECG directory."""
    if not PREPROCESSED_DIR.exists():
        pytest.skip(f"Preprocessed ECG dir not found: {PREPROCESSED_DIR}")

    files = [f for f in PREPROCESSED_DIR.iterdir() if f.suffix == ".mat"]
    if not files:
        pytest.skip("No .mat files found in preprocessed ECG directory")

    return files


def test_preprocessed_dir_exists():
    """Verify the preprocessed ECG directory exists."""
    assert PREPROCESSED_DIR.exists(), (
        f"Preprocessed ECG directory not found: {PREPROCESSED_DIR}"
    )


def test_preprocessed_files_count(mat_files):
    """Verify the exact number of preprocessed ECG files matches the cohort (9488)."""
    count = len(mat_files)
    assert count == 9488, f"Expected 9488 preprocessed .mat files, found {count}"


def test_preprocessed_files_have_feats_key(mat_files):
    """Every .mat file should contain a 'feats' key."""
    missing_key = []
    for fpath in mat_files[:50]:  # sample for speed
        try:
            mat = sio.loadmat(fpath)
            if "feats" not in mat:
                missing_key.append(fpath.name)
        except Exception as e:
            pytest.fail(f"Could not read {fpath.name}: {e}")

    assert len(missing_key) == 0, (
        f"{len(missing_key)} files missing 'feats' key: {missing_key[:10]}"
    )


def test_preprocessed_files_shape(mat_files):
    """Verify feats have expected shape: (12, num_samples)."""
    bad_shape = []
    for fpath in mat_files[:50]:  # sample for speed
        try:
            mat = sio.loadmat(fpath)
            if "feats" not in mat:
                continue
            data = mat["feats"]
            if data.shape[0] != 12:
                bad_shape.append((fpath.name, data.shape))
        except Exception as e:
            pytest.fail(f"Could not read {fpath.name}: {e}")

    assert len(bad_shape) == 0, (
        f"{len(bad_shape)} files with unexpected shape (expected 12 leads): {bad_shape[:10]}"
    )


def test_ecg_preprocessed_files_no_nans(mat_files):
    """
    Verify preprocessed ECG .mat files do not contain NaN values.
    NaNs indicate broken leads or preprocessing failures.
    """
    files_with_nans = []
    for fpath in mat_files:
        try:
            mat = sio.loadmat(fpath)
            if "feats" not in mat:
                continue
            data = mat["feats"]
            if np.isnan(data).any():
                nan_leads = []
                if data.shape[0] == 12:
                    nan_leads = [i for i in range(12) if np.isnan(data[i, :]).any()]
                files_with_nans.append((fpath.name, nan_leads))
        except Exception as e:
            pytest.fail(f"Could not read {fpath.name}: {e}")

    assert len(files_with_nans) == 0, (
        f"Found {len(files_with_nans)}/{len(mat_files)} files with NaN values: "
        f"{files_with_nans[:10]}"
    )
