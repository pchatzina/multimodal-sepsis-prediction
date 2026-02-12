import pytest
from pathlib import Path
from sqlalchemy import text
from src.utils.config import Config

import zipfile
from PIL import Image
import wfdb

# --- FIXTURES ---


@pytest.fixture(scope="module")
def cohort_stats(db_engine):
    """Fetch basic stats once to use in multiple tests."""
    with db_engine.connect() as conn:
        result = conn.execute(
            text(
                """
            SELECT 
                COUNT(*) as total,
                SUM(sepsis_label) as positives
            FROM mimiciv_ext.cohort
        """
            )
        ).fetchone()
    return result


# --- DATABASE INTEGRITY TESTS ---


def test_db_connection(db_engine):
    """Can we connect to the database?"""
    with db_engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).scalar()
    assert result == 1


def test_final_cohort_exists_and_has_data(cohort_stats):
    """
    Does the final cohort table exist and have a reasonable number of patients?
    Thesis: Total cases ~15,513.
    """
    total = cohort_stats.total
    assert total > 0, "Cohort table is empty!"

    # We use a loose lower bound in case you are working with a subset,
    # but for the full dataset, this should pass.
    if total < 1000:
        pytest.warns(
            UserWarning, match=f"Cohort size ({total}) is smaller than expected (~15k)."
        )


def test_class_balance(cohort_stats):
    """
    Is the class balance roughly correct?
    Thesis: ~54% positive.
    """
    total = cohort_stats.total
    positives = cohort_stats.positives
    positive_rate = positives / total

    # Check if rate is within a reasonable margin of 0.54 (e.g., 0.40 to 0.70)
    assert 0.40 <= positive_rate <= 0.70, (
        f"Unexpected positive rate: {positive_rate:.2f}"
    )


def test_one_admission_per_patient(db_engine):
    """
    Thesis: 'kept a single unique admission per patient'.
    """
    with db_engine.connect() as conn:
        result = conn.execute(
            text(
                """
            SELECT subject_id, COUNT(hadm_id) 
            FROM mimiciv_ext.cohort 
            GROUP BY subject_id 
            HAVING COUNT(hadm_id) > 1
        """
            )
        ).fetchall()
    assert len(result) == 0, f"Found {len(result)} patients with multiple admissions!"


# --- MODALITY TESTS ---


def test_cxr_cohort_integrity(db_engine):
    """
    Verify CXR cohort links correctly and follows rules.
    Thesis: 'Only one image... PA > AP'.
    """
    with db_engine.connect() as conn:
        # Check 1: Are there orphans? (CXR without valid patient)
        orphans = conn.execute(
            text(
                """
            SELECT COUNT(*) FROM mimiciv_ext.cohort_cxr c
            LEFT JOIN mimiciv_ext.cohort p ON c.subject_id = p.subject_id
            WHERE p.subject_id IS NULL
        """
            )
        ).scalar()
        assert orphans == 0, "Found CXR records linked to non-existent patients!"

        # Check 2: Uniqueness (One study per patient)
        duplicates = conn.execute(
            text(
                """
            SELECT subject_id FROM mimiciv_ext.cohort_cxr
            GROUP BY subject_id HAVING COUNT(*) > 1
        """
            )
        ).fetchall()
        assert len(duplicates) == 0, "Found patients with multiple CXR entries!"


def test_ecg_cohort_integrity(db_engine):
    """Verify ECG cohort links correctly."""
    with db_engine.connect() as conn:
        orphans = conn.execute(
            text(
                """
            SELECT COUNT(*) FROM mimiciv_ext.cohort_ecg e
            LEFT JOIN mimiciv_ext.cohort p ON e.subject_id = p.subject_id
            WHERE p.subject_id IS NULL
        """
            )
        ).scalar()
        assert orphans == 0, "Found ECG records linked to non-existent patients!"


# --- FILE SYSTEM TESTS ---


@pytest.fixture(scope="module")
def cxr_paths(db_engine):
    """Fetch all CXR study paths once."""
    with db_engine.connect() as conn:
        rows = conn.execute(
            text("SELECT study_path FROM mimiciv_ext.cohort_cxr")
        ).fetchall()
    return [row[0] for row in rows]


@pytest.fixture(scope="module")
def ecg_paths(db_engine):
    """Fetch all ECG study paths once."""
    with db_engine.connect() as conn:
        rows = conn.execute(
            text("SELECT study_path FROM mimiciv_ext.cohort_ecg")
        ).fetchall()
    return [row[0] for row in rows]


def test_cxr_dir_exists():
    """Verify the CXR image directory exists."""
    assert Config.RAW_CXR_IMG_DIR.is_dir(), (
        f"CXR image directory not found: {Config.RAW_CXR_IMG_DIR}"
    )


def test_ecg_dir_exists():
    """Verify the ECG directory exists."""
    assert Config.RAW_ECG_DIR.is_dir(), f"ECG directory not found: {Config.RAW_ECG_DIR}"


def test_cxr_files_on_disk(cxr_paths):
    """Verify all CXR files exist and are non-empty."""
    assert len(cxr_paths) > 0, "No CXR records found in DB."

    missing, empty = [], []
    for db_path in cxr_paths:
        full_path = Config.RAW_CXR_IMG_DIR / db_path
        if not full_path.exists():
            missing.append(str(full_path))
        elif full_path.stat().st_size == 0:
            empty.append(str(full_path))

    assert len(missing) == 0, f"{len(missing)} missing: {missing[:3]}..."
    assert len(empty) == 0, f"{len(empty)} are 0-byte: {empty[:3]}..."


def test_ecg_files_on_disk(ecg_paths):
    """Verify all ECG .hea/.dat files exist and are non-empty."""
    assert len(ecg_paths) > 0, "No ECG records found in DB."

    missing, empty = [], []
    for db_path in ecg_paths:
        for ext in (".hea", ".dat"):
            fpath = Config.RAW_ECG_DIR / f"{db_path}{ext}"
            if not fpath.exists():
                missing.append(str(fpath))
            elif fpath.stat().st_size == 0:
                empty.append(str(fpath))

    assert len(missing) == 0, f"{len(missing)} missing: {missing[:3]}..."
    assert len(empty) == 0, f"{len(empty)} are 0-byte: {empty[:3]}..."


def test_cxr_files_not_corrupted(cxr_paths):
    """Verify all CXR images can be opened/decoded."""
    corrupted = []
    for db_path in cxr_paths:
        full_path = Config.RAW_CXR_IMG_DIR / db_path
        if not full_path.exists():
            continue
        if full_path.stat().st_size == 0:
            corrupted.append((str(full_path), "0-byte file"))
            continue
        try:
            with Image.open(full_path) as img:
                img.verify()
        except Exception as e:
            corrupted.append((str(full_path), str(e)))

    assert len(corrupted) == 0, (
        f"Found {len(corrupted)} corrupted CXR files: {corrupted[:3]}"
    )


def test_ecg_files_not_corrupted(ecg_paths):
    """Verify all ECG records can be parsed by wfdb."""
    corrupted = []
    for db_path in ecg_paths:
        record_path = Config.RAW_ECG_DIR / db_path
        hea_path = record_path.with_suffix(".hea")
        dat_path = record_path.with_suffix(".dat")
        if not hea_path.exists() or not dat_path.exists():
            continue
        if hea_path.stat().st_size == 0:
            corrupted.append((str(hea_path), "0-byte file"))
            continue
        if dat_path.stat().st_size == 0:
            corrupted.append((str(dat_path), "0-byte file"))
            continue
        try:
            wfdb.rdrecord(str(record_path))
        except Exception as e:
            corrupted.append((str(record_path), str(e)))

    assert len(corrupted) == 0, (
        f"Found {len(corrupted)} corrupted ECG records: {corrupted[:3]}"
    )


def test_cxr_reports_zip_not_corrupted():
    """
    Verify the CXR reports ZIP is valid and all entries pass CRC checks.
    """
    reports_path = Config.RAW_CXR_TXT_DIR / Config.CXR_REPORTS_FILE
    if not reports_path.exists():
        pytest.skip("CXR reports file not found — skipping corruption check.")

    assert zipfile.is_zipfile(reports_path), f"{reports_path} is not a valid ZIP file"

    with zipfile.ZipFile(reports_path, "r") as zf:
        bad_file = zf.testzip()  # returns first bad filename or None
        assert bad_file is None, f"Corrupted entry in CXR reports ZIP: {bad_file}"
