import pytest
from pathlib import Path
from sqlalchemy import text
from src.utils.config import Config

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


def test_cxr_files_exist_on_disk(db_engine):
    """
    Sample 10 random CXR paths from DB and verify they exist on disk.
    This confirms download_cxr_files.py worked.
    """
    with db_engine.connect() as conn:
        rows = conn.execute(
            text(
                """
            SELECT study_path FROM mimiciv_ext.cohort_cxr 
            ORDER BY RANDOM() LIMIT 10
        """
            )
        ).fetchall()

    assert len(rows) > 0, "No CXR records found in DB to test."

    missing_files = []
    for (db_path,) in rows:
        # DB path: files/p10/...
        # Config.RAW_CXR_IMG_DIR points to the root of downloads
        full_path = Config.RAW_CXR_IMG_DIR / db_path
        if not full_path.exists():
            missing_files.append(str(full_path))

    assert len(missing_files) == 0, (
        f"Sampled files missing from disk: {missing_files[:3]}..."
    )


def test_ecg_files_exist_on_disk(db_engine):
    """
    Sample 10 random ECG paths from DB and verify .dat/.hea exist.
    """
    with db_engine.connect() as conn:
        rows = conn.execute(
            text(
                """
            SELECT study_path FROM mimiciv_ext.cohort_ecg 
            ORDER BY RANDOM() LIMIT 10
        """
            )
        ).fetchall()

    assert len(rows) > 0, "No ECG records found in DB to test."

    missing_files = []
    for (db_path,) in rows:
        # ECG path in DB often lacks extension or is relative
        # Assuming db_path is like "files/p10/..."

        # Check .hea
        hea_path = Config.RAW_ECG_DIR / f"{db_path}.hea"
        if not hea_path.exists():
            missing_files.append(str(hea_path))

        # Check .dat
        dat_path = Config.RAW_ECG_DIR / f"{db_path}.dat"
        if not dat_path.exists():
            missing_files.append(str(dat_path))

    assert len(missing_files) == 0, f"Sampled ECG files missing: {missing_files[:3]}..."
