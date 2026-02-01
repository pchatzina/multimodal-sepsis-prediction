import pytest
from sqlalchemy import text
from src.utils.database import get_engine


@pytest.fixture(scope="module")
def db_engine():
    """
    Create a single DB connection for all tests in this module.
    We use your utility to ensure consistent configuration.
    """
    engine = get_engine()
    yield engine
    # Clean up connection after tests finish
    engine.dispose()


def test_split_completeness(db_engine):
    """
    Verify that EVERY patient in the cohort has been assigned a split.
    """
    with db_engine.connect() as conn:
        # Count patients in main cohort
        total_cohort = conn.execute(
            text("SELECT COUNT(*) FROM mimiciv_ext.cohort")
        ).scalar()

        # Count patients in splits table
        total_splits = conn.execute(
            text("SELECT COUNT(*) FROM mimiciv_ext.dataset_splits")
        ).scalar()

        # Check for unassigned patients
        unassigned = total_cohort - total_splits

    assert (
        unassigned == 0
    ), f"Found {unassigned} patients who were not assigned to a split!"


def test_no_data_leakage(db_engine):
    """
    CRITICAL: Verify no patient appears in multiple splits (e.g., Train AND Test).
    """
    with db_engine.connect() as conn:
        duplicates = conn.execute(
            text(
                """
            SELECT subject_id, COUNT(DISTINCT dataset_split) 
            FROM mimiciv_ext.dataset_splits 
            GROUP BY subject_id 
            HAVING COUNT(DISTINCT dataset_split) > 1
        """
            )
        ).fetchall()

    assert (
        len(duplicates) == 0
    ), f"DATA LEAKAGE DETECTED: {len(duplicates)} patients are in multiple splits!"


def test_stratification_ratios_global(db_engine):
    """
    Check if the global split roughly matches 70/15/15.
    """
    with db_engine.connect() as conn:
        stats = conn.execute(
            text(
                """
            SELECT dataset_split, COUNT(*) 
            FROM mimiciv_ext.dataset_splits 
            GROUP BY dataset_split
        """
            )
        ).fetchall()

    counts = {row[0]: row[1] for row in stats}
    total = sum(counts.values())

    train_ratio = counts.get("train", 0) / total
    val_ratio = counts.get("validate", 0) / total
    test_ratio = counts.get("test", 0) / total

    print(
        f"Global Ratios -> Train: {train_ratio:.3f}, Val: {val_ratio:.3f}, Test: {test_ratio:.3f}"
    )

    # Allow small margin of error (e.g. +/- 2%) due to rounding in small groups
    assert 0.68 <= train_ratio <= 0.72
    assert 0.13 <= val_ratio <= 0.17
    assert 0.13 <= test_ratio <= 0.17
