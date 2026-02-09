"""Create DB schemas and load CXR/ECG metadata into PostgreSQL."""

import gzip
import logging
from pathlib import Path

from sqlalchemy import text

from src.utils.config import Config
from src.utils.database import get_engine

logger = logging.getLogger(__name__)


def run_ddl_script(engine, file_path: Path):
    """Execute a SQL file containing DDL statements (CREATE TABLE, DROP SCHEMA, etc.)."""
    if not file_path.exists():
        logger.error("Script not found: %s", file_path)
        return

    logger.info("Running DDL: %s", file_path.name)
    with open(file_path, "r", encoding="utf-8") as f:
        sql_content = f.read()

    with engine.connect() as conn:
        conn.execute(text(sql_content))
        conn.commit()


def load_table_from_csv(
    engine, table_name: str, file_path: Path, compressed: bool = False
):
    """Load CSV data into a table using the Postgres COPY protocol.

    Streams data from Python directly to the DB, bypassing file
    permission issues on the server.
    """
    if not file_path.exists():
        logger.warning("Skipping %s — file not found: %s", table_name, file_path)
        return

    logger.info("Loading table %s from %s", table_name, file_path.name)

    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cursor:
            sql = f"COPY {table_name} FROM STDIN WITH CSV HEADER NULL ''"

            if compressed:
                with gzip.open(file_path, "rt", encoding="utf-8") as f:
                    cursor.copy_expert(sql, f)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    cursor.copy_expert(sql, f)

        raw_conn.commit()
        logger.info("Loaded %s", table_name)
    except Exception as e:
        raw_conn.rollback()
        logger.error("Error loading %s: %s", table_name, e)
    finally:
        raw_conn.close()


def main():
    engine = get_engine()
    script_root = Config.PROJECT_ROOT / "scripts" / "db" / "setup"

    # ==========================================
    # PHASE 1: CREATE SCHEMAS (DDL)
    # ==========================================
    logger.info("Phase 1: Creating Schemas...")
    run_ddl_script(engine, script_root / "create_cxr.sql")
    run_ddl_script(engine, script_root / "create_ecg.sql")

    # ==========================================
    # PHASE 2: LOAD CXR DATA
    # ==========================================
    logger.info("Phase 2: Loading CXR Data...")
    cxr_tasks = [
        ("mimiciv_cxr.record_list", Config.RAW_CXR_IMG_DIR / "cxr-record-list.csv.gz"),
        ("mimiciv_cxr.study_list", Config.RAW_CXR_IMG_DIR / "cxr-study-list.csv.gz"),
        (
            "mimiciv_cxr.metadata",
            Config.RAW_CXR_IMG_DIR / "mimic-cxr-2.0.0-metadata.csv.gz",
        ),
    ]

    for table, fpath in cxr_tasks:
        load_table_from_csv(engine, table, fpath, compressed=True)

    # ==========================================
    # PHASE 3: LOAD ECG DATA
    # ==========================================
    logger.info("Phase 3: Loading ECG Data...")
    ecg_tasks = [
        ("mimiciv_ecg.record_list", Config.RAW_ECG_DIR / "record_list.csv"),
        (
            "mimiciv_ecg.machine_measurements",
            Config.RAW_ECG_DIR / "machine_measurements.csv",
        ),
    ]

    for table, fpath in ecg_tasks:
        load_table_from_csv(engine, table, fpath, compressed=False)

    logger.info("Database setup complete.")


if __name__ == "__main__":
    Config.setup_logging()
    main()
