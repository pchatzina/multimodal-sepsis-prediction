import gzip
from pathlib import Path
from sqlalchemy import text
from src.utils.config import Config
from src.utils.database import get_engine


def run_ddl_script(engine, file_path: Path):
    """
    Executes a SQL file containing DDL (CREATE TABLE, DROP SCHEMA, etc.).
    """
    if not file_path.exists():
        print(f"ERROR: Script not found: {file_path}")
        return

    print(f"--- Running DDL: {file_path.name} ---")
    with open(file_path, "r", encoding="utf-8") as f:
        sql_content = f.read()

    # Split by semicolon to handle multiple statements if needed,
    # but usually execute() handles blocks fine if they are valid SQL.
    with engine.connect() as conn:
        conn.execute(text(sql_content))
        conn.commit()


def load_table_from_csv(
    engine, table_name: str, file_path: Path, compressed: bool = False
):
    """
    High-performance data loading using Postgres COPY protocol.
    Streams data from Python directly to the DB, bypassing file permission issues on the server.
    """
    if not file_path.exists():
        print(f"SKIPPING: {table_name} - File not found: {file_path}")
        return

    print(f"--- Loading Table: {table_name} from {file_path.name} ---")

    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cursor:
            # Prepare the COPY statement
            # FROM STDIN tells Postgres to read from the data stream we send
            sql = f"COPY {table_name} FROM STDIN WITH CSV HEADER NULL ''"

            if compressed:
                with gzip.open(file_path, "rt", encoding="utf-8") as f:
                    cursor.copy_expert(sql, f)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    cursor.copy_expert(sql, f)

        raw_conn.commit()
        print(f"SUCCESS: Loaded {table_name}")
    except Exception as e:
        raw_conn.rollback()
        print(f"ERROR loading {table_name}: {e}")
    finally:
        raw_conn.close()


def main():
    engine = get_engine()
    script_root = Config.PROJECT_ROOT / "scripts" / "db" / "setup"

    # ==========================================
    # PHASE 1: CREATE SCHEMAS (DDL)
    # ==========================================
    print("\n>>> Phase 1: Creating Schemas...")
    run_ddl_script(engine, script_root / "create_cxr.sql")
    run_ddl_script(engine, script_root / "create_ecg.sql")

    # ==========================================
    # PHASE 2: LOAD CXR DATA
    # ==========================================
    print("\n>>> Phase 2: Loading CXR Data...")
    # Map Table Name -> File Name
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
    print("\n>>> Phase 3: Loading ECG Data...")
    # Map Table Name -> File Name
    ecg_tasks = [
        ("mimiciv_ecg.record_list", Config.RAW_ECG_DIR / "record_list.csv"),
        (
            "mimiciv_ecg.machine_measurements",
            Config.RAW_ECG_DIR / "machine_measurements.csv",
        ),
    ]

    for table, fpath in ecg_tasks:
        load_table_from_csv(engine, table, fpath, compressed=False)

    print("\n>>> Database setup complete.")


if __name__ == "__main__":
    main()
