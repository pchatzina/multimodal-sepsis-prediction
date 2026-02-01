import sys
import subprocess
from sqlalchemy import text
from src.utils.config import Config
from src.utils.download import download_with_wget
from src.utils.database import get_engine


def main():
    print(">>> Starting ECG Record Download for Cohort...")
    engine = get_engine()

    query = text("SELECT subject_id, study_id, study_path FROM mimiciv_ext.cohort_ecg")

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    total = len(rows)
    print(f"Found {total} ECG records to download.")

    for i, (sub_id, _, db_path) in enumerate(rows, 1):
        # Download both .dat and .hea files
        for ext in [".dat", ".hea"]:
            url = f"{Config.URL_ECG_BASE}{db_path}{ext}"
            local_file = Config.RAW_ECG_DIR / f"{db_path}{ext}"

            try:
                download_with_wget(
                    url, local_file, Config.PHYSIONET_USER, Config.PHYSIONET_PASS
                )
            except subprocess.CalledProcessError:
                print(
                    f"ERROR: Failed to download ECG part {ext} for Subject {sub_id}",
                    file=sys.stderr,
                )

        if i % 50 == 0:
            print(f"[{i}/{total}] Processed ECG for Subject {sub_id}")

    print(">>> ECG Download Complete.")


if __name__ == "__main__":
    main()
