import sys
import subprocess
from sqlalchemy import text
from src.utils.config import Config
from src.utils.download import download_with_wget
from src.utils.database import get_engine


def main():
    print(">>> Starting CXR Image Download for Cohort...")
    engine = get_engine()

    query = text("SELECT subject_id, study_id, study_path FROM mimiciv_ext.cohort_cxr")

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    total = len(rows)
    print(f"Found {total} CXR images to download.")

    for i, (sub_id, _, db_path) in enumerate(rows, 1):
        url = f"{Config.URL_CXR_JPG_BASE}{db_path}"
        local_file = Config.RAW_CXR_IMG_DIR / db_path

        try:
            download_with_wget(
                url, local_file, Config.PHYSIONET_USER, Config.PHYSIONET_PASS
            )
        except subprocess.CalledProcessError:
            print(
                f"ERROR: Failed to download CXR for Subject {sub_id} ({url})",
                file=sys.stderr,
            )

        if i % 50 == 0:
            print(f"[{i}/{total}] Downloaded CXR for Subject {sub_id}")

    print(">>> CXR Download Complete.")


if __name__ == "__main__":
    main()
