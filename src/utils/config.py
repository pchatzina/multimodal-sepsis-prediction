import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    # --- BASE PATHS ---
    # Points to the root of the project
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    # Environment Variables
    _raw_env = os.getenv("RAW_DATA_DIR")
    _processed_env = os.getenv("PROCESSED_DATA_DIR")
    _models_env = os.getenv("MODELS_DATA_DIR")

    # Validation: Ensure critical env vars exist
    if not all([_raw_env, _processed_env, _models_env]):
        print(
            "ERROR: Missing data directories in .env file (RAW/PROCESSED/MODELS)",
            file=sys.stderr,
        )

    # --- DATABASE CREDENTIALS ---
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "mimiciv")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

    # --- PHYSIONET CREDENTIALS ---
    PHYSIONET_USER = os.getenv("PHYSIONET_USER")
    PHYSIONET_PASS = os.getenv("PHYSIONET_PASS")

    # --- REMOTE URLS ---
    URL_CXR_BASE = "https://physionet.org/files/mimic-cxr/2.1.0/"
    URL_CXR_JPG_BASE = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/"
    URL_ECG_BASE = "https://physionet.org/files/mimic-iv-ecg/1.0/"

    # --- METADATA FILENAMES ---
    CXR_METADATA_FILES = [
        ("cxr-record-list.csv.gz", URL_CXR_BASE),
        ("cxr-study-list.csv.gz", URL_CXR_BASE),
        ("mimic-cxr-2.0.0-metadata.csv.gz", URL_CXR_JPG_BASE),
    ]
    ECG_METADATA_FILES = ["record_list.csv", "machine_measurements.csv"]

    # --- LOCAL DATA PATHS ---
    RAW_ECG_DIR = Path(_raw_env) / "ecg"
    RAW_CXR_IMG_DIR = Path(_raw_env) / "cxr_img"
    RAW_CXR_TXT_DIR = Path(_raw_env) / "cxr_txt"
    MODELS_DIR = Path(_models_env)

    # --- PROCESSED / FEATURES PATHS ---
    MOTOR_FEATURES_DIR = Path(_processed_env) / "ehr/features"

    @classmethod
    def check_dirs(cls):
        """
        Creates essential directories if they don't exist.
        Should be called at the start of data acquisition scripts.
        """
        paths_to_create = [
            cls.RAW_ECG_DIR,
            cls.RAW_CXR_IMG_DIR,
            cls.RAW_CXR_TXT_DIR,
            cls.MOTOR_FEATURES_DIR,
        ]

        for path in paths_to_create:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                print(f"Created directory: {path}")

    @classmethod
    def get_db_url(cls):
        """Returns the connection URL for SQLAlchemy/Pandas."""
        return f"postgresql://{cls.DB_USER}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
