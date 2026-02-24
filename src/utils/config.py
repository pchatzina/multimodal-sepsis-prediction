import logging
import os
import random  # Added for seeding
import sys
from pathlib import Path
from dotenv import load_dotenv

import numpy as np  # Added for seeding
import torch  # Added for seeding

# Load environment variables from .env file
load_dotenv()


logger = logging.getLogger(__name__)


class Config:
    # ── BASE PATHS ──────────────────────────────────────────────────────
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    _raw_env = os.getenv("RAW_DATA_DIR")
    _processed_env = os.getenv("PROCESSED_DATA_DIR")
    _models_env = os.getenv("MODELS_DATA_DIR")
    _results_env = os.getenv("RESULTS_DATA_DIR")

    if not all([_raw_env, _processed_env, _models_env, _results_env]):
        print("ERROR: Missing data directories in .env file", file=sys.stderr)
        sys.exit(1)

    # ... (Keep all your existing paths and DB variables here exactly as they were) ...
    # ── DATABASE CREDENTIALS ────────────────────────────────────────────
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "mimiciv")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

    # ── PHYSIONET CREDENTIALS ───────────────────────────────────────────
    PHYSIONET_USER = os.getenv("PHYSIONET_USER")
    PHYSIONET_PASS = os.getenv("PHYSIONET_PASS")

    # ── REMOTE URLS & METADATA ──────────────────────────────────────────
    URL_CXR_BASE = "https://physionet.org/files/mimic-cxr/2.1.0/"
    URL_CXR_JPG_BASE = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/"
    URL_ECG_BASE = "https://physionet.org/files/mimic-iv-ecg/1.0/"

    CXR_METADATA_FILES = [
        ("cxr-record-list.csv.gz", URL_CXR_BASE),
        ("cxr-study-list.csv.gz", URL_CXR_BASE),
        ("mimic-cxr-2.0.0-metadata.csv.gz", URL_CXR_JPG_BASE),
    ]
    CXR_REPORTS_FILE = "mimic-cxr-reports.zip"
    ECG_METADATA_FILES = ["record_list.csv", "machine_measurements.csv"]

    # ── RAW DATA PATHS ──────────────────────────────────────────────────
    # ECG
    RAW_ECG_DIR = Path(_raw_env) / "ecg"
    # CXR
    RAW_CXR_IMG_DIR = Path(_raw_env) / "cxr_img"
    RAW_CXR_TXT_DIR = Path(_raw_env) / "cxr_txt"
    # EHR
    RAW_EHR_COHORT_DIR = Path(_raw_env) / "ehr" / "cohort" / "2.2"
    RAW_EHR_PRETRAINING_DIR = Path(_raw_env) / "ehr" / "pretraining" / "2.2"

    # ── PROCESSED DATA PATHS ────────────────────────────────────────────
    # ECG
    ECG_PROCESSED_ROOT_DIR = Path(_processed_env) / "ecg"
    ECG_LABELS_DIR = Path(_processed_env) / "ecg" / "labels"
    ECG_MANIFEST_DIR = Path(_processed_env) / "ecg" / "manifests"
    ECG_EMBEDDINGS_DIR = Path(_processed_env) / "ecg" / "embeddings"
    # EHR
    PROCESSED_EHR_COHORT_DIR = Path(_processed_env) / "ehr" / "cohort"
    PROCESSED_EHR_PRETRAINING_DIR = Path(_processed_env) / "ehr" / "pretraining"
    COHORT_MEDS_READER_DIR = Path(_processed_env) / "ehr/cohort/mimic-iv-meds-reader"
    PRETRAINING_MEDS_READER_DIR = (
        Path(_processed_env) / "ehr/pretraining/mimic-iv-meds-reader"
    )
    METADATA_MEDS_READER_FILE = (
        Path(_processed_env) / "ehr/pretraining/mimic-iv-meds/metadata/codes.parquet"
    )
    EHR_LABELS_DIR = Path(_processed_env) / "ehr/labels"
    EHR_EMBEDDINGS_DIR = Path(_processed_env) / "ehr" / "embeddings"
    # CXR Images
    CXR_IMG_EMBEDDINGS_DIR = Path(_processed_env) / "cxr_img" / "embeddings"

    # ── MODEL ARTIFACT PATHS ────────────────────────────────────────────
    # ECG
    ECG_PRETRAINED_MODEL_DIR = Path(_models_env) / "ecg/pretrained"
    ECG_XGBOOST_MODEL_DIR = Path(_models_env) / "ecg/xgboost"
    ECG_LR_MODEL_DIR = Path(_models_env) / "ecg/lr"
    ECG_MLP_MODEL_DIR = Path(_models_env) / "ecg/mlp"
    # EHR — MOTOR foundation model
    ATHENA_VOCABULARY_DIR = Path(_models_env) / "ehr/motor/athena_vocabulary"
    MOTOR_PRETRAINING_FILES_DIR = Path(_models_env) / "ehr/motor/pretraining_files"
    MOTOR_MODEL_OUTPUT_DIR = Path(_models_env) / "ehr/motor/pretraining_output"
    MOTOR_MODEL_DIR = Path(_models_env) / "ehr/motor/model"
    # EHR — downstream classifiers
    EHR_LR_MODEL_DIR = Path(_models_env) / "ehr/lr"
    EHR_XGBOOST_MODEL_DIR = Path(_models_env) / "ehr/xgboost"
    EHR_MLP_MODEL_DIR = Path(_models_env) / "ehr/mlp"
    # CXR Images
    CXR_IMG_PRETRAINED_MODEL_DIR = Path(_models_env) / "cxr_img/pretrained"
    CXR_IMG_XGBOOST_MODEL_DIR = Path(_models_env) / "cxr_img/xgboost"
    CXR_IMG_LR_MODEL_DIR = Path(_models_env) / "cxr_img/lr"
    CXR_IMG_MLP_MODEL_DIR = Path(_models_env) / "cxr_img/mlp"

    # ── RESULTS & TENSORBOARD ───────────────────────────────────────────
    RESULTS_DIR = Path(_results_env)
    TENSORBOARD_LOG_DIR = Path(_results_env) / "tensorboard"
    REPORTS_DIR = PROJECT_ROOT / "reports"

    # ── METHODS ─────────────────────────────────────────────────────────
    @classmethod
    def check_dirs(cls):
        """Creates essential directories if they don't exist."""
        paths_to_create = [
            cls.RAW_ECG_DIR,
            cls.RAW_CXR_IMG_DIR,
            cls.RAW_CXR_TXT_DIR,
            cls.MOTOR_PRETRAINING_FILES_DIR,
            cls.MOTOR_MODEL_OUTPUT_DIR,
            cls.MOTOR_MODEL_DIR,
            cls.ECG_EMBEDDINGS_DIR,
            cls.EHR_EMBEDDINGS_DIR,
            cls.ATHENA_VOCABULARY_DIR,
            cls.RESULTS_DIR,
            cls.REPORTS_DIR,
            cls.EHR_LABELS_DIR,
            cls.ECG_PRETRAINED_MODEL_DIR,
            cls.ECG_MLP_MODEL_DIR,
            cls.CXR_IMG_EMBEDDINGS_DIR,
            cls.CXR_IMG_PRETRAINED_MODEL_DIR,
            cls.CXR_IMG_XGBOOST_MODEL_DIR,
            cls.CXR_IMG_LR_MODEL_DIR,
            cls.CXR_IMG_MLP_MODEL_DIR,
        ]

        for path in paths_to_create:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                logger.info("Created directory: %s", path)

    @classmethod
    def get_db_url(cls):
        """Returns the connection URL for SQLAlchemy/Pandas."""
        return f"postgresql://{cls.DB_USER}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"

    @staticmethod
    def setup_logging(level: int = logging.INFO) -> None:
        """Configure root logger with the project-wide format."""
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    # ── ADDED FOR REPRODUCIBILITY ───────────────────────────────────────
    @staticmethod
    def set_seed(seed: int = 42) -> None:
        """Locks all random seeds for strict reproducibility across the project."""
        logger.info(f"Setting global random seed to {seed}...")

        # 1. Core Python
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)

        # 2. NumPy (Affects Scikit-Learn Logistic Regression & Data splits)
        np.random.seed(seed)

        # 3. PyTorch (Affects MLPs and Foundation Model embeddings)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
