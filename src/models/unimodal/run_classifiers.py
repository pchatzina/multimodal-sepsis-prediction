"""
Run all unimodal classifier training scripts sequentially, or only those for a specific modality.

Usage:
    python -m src.models.unimodal.run_classifiers
    python -m src.models.unimodal.run_classifiers --modality ehr
    python -m src.models.unimodal.run_classifiers --modality ecg
"""

import subprocess
import logging
from pathlib import Path
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_classifiers")

# Root directory for classifier scripts
UNIMODAL_DIR = Path(__file__).parent


# Find all train_*.py scripts in subfolders (e.g., ecg, ehr), or only in a specific modality if given
def find_classifier_scripts(modality=None):
    scripts = []
    if modality:
        subdir = UNIMODAL_DIR / modality
        if subdir.is_dir():
            scripts.extend(subdir.glob("train_*.py"))
    else:
        for subdir in UNIMODAL_DIR.iterdir():
            if subdir.is_dir():
                scripts.extend(subdir.glob("train_*.py"))
    return scripts


def run_script(script_path):
    # Convert to module path (e.g., src.models.unimodal.ecg.train_ecg_lr)
    rel_path = script_path.relative_to(UNIMODAL_DIR.parent.parent.parent)
    module = ".".join(rel_path.with_suffix("").parts)
    cmd = ["python", "-m", module]
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=UNIMODAL_DIR.parent.parent.parent,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"SUCCESS: {module}\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FAILED: {module}\n{e.stderr}")


def main():
    parser = argparse.ArgumentParser(
        description="Run all or specific unimodal classifier training scripts."
    )
    parser.add_argument(
        "--modality",
        type=str,
        default=None,
        help="Specify a modality (e.g., 'ehr', 'ecg') to run only its classifiers.",
    )
    args = parser.parse_args()

    scripts = find_classifier_scripts(args.modality)
    if not scripts:
        if args.modality:
            logger.warning(f"No classifier scripts found for modality: {args.modality}")
        else:
            logger.warning("No classifier scripts found.")
        return
    for script in scripts:
        run_script(script)


if __name__ == "__main__":
    main()
