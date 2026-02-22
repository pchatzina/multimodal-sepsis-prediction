"""
Run unimodal classifiers (LR, XGBoost, MLP) for all modalities, or a specific one.

Usage:
    python -m src.scripts.training.unimodal.run_classifiers
    python -m src.scripts.training.unimodal.run_classifiers --modality ehr
    python -m src.scripts.training.unimodal.run_classifiers --algorithm xgboost
"""

import argparse
import importlib.util
import logging
import subprocess

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_classifiers")

# ==========================================
# CONFIGURATION
# ==========================================
SUPPORTED_MODALITIES = ["ehr", "ecg", "cxr_img", "cxr_txt"]
SUPPORTED_ALGORITHMS = ["lr", "xgboost", "mlp"]

# Unified MLP script
MLP_TRAIN_SCRIPT = "src.models.unimodal.mlp.train_unimodal_mlp"


def is_implemented(module_name: str) -> bool:
    """Checks if a Python module exists before trying to run it."""
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except Exception:
        return False


def run_command(cmd: list):
    """Executes a terminal command."""
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"SUCCESS: {' '.join(cmd)}\n")
    except subprocess.CalledProcessError:
        logger.error(f"FAILED: {' '.join(cmd)}\n")


def get_module_path(algorithm: str, modality: str) -> str:
    """Constructs the expected module path for hardcoded algorithms."""
    if algorithm == "lr":
        return f"src.models.unimodal.logistic_regression.train_{modality}_lr"
    elif algorithm == "xgboost":
        return f"src.models.unimodal.xgboost.train_{modality}_xgboost"
    return ""


def main():
    parser = argparse.ArgumentParser(description="Run unimodal classifier training.")
    parser.add_argument(
        "--modality",
        type=str,
        default="all",
        choices=["all"] + SUPPORTED_MODALITIES,
        help="Specify a modality to run. Defaults to all.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="all",
        choices=["all"] + SUPPORTED_ALGORITHMS,
        help="Specify an algorithm to run. Defaults to all.",
    )
    args = parser.parse_args()

    # Determine what to run
    modalities = SUPPORTED_MODALITIES if args.modality == "all" else [args.modality]
    algorithms = SUPPORTED_ALGORITHMS if args.algorithm == "all" else [args.algorithm]

    for mod in modalities:
        logger.info("=" * 50)
        logger.info(f"  STARTING PIPELINE FOR: {mod.upper()}")
        logger.info("=" * 50)

        for algo in algorithms:
            logger.info(f"--- Checking {algo.upper()} for {mod.upper()} ---")

            # 1. MLP Logic (Unified)
            if algo == "mlp":
                train_cmd = ["python", "-m", MLP_TRAIN_SCRIPT, "--modality", mod]
                run_command(train_cmd)

            # 2. Logistic Regression & XGBoost (Fragmented)
            else:
                module_path = get_module_path(algo, mod)

                if is_implemented(module_path):
                    cmd = ["python", "-m", module_path]
                    run_command(cmd)
                else:
                    logger.warning(
                        f"Skipping {algo.upper()} for {mod.upper()}: "
                        f"Script '{module_path}' not implemented yet.\n"
                    )


if __name__ == "__main__":
    main()
