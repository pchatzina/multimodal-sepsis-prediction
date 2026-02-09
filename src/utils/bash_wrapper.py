"""
Bash Wrapper for MIMIC-IV Data Processing.

Provides CLI commands to run export and ETL bash scripts with the correct
environment variables sourced from the project Config.
"""

import argparse
import logging
import subprocess
import os
import sys
from pathlib import Path

from src.utils.config import Config

logger = logging.getLogger(__name__)

# Resolve script directories once relative to this project's source tree
_EXPORTS_DIR = (
    Path(__file__).resolve().parent.parent / "data" / "preprocess" / "ehr" / "exports"
)
_TRANSFORMATIONS_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "preprocess"
    / "ehr"
    / "transformations"
)


def run_script(
    script_path: str,
    script_args: list[str] | None = None,
    env_vars: dict[str, str] | None = None,
) -> None:
    """Run a bash script with specific environment variables.

    Args:
        script_path: Absolute path to the bash script.
        script_args: Optional positional arguments passed to the script.
        env_vars: Optional environment variable overrides.

    Raises:
        SystemExit: If the script exits with a non-zero status.
        FileNotFoundError: If the script does not exist.
    """
    script = Path(script_path)
    if not script.is_file():
        logger.error("Script not found: %s", script)
        sys.exit(1)

    # Merge current environment with our custom config
    full_env = os.environ.copy()
    if env_vars:
        full_env.update(env_vars)

    # Build command
    cmd = ["bash", str(script)]
    if script_args:
        cmd.extend(script_args)

    logger.info("Running: %s with args %s", script, script_args)

    try:
        subprocess.run(cmd, check=True, env=full_env)
        logger.info("Success: %s", script)
    except subprocess.CalledProcessError as e:
        logger.error("Error running %s. Exit code: %d", script, e.returncode)
        sys.exit(e.returncode)


def export_cohort() -> None:
    """Export MIMIC-IV data filtered to cohort subjects."""
    env = {
        "BASE_OUTPUT_DIR": str(Config.RAW_EHR_COHORT_DIR),
        "DB": Config.DB_NAME,
    }
    run_script(str(_EXPORTS_DIR / "export_cohort_data.sh"), env_vars=env)


def export_pretraining() -> None:
    """Export MIMIC-IV data for pretraining (excluding test split)."""
    env = {
        "BASE_OUTPUT_DIR": str(Config.RAW_EHR_PRETRAINING_DIR),
        "DB": Config.DB_NAME,
    }
    run_script(str(_EXPORTS_DIR / "export_pretraining_data.sh"), env_vars=env)


def run_pipeline(dataset: str) -> None:
    """Run the full MEDS ETL pipeline for the specified dataset.

    Args:
        dataset: One of ``"pretraining"`` or ``"cohort"``.
    """
    dataset_config = {
        "pretraining": (
            Config.RAW_EHR_PRETRAINING_DIR,
            Config.PROCESSED_EHR_PRETRAINING_DIR,
        ),
        "cohort": (Config.RAW_EHR_COHORT_DIR, Config.PROCESSED_EHR_COHORT_DIR),
    }

    if dataset not in dataset_config:
        logger.error(
            "Unknown dataset: %s. Must be one of %s.", dataset, list(dataset_config)
        )
        sys.exit(1)

    raw_base, processed_base = dataset_config[dataset]

    env = {
        "RAW_BASE": str(raw_base),
        "PROCESSED_BASE": str(processed_base),
    }
    run_script(
        str(_TRANSFORMATIONS_DIR / "mimic_to_meds.sh"),
        script_args=[dataset],
        env_vars=env,
    )


def main() -> None:
    """CLI entry point for MIMIC-IV bash script wrapper."""
    Config.setup_logging()

    parser = argparse.ArgumentParser(
        description="Bash Wrapper for MIMIC-IV Data Processing",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: export-cohort
    subparsers.add_parser("export-cohort", help="Export cohort data")

    # Command: export-pretraining
    subparsers.add_parser(
        "export-pretraining",
        help="Export pretraining data (excludes test split)",
    )

    # Command: meds-pipeline
    meds_parser = subparsers.add_parser("meds-pipeline", help="Run MEDS ETL pipeline")
    meds_parser.add_argument(
        "dataset",
        choices=["pretraining", "cohort"],
        help="Name of the dataset to process",
    )

    args = parser.parse_args()

    if args.command == "export-cohort":
        export_cohort()
    elif args.command == "export-pretraining":
        export_pretraining()
    elif args.command == "meds-pipeline":
        run_pipeline(args.dataset)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
