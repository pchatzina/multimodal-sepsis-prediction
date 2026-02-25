"""
Batch execution script for Late-Fusion Sepsis Prediction Model.

Runs hyperparameter tuning (Option A and B) followed by the 4 distinct
training experiments:
1. Option A (Scratch) + No Dropout
2. Option A (Scratch) + EHR Dropout
3. Option B (Pre-trained) + No Dropout
4. Option B (Pre-trained) + EHR Dropout

Usage:
    python -m src.scripts.training.fusion.run_fusion_experiments --tune_trials 30 --dropout_rate 0.3
"""

import argparse
import logging
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_fusion_experiments")


def run_command(cmd: list):
    """Executes a terminal command and handles errors."""
    cmd_str = " ".join(cmd)
    logger.info(f"Running: {cmd_str}")
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"SUCCESS: {cmd_str}\n")
    except subprocess.CalledProcessError as e:
        logger.error(f"FAILED: {cmd_str} (Exit code: {e.returncode})\n")
        sys.exit(e.returncode)


def main():
    parser = argparse.ArgumentParser(description="Batch run Fusion Experiments.")
    parser.add_argument(
        "--tune_trials",
        type=int,
        default=0,
        help="Number of Optuna trials (0 to skip tuning).",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.3,
        help="The EHR dropout rate to use for the dropout experiments.",
    )
    args = parser.parse_args()

    # ==========================================
    # 1. OPTIONAL: RUN TUNING
    # ==========================================
    if args.tune_trials > 0:
        logger.info("=" * 50)
        logger.info("  STARTING HYPERPARAMETER TUNING")
        logger.info("=" * 50)

        # Tune Option A
        run_command(
            [
                "python",
                "-m",
                "src.models.fusion.tune_late_fusion",
                "--n_trials",
                str(args.tune_trials),
            ]
        )
        # Tune Option B
        run_command(
            [
                "python",
                "-m",
                "src.models.fusion.tune_late_fusion",
                "--n_trials",
                str(args.tune_trials),
                "--load_pretrained",
            ]
        )

    # ==========================================
    # 2. RUN 4 TRAINING EXPERIMENTS
    # ==========================================
    logger.info("=" * 50)
    logger.info("  STARTING 4 FUSION TRAINING EXPERIMENTS")
    logger.info("=" * 50)

    experiments = [
        {
            "name": "Option A (Scratch) + No Dropout",
            "pretrained": False,
            "dropout": 0.0,
        },
        {
            "name": f"Option A (Scratch) + EHR Dropout ({args.dropout_rate})",
            "pretrained": False,
            "dropout": args.dropout_rate,
        },
        {
            "name": "Option B (Pre-trained) + No Dropout",
            "pretrained": True,
            "dropout": 0.0,
        },
        {
            "name": f"Option B (Pre-trained) + EHR Dropout ({args.dropout_rate})",
            "pretrained": True,
            "dropout": args.dropout_rate,
        },
    ]

    base_cmd = ["python", "-m", "src.models.fusion.train_late_fusion"]

    for exp in experiments:
        logger.info(f"--- Starting: {exp['name']} ---")
        cmd = base_cmd.copy()

        if exp["pretrained"]:
            cmd.append("--load_pretrained")

        if exp["dropout"] > 0.0:
            cmd.extend(["--ehr_dropout_rate", str(exp["dropout"])])

        run_command(cmd)

    logger.info("=== ALL BATCH EXPERIMENTS COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
