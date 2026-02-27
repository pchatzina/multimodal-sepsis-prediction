"""
Batch execution script for Late-Fusion Sepsis Prediction Model.

Runs hyperparameter tuning (Option A and B) followed by the 4 distinct
training experiments:
1. Option A (Scratch) + No Dropout
2. Option A (Scratch) + EHR Dropout
3. Option B (Pre-trained) + No Dropout
4. Option B (Pre-trained) + EHR Dropout

Usage:
    # OPTION 1: Initial 4-Modality Experiment
    python -m src.scripts.training.fusion.run_fusion_experiments --tune_trials 30 --dropout_rate 0.3 --modalities ehr ecg img txt

    # OPTION 2: Pruned 3-Modality Champion Experiment
    python -m src.scripts.training.fusion.run_fusion_experiments --tune_trials 30 --dropout_rate 0.3 --modalities ehr img txt
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
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["ehr", "ecg", "img", "txt"],
        help="List of active modalities to include in the fusion model.",
    )
    args = parser.parse_args()

    # Create the modality arguments suffix to pass to underlying scripts
    modality_args = ["--modalities"] + args.modalities

    # ==========================================
    # 1. OPTIONAL: RUN TUNING
    # ==========================================
    if args.tune_trials > 0:
        logger.info("=" * 60)
        logger.info(
            f"  STARTING TUNING FOR {len(args.modalities)} MODALITIES: {args.modalities}"
        )
        logger.info("=" * 60)

        # Tune Option A (Scratch)
        run_command(
            [
                "python",
                "-m",
                "src.models.fusion.tune_late_fusion",
                "--n_trials",
                str(args.tune_trials),
            ]
            + modality_args
        )

        # Tune Option B (Pre-trained)
        run_command(
            [
                "python",
                "-m",
                "src.models.fusion.tune_late_fusion",
                "--n_trials",
                str(args.tune_trials),
                "--load_pretrained",
            ]
            + modality_args
        )

    # ==========================================
    # 2. RUN 4 TRAINING EXPERIMENTS
    # ==========================================
    logger.info("=" * 60)
    logger.info(
        f"  STARTING 4 TRAINING EXPERIMENTS FOR {len(args.modalities)} MODALITIES"
    )
    logger.info("=" * 60)

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

        # Always append the modalities list
        cmd.extend(modality_args)

        run_command(cmd)

    logger.info("=== ALL BATCH EXPERIMENTS COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
