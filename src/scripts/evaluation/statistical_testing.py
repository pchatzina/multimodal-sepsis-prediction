"""
Statistical Significance Testing for Multimodal Sepsis Prediction.

Uses paired bootstrapping to compute 95% Confidence Intervals and empirical p-values
for AUROC, AUPRC, and F1 scores, comparing the EHR baseline against viable
multimodal combinations.

Usage:
    python -m src.scripts.evaluation.statistical_testing
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm

from src.utils.config import Config

logger = logging.getLogger(__name__)

# The combinations we actually care about (excluding the clinically invalid TXT-without-IMG ones)
BASELINE_COMBO = "1_EHR_Only"
COMPARISONS = [
    "2_EHR_ECG",
    "3_EHR_IMG",
    "5_EHR_ECG_IMG",
    "7_EHR_IMG_TXT",
    "8_All_Modalities",
]

N_BOOTSTRAP = 1000
ALPHA = 0.05
THRESHOLD = 0.5


def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Calculates the core metrics for a single array of predictions."""
    y_pred = (y_prob >= THRESHOLD).astype(int)
    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
    }


def paired_bootstrap_test(
    y_true: np.ndarray,
    y_prob_base: np.ndarray,
    y_prob_comp: np.ndarray,
    n_iterations: int = N_BOOTSTRAP,
) -> Dict[str, Dict[str, float]]:
    """
    Performs paired bootstrapping to get 95% CIs and p-values for the difference
    between a comparison model and a baseline model.
    """
    np.random.seed(42)
    n_samples = len(y_true)

    # Store metric differences: Metric_Comp - Metric_Base
    diffs = {"auroc": [], "auprc": [], "f1": []}

    # Store absolute metrics for the comparison model to get its 95% CIs
    abs_comp = {"auroc": [], "auprc": [], "f1": []}

    for _ in range(n_iterations):
        # 1. Sample with replacement
        indices = np.random.randint(0, n_samples, n_samples)
        y_t_b = y_true[indices]
        y_p_base_b = y_prob_base[indices]
        y_p_comp_b = y_prob_comp[indices]

        # Ensure we have both classes in the bootstrap sample (rarely an issue with large N)
        if len(np.unique(y_t_b)) < 2:
            continue

        # 2. Compute metrics for both
        base_m = calculate_metrics(y_t_b, y_p_base_b)
        comp_m = calculate_metrics(y_t_b, y_p_comp_b)

        # 3. Store absolute values and differences
        for metric in diffs.keys():
            abs_comp[metric].append(comp_m[metric])
            diffs[metric].append(comp_m[metric] - base_m[metric])

    # Calculate CIs and p-values
    results = {}
    for metric in diffs.keys():
        diff_array = np.array(diffs[metric])
        comp_array = np.array(abs_comp[metric])

        # 95% CI for the comparison model's absolute metric
        ci_lower = np.percentile(comp_array, (ALPHA / 2) * 100)
        ci_upper = np.percentile(comp_array, (1 - ALPHA / 2) * 100)

        # Empirical p-value: How often was the comparison model NOT better than the baseline?
        # (1-tailed test asking if comparison > baseline)
        p_value = np.mean(diff_array <= 0.0)

        results[metric] = {
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "mean_diff": np.mean(diff_array),
        }

    return results


def main():
    Config.setup_logging()

    base_dir = Config.RESULTS_DIR / "fusion" / "incremental_value"

    # 1. Load Baseline Predictions
    base_csv = base_dir / BASELINE_COMBO / "preds_calibrated.csv"
    if not base_csv.exists():
        logger.error(f"Cannot find baseline predictions at {base_csv}")
        return

    df_base = pd.read_csv(base_csv)
    y_true = df_base["label"].values
    y_prob_base = df_base["probability"].values

    base_metrics = calculate_metrics(y_true, y_prob_base)
    logger.info(f"Baseline ({BASELINE_COMBO}) Metrics:")
    logger.info(
        f"  AUROC: {base_metrics['auroc']:.4f} | AUPRC: {base_metrics['auprc']:.4f} | F1: {base_metrics['f1']:.4f}\n"
    )

    results_summary = []

    # 2. Iterate and Compare
    logger.info(
        f"Running {N_BOOTSTRAP} bootstrap iterations for significance testing..."
    )

    for combo in tqdm(COMPARISONS, desc="Evaluating Combinations"):
        comp_csv = base_dir / combo / "preds_calibrated.csv"
        if not comp_csv.exists():
            logger.warning(f"Missing {comp_csv}. Skipping.")
            continue

        df_comp = pd.read_csv(comp_csv)
        y_prob_comp = df_comp["probability"].values

        comp_metrics = calculate_metrics(y_true, y_prob_comp)
        stats = paired_bootstrap_test(y_true, y_prob_base, y_prob_comp)

        results_summary.append(
            {
                "Model": combo,
                "AUROC": f"{comp_metrics['auroc']:.4f} [{stats['auroc']['ci_lower']:.4f}-{stats['auroc']['ci_upper']:.4f}]",
                "AUROC p-value": stats["auroc"]["p_value"],
                "AUPRC": f"{comp_metrics['auprc']:.4f} [{stats['auprc']['ci_lower']:.4f}-{stats['auprc']['ci_upper']:.4f}]",
                "AUPRC p-value": stats["auprc"]["p_value"],
                "F1": f"{comp_metrics['f1']:.4f} [{stats['f1']['ci_lower']:.4f}-{stats['f1']['ci_upper']:.4f}]",
                "F1 p-value": stats["f1"]["p_value"],
            }
        )

    # 3. Print and Save Summary
    df_results = pd.DataFrame(results_summary)

    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE REPORT (vs. 1_EHR_Only)")
    print("=" * 80)
    # Print formatted table to console
    print(df_results.to_string(index=False))
    print("=" * 80)

    out_dir = Config.RESULTS_DIR / "fusion" / "statistical_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_results.to_csv(out_dir / "significance_results.csv", index=False)
    logger.info(
        f"\nSaved full statistical report to {out_dir / 'significance_results.csv'}"
    )


if __name__ == "__main__":
    main()
