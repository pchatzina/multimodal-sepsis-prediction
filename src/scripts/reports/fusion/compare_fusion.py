"""
Compare Late-Fusion Sepsis Prediction results and generate a Markdown report.

Reads test and val metrics from the fusion results directory for the 4
distinct experiment types (Option A/B x No Dropout/EHR Dropout), prints
a formatted comparison table, and writes a Markdown file to
`Config.REPORTS_DIR / "fusion"`.

Usage:
    python -m src.scripts.reports.fusion.compare_fusion
"""

import argparse
import json
import logging
from pathlib import Path

from src.utils.config import Config

logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION
# ==========================================

RESULTS_DIR = Config.RESULTS_DIR / "fusion"
OUTPUT_DIR = Config.REPORTS_DIR / "fusion"

# List of (Display Name, Test JSON Filename, Val JSON Filename)
EXPERIMENTS = [
    (
        "Option A (No Dropout)",
        "test_metrics_fusion_scratch_no_dropout.json",
        "val_metrics_fusion_scratch_no_dropout.json",
    ),
    (
        "Option A (EHR Dropout)",
        "test_metrics_fusion_scratch_ehr_dropout.json",
        "val_metrics_fusion_scratch_ehr_dropout.json",
    ),
    (
        "Option B (No Dropout)",
        "test_metrics_fusion_pretrained_no_dropout.json",
        "val_metrics_fusion_pretrained_no_dropout.json",
    ),
    (
        "Option B (EHR Dropout)",
        "test_metrics_fusion_pretrained_ehr_dropout.json",
        "val_metrics_fusion_pretrained_ehr_dropout.json",
    ),
]

# Metrics to display and their display names (in order).
DISPLAY_METRICS = [
    ("auroc", "AUROC"),
    ("auprc", "AUPRC"),
    ("accuracy", "Accuracy"),
    ("f1", "F1"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("specificity", "Specificity"),
]

CONFUSION_KEYS = ["tp", "fp", "fn", "tn"]

# ==========================================
# HELPERS
# ==========================================


def load_metrics(results_dir: Path, filename: str) -> dict | None:
    """Load a metrics JSON file, return None if missing."""
    path = results_dir / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def find_best(values: list[float | None]) -> int | None:
    """Return the index of the highest non-None value."""
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        return None
    return max(valid, key=lambda x: x[1])[0]


def build_report(experiments: list[tuple[str, str, str]]) -> str | None:
    """Build a Markdown report string comparing all valid experiments."""
    all_test = []
    all_val = []
    names = []

    for name, test_fn, val_fn in experiments:
        test_m = load_metrics(RESULTS_DIR, test_fn)
        val_m = load_metrics(RESULTS_DIR, val_fn)

        if test_m is None:
            logger.warning(f"Skipping '{name}' — {test_fn} not found in {RESULTS_DIR}")
            continue

        all_test.append(test_m)
        all_val.append(val_m)
        names.append(name)

    if not names:
        return None

    n_samples = all_test[0].get("n_samples", "?")
    n_pos = all_test[0].get("n_positive", "?")
    n_neg = all_test[0].get("n_negative", "?")

    lines = []
    lines.append("# Late-Fusion Sepsis Model - Experiment Comparison\n")
    lines.append(
        f"**Test set:** N = {n_samples} (+{n_pos} / −{n_neg}), threshold = 0.5\n"
    )

    # --- Main metrics table ---
    header = "| Metric |" + "|".join(f" {n} " for n in names) + "|"
    sep = "|---|" + "|".join(":---:" for _ in names) + "|"
    lines.append(header)
    lines.append(sep)

    for key, display_name in DISPLAY_METRICS:
        values = [m.get(key) for m in all_test]
        best_idx = find_best(values)
        cells = []
        for i, v in enumerate(values):
            if v is None:
                cells.append("—")
            elif i == best_idx and len(names) > 1:
                cells.append(f"**{v:.4f}**")
            else:
                cells.append(f"{v:.4f}")
        lines.append(f"| {display_name} |" + "|".join(f" {c} " for c in cells) + "|")

    # --- Confusion matrix ---
    lines.append("")
    lines.append("### Confusion Matrix\n")
    header = "| |" + "|".join(f" {n} " for n in names) + "|"
    sep = "|---|" + "|".join(":---:" for _ in names) + "|"
    lines.append(header)
    lines.append(sep)
    for key in CONFUSION_KEYS:
        values = [str(m.get(key, "—")) for m in all_test]
        lines.append(f"| {key.upper()} |" + "|".join(f" {v} " for v in values) + "|")

    # --- Val → Test gap ---
    if any(v is not None for v in all_val):
        lines.append("")
        lines.append("### Generalisation Gap (Val → Test)\n")
        header = "| Metric |" + "|".join(f" {n} " for n in names) + "|"
        sep = "|---|" + "|".join(":---:" for _ in names) + "|"
        lines.append(header)
        lines.append(sep)
        for key, display_name in [("auroc", "AUROC"), ("auprc", "AUPRC")]:
            cells = []
            for test_m, val_m in zip(all_test, all_val):
                if val_m is None or key not in val_m:
                    cells.append("—")
                else:
                    gap = test_m[key] - val_m[key]
                    cells.append(f"{gap:+.4f}")
            lines.append(
                f"| {display_name} |" + "|".join(f" {c} " for c in cells) + "|"
            )

    lines.append("")
    return "\n".join(lines)


# ==========================================
# MAIN
# ==========================================


def main():
    Config.setup_logging()
    parser = argparse.ArgumentParser(description="Compare Late-Fusion experiments.")
    parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report(EXPERIMENTS)

    if report is None:
        logger.error(f"No results found in {RESULTS_DIR} to generate a report.")
        return

    # Print to console
    print(report)

    # Write Markdown file
    out_path = OUTPUT_DIR / "late_fusion_comparison.md"
    out_path.write_text(report)
    logger.info(f"\nReport successfully written to -> {out_path}")


if __name__ == "__main__":
    main()
