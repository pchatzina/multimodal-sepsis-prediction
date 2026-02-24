"""Compare unimodal classifier results and generate Markdown reports.

Reads test and val metrics from each classifier's results directory,
prints a formatted comparison table, and writes a Markdown file per
modality to ``Config.REPORTS_DIR / "unimodal"``.

Usage:
    python -m src.scripts.reports.unimodal.compare_classifiers              # all modalities
    python -m src.scripts.reports.unimodal.compare_classifiers --modality ehr
    python -m src.scripts.reports.unimodal.compare_classifiers --modality ecg
"""

import argparse
import json
import logging
from pathlib import Path

from src.utils.config import Config

logger = logging.getLogger(__name__)

# ==========================================
# REGISTRY
# ==========================================

# (display_name, results_dir, test_filename, val_filename) per classifier, grouped by modality.
MODALITY_CLASSIFIERS = {
    "ehr": [
        (
            "LR",
            Config.RESULTS_DIR / "ehr" / "lr",
            "test_metrics.json",
            "val_metrics.json",
        ),
        (
            "XGBoost",
            Config.RESULTS_DIR / "ehr" / "xgboost",
            "test_metrics.json",
            "val_metrics.json",
        ),
        (
            "MLP",
            Config.RESULTS_DIR / "ehr" / "mlp",
            "test_metrics_mlp.json",
            "val_metrics_mlp.json",
        ),
    ],
    "ecg": [
        (
            "LR",
            Config.RESULTS_DIR / "ecg" / "lr",
            "test_metrics.json",
            "val_metrics.json",
        ),
        (
            "XGBoost",
            Config.RESULTS_DIR / "ecg" / "xgboost",
            "test_metrics.json",
            "val_metrics.json",
        ),
        (
            "MLP",
            Config.RESULTS_DIR / "ecg" / "mlp",
            "test_metrics_mlp.json",
            "val_metrics_mlp.json",
        ),
    ],
    "cxr_img": [
        (
            "LR",
            Config.RESULTS_DIR / "cxr_img" / "lr",
            "test_metrics.json",
            "val_metrics.json",
        ),
        (
            "XGBoost",
            Config.RESULTS_DIR / "cxr_img" / "xgboost",
            "test_metrics.json",
            "val_metrics.json",
        ),
        (
            "MLP",
            Config.RESULTS_DIR / "cxr_img" / "mlp",
            "test_metrics_mlp.json",
            "val_metrics_mlp.json",
        ),
    ],
    "cxr_txt": [
        (
            "LR",
            Config.RESULTS_DIR / "cxr_txt" / "lr",
            "test_metrics.json",
            "val_metrics.json",
        ),
        (
            "XGBoost",
            Config.RESULTS_DIR / "cxr_txt" / "xgboost",
            "test_metrics.json",
            "val_metrics.json",
        ),
        (
            "MLP",
            Config.RESULTS_DIR / "cxr_txt" / "mlp",
            "test_metrics_mlp.json",
            "val_metrics_mlp.json",
        ),
    ],
}

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

OUTPUT_DIR = Config.REPORTS_DIR / "unimodal"


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


def build_report(
    modality: str, classifiers: list[tuple[str, Path, str, str | None]]
) -> str | None:
    """Build a Markdown report string for one modality. Returns None if no data."""
    all_test = []
    all_val = []
    names = []

    # Unpack the 4-tuple configuration
    for name, results_dir, test_fn, val_fn in classifiers:
        test_m = load_metrics(results_dir, test_fn) if test_fn else None
        val_m = load_metrics(results_dir, val_fn) if val_fn else None

        if test_m is None:
            logger.warning(
                "Skipping %s — %s not found at %s", name, test_fn, results_dir
            )
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
    lines.append(f"# {modality.upper()} - Unimodal Classifier Comparison\n")
    lines.append(f"Test set: N = {n_samples} (+{n_pos} / −{n_neg}), threshold = 0.5\n")

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

    parser = argparse.ArgumentParser(description="Compare unimodal classifier results")
    parser.add_argument(
        "--modality",
        choices=list(MODALITY_CLASSIFIERS.keys()),
        default=None,
        help="Restrict to one modality (default: show all)",
    )
    args = parser.parse_args()

    modalities = [args.modality] if args.modality else list(MODALITY_CLASSIFIERS.keys())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for modality in modalities:
        classifiers = MODALITY_CLASSIFIERS[modality]
        report = build_report(modality, classifiers)

        if report is None:
            logger.warning("No results found for %s — skipping", modality)
            continue

        # Print to console
        print(report)

        # Write Markdown file
        out_path = OUTPUT_DIR / f"{modality}.md"
        out_path.write_text(report)
        logger.info("Report written → %s", out_path)


if __name__ == "__main__":
    main()
