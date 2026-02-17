"""Train a Logistic Regression classifier on EHR (MOTOR) embeddings.

Loads the standardised ``.pt`` embedding files produced by
``extract_ehr_embeddings.py``, fits a ``LogisticRegression`` with
``StandardScaler``, evaluates on validation and test splits, and persists:

- Model artifact (``model.pkl``)
- Scaler artifact (``scaler.pkl``)
- Per-sample test predictions (``test_predictions.csv``)
- Metrics JSON (``test_metrics.json``, ``val_metrics.json``)
- TensorBoard scalars

Usage:
    python -m src.models.unimodal.logistic_regression.train_ehr_lr
"""

import logging
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter

from src.utils.config import Config
from src.utils.evaluation import (
    compute_metrics,
    load_embeddings,
    log_metrics_to_tensorboard,
    print_metrics,
    save_metrics,
    save_predictions,
)

logger = logging.getLogger(__name__)

# ==========================================
# CONSTANTS
# ==========================================

SOLVER = "lbfgs"
MAX_ITER = 2000
RANDOM_STATE = 42


# ==========================================
# MAIN
# ==========================================


def main():
    Config.setup_logging()

    embeddings_dir = Config.EHR_EMBEDDINGS_DIR
    output_dir = Config.EHR_LR_MODEL_DIR
    results_dir = Config.RESULTS_DIR / "ehr" / "lr"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load embeddings
    # ------------------------------------------------------------------
    X_train, y_train, _ = load_embeddings(embeddings_dir / "train_embeddings.pt")
    X_val, y_val, _ = load_embeddings(embeddings_dir / "valid_embeddings.pt")
    X_test, y_test, test_ids = load_embeddings(embeddings_dir / "test_embeddings.pt")

    # ------------------------------------------------------------------
    # 2. Scale features (fit on train only)
    # ------------------------------------------------------------------
    logger.info("Scaling features (StandardScaler, fit on train)")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    logger.info(
        "Training Logistic Regression (solver=%s, max_iter=%d)", SOLVER, MAX_ITER
    )
    model = LogisticRegression(
        solver=SOLVER,
        max_iter=MAX_ITER,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    val_metrics = compute_metrics(y_val, val_prob)
    test_metrics = compute_metrics(y_test, test_prob)

    print_metrics(val_metrics, "EHR LR — Validation")
    print_metrics(test_metrics, "EHR LR — Test")

    # ------------------------------------------------------------------
    # 5. TensorBoard
    # ------------------------------------------------------------------
    tb_dir = Config.TENSORBOARD_LOG_DIR / "ehr" / "lr"
    writer = SummaryWriter(log_dir=str(tb_dir))
    log_metrics_to_tensorboard(writer, val_metrics, global_step=0, prefix="val")
    log_metrics_to_tensorboard(writer, test_metrics, global_step=0, prefix="test")
    writer.close()
    logger.info("TensorBoard logs → %s", tb_dir)

    # ------------------------------------------------------------------
    # 6. Save artifacts
    # ------------------------------------------------------------------
    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Model + scaler saved → %s", output_dir)

    save_predictions(results_dir / "test_predictions.csv", test_ids, y_test, test_prob)
    save_metrics(results_dir / "val_metrics.json", val_metrics)
    save_metrics(results_dir / "test_metrics.json", test_metrics)

    logger.info("EHR LR training complete")


if __name__ == "__main__":
    main()
