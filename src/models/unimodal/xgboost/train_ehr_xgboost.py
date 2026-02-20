"""Train an XGBoost classifier on EHR (MOTOR) embeddings.

Loads the standardised ``.pt`` embedding files produced by
``extract_ehr_embeddings.py``, trains XGBoost with GPU acceleration and
early stopping, evaluates on validation and test splits, and persists:

- Model artifact (``model.json``)
- Per-sample test predictions (``test_predictions.csv``)
- Metrics JSON (``test_metrics.json``, ``val_metrics.json``)
- TensorBoard scalars

Usage:
    python -m src.models.unimodal.xgboost.train_ehr_xgboost
"""

import logging

import xgboost as xgb
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

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": ["auc", "aucpr"],
    "tree_method": "hist",
    "device": "cuda",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": 1,
}

NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 50


# ==========================================
# MAIN
# ==========================================


def main():
    Config.setup_logging()
    Config.set_seed(42)

    embeddings_dir = Config.EHR_EMBEDDINGS_DIR
    output_dir = Config.EHR_XGBOOST_MODEL_DIR
    results_dir = Config.RESULTS_DIR / "ehr" / "xgboost"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load embeddings
    # ------------------------------------------------------------------
    X_train, y_train, _ = load_embeddings(embeddings_dir / "train_embeddings.pt")
    X_val, y_val, val_ids = load_embeddings(embeddings_dir / "valid_embeddings.pt")
    X_test, y_test, test_ids = load_embeddings(embeddings_dir / "test_embeddings.pt")

    # ------------------------------------------------------------------
    # 2. Create DMatrices
    # ------------------------------------------------------------------
    logger.info("Creating DMatrix objects")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    logger.info(
        "Training XGBoost (GPU, %d rounds, early_stop=%d)",
        NUM_BOOST_ROUND,
        EARLY_STOPPING_ROUNDS,
    )
    model = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=VERBOSE_EVAL,
    )
    logger.info("Best iteration: %d", model.best_iteration)

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    val_prob = model.predict(dval)
    test_prob = model.predict(dtest)

    val_metrics = compute_metrics(y_val, val_prob)
    test_metrics = compute_metrics(y_test, test_prob)

    print_metrics(val_metrics, "EHR XGBoost — Validation")
    print_metrics(test_metrics, "EHR XGBoost — Test")

    # ------------------------------------------------------------------
    # 5. TensorBoard
    # ------------------------------------------------------------------
    tb_dir = Config.TENSORBOARD_LOG_DIR / "ehr" / "xgboost"
    writer = SummaryWriter(log_dir=str(tb_dir))
    log_metrics_to_tensorboard(writer, val_metrics, global_step=0, prefix="val")
    log_metrics_to_tensorboard(writer, test_metrics, global_step=0, prefix="test")
    writer.close()
    logger.info("TensorBoard logs → %s", tb_dir)

    # ------------------------------------------------------------------
    # 6. Save artifacts
    # ------------------------------------------------------------------
    model_path = output_dir / "model.json"
    model.save_model(str(model_path))
    logger.info("Model saved → %s", model_path)

    save_predictions(results_dir / "test_predictions.csv", test_ids, y_test, test_prob)
    save_metrics(results_dir / "val_metrics.json", val_metrics)
    save_metrics(results_dir / "test_metrics.json", test_metrics)

    logger.info("EHR XGBoost training complete")


if __name__ == "__main__":
    main()
