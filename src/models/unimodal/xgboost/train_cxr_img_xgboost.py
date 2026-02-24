"""
Trains an XGBoost classifier on the extracted CXR Image embeddings.

Features:
- GPU-accelerated training with early stopping.
- Logs training and validation metrics to TensorBoard.
- Centralized metric computation using `src.utils.evaluation`.

Usage:
    python -m src.models.unimodal.xgboost.train_cxr_img_xgboost
"""

import logging
import xgboost as xgb
from torch.utils.tensorboard import SummaryWriter

from src.utils.config import Config
from src.utils.evaluation import (
    load_embeddings,
    compute_metrics,
    log_metrics_to_tensorboard,
    print_metrics,
    save_metrics,
    save_predictions,
)

logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION
# ==========================================
embeddings_dir = Config.CXR_IMG_EMBEDDINGS_DIR
output_dir = Config.CXR_IMG_XGBOOST_MODEL_DIR
results_dir = Config.RESULTS_DIR / "cxr_img" / "xgboost"
model_save_path = output_dir / "model.json"
metrics_save_path = results_dir / "test_metrics.json"
predictions_save_path = results_dir / "test_predictions.csv"
val_metrics_save_path = results_dir / "val_metrics.json"
tb_log_dir = Config.TENSORBOARD_LOG_DIR / "cxr_img" / "xgboost"

# XGBoost Hyperparameters
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
# MAIN EXECUTION
# ==========================================


def main():
    Config.setup_logging()
    Config.set_seed(42)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing TensorBoard at %s", tb_log_dir)
    writer = SummaryWriter(log_dir=str(tb_log_dir))

    logger.info("--- Loading Data ---")
    X_train, y_train, _ = load_embeddings(embeddings_dir / "train_embeddings.pt")
    X_val, y_val, _ = load_embeddings(embeddings_dir / "valid_embeddings.pt")
    X_test, y_test, test_ids = load_embeddings(embeddings_dir / "test_embeddings.pt")

    logger.info("--- Creating DMatrix objects ---")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    logger.info("--- Starting XGBoost Training on GPU ---")
    model = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=VERBOSE_EVAL,
    )
    logger.info("Best iteration: %d", model.best_iteration)

    logger.info("--- Saving Model ---")
    model.save_model(model_save_path)
    logger.info("Model saved to %s", model_save_path)

    logger.info("--- Evaluating ---")
    test_prob = model.predict(dtest)
    test_pred = (test_prob >= 0.5).astype(int)

    # Compute, print, and save metrics/predictions using evaluation.py
    test_metrics = compute_metrics(y_test, test_prob)
    print_metrics(test_metrics, name="CXR IMG TEST SET (XGBoost)")

    # Evaluate on validation set
    val_prob = model.predict(dval)
    val_metrics = compute_metrics(y_val, val_prob)
    save_metrics(val_metrics_save_path, val_metrics)

    save_metrics(metrics_save_path, test_metrics)
    save_predictions(predictions_save_path, test_ids, y_test, test_prob, test_pred)
    logger.info("Test predictions saved to: %s", predictions_save_path)

    # TensorBoard logging
    log_metrics_to_tensorboard(writer, val_metrics, global_step=0, prefix="val")
    log_metrics_to_tensorboard(writer, test_metrics, global_step=0, prefix="test")
    writer.close()
    logger.info("TensorBoard logs → %s", tb_log_dir)


if __name__ == "__main__":
    main()
