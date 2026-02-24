"""
Trains a Logistic Regression linear probe on the extracted CXR Image embeddings.

Usage:
    python -m src.models.unimodal.logistic_regression.train_cxr_img_lr
"""

import logging
import joblib
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
# CONFIGURATION
# ==========================================
embeddings_dir = Config.CXR_IMG_EMBEDDINGS_DIR
output_dir = Config.CXR_IMG_LR_MODEL_DIR
results_dir = Config.RESULTS_DIR / "cxr_img" / "lr"
model_save_path = output_dir / "model.joblib"
scaler_save_path = output_dir / "scaler.joblib"
metrics_save_path = results_dir / "test_metrics.json"
predictions_save_path = results_dir / "test_predictions.csv"
val_metrics_save_path = results_dir / "val_metrics.json"


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    Config.setup_logging()
    Config.set_seed(42)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("--- Loading CXR Image Data ---")
    X_train, y_train, _ = load_embeddings(embeddings_dir / "train_embeddings.pt")
    X_val, y_val, _ = load_embeddings(embeddings_dir / "valid_embeddings.pt")
    X_test, y_test, test_ids = load_embeddings(embeddings_dir / "test_embeddings.pt")

    logger.info("--- Scaling Features ---")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Save scaler for future inference
    joblib.dump(scaler, scaler_save_path)

    logger.info("--- Training Logistic Regression ---")
    model = LogisticRegression(
        solver="lbfgs", max_iter=2000, n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)

    logger.info("--- Saving Model ---")
    joblib.dump(model, model_save_path)
    logger.info("Model saved to %s", model_save_path)

    logger.info("--- Evaluating ---")
    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)

    # Compute, print, and save metrics/predictions using evaluation.py
    test_metrics = compute_metrics(y_test, test_prob)
    print_metrics(test_metrics, name="CXR IMG TEST SET (Logistic Regression)")

    # Evaluate on validation set
    val_prob = model.predict_proba(X_val)[:, 1]
    val_metrics = compute_metrics(y_val, val_prob)
    save_metrics(val_metrics_save_path, val_metrics)

    save_metrics(metrics_save_path, test_metrics)
    save_predictions(predictions_save_path, test_ids, y_test, test_prob, test_pred)
    logger.info("Test predictions saved to: %s", predictions_save_path)

    # TensorBoard logging
    tb_dir = Config.TENSORBOARD_LOG_DIR / "cxr_img" / "lr"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    log_metrics_to_tensorboard(writer, val_metrics, global_step=0, prefix="val")
    log_metrics_to_tensorboard(writer, test_metrics, global_step=0, prefix="test")
    writer.close()
    logger.info("TensorBoard logs → %s", tb_dir)


if __name__ == "__main__":
    main()
