"""
Trains a PyTorch Multilayer Perceptron (MLP) on the extracted ECG embeddings.

Features:
- Includes an input LayerNorm to stabilize the high-variance foundation model embeddings.
- Implements Early Stopping based on validation AUROC.
- Logs training and validation metrics to TensorBoard.
- Centralized metric computation using `src.utils.evaluation`.

Usage:
    python -m src.models.unimodal.mlp.train_ecg_mlp
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
INPUT_DIM = 768  # Standard ecg-fm output dimension
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 100
PATIENCE = 15  # Early stopping patience

DATA_DIR = Config.ECG_EMBEDDINGS_DIR
OUTPUT_DIR = Config.ECG_MLP_MODEL_DIR
RESULTS_DIR = Config.RESULTS_DIR / "ecg/mlp"
MODEL_SAVE_PATH = OUTPUT_DIR / "best_ecg_mlp.pt"
METRICS_SAVE_PATH = RESULTS_DIR / "test_metrics_mlp.json"
PREDICTIONS_SAVE_PATH = RESULTS_DIR / "test_predictions_mlp.csv"
VAL_METRICS_SAVE_PATH = RESULTS_DIR / "val_metrics_mlp.json"

TB_LOG_DIR = Config.TENSORBOARD_LOG_DIR / "ecg" / "mlp"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# MODEL ARCHITECTURE
# ==========================================


class ECGClassifierMLP(nn.Module):
    def __init__(self, input_dim: int, dropout_rate: float = 0.3):
        super().__init__()

        self.network = nn.Sequential(
            # 1. Input Normalization (Crucial for std ~ 12.7)
            nn.LayerNorm(input_dim),
            # 2. First Hidden Layer
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            # 3. Second Hidden Layer
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            # 4. Output Layer (Unnormalized logits for BCEWithLogitsLoss)
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)


# ==========================================
# HELPER FUNCTIONS
# ==========================================


def create_dataloader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    """Converts numpy arrays to PyTorch tensors and returns a DataLoader."""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Shape: (N, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_model(
    model: nn.Module, dataloader: DataLoader
) -> tuple[np.ndarray, np.ndarray, float]:
    """Runs inference and returns true labels, probabilities, and average loss."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            logits = model(inputs)
            loss = criterion(logits, targets)

            total_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    return np.array(all_targets).flatten(), np.array(all_probs).flatten(), avg_loss


# ==========================================
# MAIN EXECUTION
# ==========================================


def main():
    Config.setup_logging()
    Config.set_seed(42)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TB_LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing TensorBoard at %s", TB_LOG_DIR)
    writer = SummaryWriter(log_dir=str(TB_LOG_DIR))

    # 1. Load Data using Centralized Evaluator
    logger.info("--- Loading Data ---")
    X_train, y_train, _ = load_embeddings(DATA_DIR / "train_embeddings.pt")
    X_val, y_val, _ = load_embeddings(DATA_DIR / "valid_embeddings.pt")
    X_test, y_test, test_ids = load_embeddings(DATA_DIR / "test_embeddings.pt")

    train_loader = create_dataloader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, BATCH_SIZE, shuffle=False)
    test_loader = create_dataloader(X_test, y_test, BATCH_SIZE, shuffle=False)

    # 2. Setup Model, Loss, and Optimizer
    logger.info("--- Initializing Model on %s ---", DEVICE)
    model = ECGClassifierMLP(input_dim=INPUT_DIM).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 3. Training Loop with Early Stopping
    logger.info("--- Starting Training ---")
    best_val_auroc = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        val_true, val_prob, val_loss = evaluate_model(model, val_loader)
        val_metrics = compute_metrics(val_true, val_prob)

        # Logging
        logger.info(
            "Epoch %03d | Train Loss: %.4f | Val Loss: %.4f | Val AUROC: %.4f",
            epoch,
            train_loss,
            val_loss,
            val_metrics["auroc"],
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        log_metrics_to_tensorboard(writer, val_metrics, epoch, prefix="Val")

        # Early Stopping Check
        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info("  -> Best model saved! (AUROC: %.4f)", best_val_auroc)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                logger.warning("Early stopping triggered at epoch %d", epoch)
                break

    # 4. Final Evaluation on Test Set
    logger.info("--- Evaluating Best Model on Test Set ---")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    # Evaluate best model on validation set
    val_true_final, val_prob_final, _ = evaluate_model(model, val_loader)
    val_metrics_final = compute_metrics(val_true_final, val_prob_final)
    save_metrics(VAL_METRICS_SAVE_PATH, val_metrics_final)

    test_true, test_prob, _ = evaluate_model(model, test_loader)
    test_pred = (test_prob >= 0.5).astype(int)

    test_metrics = compute_metrics(test_true, test_prob)
    print_metrics(test_metrics, name="TEST SET (PyTorch MLP)")

    log_metrics_to_tensorboard(writer, test_metrics, global_step=0, prefix="Test")
    writer.close()

    # 5. Save Artifacts using Centralized Evaluator
    save_metrics(METRICS_SAVE_PATH, test_metrics)
    save_predictions(PREDICTIONS_SAVE_PATH, test_ids, test_true, test_prob, test_pred)

    logger.info("ECG MLP training pipeline complete!")


if __name__ == "__main__":
    main()
