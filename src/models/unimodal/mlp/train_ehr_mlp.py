"""Train a two-layer MLP classifier on EHR (MOTOR) embeddings.

Loads the standardised ``.pt`` embedding files produced by
``extract_ehr_embeddings.py``, trains a small feed-forward network with
dropout and batch normalisation, evaluates on validation and test splits,
and persists:

- Model checkpoint (``model.pt``)
- Per-sample test predictions (``test_predictions.csv``)
- Metrics JSON (``test_metrics.json``, ``val_metrics.json``)
- TensorBoard scalars (loss curves + final metrics)

Architecture::

    Linear(D, 256) → BN → ReLU → Dropout(0.3)
    Linear(256, 64) → BN → ReLU → Dropout(0.3)
    Linear(64, 1)   → Sigmoid

Usage:
    python -m src.models.unimodal.mlp.train_ehr_mlp
"""

import logging

import numpy as np
import torch
import torch.nn as nn
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
# CONSTANTS
# ==========================================

HIDDEN_1 = 256
HIDDEN_2 = 64
DROPOUT = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
NUM_EPOCHS = 100
PATIENCE = 10  # early-stopping patience (val AUROC)
RANDOM_STATE = 42


# ==========================================
# MODEL
# ==========================================


class SepsisMLPClassifier(nn.Module):
    """Two-hidden-layer MLP for binary classification."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_1),
            nn.BatchNorm1d(HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.BatchNorm1d(HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==========================================
# HELPERS
# ==========================================


def _make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    """Wrap numpy arrays into a DataLoader."""
    ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, pin_memory=True)


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Return predicted probabilities for every sample in *loader*."""
    model.eval()
    probs = []
    for X_batch, _ in loader:
        logits = model(X_batch.to(device)).squeeze(-1)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


# ==========================================
# MAIN
# ==========================================


def main():
    Config.setup_logging()
    torch.manual_seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    embeddings_dir = Config.EHR_EMBEDDINGS_DIR
    output_dir = Config.EHR_MLP_MODEL_DIR
    results_dir = Config.RESULTS_DIR / "ehr" / "mlp"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load embeddings
    # ------------------------------------------------------------------
    X_train, y_train, _ = load_embeddings(embeddings_dir / "train_embeddings.pt")
    X_val, y_val, val_ids = load_embeddings(embeddings_dir / "valid_embeddings.pt")
    X_test, y_test, test_ids = load_embeddings(embeddings_dir / "test_embeddings.pt")

    train_loader = _make_loader(X_train, y_train, shuffle=True)
    val_loader = _make_loader(X_val, y_val, shuffle=False)
    test_loader = _make_loader(X_test, y_test, shuffle=False)

    input_dim = X_train.shape[1]
    logger.info("Input dim: %d", input_dim)

    # ------------------------------------------------------------------
    # 2. Model, loss, optimizer
    # ------------------------------------------------------------------
    model = SepsisMLPClassifier(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    tb_dir = Config.TENSORBOARD_LOG_DIR / "ehr" / "mlp"
    writer = SummaryWriter(log_dir=str(tb_dir))
    logger.info("TensorBoard logs → %s", tb_dir)

    # ------------------------------------------------------------------
    # 3. Training loop with early stopping
    # ------------------------------------------------------------------
    best_val_auroc = 0.0
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch).squeeze(-1)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # --- Validation ---
        val_prob = _predict(model, val_loader, device)
        val_m = compute_metrics(y_val, val_prob)

        writer.add_scalar("train/loss", avg_train_loss, epoch)
        writer.add_scalar("val/auroc", val_m["auroc"], epoch)
        writer.add_scalar("val/auprc", val_m["auprc"], epoch)
        writer.add_scalar("val/f1", val_m["f1"], epoch)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d | train_loss=%.4f | val_auroc=%.4f | val_auprc=%.4f",
                epoch,
                avg_train_loss,
                val_m["auroc"],
                val_m["auprc"],
            )

        # --- Early stopping ---
        if val_m["auroc"] > best_val_auroc:
            best_val_auroc = val_m["auroc"]
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "model.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(
                    "Early stopping at epoch %d (best val AUROC %.4f)",
                    epoch,
                    best_val_auroc,
                )
                break

    # ------------------------------------------------------------------
    # 4. Load best model and evaluate on test
    # ------------------------------------------------------------------
    model.load_state_dict(
        torch.load(output_dir / "model.pt", map_location=device, weights_only=True)
    )
    val_prob = _predict(model, val_loader, device)
    test_prob = _predict(model, test_loader, device)

    val_metrics = compute_metrics(y_val, val_prob)
    test_metrics = compute_metrics(y_test, test_prob)

    print_metrics(val_metrics, "EHR MLP — Validation")
    print_metrics(test_metrics, "EHR MLP — Test")

    # Final metrics to TensorBoard
    log_metrics_to_tensorboard(writer, val_metrics, global_step=0, prefix="final_val")
    log_metrics_to_tensorboard(writer, test_metrics, global_step=0, prefix="final_test")
    writer.close()

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    save_predictions(results_dir / "test_predictions.csv", test_ids, y_test, test_prob)
    save_metrics(results_dir / "val_metrics.json", val_metrics)
    save_metrics(results_dir / "test_metrics.json", test_metrics)

    logger.info("EHR MLP training complete (best val AUROC: %.4f)", best_val_auroc)


if __name__ == "__main__":
    main()
