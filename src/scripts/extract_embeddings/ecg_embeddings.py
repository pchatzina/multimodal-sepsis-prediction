"""
Extract ECG embeddings using the pretrained wanglab/ecg-fm Foundation Model.

This script automatically downloads the pretrained checkpoint, loads 10-second
whole-ECG waveforms based on manifest files, dynamically chunks them into
5-second windows (to match model pretraining), and pools the outputs into
a single patient-level representation.

Outputs are saved as .pt files ready for the unimodal classifiers.

Usage:
    python -m src.scripts.extract_embeddings.ecg_embeddings
"""

import logging
import torch
import scipy.io as sio
from tqdm import tqdm
from pathlib import Path

from huggingface_hub import hf_hub_download
from fairseq_signals.models import build_model_from_checkpoint
from src.utils.config import Config

logger = logging.getLogger(__name__)


class ECGEmbeddingExtractor:
    def __init__(self, checkpoint_path: Path, device: str = "cuda"):
        """Initialize the embedding extractor and load the Foundation Model."""
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = self._download_and_load_model()

    def _download_and_load_model(self):
        """Downloads the ecg-fm model if missing, then loads it into memory."""
        if not self.checkpoint_path.exists():
            logger.info(
                "Downloading wanglab/ecg-fm to %s...", self.checkpoint_path.parent
            )
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id="wanglab/ecg-fm",
                filename="mimic_iv_ecg_physionet_pretrained.pt",
                local_dir=self.checkpoint_path.parent,
                local_dir_use_symlinks=False,
            )

        logger.info("Loading ecg-fm checkpoint from %s", self.checkpoint_path)

        # Depending on fairseq_signals version, this may return a tuple or just the model.
        # We handle both cases safely.
        result = build_model_from_checkpoint(str(self.checkpoint_path))
        if isinstance(result, tuple) or isinstance(result, list):
            model = result[0][0] if isinstance(result[0], list) else result[0]
        else:
            model = result

        model = model.to(self.device)
        model.eval()
        logger.info("Foundation Model successfully loaded and set to eval mode.")
        return model

    def load_ecg_from_mat(self, mat_path: Path) -> torch.Tensor:
        """Loads ECG data, handles flat/missing leads (NaNs), and ensures shape."""
        mat_data = sio.loadmat(str(mat_path))

        if "feats" not in mat_data:
            raise KeyError(
                f"Cannot find 'feats' in {mat_path.name}. Keys: {mat_data.keys()}"
            )

        signal = torch.from_numpy(mat_data["feats"]).float()

        # Handle NaNs (flat leads) found during our pytest audit
        if torch.isnan(signal).any() or torch.isinf(signal).any():
            signal = torch.nan_to_num(signal, nan=0.0, posinf=1e5, neginf=-1e5)

        # Ensure standard shape: [12 leads, N samples]
        if signal.shape[0] > signal.shape[1]:
            signal = signal.T

        return signal

    def extract_embedding(self, ecg_signal: torch.Tensor) -> torch.Tensor:
        """
        Extracts embedding from a whole-ECG signal.
        Dynamically slices 10s waveforms into 5s chunks, processes them,
        and averages the outputs for a single patient-level vector.
        """
        n_leads, n_samples = ecg_signal.shape
        chunk_size = 2500  # 5 seconds at 500 Hz (ecg-fm standard)

        chunks = []
        # Slice the waveform into chunks
        for i in range(0, n_samples, chunk_size):
            chunk = ecg_signal[:, i : i + chunk_size]
            # Pad if the last chunk is shorter than 5 seconds
            if chunk.shape[1] < chunk_size:
                pad_len = chunk_size - chunk.shape[1]
                chunk = torch.nn.functional.pad(chunk, (0, pad_len))
            chunks.append(chunk)

        # Batch the chunks: [num_chunks, 12, 2500]
        batch = torch.stack(chunks).to(self.device)

        with torch.no_grad():
            output = self.model.extract_features(batch, padding_mask=None)

            # Safely extract the encoder output depending on fairseq dictionary/tuple returns
            if isinstance(output, dict) and "x" in output:
                encoder_out = output["x"]
            elif isinstance(output, tuple):
                encoder_out = output[0]
            else:
                encoder_out = output

            # 1. Temporal Pooling: Mean pool across the sequence length for each chunk
            # encoder_out shape: [num_chunks, seq_len, hidden_dim]
            chunk_embeddings = encoder_out.mean(dim=1)  # [num_chunks, hidden_dim]

            # 2. Window Pooling: Mean pool across the chunks to get 1 representation per patient
            patient_embedding = chunk_embeddings.mean(dim=0)  # [hidden_dim]

        return patient_embedding.cpu()

    def process_manifest(self, manifest_dir: Path, split_name: str, mat_dir: Path):
        """Processes all ECGs in a manifest and extracts their embeddings."""
        tsv_path = manifest_dir / f"{split_name}.tsv"
        lbl_path = manifest_dir / f"{split_name}.lbl"

        if not tsv_path.exists():
            logger.warning("Manifest %s not found. Skipping.", tsv_path.name)
            return None, None, None

        # Read TSV (skipping the first line which is just the root path)
        with open(tsv_path, "r") as f:
            lines = f.readlines()
        file_info = [line.strip().split("\t") for line in lines[1:]]

        # Read LBL labels
        labels = []
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                labels = [int(line.strip()) for line in f.readlines()]

        embeddings_list = []
        sample_ids = []
        valid_labels = []

        logger.info("Processing %s split: %d samples", split_name, len(file_info))

        for idx, file_entry in enumerate(
            tqdm(file_info, desc=f"Extracting {split_name}")
        ):
            fname = file_entry[0]
            sample_id = int(fname.replace(".mat", ""))  # Maps strictly to subject_id

            mat_path = mat_dir / fname

            if not mat_path.exists():
                logger.debug("Missing file %s, skipping.", fname)
                continue

            try:
                ecg_signal = self.load_ecg_from_mat(mat_path)
                embedding = self.extract_embedding(ecg_signal)

                embeddings_list.append(embedding)
                sample_ids.append(sample_id)

                if labels:
                    valid_labels.append(labels[idx])

            except Exception as e:
                logger.error("Error processing %s: %s", fname, str(e))
                continue

        if not embeddings_list:
            return None, None, None

        # Stack into final matrix: [n_samples, hidden_dim]
        embeddings = torch.stack(embeddings_list, dim=0)
        logger.info(
            "Extracted %d embeddings of dimension %d.",
            embeddings.shape[0],
            embeddings.shape[1],
        )

        return embeddings, valid_labels, sample_ids


def main():
    Config.setup_logging()

    # Configuration
    CHECKPOINT_PATH = (
        Config.ECG_PRETRAINED_MODEL_DIR / "mimic_iv_ecg_physionet_pretrained.pt"
    )
    MANIFEST_DIR = Config.ECG_MANIFEST_DIR
    MAT_DIR = Config.ECG_PROCESSED_ROOT_DIR / "preprocessed"
    OUTPUT_DIR = Config.ECG_EMBEDDINGS_DIR

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Initializing ECG Embedding Extractor on %s...", device.upper())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extractor = ECGEmbeddingExtractor(CHECKPOINT_PATH, device=device)

    splits = ["train", "valid", "test"]

    for split in splits:
        logger.info("--- Starting %s split ---", split.upper())

        embeddings, labels, sample_ids = extractor.process_manifest(
            MANIFEST_DIR, split, MAT_DIR
        )

        if embeddings is None:
            continue

        output_data = {
            "embeddings": embeddings,
            "labels": labels,
            "sample_ids": sample_ids,
        }

        output_path = OUTPUT_DIR / f"{split}_embeddings.pt"
        torch.save(output_data, output_path)

        logger.info("Saved %s to %s", split.upper(), output_path.name)
        logger.info("  -> Embeddings shape: %s", list(embeddings.shape))

    logger.info("All embedding extractions complete!")


if __name__ == "__main__":
    main()
