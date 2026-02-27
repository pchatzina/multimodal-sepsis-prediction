"""
End-to-End Inference & Local Explainability Pipeline.

Simulates a retrospective Clinical Decision Support system. Takes a single patient's
subject_id, dynamically extracts features from raw CXR images and text reports using
Foundation Models, retrieves their EHR representation, and processes them through the
calibrated 3-Modality Late-Fusion Champion Model. Outputs the clinical risk score
alongside hierarchical macro (fusion gating) and micro (EHR token) explainability.

Change captum_path to explore different cases (ehr_token_attributions.csv or champion_ehr_token_attributions.csv).

Usage:
    python -m src.scripts.inference.end_to_end_pipeline --subject_id 10001884
"""

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchxrayvision as xrv
from transformers import AutoModel, AutoTokenizer

from src.models.fusion.late_fusion_model import LateFusionSepsisModel
from src.utils.config import Config
from src.utils.database import query_to_df

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTIVE_MODALITIES = ["ehr", "img", "txt"]
NUM_MODS = len(ACTIVE_MODALITIES)
CHAMPION_VARIANT = "pretrained_ehr_dropout"


class PatientIngestor:
    """Handles raw data location and pre-computed EHR tensor retrieval."""

    def __init__(self, split: str = "test"):
        self.split = split
        self.ehr_data = torch.load(
            Config.EHR_EMBEDDINGS_DIR / f"{split}_embeddings.pt", map_location="cpu"
        )
        self.ehr_idx_map = {
            sid: idx for idx, sid in enumerate(self.ehr_data["sample_ids"])
        }

    def get_patient_data(self, subject_id: int) -> dict:
        if subject_id not in self.ehr_idx_map:
            raise ValueError(
                f"Subject {subject_id} not found in {self.split} EHR embeddings."
            )

        ehr_idx = self.ehr_idx_map[subject_id]
        ehr_tensor = self.ehr_data["embeddings"][ehr_idx].unsqueeze(0)
        true_label = self.ehr_data["labels"][ehr_idx]

        query = f"SELECT study_id, study_path FROM mimiciv_ext.cohort_cxr WHERE subject_id = {subject_id} LIMIT 1;"
        df = query_to_df(query)

        # Gracefully handle missing CXR
        if df.empty:
            img_path = None
            txt_path = None
            logger.info(
                f"Subject {subject_id} has no CXR records. Proceeding with EHR-Only inference."
            )
        else:
            study_id = df.iloc[0]["study_id"]
            img_path = Config.RAW_CXR_IMG_DIR / df.iloc[0]["study_path"]
            subject_str = str(subject_id)
            txt_path = (
                Config.RAW_CXR_TXT_DIR
                / "mimic-cxr-reports"
                / "files"
                / f"p{subject_str[:2]}"
                / f"p{subject_str}"
                / f"s{study_id}.txt"
            )

        return {
            "subject_id": subject_id,
            "true_label": true_label,
            "ehr_tensor": ehr_tensor,
            "img_path": img_path,
            "txt_path": txt_path,
            "has_img": img_path is not None and img_path.exists(),
            "has_txt": txt_path is not None and txt_path.exists(),
            "patient_index": ehr_idx,
        }


class FeatureEmbedder:
    """Dynamically embeds raw unstructured data using frozen Foundation Models."""

    def __init__(self):
        logger.info("Loading DenseNet121 Foundation Model...")
        self.img_model = xrv.models.DenseNet(weights="densenet121-res224-all").to(
            DEVICE
        )
        self.img_model.eval()
        self.img_transform = transforms.Compose(
            [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
        )

        logger.info("Loading Bio_ClinicalBERT Foundation Model...")
        model_name = "emilyalsentzer/Bio_ClinicalBERT"
        cache_dir = Config.CXR_TXT_PRETRAINED_MODEL_DIR
        self.txt_tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.txt_model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(
            DEVICE
        )
        self.txt_model.eval()

    @torch.inference_mode()
    def embed_image(self, img_path: Path | None) -> torch.Tensor:
        if img_path is None or not img_path.exists():
            return torch.zeros((1, 1024), device=DEVICE)

        img = skimage.io.imread(img_path, as_gray=True)
        img = xrv.datasets.normalize(img, 255)[None, ...]
        img_tensor = (
            torch.from_numpy(self.img_transform(img)).float().unsqueeze(0).to(DEVICE)
        )
        features = self.img_model.features(img_tensor)
        features = F.relu(features, inplace=True)
        return F.adaptive_avg_pool2d(features, (1, 1)).view(1, 1024)

    @torch.inference_mode()
    def embed_text(self, txt_path: Path | None) -> torch.Tensor:
        if txt_path is None or not txt_path.exists():
            return torch.zeros((1, 768), device=DEVICE)

        with open(txt_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        clean_text = re.sub(r"\s+", " ", re.sub(r"_+", " ", raw_text)).strip().lower()
        inputs = self.txt_tokenizer(
            clean_text, return_tensors="pt", truncation=True, max_length=512
        ).to(DEVICE)
        return self.txt_model(**inputs).last_hidden_state[:, 0, :]


class SepsisPredictor:
    """Executes the fusion forward pass and applies post-hoc temperature scaling."""

    def __init__(self):
        logger.info("Loading 3-Modality Champion Model...")

        # 1. Load configs
        fusion_dir = Config.RESULTS_DIR / "fusion"
        tuning_file = (
            fusion_dir
            / "tuning"
            / f"best_hyperparameters_{NUM_MODS}mod_pretrained.json"
        )
        with open(tuning_file, "r") as f:
            best_params = json.load(f)["params"]

        unimodal_configs = {}
        mod_map = {"ehr": "ehr", "img": "cxr_img", "txt": "cxr_txt"}
        for mod in ACTIVE_MODALITIES:
            with open(
                Config.RESULTS_DIR
                / mod_map[mod]
                / "tuning"
                / "best_hyperparameters.json",
                "r",
            ) as f:
                unimodal_configs[mod] = json.load(f)["params"]

        # 2. Instantiate Model
        input_dims = {"ehr": 768, "img": 1024, "txt": 768}
        self.model = LateFusionSepsisModel(
            input_dims=input_dims,
            config=best_params,
            unimodal_configs=unimodal_configs,
            common_dim=768,
            active_modalities=ACTIVE_MODALITIES,
        ).to(DEVICE)

        weights_path = (
            Config.FUSION_MODEL_DIR
            / f"best_late_fusion_model_{NUM_MODS}mod_{CHAMPION_VARIANT}.pt"
        )
        self.model.load_state_dict(
            torch.load(weights_path, map_location=DEVICE, weights_only=True)
        )
        self.model.eval()

        # 3. Load Calibration Temperature
        temps_path = fusion_dir / f"master_calibration_temperatures_{NUM_MODS}mod.json"
        with open(temps_path, "r") as f:
            self.t_val = json.load(f).get(CHAMPION_VARIANT, {}).get("final", 1.0)
        logger.info(f"Loaded Optimal Calibration Temperature: T={self.t_val:.3f}")

    @torch.inference_mode()
    def predict(
        self, ehr_tensor, img_tensor, txt_tensor, has_img: bool, has_txt: bool
    ) -> dict:
        embeddings = {
            "ehr": ehr_tensor.to(DEVICE),
            "img": img_tensor.to(DEVICE),
            "txt": txt_tensor.to(DEVICE),
        }

        # Dynamically build masks based on actual file existence
        masks = {
            "ehr": torch.tensor([[1.0]], device=DEVICE),
            "img": torch.tensor([[1.0 if has_img else 0.0]], device=DEVICE),
            "txt": torch.tensor([[1.0 if has_txt else 0.0]], device=DEVICE),
        }

        outputs = self.model(embeddings, masks)

        p_uncalibrated = outputs["p_final"].item()
        p_clipped = max(min(p_uncalibrated, 1 - 1e-7), 1e-7)
        logit_final = torch.tensor(np.log(p_clipped / (1 - p_clipped)))
        p_calibrated = torch.sigmoid(logit_final / self.t_val).item()

        return {
            "risk_score": p_calibrated,
            "weights": outputs["weights"].cpu().numpy().flatten(),
            "beta": outputs["beta"].item(),
            "masks": masks,  # Pass back for debugging if needed
        }


class HierarchicalExplainer:
    """Provides local macro and micro explainability for the prediction."""

    def __init__(self):
        self.captum_path = (
            Config.RESULTS_DIR
            / "explainability"
            / "champion_ehr_token_attributions.csv"
        )

    def explain_macro(self, weights: list, beta: float):
        print("\n" + "=" * 60)
        print("🧠 MACRO-LEVEL EXPLAINABILITY (Gating Network Fusion)")
        print("=" * 60)
        print(f"Synergy Contribution (Beta): {beta:.2%}")
        print("Modality Attention Weights:")
        for mod, w in zip(ACTIVE_MODALITIES, weights):
            print(f"  - {mod.upper()}: {w:.2%}")

    def explain_micro(self, subject_id: int):  # Changed argument
        if not self.captum_path.exists():
            print(
                "\n[Warning] Micro-level explainer requires pre-computed Captum attributions."
            )
            return

        df = pd.read_csv(self.captum_path)
        patient_df = df[df["subject_id"] == subject_id]  # Changed filter

        if patient_df.empty:
            print(
                f"\n[Warning] No Captum attributions found for subject ID {subject_id}."
            )
            return

        patient_df = patient_df.sort_values(by="attribution_score", ascending=False)

        print("\n" + "=" * 60)
        print("🏥 MICRO-LEVEL EXPLAINABILITY (EHR Token Vignette)")
        print("=" * 60)
        print("🔴 TOP 3 CLINICAL FACTORS DRIVING SEPSIS RISK UP:")
        for _, row in patient_df.head(3).iterrows():
            print(f"  [+{row['attribution_score']:.2e}] {row['token_string']}")

        print("\n🔵 TOP 3 CLINICAL FACTORS DRIVING SEPSIS RISK DOWN:")
        for _, row in (
            patient_df.tail(3)
            .sort_values(by="attribution_score", ascending=True)
            .iterrows()
        ):
            print(f"  [{row['attribution_score']:.2e}] {row['token_string']}")
        print("=" * 60 + "\n")


def main():
    Config.setup_logging()
    Config.set_seed(42)

    parser = argparse.ArgumentParser(
        description="End-to-End Multimodal Sepsis Inference."
    )
    parser.add_argument(
        "--subject_id", type=int, required=True, help="Patient Subject ID to analyze."
    )
    args = parser.parse_args()

    print(f"\nInitializing End-to-End Pipeline for Subject ID: {args.subject_id}...\n")

    # 1. Initialize Pipeline Modules
    ingestor = PatientIngestor()
    embedder = FeatureEmbedder()
    predictor = SepsisPredictor()
    explainer = HierarchicalExplainer()

    # 2. Ingest Data
    patient_data = ingestor.get_patient_data(args.subject_id)

    # 3. Dynamic Feature Embedding
    if patient_data["has_img"]:
        logger.info(f"Processing Raw CXR Image: {patient_data['img_path'].name}")
    else:
        logger.info("No CXR Image found. Using zero-tensor mask.")
    img_tensor = embedder.embed_image(patient_data["img_path"])

    if patient_data["has_txt"]:
        logger.info(f"Processing Raw Clinical Note: {patient_data['txt_path'].name}")
    else:
        logger.info("No Clinical Note found. Using zero-tensor mask.")
    txt_tensor = embedder.embed_text(patient_data["txt_path"])

    # 4. Predict
    logger.info("Executing Late-Fusion Forward Pass & Calibration...")
    results = predictor.predict(
        patient_data["ehr_tensor"],
        img_tensor,
        txt_tensor,
        patient_data["has_img"],
        patient_data["has_txt"],
    )

    # 5. Output
    print("\n" + "!" * 60)
    print(f"⚕️  CLINICAL PREDICTION")
    print("!" * 60)
    print(f"Patient Subject ID: {args.subject_id}")
    print(f"True Sepsis Label:  {bool(patient_data['true_label'])}")
    print(f"Model Sepsis Risk:  {results['risk_score']:.2%} (Calibrated)")

    # 6. Explainability
    explainer.explain_macro(results["weights"], results["beta"])

    # We pass the subject_id instead of patient_index for the updated micro-explainer
    explainer.explain_micro(args.subject_id)


if __name__ == "__main__":
    main()
