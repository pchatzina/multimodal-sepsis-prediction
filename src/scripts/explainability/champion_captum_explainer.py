"""
Applies Captum Layer Gradient X Activation to the 3-Modality Champion Model.
Maps Sepsis risk predictions back to specific medical tokens in a patient's history
by explicitly slicing the femr memory batches.

Usage:
    python -m src.scripts.explainability.champion_captum_explainer --subject_ids 11540283 19004463
"""

import argparse
import json
import logging
import pickle
import msgpack
import datasets
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import LayerGradientXActivation

import femr.models.tokenizer
import femr.models.transformer

from src.utils.config import Config
from src.models.fusion.late_fusion_model import LateFusionSepsisModel

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTIVE_MODALITIES = ["ehr", "img", "txt"]
NUM_MODS = len(ACTIVE_MODALITIES)
CHAMPION_VARIANT = "pretrained_ehr_dropout"


class ChampionEHRBranch(nn.Module):
    """Seamlessly links the fusion model's EHR projector and EHR MLP."""

    def __init__(self, projector: nn.Module, mlp: nn.Module):
        super().__init__()
        self.projector = projector
        self.mlp = mlp

    def forward(self, x):
        return self.mlp(self.projector(x))


class EHREndToEndWrapper(nn.Module):
    """Fuses the frozen MOTOR Transformer and the extracted Champion EHR Branch."""

    def __init__(self, motor_model: nn.Module, champion_ehr_branch: nn.Module):
        super().__init__()
        self.motor = motor_model
        self.motor.config.task_config = None  # Bypass pretraining task head
        self.ehr_branch = champion_ehr_branch

        self.motor.eval()
        for param in self.motor.parameters():
            param.requires_grad = False
        self.ehr_branch.eval()

    def forward(self, dummy_input, batch_dict):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, motor_outputs = self.motor(
                batch_dict, return_loss=True, return_reprs=True
            )
            patient_rep = motor_outputs["representations"]
            logits = self.ehr_branch(patient_rep)
        return torch.sigmoid(logits).to(torch.float32)


def load_champion_end_to_end_models():
    """Loads MOTOR and extracts the fine-tuned EHR branch from the Champion Fusion Model."""
    logger.info("Loading Tokenizer and MOTOR Foundation Model...")
    ontology_path = Config.MOTOR_PRETRAINING_FILES_DIR / "ontology.pkl"
    with open(ontology_path, "rb") as f:
        ontology = pickle.load(f)

    tokenizer = femr.models.tokenizer.HierarchicalTokenizer.from_pretrained(
        Config.MOTOR_MODEL_DIR, ontology=ontology
    )
    motor_model = femr.models.transformer.FEMRModel.from_pretrained(
        str(Config.MOTOR_MODEL_DIR)
    ).to(DEVICE)

    logger.info("Loading 3-Modality Champion Late-Fusion Model...")
    fusion_dir = Config.RESULTS_DIR / "fusion"
    tuning_file = (
        fusion_dir / "tuning" / f"best_hyperparameters_{NUM_MODS}mod_pretrained.json"
    )
    with open(tuning_file, "r") as f:
        best_params = json.load(f)["params"]

    unimodal_configs = {}
    mod_map = {"ehr": "ehr", "img": "cxr_img", "txt": "cxr_txt"}
    for mod in ACTIVE_MODALITIES:
        with open(
            Config.RESULTS_DIR / mod_map[mod] / "tuning" / "best_hyperparameters.json",
            "r",
        ) as f:
            unimodal_configs[mod] = json.load(f)["params"]

    input_dims = {"ehr": 768, "img": 1024, "txt": 768}
    fusion_model = LateFusionSepsisModel(
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
    fusion_model.load_state_dict(
        torch.load(weights_path, map_location=DEVICE, weights_only=True)
    )

    champion_ehr_branch = ChampionEHRBranch(
        projector=fusion_model.projectors["ehr"], mlp=fusion_model.unimodal_mlps["ehr"]
    )

    return EHREndToEndWrapper(motor_model, champion_ehr_branch), tokenizer


def prepare_batch(data):
    """Recursively pushes batch dictionaries to the GPU."""
    if isinstance(data, dict):
        return {k: prepare_batch(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.unsqueeze(0).to(DEVICE)
    return data


def load_raw_msgpack_dictionary():
    dict_path = Config.MOTOR_MODEL_DIR / "dictionary.msgpack"
    with open(dict_path, "rb") as f:
        return msgpack.unpackb(f.read())["vocab"]


def get_token_string_resilient(vocab_list, token_id):
    """Parses FEMR dictionary schemas robustly."""
    token_id = int(token_id)
    if token_id >= len(vocab_list) or token_id < 0:
        return f"Out_of_Bounds_ID_{token_id}"
    item = vocab_list[token_id]
    try:
        if item.get("type") == "code" and "code_string" in item:
            return item["code_string"].strip()
        elif item.get("type") == "text" and "text_string" in item:
            return (
                f"{item.get('property', 'text').strip()}: {item['text_string'].strip()}"
            )
        elif "val_start" in item and "val_end" in item:
            prop = item.get(
                "property", item.get("code_string", "Numeric_Feature")
            ).strip()
            start_str = (
                f"{item['val_start']:.2f}"
                if isinstance(item["val_start"], float)
                else str(item["val_start"])
            )
            end_str = (
                f"{item['val_end']:.2f}"
                if isinstance(item["val_end"], float)
                else str(item["val_end"])
            )
            return f"{prop}: [{start_str}, {end_str})"
        else:
            return str({k: v for k, v in item.items() if k != "weight"})
    except Exception:
        return f"Parse_Error_ID_{token_id}"


def run_captum_explainability(wrapper, test_batches_path, target_sids: list):
    logger.info("Initializing Captum Layer Gradient X Activation...")
    embedding_layer = wrapper.motor.transformer.embed_bag
    attr_algo = LayerGradientXActivation(forward_func=wrapper, layer=embedding_layer)

    test_batches = datasets.Dataset.load_from_disk(test_batches_path)
    test_batches.set_format("pt")
    raw_vocab = load_raw_msgpack_dictionary()
    attributions_list = []

    for target_sid in target_sids:
        # 1. Locate the batch and the exact patient position
        target_batch_idx, patient_pos = None, None
        for b_idx, batch in enumerate(test_batches):
            sids = batch["subject_ids"].numpy().tolist()
            if target_sid in sids:
                target_batch_idx = b_idx

                # --- BUG FIX: Map the flat token array back to the unique patient array ---
                unique_sids = []
                for sid in sids:
                    if not unique_sids or unique_sids[-1] != sid:
                        unique_sids.append(sid)

                patient_pos = unique_sids.index(target_sid)
                break

        if target_batch_idx is None:
            logger.warning(f"Subject {target_sid} not found in test batches. Skipping.")
            continue

        # 2. Run Captum on the specific batch
        batch = test_batches[target_batch_idx]
        batch_gpu = prepare_batch(batch)
        dummy_input = torch.empty(0).to(DEVICE)

        attributions = attr_algo.attribute(
            inputs=dummy_input, additional_forward_args=(batch_gpu,), target=0
        )
        token_attributions = attributions.sum(dim=-1).squeeze().cpu().detach().numpy()
        tokens_flat = (
            batch_gpu["transformer"]["hierarchical_tokens"].squeeze().cpu().numpy()
        )

        # 3. Mathematically slice the exact patient tokens from the packed batch array
        lengths = batch["transformer"]["subject_lengths"].numpy()
        start_idx = int(lengths[:patient_pos].sum())
        end_idx = start_idx + int(lengths[patient_pos])

        patient_tokens = tokens_flat[start_idx:end_idx]
        patient_attrs = token_attributions[start_idx:end_idx]

        for token_id, attr_score in zip(patient_tokens, patient_attrs):
            token_str = get_token_string_resilient(raw_vocab, token_id)
            attributions_list.append(
                {
                    "subject_id": target_sid,
                    "token_id": token_id,
                    "token_string": token_str,
                    "attribution_score": float(attr_score),
                }
            )

        logger.info(f"Extracted {end_idx - start_idx} tokens for Subject {target_sid}")

    # 4. Save
    if attributions_list:
        df = pd.DataFrame(attributions_list)
        out_path = (
            Config.RESULTS_DIR
            / "explainability"
            / "champion_ehr_token_attributions.csv"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info(f"Saved specific patient attributions to {out_path}")


def main():
    Config.setup_logging()

    parser = argparse.ArgumentParser(description="Run Captum on explicit Subject IDs.")
    parser.add_argument(
        "--subject_ids",
        nargs="+",
        type=int,
        required=True,
        help="List of explicit Subject IDs",
    )
    args = parser.parse_args()

    wrapper, _ = load_champion_end_to_end_models()
    test_batches_path = Config.RESULTS_DIR / "explainability" / "test_batches"

    run_captum_explainability(wrapper, test_batches_path, args.subject_ids)


if __name__ == "__main__":
    main()
