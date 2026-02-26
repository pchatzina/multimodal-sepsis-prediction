"""
Applies Captum Layer Gradient X Activation to the end-to-end EHR pipeline.
Maps Sepsis risk predictions back to specific medical tokens in the patient's history.

Usage:
    python -m src.scripts.explainability.ehr_captum_explainer
"""

import json
import logging
import pickle
import traceback
from pathlib import Path
import msgpack

import datasets
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import LayerGradientXActivation

import femr.models.processor
import femr.models.tokenizer
import femr.models.transformer

from src.utils.config import Config
from src.models.unimodal.mlp.train_unimodal_mlp import DynamicModalityMLP

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EHREndToEndWrapper(nn.Module):
    """
    Fuses the frozen MOTOR Transformer and the trained EHR MLP into a single module.
    """

    def __init__(self, motor_model: nn.Module, mlp_model: nn.Module):
        super().__init__()
        self.motor = motor_model

        # Bypass the pretraining task head. We only want embeddings!
        self.motor.config.task_config = None

        self.mlp = mlp_model

        self.motor.eval()
        for param in self.motor.parameters():
            param.requires_grad = False

        self.mlp.eval()

    def forward(self, dummy_input, batch_dict):
        # Force bfloat16 to satisfy xformers FlashAttention constraints on modern GPUs
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, motor_outputs = self.motor(
                batch_dict, return_loss=True, return_reprs=True
            )

            patient_rep = motor_outputs["representations"]
            logits = self.mlp(patient_rep)

        # Cast back to float32 so Captum's gradient accumulators don't underflow
        return torch.sigmoid(logits).to(torch.float32)


def load_end_to_end_models():
    """Loads the tokenizer, natively loaded MOTOR model, and trained EHR MLP."""
    logger.info("Loading Tokenizer and MOTOR Foundation Model...")

    ontology_path = Config.MOTOR_PRETRAINING_FILES_DIR / "ontology.pkl"
    with open(ontology_path, "rb") as f:
        ontology = pickle.load(f)

    tokenizer = femr.models.tokenizer.HierarchicalTokenizer.from_pretrained(
        Config.MOTOR_MODEL_DIR, ontology=ontology
    )

    motor_model = femr.models.transformer.FEMRModel.from_pretrained(
        str(Config.MOTOR_MODEL_DIR)
    )
    motor_model.to(DEVICE)

    logger.info("Loading Trained Unimodal EHR MLP...")
    tuning_file = Config.RESULTS_DIR / "ehr" / "tuning" / "best_hyperparameters.json"
    with open(tuning_file, "r") as f:
        best_params = json.load(f)["params"]

    mlp_model = DynamicModalityMLP(input_dim=768, config=best_params)
    mlp_model.load_state_dict(
        torch.load(
            Config.EHR_MLP_MODEL_DIR / "best_ehr_mlp.pt",
            map_location=DEVICE,
            weights_only=True,
        )
    )
    mlp_model.to(DEVICE)

    wrapper = EHREndToEndWrapper(motor_model, mlp_model)
    return wrapper, tokenizer


def prepare_batch(data):
    """Recursively adds a batch dimension (shape[0]=1) to all nested tensors."""
    if isinstance(data, dict):
        return {k: prepare_batch(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.unsqueeze(0).to(DEVICE)
    else:
        return data


def load_raw_msgpack_dictionary():
    """Loads the msgpack and explicitly returns the 'vocab' list."""
    dict_path = Config.MOTOR_MODEL_DIR / "dictionary.msgpack"
    with open(dict_path, "rb") as f:
        data = msgpack.unpackb(f.read())
        return data["vocab"]  # This is the list of dictionaries you found!


def get_token_string_resilient(vocab_list, token_id):
    """Parses the exact 3 schemas found in the FEMR vocab list."""
    token_id = int(token_id)

    # Boundary check
    if token_id >= len(vocab_list) or token_id < 0:
        return f"Out_of_Bounds_ID_{token_id}"

    item = vocab_list[token_id]

    try:
        # Schema 1: Standard Clinical Code (e.g., ICD9, LOINC, RxNorm)
        if item.get("type") == "code" and "code_string" in item:
            return item["code_string"].strip()

        # Schema 2: Text Property (e.g., caregiver_id, text_value)
        elif item.get("type") == "text" and "text_string" in item:
            prop = item.get("property", "text").strip()
            val = item["text_string"].strip()
            return f"{prop}: {val}"

        # Schema 3: Binned Numeric Value (e.g., Labs, Vitals)
        elif "val_start" in item and "val_end" in item:
            # Sometimes the property name is stored under 'property', sometimes under 'code_string'
            prop = item.get(
                "property", item.get("code_string", "Numeric_Feature")
            ).strip()

            # Format the bin nicely for the CSV
            start = item["val_start"]
            end = item["val_end"]

            # Format to 2 decimal places if it's a float
            start_str = f"{start:.2f}" if isinstance(start, float) else str(start)
            end_str = f"{end:.2f}" if isinstance(end, float) else str(end)

            return f"{prop}: [{start_str}, {end_str})"

        # Fallback: Just stringify the dict without the 'weight' key for readability
        else:
            safe_item = {k: v for k, v in item.items() if k != "weight"}
            return str(safe_item)

    except Exception as e:
        logger.error(f"Error parsing token {token_id}: {e}")
        return f"Parse_Error_ID_{token_id}"


def run_captum_explainability(wrapper, tokenizer, test_batches_path):
    """Runs Layer Gradient X Activation on the test set."""
    logger.info("Initializing Captum Layer Gradient X Activation...")

    embedding_layer = wrapper.motor.transformer.embed_bag

    attr_algo = LayerGradientXActivation(forward_func=wrapper, layer=embedding_layer)

    logger.info(f"Loading test batches from {test_batches_path}")
    test_batches = datasets.Dataset.load_from_disk(test_batches_path)
    test_batches.set_format("pt")

    attributions_list = []

    raw_vocab = load_raw_msgpack_dictionary()

    logger.info("Calculating attributions (this may take a while)...")
    for i in range(min(50, len(test_batches))):
        batch = test_batches[i]
        batch_gpu = prepare_batch(batch)
        dummy_input = torch.empty(0).to(DEVICE)

        try:
            attributions = attr_algo.attribute(
                inputs=dummy_input,
                additional_forward_args=(batch_gpu,),
                target=0,
            )

            token_attributions = (
                attributions.sum(dim=-1).squeeze().cpu().detach().numpy()
            )
            tokens_flat = (
                batch_gpu["transformer"]["hierarchical_tokens"].squeeze().cpu().numpy()
            )

            for token_id, attr_score in zip(tokens_flat, token_attributions):
                # USE THE NEW DIRECT MSGPACK MAPPER
                token_str = get_token_string_resilient(raw_vocab, token_id)

                attributions_list.append(
                    {
                        "patient_index": i,
                        "token_id": token_id,
                        "token_string": token_str,
                        "attribution_score": float(attr_score),
                    }
                )

        except Exception as e:
            logger.error(
                f"Failed Captum attribution for patient {i}: {type(e).__name__} - {e}"
            )
            continue

    if not attributions_list:
        logger.error("No attributions were successfully computed. See errors above.")
        return

    df = pd.DataFrame(attributions_list)
    output_path = Config.RESULTS_DIR / "explainability" / "ehr_token_attributions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = df.groupby("token_string")["attribution_score"].mean().reset_index()
    summary_df = summary_df.sort_values(by="attribution_score", ascending=False)

    df.to_csv(output_path, index=False)
    summary_df.to_csv(
        Config.RESULTS_DIR / "explainability" / "ehr_token_summary.csv", index=False
    )

    logger.info(f"Saved token attributions to {output_path}")
    logger.info(f"Top 5 predictive tokens:\n{summary_df.head(5)}")


def main():
    Config.setup_logging()

    wrapper, tokenizer = load_end_to_end_models()
    test_batches_path = Config.RESULTS_DIR / "explainability" / "test_batches"

    if not test_batches_path.exists():
        logger.error(f"Test batches not found at {test_batches_path}")
        return

    run_captum_explainability(wrapper, tokenizer, test_batches_path)
    logger.info("=== Step 3.2: EHR Token Explainability Complete ===")


if __name__ == "__main__":
    main()
