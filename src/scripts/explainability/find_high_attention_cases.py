"""
Identifies multimodal patients where the Gating Network assigned the highest
attention weights to the auxiliary modalities (Chest X-ray Image and Text).

Usage:
    python -m src.scripts.explainability.find_high_attention_cases
"""

import pandas as pd
from src.utils.config import Config


def clean_tensor_string(val):
    """Strips the 'tensor()' string wrapper if it exists and returns an int."""
    val_str = str(val).strip()
    if val_str.startswith("tensor("):
        val_str = val_str.replace("tensor(", "").replace(")", "")
    return int(val_str)


def main():
    # Path to the 3-Modality weights CSV
    csv_path = (
        Config.RESULTS_DIR
        / "explainability"
        / "modality_weights"
        / "3mod_architecture"
        / "test_set_modality_weights.csv"
    )

    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        print(
            "Please run explain_modality_weights.py for the 3-modality champion first."
        )
        return

    df = pd.read_csv(csv_path)

    # Clean the subject_id column immediately
    df["subject_id"] = df["subject_id"].apply(clean_tensor_string)

    # 1. Filter for true multimodal patients (Masked Softmax ensures w > 0 only if present)
    multimodal_df = df[(df["w_img"] > 0) & (df["w_txt"] > 0)].copy()

    if multimodal_df.empty:
        print("No multimodal patients found in the weights CSV.")
        return

    # 2. Calculate the combined auxiliary weight contribution
    multimodal_df["aux_weight"] = multimodal_df["w_img"] + multimodal_df["w_txt"]

    # 3. Sort by the highest auxiliary contribution
    top_cases = multimodal_df.sort_values(by="aux_weight", ascending=False)

    def print_top_cases(target_df, label_name, max_cases=5):
        print("\n" + "=" * 80)
        print(
            f"🏥 TOP {max_cases} MULTIMODAL CASES RELYING ON CXR + TEXT | True Label: {label_name}"
        )
        print("=" * 80)

        for idx, row in target_df.head(max_cases).iterrows():
            sid = int(row["subject_id"])
            p_final = row["p_final"]
            w_ehr = row["w_ehr"]
            w_img = row["w_img"]
            w_txt = row["w_txt"]
            aux = row["aux_weight"]

            print(
                f"Subject ID: {sid:<10} | Risk: {p_final:05.2%} | "
                f"Aux Contribution: {aux:05.2%} (IMG: {w_img:05.2%}, TXT: {w_txt:05.2%}) | EHR: {w_ehr:05.2%}"
            )

    # Print Top 5 Sepsis Positive and Top 5 Sepsis Negative cases
    print_top_cases(top_cases[top_cases["true_label"] == 1], "SEPSIS POSITIVE (1)")
    print_top_cases(top_cases[top_cases["true_label"] == 0], "SEPSIS NEGATIVE (0)")

    print(
        "\nTip: Pass these Subject IDs into your end_to_end_pipeline.py to see their full vignette!"
    )


if __name__ == "__main__":
    main()
