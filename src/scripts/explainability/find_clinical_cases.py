"""
Reads the exact patient-level clinical vignettes from the Champion Captum attributions.
Prints the top and bottom clinical factors driving the Sepsis risk score.

Change captum_path to explore different cases (ehr_token_attributions.csv or champion_ehr_token_attributions.csv).

Usage:
    python -m src.scripts.explainability.find_clinical_cases
"""

import pandas as pd
from src.utils.config import Config


def main():
    attr_path = (
        Config.RESULTS_DIR / "explainability" / "champion_ehr_token_attributions.csv"
    )
    if not attr_path.exists():
        print(
            f"Error: {attr_path} not found. Run the champion_captum_explainer script first."
        )
        return

    df = pd.read_csv(attr_path)
    subject_ids = df["subject_id"].unique()

    for sid in subject_ids:
        # Isolate patient and sort by attribution score
        patient_df = df[df["subject_id"] == sid].sort_values(
            by="attribution_score", ascending=False
        )

        print("\n" + "=" * 70)
        print(f"🏥 CLINICAL VIGNETTE | Subject ID: {sid}")
        print("=" * 70)

        print("🔴 TOP 3 FACTORS DRIVING SEPSIS RISK UP:")
        for _, row in patient_df.head(3).iterrows():
            print(f"  [+{row['attribution_score']:.2e}] {row['token_string']}")

        print("\n🔵 TOP 3 FACTORS DRIVING SEPSIS RISK DOWN:")
        for _, row in (
            patient_df.tail(3)
            .sort_values(by="attribution_score", ascending=True)
            .iterrows()
        ):
            print(f"  [{row['attribution_score']:.2e}] {row['token_string']}")


if __name__ == "__main__":
    main()
