# Explainable Late-Fusion Sepsis Prediction using Multimodal Data

![Status](https://img.shields.io/badge/Status-Work_in_Progress-yellow)
![License](https://img.shields.io/badge/License-MIT-blue)

## 📌 Project Overview
This repository contains the implementation of a multimodal neural network designed for the **early prediction of sepsis** in ICU patients.

The system utilizes a **Late-Fusion architecture** to process four distinct modalities of clinical data:
1.  **Tabular EHR Data** (Vitals, Labs, Demographics)
2.  **Electrocardiograms** (ECG signals)
3.  **Chest X-Ray Images** (Visual data)
4.  **Chest X-Ray Reports** (Textual data)

By processing each modality independently before fusion, the model aims to maximize **interpretability**, allowing clinicians to see exactly which data source contributed to a positive sepsis prediction.

## 📂 Repository Structure
* `src/`: Source code for model definitions, preprocessing logic, and utilities.
* `scripts/`: Execution scripts for data acquisition, database setup, and pipeline orchestration.
* `tests/`: Validation tests to ensure data integrity and split stratification.
* `reports/`: Results of models currently in markdown format.

## 🚀 Getting Started

### 🛠️ Prerequisites & Setup

**1. Data Access**<br>
You must be a credentialed researcher on [PhysioNet](https://physionet.org/) with signed Data Use Agreements (DUA) for:
* [**MIMIC-IV (v2.2)**](https://physionet.org/content/mimiciv/2.2/) (Core clinical data)
* [**MIMIC-IV-ECG**](https://physionet.org/content/mimic-iv-ecg/1.0/) (Waveform signals)
* [**MIMIC-CXR-JPG**](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) (Chest X-Ray images)
* [**MIMIC-CXR**](https://physionet.org/content/mimic-cxr/2.1.0/) (Radiology reports)

**2. Database Infrastructure**<br>
A local Postgres database with the core [**MIMIC-IV**](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/buildmimic/postgres) and [**MIMIC-IV Concepts**](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/concepts_postgres) schemas pre-installed.

**3. Python Environment**<br>
This project uses Python 3.10. We recommend managing dependencies via Conda.
```bash
# Create the environment from the file
conda env create -f environment.yml

# Activate the environment
conda activate thesis_py310
```

**4. Configuration**<br>
The pipeline requires a local configuration file to locate your database and raw files.<br>
Rename .env.example to .env. and fill in your Database credentials and PhysioNet login details.

### Data Acquisition & DB Setup
The complete pipeline for downloading raw files, building the database schemas, and creating the patient cohort is detailed in the scripts directory.

👉 **[Go to Data Setup Guide](scripts/)**


## 🧩 Pipeline Overview

1. **Data Preprocessing**
	- [EHR Preprocessing](src/data/preprocess/ehr/): Export, clean, and transform tabular EHR data for downstream modeling.
	- [ECG Preprocessing](src/data/preprocess/ecg/): Extract, clean, and standardize raw ECG waveform data for downstream modeling. See subfolder for details.

2. **Embeddings Extraction**
	- [EHR Embeddings](src/scripts/extract_embeddings/): Generate patient-level embeddings using the pretrained MOTOR foundation model.
	  - _Requires MOTOR foundation model pretraining. See [src/models/foundation/ehr/README.md](src/models/foundation/ehr/) for instructions._
	- [ECG Embeddings](src/scripts/extract_embeddings/): Generate patient-level ECG embeddings using the frozen ECG-FM model. See subfolder for details.
	- [CXR Image Embeddings](src/scripts/extract_embeddings/): Generate patient-level CXR image embeddings using the planned/frozen CXR foundation model. See subfolder for details.

3. **Unimodal Classifiers**
	- [Run Classifiers](src/models/unimodal/): Train and evaluate unimodal models (LR, XGBoost, MLP) on EHR embeddings.

4. **Results**
	- [Comparison Reports](src/scripts/reports/unimodal/): View performance metrics and comparison reports.



## 🧪 Validation & Testing

After completing each pipeline stage, you can validate outputs and data integrity using the provided integration tests:

```bash
pytest tests/ -v
```

See [tests/README.md](tests/) for details on individual test files and their requirements.

---

## ⚖️ License & Data Usage

### Code License
The source code in this repository is released under the **MIT License**. See the [LICENSE](./LICENSE.md) file for details.

### Data License (Important)
This project relies on the MIMIC-IV dataset, which is a restricted-access resource. The data itself is **not** included in this repository.
* Users must be credentialed researchers on [PhysioNet](https://physionet.org/).
* Users must sign the Data Use Agreement (DUA) for MIMIC-IV, MIMIC-CXR, and MIMIC-IV-ECG.

## 🚧 Status
**Current Status:**
- Data acquisition: complete
- EHR, ECG, CXR image preprocessing: complete
- EHR, ECG, CXR image embeddings extraction: complete
- Unimodal classifiers (EHR, ECG, CXR image): complete
- Testing for all the above: complete