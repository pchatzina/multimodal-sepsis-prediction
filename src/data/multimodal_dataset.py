import torch
from torch.utils.data import Dataset
from src.utils.config import Config


class MultimodalSepsisDataset(Dataset):
    """
    A custom PyTorch Dataset for loading and fusing multimodal sepsis prediction data.

    This dataset performs an outer-join of 4 modalities (EHR, ECG, CXR Image, CXR Text)
    based on the patient's integer `subject_id`. It handles missing modalities by
    yielding zero-vectors and generates binary missingness masks for the Gating Network.
    EHR data is assumed to be the base modality and must be present for all patients.

    Args:
        split (str): The dataset split to load. Must be one of 'train', 'valid', or 'test'.

    Returns:
        dict: A dictionary containing:
            - 'subject_id' (int): The unique patient identifier.
            - 'embeddings' (dict): Tensors of embeddings keyed by modality ('ehr', 'ecg', 'img', 'txt').
            - 'masks' (dict): Binary missingness masks (1.0 for present, 0.0 for missing) keyed by modality.
            - 'label' (Tensor): The binary ground truth label for sepsis prediction.
    """

    def __init__(self, split: str = "train", ehr_dropout_rate: float = 0.0):
        super().__init__()
        self.split = split
        self.ehr_dropout_rate = ehr_dropout_rate

        # 1. Define paths for the requested split using the corrected filenames
        ehr_path = Config.EHR_EMBEDDINGS_DIR / f"{split}_embeddings.pt"
        ecg_path = Config.ECG_EMBEDDINGS_DIR / f"{split}_embeddings.pt"
        img_path = Config.CXR_IMG_EMBEDDINGS_DIR / f"{split}_embeddings.pt"
        txt_path = Config.CXR_TXT_EMBEDDINGS_DIR / f"{split}_embeddings.pt"

        # 2. Load the data dictionaries
        # Using map_location='cpu' ensures we don't load onto GPU before the DataLoader
        self.ehr_data = torch.load(ehr_path, map_location="cpu")
        self.ecg_data = (
            torch.load(ecg_path, map_location="cpu") if ecg_path.exists() else None
        )
        self.img_data = (
            torch.load(img_path, map_location="cpu") if img_path.exists() else None
        )
        self.txt_data = (
            torch.load(txt_path, map_location="cpu") if txt_path.exists() else None
        )

        # 3. Establish the base cohort
        # We rely on EHR subject_ids as our primary source of truth.
        self.subject_ids = self.ehr_data["sample_ids"]
        self.labels = self.ehr_data["labels"]

        # 4. Create O(1) lookup dictionaries: subject_id -> tensor row index
        self.ehr_idx_map = {sid: idx for idx, sid in enumerate(self.subject_ids)}
        self.ecg_idx_map = self._build_map(self.ecg_data)
        self.img_idx_map = self._build_map(self.img_data)
        self.txt_idx_map = self._build_map(self.txt_data)

        # 5. Store embedding dimensions to generate zero-vectors for missing data
        self.dim_ehr = self.ehr_data["embeddings"].shape[1]
        self.dim_ecg = self.ecg_data["embeddings"].shape[1] if self.ecg_data else 0
        self.dim_img = self.img_data["embeddings"].shape[1] if self.img_data else 0
        self.dim_txt = self.txt_data["embeddings"].shape[1] if self.txt_data else 0

    def _build_map(self, modality_data):
        """Helper to map subject_id to tensor index for a given modality."""
        if modality_data is None:
            return {}
        return {sid: idx for idx, sid in enumerate(modality_data["sample_ids"])}

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx]
        label = self.labels[idx]

        embeddings = {}
        masks = {}

        # Check if the patient has any alternative modalities
        has_other_modality = (
            (subject_id in self.ecg_idx_map)
            or (subject_id in self.img_idx_map)
            or (subject_id in self.txt_idx_map)
        )

        # --- EHR Conditional Dropout ---
        drop_ehr = False
        if self.ehr_dropout_rate > 0.0 and has_other_modality:
            if torch.rand(1).item() < self.ehr_dropout_rate:
                drop_ehr = True

        if drop_ehr:
            embeddings["ehr"] = torch.zeros(self.dim_ehr, dtype=torch.float32)
            masks["ehr"] = torch.tensor([0.0], dtype=torch.float32)
        else:
            ehr_row = self.ehr_idx_map[subject_id]
            embeddings["ehr"] = self.ehr_data["embeddings"][ehr_row]
            masks["ehr"] = torch.tensor([1.0], dtype=torch.float32)

        # --- ECG ---
        if subject_id in self.ecg_idx_map:
            ecg_row = self.ecg_idx_map[subject_id]
            embeddings["ecg"] = self.ecg_data["embeddings"][ecg_row]
            masks["ecg"] = torch.tensor([1.0], dtype=torch.float32)
        else:
            embeddings["ecg"] = torch.zeros(self.dim_ecg, dtype=torch.float32)
            masks["ecg"] = torch.tensor([0.0], dtype=torch.float32)

        # --- CXR Image ---
        if subject_id in self.img_idx_map:
            img_row = self.img_idx_map[subject_id]
            embeddings["img"] = self.img_data["embeddings"][img_row]
            masks["img"] = torch.tensor([1.0], dtype=torch.float32)
        else:
            embeddings["img"] = torch.zeros(self.dim_img, dtype=torch.float32)
            masks["img"] = torch.tensor([0.0], dtype=torch.float32)

        # --- CXR Text ---
        if subject_id in self.txt_idx_map:
            txt_row = self.txt_idx_map[subject_id]
            embeddings["txt"] = self.txt_data["embeddings"][txt_row]
            masks["txt"] = torch.tensor([1.0], dtype=torch.float32)
        else:
            embeddings["txt"] = torch.zeros(self.dim_txt, dtype=torch.float32)
            masks["txt"] = torch.tensor([0.0], dtype=torch.float32)

        return {
            "subject_id": subject_id,
            "embeddings": embeddings,
            "masks": masks,
            "label": torch.tensor(label, dtype=torch.float32),
        }
