import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    """Standard MLP block for the Gating Network and Synergy Head."""

    def __init__(
        self,
        input_dim: int,
        hidden_1: int,
        hidden_2: int,
        output_dim: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_1),
            nn.LayerNorm(hidden_1),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_1, hidden_2),
            nn.LayerNorm(hidden_2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_2, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class DynamicUnimodalBlock(nn.Module):
    """Dynamically reconstructs the unimodal MLP architecture based on Optuna JSON configs."""

    def __init__(self, input_dim: int, config: dict):
        super().__init__()
        hidden_1 = config["hidden_dim_1"]
        hidden_2 = config["hidden_dim_2"]
        dropout_rate = config.get("dropout_rate", 0.0)
        norm_type = config.get("norm_type", "layer")
        use_dropout = config.get("use_dropout", False)
        use_input_norm = config.get("use_input_norm", True)
        activation_name = config.get("activation", "GELU")

        activation_layer = nn.ReLU() if activation_name == "ReLU" else nn.GELU()
        layers = []

        if use_input_norm:
            layers.append(nn.LayerNorm(input_dim))

        layers.append(nn.Linear(input_dim, hidden_1))
        layers.append(
            nn.BatchNorm1d(hidden_1) if norm_type == "batch" else nn.LayerNorm(hidden_1)
        )
        layers.append(activation_layer)
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_1, hidden_2))
        layers.append(
            nn.BatchNorm1d(hidden_2) if norm_type == "batch" else nn.LayerNorm(hidden_2)
        )
        layers.append(activation_layer)
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_2, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LateFusionSepsisModel(nn.Module):
    def __init__(
        self,
        input_dims: dict,
        config: dict,
        unimodal_configs: dict = None,
        common_dim: int = 768,
    ):
        """
        Args:
            input_dims (dict): Dictionary of raw dimensions.
            config (dict): Hyperparameters for the fusion components (Gating & Synergy).
            unimodal_configs (dict): Nested dictionary of loaded JSON configs for each modality.
                                     If None, falls back to a standard MLPBlock (Option A).
            common_dim (int): The target dimension for all modalities.
        """
        super().__init__()
        self.common_dim = common_dim
        self.modalities = ["ehr", "ecg", "img", "txt"]

        # 1. Linear Projectors
        self.projectors = nn.ModuleDict(
            {mod: nn.Linear(input_dims[mod], common_dim) for mod in self.modalities}
        )

        # 2. Unimodal MLPs (Dynamic reconstruction for Option B)
        self.unimodal_mlps = nn.ModuleDict()
        for mod in self.modalities:
            if unimodal_configs and mod in unimodal_configs:
                # Option B: Rebuild exact architecture
                self.unimodal_mlps[mod] = DynamicUnimodalBlock(
                    common_dim, unimodal_configs[mod]
                )
            else:
                # Option A: Standard fallback block
                self.unimodal_mlps[mod] = MLPBlock(
                    input_dim=common_dim,
                    hidden_1=config.get("uni_hidden_1", 256),
                    hidden_2=config.get("uni_hidden_2", 128),
                    output_dim=1,
                    dropout_rate=config.get("dropout_rate", 0.1),
                )

        # 3. Gating Network (unchanged)
        gate_input_dim = (common_dim * 4) + 4
        self.gating_network = MLPBlock(
            input_dim=gate_input_dim,
            hidden_1=config.get("gate_hidden_1", 512),
            hidden_2=config.get("gate_hidden_2", 128),
            output_dim=4,
            dropout_rate=config.get("dropout_rate", 0.1),
        )

        # 4. Synergy Head (unchanged)
        syn_input_dim = common_dim * 4
        self.synergy_head = MLPBlock(
            input_dim=syn_input_dim,
            hidden_1=config.get("syn_hidden_1", 512),
            hidden_2=config.get("syn_hidden_2", 128),
            output_dim=1,
            dropout_rate=config.get("dropout_rate", 0.1),
        )

    def forward(self, embeddings: dict, masks: dict):
        batch_size = embeddings["ehr"].size(0)
        device = embeddings["ehr"].device

        z_dict = {}
        p_dict = {}
        logit_dict = {}

        # --- A & B: Projection and Unimodal Prediction ---
        for mod in self.modalities:
            # Mask out missing modalities before projection (just in case they aren't exact zeros)
            mask_tensor = masks[mod]  # Shape: [B, 1]
            raw_emb = embeddings[mod] * mask_tensor

            # Linear projection: z_i = Linear(z_raw_i)
            z_i = self.projectors[mod](raw_emb)
            # Re-apply mask to ensure missing modalities remain exactly zero vectors
            z_i = z_i * mask_tensor
            z_dict[mod] = z_i

            # Unimodal prediction: p_i = sigmoid(MLP(z_i))
            logit_i = self.unimodal_mlps[mod](z_i)
            logit_dict[mod] = logit_i
            p_dict[mod] = torch.sigmoid(logit_i)

        # --- C: Gating Network Mechanism ---
        # Concatenate [z_ehr, z_ecg, z_img, z_txt, M_ehr, M_ecg, M_img, M_txt]
        z_list = [z_dict[mod] for mod in self.modalities]
        m_list = [masks[mod] for mod in self.modalities]

        v_gate = torch.cat(z_list + m_list, dim=1)
        logits_gate = self.gating_network(v_gate)  # Shape: [B, 4]

        # Masked Softmax: Apply large negative value where mask == 0 so exp(logits) becomes 0
        mask_concat = torch.cat(m_list, dim=1)  # Shape: [B, 4]
        masked_logits_gate = logits_gate.masked_fill(mask_concat == 0, -1e9)
        w = F.softmax(masked_logits_gate, dim=-1)  # Shape: [B, 4]

        # --- D: Late Fusion (Additive) ---
        p_add = torch.zeros((batch_size, 1), device=device)
        for i, mod in enumerate(self.modalities):
            # w[:, i:i+1] extracts the specific weight column keeping shape [B, 1]
            p_add += w[:, i : i + 1] * p_dict[mod]

        # --- E: Synergy Head ---
        z_joint = torch.cat(z_list, dim=1)
        logit_syn = self.synergy_head(z_joint)
        p_syn = torch.sigmoid(logit_syn)

        # --- F: Final Prediction ---
        # Calculate beta = 1 - max(w_i)
        max_w, _ = torch.max(w, dim=1, keepdim=True)
        beta = 1.0 - max_w

        p_final = (1 - beta) * p_add + beta * p_syn

        return {
            "p_final": p_final,
            "p_unimodal": p_dict,  # Dictionary of individual probabilities
            "weights": w,  # Useful for explainability
            "beta": beta,  # Useful for monitoring
            "logits": {**logit_dict, "syn": logit_syn},
        }
