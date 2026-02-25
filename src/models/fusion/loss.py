# src/models/fusion/loss.py
import torch
import torch.nn as nn


def composite_sepsis_loss(p_final, p_unimodal, masks, targets, lambda_weight=0.4):
    """
    Calculates the composite loss enforcing unimodal accountability.
    """
    bce_unreduced = nn.BCELoss(reduction="none")
    bce_reduced = nn.BCELoss(reduction="mean")

    targets = targets.view(-1, 1).float()
    main_loss = bce_reduced(p_final, targets)

    aux_loss = 0.0
    modalities = ["ehr", "ecg", "img", "txt"]

    for mod in modalities:
        p_i = p_unimodal[mod]
        m_i = masks[mod]
        bce_i = bce_unreduced(p_i, targets)
        masked_bce = bce_i * m_i
        aux_loss += masked_bce.mean()

    total_loss = main_loss + (lambda_weight * aux_loss)
    return total_loss, main_loss, aux_loss
