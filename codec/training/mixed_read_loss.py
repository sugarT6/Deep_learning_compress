"""Preserve source exposure weights when different-length reads share an update."""

import torch
from torch.nn import functional as F


def source_weighted_cross_entropy(logits, qualities, active_mask,
                                  row_dataset_indices, source_probabilities,
                                  source_quality_symbols):
    """Return objective and per-source NLL sums for a full batch or microbatch.

    Denominators describe the *whole optimizer batch*, including when logits
    contain only a microbatch. Adding the returned objectives then gives the
    same loss and gradients as processing the complete batch at once.
    """
    if logits.shape[:-1] != qualities.shape or qualities.shape != active_mask.shape:
        raise ValueError("logits, qualities and active_mask have incompatible shapes")
    if active_mask.dtype != torch.bool:
        raise ValueError("active_mask must be boolean")
    if row_dataset_indices.shape != (qualities.shape[0],):
        raise ValueError("one source index is required for every read")
    if (source_probabilities.ndim != 1
            or source_quality_symbols.shape != source_probabilities.shape
            or torch.any(source_quality_symbols <= 0)):
        raise ValueError("every source requires a positive full-batch Q count")
    if (torch.any(source_probabilities <= 0)
            or not torch.isclose(source_probabilities.sum(),
                                 source_probabilities.new_tensor(1.))):
        raise ValueError("source probabilities must be positive and sum to one")
    if (torch.any(row_dataset_indices < 0)
            or torch.any(row_dataset_indices >= source_probabilities.numel())):
        raise ValueError("source index is out of range")
    # Replace padding targets before CE: padding is id 42, outside Q0..Q41.
    targets = qualities.masked_fill(~active_mask, 0)
    values = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    row_nll = (values * active_mask.to(values.dtype)).sum(dim=1)
    source_nll = row_nll.new_zeros(source_probabilities.numel()).scatter_add(
        0, row_dataset_indices, row_nll)
    objective = (source_nll * source_probabilities.to(source_nll.dtype)
                 / source_quality_symbols.to(source_nll.dtype)).sum()
    return objective, source_nll
