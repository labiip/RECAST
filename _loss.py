from __future__ import annotations

import torch

from ._distributions import ZeroInflatedNegativeBinomial


def zinb_reconstruction_loss(
    X: torch.Tensor,
    *,
    mu: torch.Tensor,
    theta: torch.Tensor,
    gate_logits: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    logits = (mu / (theta + 1e-8)).log()
    total_counts = theta + 1e-6
    znb = ZeroInflatedNegativeBinomial(
        total_count=total_counts,
        logits=logits,
        gate_logits=gate_logits,
    )
    nll = -znb.log_prob(X)
    if reduction == "sum":
        return nll.sum(dim=1)
    if reduction == "mean":
        return nll.mean(dim=1)
    if reduction == "none":
        return nll
    raise ValueError(reduction)
