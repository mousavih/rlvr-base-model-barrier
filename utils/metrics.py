from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn.functional as F

from .model import AutoregressivePolicy


@torch.no_grad()
def eval_sequence_error(model: AutoregressivePolicy, x: torch.Tensor, y_true: torch.Tensor) -> float:
    _, seq_length = y_true.shape
    y_samples = []
    for t in range(seq_length):
        y_prefix = torch.stack(y_samples, dim=1) if y_samples else y_true[:, :0]
        y_t = model.sample_step(x, y_prefix)
        y_samples.append(y_t)
    y_samples = torch.stack(y_samples, dim=1)
    matches = (y_samples == y_true).all(dim=1)
    return 1.0 - matches.float().mean().item()


@torch.no_grad()
def compute_sequence_likelihood(model: AutoregressivePolicy, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model.logits(x, y)
    log_probs = F.log_softmax(logits, dim=-1)
    target_logp = log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    logp_seq = target_logp.sum(dim=1)
    return logp_seq.exp()


@torch.no_grad()
def estimate_cdf_p(
    models: Iterable[AutoregressivePolicy],
    x: torch.Tensor,
    y: torch.Tensor,
) -> List[torch.Tensor]:
    p_sorteds = []
    for model in models:
        p_seq = compute_sequence_likelihood(model, x, y).cpu()
        p_sorteds.append(torch.sort(p_seq).values)
    return p_sorteds
