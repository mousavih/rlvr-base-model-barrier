from __future__ import annotations

import torch
import torch.nn.functional as F


class AutoregressivePolicy(torch.nn.Module):
    def __init__(self, d: int, k: int, device: torch.device):
        super().__init__()
        self.d = d
        self.k = k
        self.w = torch.nn.Parameter(torch.zeros(d * k + k * k, device=device))

    def _split_w(self):
        w1 = self.w[: self.d * self.k].view(self.d, self.k)
        w2 = self.w[self.d * self.k :].view(self.k, self.k)
        return w1, w2

    def logits_next(self, x: torch.Tensor, y_prefix: torch.Tensor) -> torch.Tensor:
        w1, w2 = self._split_w()
        base = x @ w1
        if y_prefix.shape[1] == 0:
            return base
        y_prev = y_prefix[:, -1]
        add = w2[:, y_prev].T
        return base + add

    def logits(self, x: torch.Tensor, y_seq: torch.Tensor) -> torch.Tensor:
        seq_length = y_seq.shape[1]
        w1, w2 = self._split_w()
        base = x @ w1
        logits = base.unsqueeze(1).expand(-1, seq_length, -1).clone()

        if seq_length > 1:
            y_prev = y_seq[:, :-1]
            add = w2[:, y_prev].permute(1, 2, 0)
            logits[:, 1:, :] += add

        return logits

    def log_prob_step(self, x: torch.Tensor, y_prefix: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.logits_next(x, y_prefix)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def sample_step(self, x: torch.Tensor, y_prefix: torch.Tensor):
        logits = self.logits_next(x, y_prefix)
        probs = F.softmax(logits, dim=-1)
        y = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return y
