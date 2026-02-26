from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GroundTruth:
    w1_star: Optional[torch.Tensor] = None
    w2_star: Optional[torch.Tensor] = None


def _normalize_to_sqrt_d(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[1]
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12) * (d ** 0.5)


def _clip_norm(x: torch.Tensor, max_norm: float) -> torch.Tensor:
    norms = x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    scales = (max_norm / norms).clamp(max=1.0)
    return x * scales


class InputGenerator(ABC):
    @abstractmethod
    def sample(self, batch_size: int, d: int, device: torch.device) -> torch.Tensor:
        """Sample x with shape [batch_size, d]."""


class RademacherInputGenerator(InputGenerator):
    def sample(self, batch_size: int, d: int, device: torch.device) -> torch.Tensor:
        x = torch.randint(0, 2, (batch_size, d), device=device, dtype=torch.float32)
        return x * 2.0 - 1.0


class MixtureInputGenerator(InputGenerator):
    def __init__(self, num_mix: int, sigma: float):
        if num_mix <= 0:
            raise ValueError(f"num_mix must be positive, got {num_mix}")
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        self.num_mix = num_mix
        self.sigma = sigma
        self._means: dict[tuple[int, str], torch.Tensor] = {}

    def _get_means(self, d: int, device: torch.device) -> torch.Tensor:
        key = (d, str(device))
        means = self._means.get(key)
        if means is None:
            means = torch.zeros(self.num_mix, d, device=device)
            basis_idx = torch.arange(self.num_mix, device=device) % d
            means[torch.arange(self.num_mix, device=device), basis_idx] = 1.0
            self._means[key] = means
        return means

    def centers(self, d: int, device: torch.device) -> torch.Tensor:
        """Return normalized mixture centers with shape [num_mix, d]."""
        return _normalize_to_sqrt_d(self._get_means(d, device))

    def sample(self, batch_size: int, d: int, device: torch.device) -> torch.Tensor:
        means = self._get_means(d, device)
        mix_idx = torch.randint(0, self.num_mix, (batch_size,), device=device)
        eps = torch.randn(batch_size, d, device=device) * self.sigma
        eps = _clip_norm(eps, self.sigma)
        return _normalize_to_sqrt_d(means[mix_idx] + eps) / (d ** 0.5)


def init_ground_truth(gt: GroundTruth, d: int, k: int, device: torch.device):
    if gt.w1_star is None:
        gt.w1_star = torch.randn(k, d, device=device)
    if gt.w2_star is None:
        gt.w2_star = torch.randn(k, k, device=device)


def generate_y(x: torch.Tensor, w1_star: torch.Tensor, w2_star: torch.Tensor, seq_length: int) -> torch.Tensor:
    base = x @ w1_star.T
    y = torch.empty((x.shape[0], seq_length), device=x.device, dtype=torch.long)
    y[:, 0] = base.argmax(dim=1)
    for t in range(1, seq_length):
        prev = y[:, t - 1]
        logits = base + w2_star[:, prev].T
        y[:, t] = logits.argmax(dim=1)
    return y


def sample_batch(
    batch_size: int,
    d: int,
    k: int,
    seq_length: int,
    gt: GroundTruth,
    device: torch.device,
    data_generator: InputGenerator | None = None,
):
    init_ground_truth(gt, d=d, k=k, device=device)
    generator = data_generator if data_generator is not None else RademacherInputGenerator()
    x = generator.sample(batch_size=batch_size, d=d, device=device)
    y = generate_y(x, gt.w1_star, gt.w2_star, seq_length)
    return x, y
