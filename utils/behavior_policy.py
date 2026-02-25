from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .model import AutoregressivePolicy


class BehaviorPolicy(ABC):
    @abstractmethod
    def sample_step(
        self,
        learner: AutoregressivePolicy,
        x: torch.Tensor,
        y_prefix: torch.Tensor,
        y_true: torch.Tensor | None = None,
        token_index: int | None = None,
    ) -> torch.Tensor:
        """Sample an action from the behavior policy for one autoregressive step."""


class OnPolicyBehavior(BehaviorPolicy):
    def sample_step(
        self,
        learner: AutoregressivePolicy,
        x: torch.Tensor,
        y_prefix: torch.Tensor,
        y_true: torch.Tensor | None = None,
        token_index: int | None = None,
    ) -> torch.Tensor:
        del y_true, token_index
        return learner.sample_step(x, y_prefix)


class FrozenPolicyBehavior(BehaviorPolicy):
    def __init__(self, policy: AutoregressivePolicy):
        self.policy = policy

    def sample_step(
        self,
        learner: AutoregressivePolicy,
        x: torch.Tensor,
        y_prefix: torch.Tensor,
        y_true: torch.Tensor | None = None,
        token_index: int | None = None,
    ) -> torch.Tensor:
        del learner, y_true, token_index
        return self.policy.sample_step(x, y_prefix)


class UniformBehavior(BehaviorPolicy):
    def __init__(self, k: int):
        self.k = k

    def sample_step(
        self,
        learner: AutoregressivePolicy,
        x: torch.Tensor,
        y_prefix: torch.Tensor,
        y_true: torch.Tensor | None = None,
        token_index: int | None = None,
    ) -> torch.Tensor:
        del learner, y_prefix, y_true, token_index
        return torch.randint(0, self.k, (x.shape[0],), device=x.device)


class EpsilonMixtureBehavior(BehaviorPolicy):
    def __init__(self, k: int, epsilon: float, base_behavior: BehaviorPolicy | None = None):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self.k = k
        self.epsilon = epsilon
        self.base_behavior = base_behavior if base_behavior is not None else OnPolicyBehavior()

    def sample_step(
        self,
        learner: AutoregressivePolicy,
        x: torch.Tensor,
        y_prefix: torch.Tensor,
        y_true: torch.Tensor | None = None,
        token_index: int | None = None,
    ) -> torch.Tensor:
        base_actions = self.base_behavior.sample_step(learner, x, y_prefix, y_true, token_index)
        random_actions = torch.randint(0, self.k, (x.shape[0],), device=x.device)
        choose_random = torch.rand(x.shape[0], device=x.device) < self.epsilon
        return torch.where(choose_random, random_actions, base_actions)


class BoMProcessPolicy(BehaviorPolicy):
    def __init__(self, base_policy: BehaviorPolicy, m: int):
        if m <= 0:
            raise ValueError(f"m must be positive, got {m}")
        self.base_policy = base_policy
        self.m = m

    def sample_step(
        self,
        learner: AutoregressivePolicy,
        x: torch.Tensor,
        y_prefix: torch.Tensor,
        y_true: torch.Tensor | None = None,
        token_index: int | None = None,
    ) -> torch.Tensor:
        if y_true is None or token_index is None:
            raise ValueError("BoMProcessPolicy requires y_true and token_index")

        locked = torch.zeros(x.shape[0], device=x.device, dtype=torch.bool)
        locked_actions = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        last_actions = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)

        for _ in range(self.m):
            candidate = self.base_policy.sample_step(learner, x, y_prefix, y_true, token_index)
            last_actions = candidate
            prefix_plus = torch.cat([y_prefix, candidate.unsqueeze(1)], dim=1)
            correct_so_far = (prefix_plus == y_true[:, : token_index + 1]).all(dim=1)
            newly_locked = (~locked) & correct_so_far
            locked_actions = torch.where(newly_locked, candidate, locked_actions)
            locked = locked | newly_locked

        return torch.where(locked, locked_actions, last_actions)


class TeacherForcingBehavior(BehaviorPolicy):
    def sample_step(
        self,
        learner: AutoregressivePolicy,
        x: torch.Tensor,
        y_prefix: torch.Tensor,
        y_true: torch.Tensor | None = None,
        token_index: int | None = None,
    ) -> torch.Tensor:
        del learner, x, y_prefix
        if y_true is None or token_index is None:
            raise ValueError("TeacherForcingBehavior requires y_true and token_index")
        return y_true[:, token_index]
