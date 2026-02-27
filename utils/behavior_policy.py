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

        batch_size = x.shape[0]
        if token_index == 0:
            prefix_correct = torch.ones(batch_size, device=x.device, dtype=torch.bool)
        else:
            prefix_correct = (y_prefix == y_true[:, :token_index]).all(dim=1)

        # Base policy is batch-aware, so we expand B -> B*m and then reshape outputs to [B, m].
        x_rep = x.repeat_interleave(self.m, dim=0)
        y_prefix_rep = y_prefix.repeat_interleave(self.m, dim=0)
        y_true_rep = y_true.repeat_interleave(self.m, dim=0)
        candidates = self.base_policy.sample_step(
            learner=learner,
            x=x_rep,
            y_prefix=y_prefix_rep,
            y_true=y_true_rep,
            token_index=token_index,
        ).view(batch_size, self.m)

        correct_mask = prefix_correct[:, None] & (candidates == y_true[:, token_index : token_index + 1])
        any_correct = correct_mask.any(dim=1)

        first_correct_idx = correct_mask.to(torch.int64).argmax(dim=1, keepdim=True)
        first_correct_action = candidates.gather(1, first_correct_idx).squeeze(1)
        last_action = candidates[:, -1]
        return torch.where(any_correct, first_correct_action, last_action)


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
