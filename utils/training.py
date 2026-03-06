from __future__ import annotations

from collections.abc import Callable
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .behavior_policy import BehaviorPolicy, OnPolicyBehavior
from .data_generator import InputGenerator, GroundTruth, sample_batch
from .metrics import compute_sequence_likelihood, eval_sequence_error
from .model import AutoregressivePolicy


def supervised_train(
    d: int,
    k: int,
    seq_length: int,
    gt: GroundTruth,
    device: torch.device,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    steps: int,
    batch_size: int,
    optimizer_partial: Callable[..., torch.optim.Optimizer],
    test_every: int,
    init_model: Optional[AutoregressivePolicy] = None,
    data_generator: Optional[InputGenerator] = None,
    step_offset: int = 0,
):
    model = AutoregressivePolicy(d=d, k=k, device=device).to(device)
    if init_model is not None:
        model.w.data.copy_(init_model.w.data)

    opt = optimizer_partial(model.parameters())
    test_errors = []

    for step in range(steps):
        x, y = sample_batch(
            batch_size,
            d=d,
            k=k,
            seq_length=seq_length,
            gt=gt,
            device=device,
            data_generator=data_generator,
        )
        logits = model.logits(x, y)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % test_every == 0:
            err = eval_sequence_error(model, test_x, test_y)
            test_errors.append(err)
            global_step = step_offset + step + 1
            print(f"[sup] step {global_step} loss={loss.item():.4f} test_err={err:.3f}")

    return model, test_errors


def outcome_reward_pg(
    d: int,
    k: int,
    seq_length: int,
    gt: GroundTruth,
    device: torch.device,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    steps: int,
    batch_size: int,
    optimizer_partial: Callable[..., torch.optim.Optimizer],
    test_every: int,
    track_samples: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    track_every: Optional[int] = None,
    init_model: Optional[AutoregressivePolicy] = None,
    baseline: bool = True,
    behavior_policy: Optional[BehaviorPolicy] = None,
    data_generator: Optional[InputGenerator] = None,
):
    model = AutoregressivePolicy(d=d, k=k, device=device).to(device)
    if init_model is not None:
        model.w.data.copy_(init_model.w.data)

    opt = optimizer_partial(model.parameters())
    test_errors = []
    likelihood_history = []

    if track_every is None:
        track_every = test_every

    if track_samples is not None:
        track_x, track_y = track_samples
        likelihood_history.append(compute_sequence_likelihood(model, track_x, track_y).clone())
    test_errors.append(eval_sequence_error(model, test_x, test_y))

    behavior = behavior_policy if behavior_policy is not None else OnPolicyBehavior()

    for step in range(steps):
        x, y_true = sample_batch(
            batch_size,
            d=d,
            k=k,
            seq_length=seq_length,
            gt=gt,
            device=device,
            data_generator=data_generator,
        )
        y_samples = torch.empty((batch_size, seq_length), device=device, dtype=torch.long)

        for t in range(seq_length):
            y_prefix = y_samples[:, :t]
            y_t = behavior.sample_step(model, x, y_prefix, y_true, t)
            y_samples[:, t] = y_t

        logits = model.logits(x, y_samples)
        logps = F.log_softmax(logits, dim=-1).gather(-1, y_samples.unsqueeze(-1)).squeeze(-1)
        matches = (y_samples == y_true).all(dim=1)
        rewards = torch.where(
            matches,
            torch.ones_like(matches, dtype=torch.float32),
            -torch.ones_like(matches, dtype=torch.float32),
        )
        adv = rewards - rewards.mean() if baseline else rewards
        loss = -(adv[:, None] * logps).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if track_samples is not None and (step + 1) % track_every == 0:
            track_x, track_y = track_samples
            likelihood_history.append(compute_sequence_likelihood(model, track_x, track_y).clone())

        if (step + 1) % test_every == 0:
            err = eval_sequence_error(model, test_x, test_y)
            test_errors.append(err)
            acc = matches.float().mean().item()
            print(f"[pg] step {step + 1} loss={loss.item():.4f} acc={acc:.3f} test_err={err:.3f}")

    if track_samples is not None:
        return model, test_errors, likelihood_history
    return model, test_errors


def process_reward_pg(
    d: int,
    k: int,
    seq_length: int,
    gt: GroundTruth,
    device: torch.device,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    steps: int,
    batch_size: int,
    optimizer_partial: Callable[..., torch.optim.Optimizer],
    test_every: int,
    track_samples: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    track_every: Optional[int] = None,
    init_model: Optional[AutoregressivePolicy] = None,
    baseline: bool = True,
    behavior_policy: Optional[BehaviorPolicy] = None,
    data_generator: Optional[InputGenerator] = None,
):
    model = AutoregressivePolicy(d=d, k=k, device=device).to(device)
    if init_model is not None:
        model.w.data.copy_(init_model.w.data)

    opt = optimizer_partial(model.parameters())
    test_errors = []
    likelihood_history = []

    if track_every is None:
        track_every = test_every

    if track_samples is not None:
        track_x, track_y = track_samples
        likelihood_history.append(compute_sequence_likelihood(model, track_x, track_y).clone())
    test_errors.append(eval_sequence_error(model, test_x, test_y))

    behavior = behavior_policy if behavior_policy is not None else OnPolicyBehavior()

    for step in range(steps):
        x, y_true = sample_batch(
            batch_size,
            d=d,
            k=k,
            seq_length=seq_length,
            gt=gt,
            device=device,
            data_generator=data_generator,
        )
        y_samples = torch.empty((batch_size, seq_length), device=device, dtype=torch.long)

        for t in range(seq_length):
            y_prefix = y_samples[:, :t]
            y_t = behavior.sample_step(model, x, y_prefix, y_true, t)
            y_samples[:, t] = y_t

        logits = model.logits(x, y_samples)
        logps = F.log_softmax(logits, dim=-1).gather(-1, y_samples.unsqueeze(-1)).squeeze(-1)
        correct_prefix = (y_samples == y_true).long().cumprod(dim=1).bool()
        rewards = torch.where(
            correct_prefix,
            torch.ones_like(correct_prefix, dtype=torch.float32),
            -torch.ones_like(correct_prefix, dtype=torch.float32),
            # torch.zeros_like(correct_prefix, dtype=torch.float32),
        )

        adv = rewards - rewards.mean(dim=0, keepdim=True) if baseline else rewards
        loss = -(adv * logps).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if track_samples is not None and (step + 1) % track_every == 0:
            track_x, track_y = track_samples
            likelihood_history.append(compute_sequence_likelihood(model, track_x, track_y).clone())

        if (step + 1) % test_every == 0:
            err = eval_sequence_error(model, test_x, test_y)
            test_errors.append(err)
            acc = (y_samples == y_true).all(dim=1).float().mean().item()
            mean_reward = rewards.mean().item()
            print(
                f"[pr] step {step + 1} loss={loss.item():.4f} "
                f"acc={acc:.3f} mean_r={mean_reward:.3f} test_err={err:.3f}"
            )

    if track_samples is not None:
        return model, test_errors, likelihood_history
    return model, test_errors
