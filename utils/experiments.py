from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Tuple

import torch
from hydra.utils import instantiate

from .plotting import (
    plot_average_likelihood_over_time,
    plot_alpha_tail,
    plot_cdf,
    plot_expected_error_over_time,
    plot_likelihood_histogram,
    plot_likelihood_over_time,
    plot_quantile,
)

from .data_generator import InputGenerator, GroundTruth, sample_batch
from .metrics import compute_sequence_likelihood, estimate_cdf_p
from .model import AutoregressivePolicy
from .training import policy_gradient_train, process_reward_train, supervised_train


def _ensure_output_dir(output_dir: str | Path):
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def _select_track_samples(
    model: AutoregressivePolicy,
    x: torch.Tensor,
    y: torch.Tensor,
    quantiles: Iterable[float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    likelihoods = compute_sequence_likelihood(model, x, y)
    _, sorted_idx = torch.sort(likelihoods)
    n = sorted_idx.numel()
    q_tensor = torch.tensor(list(quantiles), dtype=torch.float32)
    positions = [int(q.item() * (n - 1)) for q in q_tensor]
    track_idx = sorted_idx[positions]
    return x[track_idx], y[track_idx], q_tensor


def _sample_track_pool(
    use_test_set: bool,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    track_set_size: int,
    d: int,
    k: int,
    seq_length: int,
    gt: GroundTruth,
    device: torch.device,
    data_generator: InputGenerator,
):
    if use_test_set:
        return test_x, test_y
    return sample_batch(
        track_set_size,
        d=d,
        k=k,
        seq_length=seq_length,
        gt=gt,
        device=device,
        data_generator=data_generator,
    )


def _resolve_quantiles(exp: Any) -> list[float]:
    if exp.get("quantiles") is not None:
        return list(exp.quantiles)
    num_quantiles = int(exp.get("track_initial_quantiles", exp.get("num_quantiles", 15)))
    return torch.linspace(0.0, 1.0, num_quantiles).tolist()


def _plot_supervised_track_set_likelihood_histogram(
    exp: Any,
    output_dir: str | Path,
    fallback_filename: str,
    likelihoods: torch.Tensor,
):
    histogram_file = exp.get("supervised_track_histogram_file", fallback_filename)
    if not histogram_file:
        return
    bins = int(exp.get("supervised_track_histogram_bins", 80))
    plot_likelihood_histogram(
        likelihoods=likelihoods,
        filename=str(Path(output_dir) / histogram_file),
        bins=bins,
    )


def _save_supervised_track_set_likelihood_tensor(
    exp: Any,
    output_dir: str | Path,
    fallback_filename: str,
    likelihoods: torch.Tensor,
):
    tensor_file = exp.get("supervised_track_likelihood_tensor_file", fallback_filename)
    if not tensor_file:
        return
    torch.save(likelihoods.detach().cpu(), Path(output_dir) / tensor_file)


def _select_low_likelihood_track_samples(
    x: torch.Tensor,
    y: torch.Tensor,
    likelihoods: torch.Tensor,
    threshold: float,
    max_samples: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected_idx = torch.nonzero(likelihoods < threshold, as_tuple=False).squeeze(1)
    if selected_idx.numel() == 0:
        raise ValueError(
            f"No track_set samples found below threshold={threshold:.6f}. "
            "Increase track_set_size or raise initial_likelihood_threshold."
        )
    if max_samples is not None and selected_idx.numel() > max_samples:
        selected_likelihoods = likelihoods[selected_idx]
        keep = torch.argsort(selected_likelihoods)[:max_samples]
        selected_idx = selected_idx[keep]
    return x[selected_idx], y[selected_idx], likelihoods[selected_idx]


def run_outcome_reward_experiment(
    exp: Any,
    d: int,
    k: int,
    seq_length: int,
    gt: GroundTruth,
    device: torch.device,
    test_set_size: int,
    track_set_size: int,
    output_dir: str,
    track_source: str = "test",
    data_generator: InputGenerator | None = None,
):
    _ensure_output_dir(output_dir)
    if data_generator is None:
        raise ValueError("data_generator must be provided")
    test_x, test_y = sample_batch(
        test_set_size,
        d=d,
        k=k,
        seq_length=seq_length,
        gt=gt,
        device=device,
        data_generator=data_generator,
    )

    sup_model, _ = supervised_train(
        d=d,
        k=k,
        seq_length=seq_length,
        gt=gt,
        device=device,
        test_x=test_x,
        test_y=test_y,
        steps=exp.supervised.steps,
        batch_size=exp.supervised.batch_size,
        optimizer_partial=instantiate(exp.supervised.optimizer),
        test_every=max(1, exp.supervised.test_every),
        data_generator=data_generator,
    )

    track_x_pool, track_y_pool = _sample_track_pool(
        use_test_set=(track_source == "test"),
        test_x=test_x,
        test_y=test_y,
        track_set_size=track_set_size,
        d=d,
        k=k,
        seq_length=seq_length,
        gt=gt,
        device=device,
        data_generator=data_generator,
    )
    track_pool_likelihoods = compute_sequence_likelihood(sup_model, track_x_pool, track_y_pool)
    _plot_supervised_track_set_likelihood_histogram(
        exp=exp,
        output_dir=output_dir,
        fallback_filename="track_set_likelihood_hist_after_supervised_outcome.pdf",
        likelihoods=track_pool_likelihoods,
    )
    _save_supervised_track_set_likelihood_tensor(
        exp=exp,
        output_dir=output_dir,
        fallback_filename="track_set_likelihoods_after_supervised_outcome.pt",
        likelihoods=track_pool_likelihoods,
    )
    track_x, track_y, q_tensor = _select_track_samples(
        sup_model,
        track_x_pool,
        track_y_pool,
        _resolve_quantiles(exp),
    )

    _, pg_errors, history = policy_gradient_train(
        d=d,
        k=k,
        seq_length=seq_length,
        gt=gt,
        device=device,
        test_x=test_x,
        test_y=test_y,
        steps=exp.pg.steps,
        batch_size=exp.pg.batch_size,
        optimizer_partial=instantiate(exp.pg.optimizer),
        test_every=exp.pg.test_every,
        track_samples=(track_x, track_y),
        track_every=exp.pg.track_every,
        init_model=sup_model,
        behavior_policy=instantiate(exp.pg.behavior),
        data_generator=data_generator,
    )

    plot_likelihood_over_time(
        history,
        filename=str(Path(output_dir) / exp.likelihood_plot_file),
        track_every=exp.pg.track_every,
        quantiles=q_tensor,
    )
    if exp.expected_error_plot_file:
        plot_expected_error_over_time(
            pg_errors,
            filename=str(Path(output_dir) / exp.expected_error_plot_file),
            test_every=exp.pg.test_every,
        )


def run_process_reward_experiment(
    exp: Any,
    d: int,
    k: int,
    seq_length: int,
    gt: GroundTruth,
    device: torch.device,
    test_set_size: int,
    track_set_size: int,
    output_dir: str,
    track_source: str = "test",
    data_generator: InputGenerator | None = None,
):
    _ensure_output_dir(output_dir)
    if data_generator is None:
        raise ValueError("data_generator must be provided")
    test_x, test_y = sample_batch(
        test_set_size,
        d=d,
        k=k,
        seq_length=seq_length,
        gt=gt,
        device=device,
        data_generator=data_generator,
    )

    sup_model, _ = supervised_train(
        d=d,
        k=k,
        seq_length=seq_length,
        gt=gt,
        device=device,
        test_x=test_x,
        test_y=test_y,
        steps=exp.supervised.steps,
        batch_size=exp.supervised.batch_size,
        optimizer_partial=instantiate(exp.supervised.optimizer),
        test_every=max(1, exp.supervised.test_every),
        data_generator=data_generator,
    )

    track_x_pool, track_y_pool = _sample_track_pool(
        use_test_set=(track_source == "test"),
        test_x=test_x,
        test_y=test_y,
        track_set_size=track_set_size,
        d=d,
        k=k,
        seq_length=seq_length,
        gt=gt,
        device=device,
        data_generator=data_generator,
    )
    track_pool_likelihoods = compute_sequence_likelihood(sup_model, track_x_pool, track_y_pool)
    _plot_supervised_track_set_likelihood_histogram(
        exp=exp,
        output_dir=output_dir,
        fallback_filename="track_set_likelihood_hist_after_supervised_process.pdf",
        likelihoods=track_pool_likelihoods,
    )
    _save_supervised_track_set_likelihood_tensor(
        exp=exp,
        output_dir=output_dir,
        fallback_filename="track_set_likelihoods_after_supervised_process.pt",
        likelihoods=track_pool_likelihoods,
    )
    track_x, track_y, q_tensor = _select_track_samples(
        sup_model,
        track_x_pool,
        track_y_pool,
        _resolve_quantiles(exp),
    )

    _, pg_errors, history = process_reward_train(
        d=d,
        k=k,
        seq_length=seq_length,
        gt=gt,
        device=device,
        test_x=test_x,
        test_y=test_y,
        steps=exp.pg.steps,
        batch_size=exp.pg.batch_size,
        optimizer_partial=instantiate(exp.pg.optimizer),
        test_every=exp.pg.test_every,
        track_samples=(track_x, track_y),
        track_every=exp.pg.track_every,
        init_model=sup_model,
        baseline=exp.pg.baseline,
        behavior_policy=instantiate(exp.pg.behavior),
        data_generator=data_generator,
    )

    plot_likelihood_over_time(
        history,
        filename=str(Path(output_dir) / exp.likelihood_plot_file),
        track_every=exp.pg.track_every,
        quantiles=q_tensor,
    )
    if exp.expected_error_plot_file:
        plot_expected_error_over_time(
            pg_errors,
            filename=str(Path(output_dir) / exp.expected_error_plot_file),
            test_every=exp.pg.test_every,
        )


def run_cdf_quantile_experiment(
    exp: Any,
    d: int,
    k: int,
    seq_length: int,
    gt: GroundTruth,
    device: torch.device,
    test_set_size: int,
    output_dir: str,
    data_generator: InputGenerator | None = None,
):
    _ensure_output_dir(output_dir)
    if data_generator is None:
        raise ValueError("data_generator must be provided")

    test_x, test_y = sample_batch(
        test_set_size,
        d=d,
        k=k,
        seq_length=seq_length,
        gt=gt,
        device=device,
        data_generator=data_generator,
    )
    model = AutoregressivePolicy(d=d, k=k, device=device).to(device)
    models = [model]
    all_steps = torch.arange(0, exp.num_models + 1) * exp.partial_steps

    for _ in range(exp.num_models):
        model, _ = supervised_train(
            d=d,
            k=k,
            seq_length=seq_length,
            gt=gt,
            device=device,
            test_x=test_x,
            test_y=test_y,
            steps=exp.partial_steps,
            batch_size=exp.batch_size,
            optimizer_partial=instantiate(exp.optimizer),
            test_every=max(1, exp.partial_steps),
            init_model=model,
            data_generator=data_generator,
        )
        models.append(model)

    cdf_x, cdf_y = sample_batch(
        exp.cdf_samples,
        d=d,
        k=k,
        seq_length=seq_length,
        gt=gt,
        device=device,
        data_generator=data_generator,
    )
    cdfs = estimate_cdf_p(models, x=cdf_x, y=cdf_y)

    plot_cdf(cdfs, all_steps, filename=str(Path(output_dir) / exp.cdf_plot_file))
    plot_quantile(cdfs, all_steps, filename=str(Path(output_dir) / exp.quantile_plot_file))
    if exp.alpha_tail_plot_file:
        plot_alpha_tail(cdfs, all_steps, filename=str(Path(output_dir) / exp.alpha_tail_plot_file))


def run_threshold_track_experiment(
    exp: Any,
    d: int,
    k: int,
    seq_length: int,
    gt: GroundTruth,
    device: torch.device,
    test_set_size: int,
    track_set_size: int,
    output_dir: str,
    track_source: str = "test",
    data_generator: InputGenerator | None = None,
):
    _ensure_output_dir(output_dir)
    if data_generator is None:
        raise ValueError("data_generator must be provided")

    test_x, test_y = sample_batch(
        test_set_size,
        d=d,
        k=k,
        seq_length=seq_length,
        gt=gt,
        device=device,
        data_generator=data_generator,
    )

    sup_model, _ = supervised_train(
        d=d,
        k=k,
        seq_length=seq_length,
        gt=gt,
        device=device,
        test_x=test_x,
        test_y=test_y,
        steps=exp.supervised.steps,
        batch_size=exp.supervised.batch_size,
        optimizer_partial=instantiate(exp.supervised.optimizer),
        test_every=max(1, exp.supervised.test_every),
        data_generator=data_generator,
    )

    track_x_pool, track_y_pool = _sample_track_pool(
        use_test_set=(track_source == "test"),
        test_x=test_x,
        test_y=test_y,
        track_set_size=track_set_size,
        d=d,
        k=k,
        seq_length=seq_length,
        gt=gt,
        device=device,
        data_generator=data_generator,
    )
    track_pool_likelihoods = compute_sequence_likelihood(sup_model, track_x_pool, track_y_pool)
    _plot_supervised_track_set_likelihood_histogram(
        exp=exp,
        output_dir=output_dir,
        fallback_filename="track_set_likelihood_hist_after_supervised_threshold_track.pdf",
        likelihoods=track_pool_likelihoods,
    )
    _save_supervised_track_set_likelihood_tensor(
        exp=exp,
        output_dir=output_dir,
        fallback_filename="track_set_likelihoods_after_supervised_threshold_track.pt",
        likelihoods=track_pool_likelihoods,
    )

    threshold = float(exp.initial_likelihood_threshold)
    max_samples = exp.get("max_tracked_samples")
    max_samples = int(max_samples) if max_samples is not None else None
    track_x, track_y, initial_selected_likelihoods = _select_low_likelihood_track_samples(
        track_x_pool,
        track_y_pool,
        track_pool_likelihoods,
        threshold=threshold,
        max_samples=max_samples,
    )
    print(
        "[track-threshold] selected "
        f"{track_x.shape[0]} / {track_x_pool.shape[0]} samples "
        f"with initial likelihood < {threshold:.6f}"
    )

    reward_type = str(exp.reward_type).lower()
    if reward_type == "outcome":
        _, pg_errors, history = policy_gradient_train(
            d=d,
            k=k,
            seq_length=seq_length,
            gt=gt,
            device=device,
            test_x=test_x,
            test_y=test_y,
            steps=exp.pg.steps,
            batch_size=exp.pg.batch_size,
            optimizer_partial=instantiate(exp.pg.optimizer),
            test_every=exp.pg.test_every,
            track_samples=(track_x, track_y),
            track_every=exp.pg.track_every,
            init_model=sup_model,
            behavior_policy=instantiate(exp.pg.behavior),
            data_generator=data_generator,
        )
    elif reward_type == "process":
        _, pg_errors, history = process_reward_train(
            d=d,
            k=k,
            seq_length=seq_length,
            gt=gt,
            device=device,
            test_x=test_x,
            test_y=test_y,
            steps=exp.pg.steps,
            batch_size=exp.pg.batch_size,
            optimizer_partial=instantiate(exp.pg.optimizer),
            test_every=exp.pg.test_every,
            track_samples=(track_x, track_y),
            track_every=exp.pg.track_every,
            init_model=sup_model,
            baseline=bool(exp.pg.get("baseline", True)),
            behavior_policy=instantiate(exp.pg.behavior),
            data_generator=data_generator,
        )
    else:
        raise ValueError(f"Unknown reward_type for threshold_track experiment: {exp.reward_type}")

    plot_average_likelihood_over_time(
        history,
        filename=str(Path(output_dir) / exp.average_likelihood_plot_file),
        track_every=exp.pg.track_every,
    )
    torch.save(
        initial_selected_likelihoods.detach().cpu(),
        Path(output_dir) / exp.selected_initial_likelihood_tensor_file,
    )

    if exp.get("expected_error_plot_file"):
        plot_expected_error_over_time(
            pg_errors,
            filename=str(Path(output_dir) / exp.expected_error_plot_file),
            test_every=exp.pg.test_every,
        )
