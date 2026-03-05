from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate

from .plotting import (
    plot_alpha_tail,
    plot_cdf,
    plot_expected_error_over_time,
    plot_likelihood_histogram,
    plot_likelihood_over_time,
    plot_quantile,
)

from .data_generator import (
    InputGenerator,
    GroundTruth,
    MixtureInputGenerator,
    generate_y,
    sample_batch,
)
from .metrics import compute_sequence_likelihood, estimate_cdf_p
from .model import AutoregressivePolicy
from .training import policy_gradient_train, process_reward_train, supervised_train


def _ensure_output_dir(output_dir: str | Path):
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def _sample_track_pool(
    track_source: str,
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
    if track_source == "test":
        return test_x, test_y
    if track_source == "mixture_centers":
        if not isinstance(data_generator, MixtureInputGenerator):
            raise ValueError("track_source='mixture_centers' requires MixtureInputGenerator")
        if gt.w1_star is None or gt.w2_star is None:
            raise ValueError("Ground-truth weights are not initialized")
        x = data_generator.centers(d=d, device=device)
        y = generate_y(x, gt.w1_star, gt.w2_star, seq_length)
        return x, y
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
    num_quantiles = int(exp.get("track_initial_quantiles", 15))
    return torch.linspace(0.0, 1.0, num_quantiles).tolist()


def _to_cpu_history(history: list[torch.Tensor]) -> list[torch.Tensor]:
    return [h.detach().cpu() for h in history]


def _fixed_plot_files(experiment_type: str) -> dict[str, str]:
    if experiment_type == "outcome_reward":
        return {
            "base_likelihood_histogram_file": "outcome_reward_base_likelihood_histogram.pdf",
            "likelihood_plot_file": "outcome_reward_likelihood_over_time.pdf",
            "expected_error_plot_file": "outcome_reward_expected_error_over_time.pdf",
        }
    if experiment_type == "process_reward":
        return {
            "base_likelihood_histogram_file": "process_reward_base_likelihood_histogram.pdf",
            "likelihood_plot_file": "process_reward_likelihood_over_time.pdf",
            "expected_error_plot_file": "process_reward_expected_error_over_time.pdf",
        }
    if experiment_type == "cdf_quantile":
        return {
            "cdf_plot_file": "cdf_quantile_cdf.pdf",
            "quantile_plot_file": "cdf_quantile_quantile.pdf",
            "alpha_tail_plot_file": "cdf_quantile_alpha_tail.pdf",
        }
    raise ValueError(f"Unknown experiment type for fixed plot files: {experiment_type}")


def _default_artifact_file(experiment_name: str) -> str:
    safe_name = str(experiment_name).replace("/", "_")
    return f"{safe_name}.pt"


def _torch_load_cpu(path: Path) -> Any:
    return torch.load(path, map_location="cpu")


def save_experiment_artifact(
    artifact: dict[str, Any],
    output_dir: str | Path,
    artifact_file: str | None = None,
    experiment_name: str | None = None,
) -> Path:
    _ensure_output_dir(output_dir)
    if artifact_file is None and not experiment_name:
        raise ValueError("experiment_name is required when artifact_file is not provided")
    filename = artifact_file or _default_artifact_file(str(experiment_name))
    artifact_path = Path(output_dir) / filename
    torch.save(artifact, artifact_path)
    return artifact_path


def load_experiment_artifact(path: str | Path) -> dict[str, Any]:
    artifact = _torch_load_cpu(Path(path))
    if not isinstance(artifact, dict):
        raise ValueError(f"Artifact at {path} is not a dictionary")
    return artifact


def find_experiment_artifact(
    experiment_name: str,
    output_dir: str | Path,
) -> Path:
    output_path = Path(output_dir)
    if not output_path.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_path}")
    expected = output_path / _default_artifact_file(experiment_name)
    if not expected.exists():
        raise FileNotFoundError(
            f"No artifact found for experiment_name='{experiment_name}'. "
            f"Expected file: {expected}"
        )
    return expected


def plot_experiment_artifact(
    artifact: dict[str, Any],
    output_dir: str | Path,
    ema_beta: float = 0.0,
    include_colorbar: bool = False,
):
    _ensure_output_dir(output_dir)
    exp_type = str(artifact["experiment_type"])
    data = artifact["data"]
    plot_cfg = artifact.get("plot", {})
    fixed_files = _fixed_plot_files(exp_type)
    title_map = {
        "outcome_reward": "With Outcome Reward",
        "process_reward": "With Process Reward",
    }
    title = title_map.get(exp_type)

    if "base_likelihood_histogram_file" in fixed_files:
        plot_likelihood_histogram(
            likelihoods=data["track_pool_likelihoods"],
            filename=str(Path(output_dir) / fixed_files["base_likelihood_histogram_file"]),
            bins=int(plot_cfg.get("base_likelihood_histogram_bins", 80)),
            title=title,
        )

    if exp_type in ("outcome_reward", "process_reward"):
        plot_likelihood_over_time(
            data["likelihood_history"],
            filename=str(Path(output_dir) / fixed_files["likelihood_plot_file"]),
            track_every=int(plot_cfg["track_every"]),
            include_colorbar=include_colorbar,
            ema_beta=ema_beta,
            title=title,
        )
        plot_expected_error_over_time(
            data["pg_errors"],
            filename=str(Path(output_dir) / fixed_files["expected_error_plot_file"]),
            test_every=int(plot_cfg["test_every"]),
            title=title,
        )
        return

    if exp_type == "cdf_quantile":
        cdfs = data["cdfs"]
        all_steps = data["all_steps"]
        plot_cdf(
            cdfs,
            all_steps,
            filename=str(Path(output_dir) / fixed_files["cdf_plot_file"]),
        )
        plot_quantile(
            cdfs,
            all_steps,
            filename=str(Path(output_dir) / fixed_files["quantile_plot_file"]),
        )
        plot_alpha_tail(
            cdfs,
            all_steps,
            filename=str(Path(output_dir) / fixed_files["alpha_tail_plot_file"]),
        )
        return

    raise ValueError(f"Unknown experiment type in artifact: {exp_type}")


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


def _select_tracking_samples(
    exp: Any,
    track_x_pool: torch.Tensor,
    track_y_pool: torch.Tensor,
    track_pool_likelihoods: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tracking_type = str(exp.get("likelihood_tracking_type", "uniform")).lower()
    if tracking_type == "uniform":
        _, sorted_idx = torch.sort(track_pool_likelihoods)
        n = sorted_idx.numel()
        quantiles = _resolve_quantiles(exp)
        q_tensor = torch.tensor(quantiles, dtype=torch.float32)
        positions = [int(q.item() * (n - 1)) for q in q_tensor]
        track_idx = sorted_idx[positions]
        return track_x_pool[track_idx], track_y_pool[track_idx]
    if tracking_type == "threshold":
        threshold = float(exp.initial_likelihood_threshold)
        max_samples = exp.get("max_tracked_samples")
        max_samples = int(max_samples) if max_samples is not None else None
        track_x, track_y, _ = _select_low_likelihood_track_samples(
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
        return track_x, track_y
    raise ValueError(f"Unknown likelihood_tracking_type: {tracking_type}")


def run_outcome_reward_experiment(
    exp: Any,
    d: int,
    k: int,
    seq_length: int,
    gt: GroundTruth,
    device: torch.device,
    test_set_size: int,
    track_set_size: int,
    track_source: str = "test",
    data_generator: InputGenerator | None = None,
) -> dict[str, Any]:
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
        track_source=track_source,
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
    track_x, track_y = _select_tracking_samples(
        exp,
        track_x_pool,
        track_y_pool,
        track_pool_likelihoods,
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
        baseline=bool(exp.pg.get("baseline", True)),
        behavior_policy=instantiate(exp.pg.behavior),
        data_generator=data_generator,
    )

    return {
        "experiment_type": "outcome_reward",
        "plot": {
            "base_likelihood_histogram_bins": int(exp.get("base_likelihood_histogram_bins", 80)),
            "track_every": int(exp.pg.track_every),
            "test_every": int(exp.pg.test_every),
        },
        "data": {
            "track_pool_likelihoods": track_pool_likelihoods.detach().cpu(),
            "likelihood_history": _to_cpu_history(history),
            "pg_errors": torch.as_tensor(pg_errors, dtype=torch.float32).detach().cpu(),
        },
    }


def run_process_reward_experiment(
    exp: Any,
    d: int,
    k: int,
    seq_length: int,
    gt: GroundTruth,
    device: torch.device,
    test_set_size: int,
    track_set_size: int,
    track_source: str = "test",
    data_generator: InputGenerator | None = None,
) -> dict[str, Any]:
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
        track_source=track_source,
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
    track_x, track_y = _select_tracking_samples(
        exp,
        track_x_pool,
        track_y_pool,
        track_pool_likelihoods,
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

    return {
        "experiment_type": "process_reward",
        "plot": {
            "base_likelihood_histogram_bins": int(exp.get("base_likelihood_histogram_bins", 80)),
            "track_every": int(exp.pg.track_every),
            "test_every": int(exp.pg.test_every),
        },
        "data": {
            "track_pool_likelihoods": track_pool_likelihoods.detach().cpu(),
            "likelihood_history": _to_cpu_history(history),
            "pg_errors": torch.as_tensor(pg_errors, dtype=torch.float32).detach().cpu(),
        },
    }


def run_cdf_quantile_experiment(
    exp: Any,
    d: int,
    k: int,
    seq_length: int,
    gt: GroundTruth,
    device: torch.device,
    test_set_size: int,
    data_generator: InputGenerator | None = None,
) -> dict[str, Any]:
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

    return {
        "experiment_type": "cdf_quantile",
        "data": {
            "all_steps": all_steps.detach().cpu(),
            "cdfs": [cdf.detach().cpu() for cdf in cdfs],
        },
    }
