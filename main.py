from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate

from utils.config import load_config
from utils.data_generator import GroundTruth, RademacherInputGenerator
from utils.experiments import (
    find_experiment_artifact,
    load_experiment_artifact,
    plot_experiment_artifact,
    run_cdf_quantile_experiment,
    run_outcome_reward_experiment,
    run_process_reward_experiment,
    save_experiment_artifact,
    run_threshold_track_experiment,
)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run experiments and plot from saved artifacts"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run experiment from config and save artifact")
    run_parser.add_argument("config_name", help="Config name under configs/")
    run_parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: config global.output_dir)",
    )
    run_parser.add_argument(
        "--artifact-file",
        default=None,
        help="Optional artifact filename under output_dir",
    )

    plot_parser = subparsers.add_parser("plot", help="Generate plots from saved artifact")
    plot_parser.add_argument(
        "experiment_name",
        help="Config stem / artifact name (e.g. process_reward or process_reward.yaml)",
    )
    plot_parser.add_argument(
        "--output-dir",
        default=None,
        help="Artifact/plot directory (default: outputs/)",
    )
    plot_parser.add_argument(
        "--artifact-file",
        default=None,
        help="Explicit artifact filename under output_dir",
    )
    return parser.parse_args()


def _normalize_config_name(config_name: str) -> str:
    if not config_name.endswith(".yaml"):
        config_name = f"{config_name}.yaml"
    return config_name


def _config_stem(config_name: str) -> str:
    return Path(_normalize_config_name(config_name)).stem


def _run_command(config_name: str, output_dir_override: str | None, artifact_file: str | None):
    config_name = _normalize_config_name(config_name)
    config_path = Path("configs") / config_name
    cfg = load_config(config_path)
    global_cfg = cfg["global"]
    exp = cfg.experiment
    exp_type = config_path.stem
    experiment_name = config_path.stem
    valid_experiment_types = {
        "outcome_reward",
        "process_reward",
        "cdf_quantile",
        "threshold_track",
    }
    if exp_type not in valid_experiment_types:
        raise ValueError(
            f"Unknown experiment type from config filename: {config_path.name}. "
            f"Expected one of {sorted(valid_experiment_types)}."
        )

    set_seed(global_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir_override or global_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(global_cfg.seed if exp.get("seed") is None else exp.seed)
    gt = GroundTruth()
    data_generator = (
        instantiate(global_cfg.data_generator)
        if global_cfg.get("data_generator") is not None
        else RademacherInputGenerator()
    )

    if exp_type == "outcome_reward":
        print(f"Running outcome-reward experiment: {experiment_name}")
        artifact = run_outcome_reward_experiment(
            exp=exp,
            d=global_cfg.d,
            k=global_cfg.k,
            seq_length=global_cfg.seq_length,
            gt=gt,
            device=device,
            test_set_size=global_cfg.test_set_size,
            track_set_size=global_cfg.track_set_size,
            track_source=global_cfg.track_source,
            data_generator=data_generator,
        )
    elif exp_type == "process_reward":
        print(f"Running process-reward experiment: {experiment_name}")
        artifact = run_process_reward_experiment(
            exp=exp,
            d=global_cfg.d,
            k=global_cfg.k,
            seq_length=global_cfg.seq_length,
            gt=gt,
            device=device,
            test_set_size=global_cfg.test_set_size,
            track_set_size=global_cfg.track_set_size,
            track_source=global_cfg.track_source,
            data_generator=data_generator,
        )
    elif exp_type == "cdf_quantile":
        print(f"Running CDF/quantile experiment: {experiment_name}")
        artifact = run_cdf_quantile_experiment(
            exp=exp,
            d=global_cfg.d,
            k=global_cfg.k,
            seq_length=global_cfg.seq_length,
            gt=gt,
            device=device,
            test_set_size=global_cfg.test_set_size,
            data_generator=data_generator,
        )
    elif exp_type == "threshold_track":
        print(f"Running threshold-track experiment: {experiment_name}")
        artifact = run_threshold_track_experiment(
            exp=exp,
            d=global_cfg.d,
            k=global_cfg.k,
            seq_length=global_cfg.seq_length,
            gt=gt,
            device=device,
            test_set_size=global_cfg.test_set_size,
            track_set_size=global_cfg.track_set_size,
            track_source=global_cfg.track_source,
            data_generator=data_generator,
        )
    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")
    saved_path = save_experiment_artifact(
        artifact=artifact,
        output_dir=output_dir,
        artifact_file=artifact_file,
        experiment_name=experiment_name,
    )
    print(f"Saved artifact: {saved_path}")


def _plot_command(experiment_name: str, output_dir_override: str | None, artifact_file: str | None):
    experiment_name = _config_stem(experiment_name)
    output_dir = Path(output_dir_override or "outputs/")
    if artifact_file:
        artifact_path = output_dir / artifact_file
    else:
        artifact_path = find_experiment_artifact(
            experiment_name=experiment_name,
            output_dir=output_dir,
        )
    artifact = load_experiment_artifact(artifact_path)
    plot_experiment_artifact(artifact=artifact, output_dir=output_dir)
    print(f"Generated plots for '{artifact_path.stem}' from artifact: {artifact_path}")


def main():
    args = parse_args()
    if args.command == "run":
        _run_command(
            config_name=args.config_name,
            output_dir_override=args.output_dir,
            artifact_file=args.artifact_file,
        )
        return
    if args.command == "plot":
        _plot_command(
            experiment_name=args.experiment_name,
            output_dir_override=args.output_dir,
            artifact_file=args.artifact_file,
        )
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
