from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from hydra.utils import instantiate

from utils.config import load_config
from utils.data_generator import GroundTruth, RademacherInputGenerator
from utils.experiments import (
    run_cdf_quantile_experiment,
    run_outcome_reward_experiment,
    run_process_reward_experiment,
    run_threshold_track_experiment,
)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments from config")
    parser.add_argument(
        "config_name",
        nargs="?",
        help="Config name under configs/",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_name = args.config_name
    if not config_name.endswith(".yaml"):
        config_name = f"{config_name}.yaml"
    config_path = Path("configs") / config_name
    cfg = load_config(config_path)
    global_cfg = cfg["global"]
    exp = cfg.experiment
    exp_type = exp.type

    set_seed(global_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(global_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(global_cfg.seed if exp.get("seed") is None else exp.seed)
    gt = GroundTruth()
    data_generator = (
        instantiate(global_cfg.data_generator)
        if global_cfg.get("data_generator") is not None
        else RademacherInputGenerator()
    )

    if exp_type == "outcome_reward":
        print(f"Running outcome-reward experiment: {exp.name}")
        run_outcome_reward_experiment(
            exp=exp,
            d=global_cfg.d,
            k=global_cfg.k,
            seq_length=global_cfg.seq_length,
            gt=gt,
            device=device,
            test_set_size=global_cfg.test_set_size,
            track_set_size=global_cfg.track_set_size,
            output_dir=output_dir,
            track_source=global_cfg.track_source,
            data_generator=data_generator,
        )
    elif exp_type == "process_reward":
        print(f"Running process-reward experiment: {exp.name}")
        run_process_reward_experiment(
            exp=exp,
            d=global_cfg.d,
            k=global_cfg.k,
            seq_length=global_cfg.seq_length,
            gt=gt,
            device=device,
            test_set_size=global_cfg.test_set_size,
            track_set_size=global_cfg.track_set_size,
            output_dir=output_dir,
            track_source=global_cfg.track_source,
            data_generator=data_generator,
        )
    elif exp_type == "cdf_quantile":
        print(f"Running CDF/quantile experiment: {exp.name}")
        run_cdf_quantile_experiment(
            exp=exp,
            d=global_cfg.d,
            k=global_cfg.k,
            seq_length=global_cfg.seq_length,
            gt=gt,
            device=device,
            test_set_size=global_cfg.test_set_size,
            output_dir=output_dir,
            data_generator=data_generator,
        )
    elif exp_type == "threshold_track":
        print(f"Running threshold-track experiment: {exp.name}")
        run_threshold_track_experiment(
            exp=exp,
            d=global_cfg.d,
            k=global_cfg.k,
            seq_length=global_cfg.seq_length,
            gt=gt,
            device=device,
            test_set_size=global_cfg.test_set_size,
            track_set_size=global_cfg.track_set_size,
            output_dir=output_dir,
            track_source=global_cfg.track_source,
            data_generator=data_generator,
        )
    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")


if __name__ == "__main__":
    main()
