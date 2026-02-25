from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    cfg = OmegaConf.load(Path(path))
    OmegaConf.resolve(cfg)
    return cfg
