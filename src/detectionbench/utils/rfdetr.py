"""Shared RF-DETR helpers for Hydra entrypoints."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

RFDETR_MODEL_CLASS_MAP = {
    "rfdetr-nano": "RFDETRNano",
    "rfdetr-small": "RFDETRSmall",
    "rfdetr-medium": "RFDETRMedium",
    "rfdetr-large": "RFDETRLarge",
}
RFDETR_MODEL_ALIASES = {
    "nano": "rfdetr-nano",
    "small": "rfdetr-small",
    "medium": "rfdetr-medium",
    "large": "rfdetr-large",
    "rfdetr_nano": "rfdetr-nano",
    "rfdetr_small": "rfdetr-small",
    "rfdetr_medium": "rfdetr-medium",
    "rfdetr_large": "rfdetr-large",
}


def seed_everything(seed: int) -> None:
    """Seed supported random number generators for reproducible runs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_path(path_value: str | None) -> str | None:
    """Resolve a config path relative to the original working directory."""
    if not path_value:
        return None
    if path_value in {"last", "best"}:
        return path_value
    return to_absolute_path(path_value)


def normalize_model_name(model_name: str) -> str:
    """Normalize RF-DETR model aliases to canonical config names."""
    normalized = model_name.strip().lower()
    return RFDETR_MODEL_ALIASES.get(normalized, normalized)


def load_rfdetr_model_class(model_name: str) -> type[Any]:
    """Resolve the configured RF-DETR model class from the installed package."""
    try:
        import rfdetr
    except ImportError as err:
        raise ImportError(
            "RF-DETR is not installed. Install it with `pip install rfdetr` "
            'or `pip install "rfdetr[train,loggers]"` for training support.'
        ) from err

    canonical_name = normalize_model_name(model_name)
    class_name = RFDETR_MODEL_CLASS_MAP.get(canonical_name)
    if class_name is None:
        supported = ", ".join(sorted(RFDETR_MODEL_CLASS_MAP))
        raise ValueError(
            f"Unsupported RF-DETR model '{model_name}'. Supported values: {supported}."
        )
    return getattr(rfdetr, class_name)


def build_model_kwargs(cfg: DictConfig, *, device_key: str) -> dict[str, Any]:
    """Build RF-DETR model constructor kwargs from Hydra config."""
    model_kwargs: dict[str, Any] = {
        "device": str(cfg[device_key].device),
    }
    if cfg.model.num_classes is not None:
        model_kwargs["num_classes"] = int(cfg.model.num_classes)
    if cfg.model.pretrain_weights:
        model_kwargs["pretrain_weights"] = resolve_path(cfg.model.pretrain_weights)
    if cfg.model.resolution is not None:
        model_kwargs["resolution"] = int(cfg.model.resolution)
    return model_kwargs


def build_training_kwargs(cfg: DictConfig) -> dict[str, Any]:
    """Build RF-DETR training kwargs from Hydra config."""
    batch_size: int | str
    if str(cfg.training.batch_size).lower() == "auto":
        batch_size = "auto"
    else:
        batch_size = int(cfg.training.batch_size)

    training_kwargs: dict[str, Any] = {
        "dataset_dir": to_absolute_path(cfg.training.dataset_dir),
        "dataset_file": str(cfg.training.dataset_file),
        "output_dir": to_absolute_path(cfg.training.output_dir),
        "epochs": int(cfg.training.epochs),
        "batch_size": batch_size,
        "grad_accum_steps": int(cfg.training.grad_accum_steps),
        "lr": float(cfg.training.learning_rate),
        "lr_encoder": float(cfg.training.learning_rate_encoder),
        "lr_scheduler": str(cfg.training.scheduler_type),
        "checkpoint_interval": int(cfg.training.checkpoint_interval),
        "eval_interval": int(cfg.training.eval_interval),
        "num_workers": int(cfg.training.workers),
        "weight_decay": float(cfg.training.weight_decay),
        "use_ema": bool(cfg.training.use_ema),
        "tensorboard": bool(cfg.training.tensorboard),
        "wandb": bool(cfg.training.wandb),
        "early_stopping": bool(cfg.training.early_stopping),
        "early_stopping_patience": int(cfg.training.early_stopping_patience),
        "early_stopping_min_delta": float(cfg.training.early_stopping_min_delta),
        "progress_bar": cfg.training.progress_bar,
        "amp_dtype": str(cfg.training.amp_dtype),
        "device": str(cfg.training.device),
        "project": cfg.training.project,
        "run": cfg.training.run_name,
        "run_test": bool(cfg.training.run_test),
        "augmentation_backend": str(cfg.training.augmentation_backend),
        "save_dataset_grids": bool(cfg.training.save_dataset_grids),
        "resolution": int(cfg.model.resolution)
        if cfg.model.resolution is not None
        else None,
        "seed": int(cfg.training.seed),
    }
    if cfg.training.resume:
        training_kwargs["resume"] = resolve_path(cfg.training.resume)
    return {key: value for key, value in training_kwargs.items() if value is not None}
