"""
Train a YOLO model on the configured dataset.

Runs evaluation afterward.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from detectionbench.datasets import get_spec
from detectionbench.scripts.evaluate import (
    EvaluationOptions,
    evaluate_yolo,
    print_metrics_table,
)
from detectionbench.utils.trainer import YOLOTrainer
from detectionbench.utils.utils import RichConsoleManager

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"


def seed_everything(seed: int) -> None:
    """Seed supported random number generators for reproducible runs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Must be False when deterministic=True


@hydra.main(version_base=None, config_path=str(CONFIGS_DIR), config_name="config")
def train_and_evaluate(cfg: DictConfig) -> None:
    """Train the configured YOLO model and evaluate the best checkpoint."""
    console = RichConsoleManager.get_console()
    console.print(OmegaConf.to_yaml(cfg), style="warning")
    trainer = YOLOTrainer(
        model_name=cfg.model.name,
        num_classes=cfg.model.num_classes,
        device=cfg.training.device,
    )
    results = trainer.train(
        dataset_yaml=cfg.training.dataset_yaml,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.learning_rate,
        imgsz=cfg.training.imgsz,
        use_amp=cfg.training.use_amp,
        output_dir=cfg.training.output_dir,
        workers=cfg.training.workers,
        patience=cfg.training.patience,
        augment=cfg.training.use_augmentation,
    )
    if not results["model_path"]:
        raise RuntimeError("Training completed without producing a model checkpoint.")

    console.print(f"  Best model saved to: {results['model_path']}")
    console.print(f"  All artifacts saved to: {results['output_dir']}")

    console.print(
        "\n[bold cyan]Evaluating the trained model on test data...[/bold cyan]"
    )
    spec = get_spec(cfg.dataset.name)
    metrics = evaluate_yolo(
        EvaluationOptions(
            checkpoint_path=results["model_path"],
            dataset_yaml=cfg.evaluation.dataset_yaml,
            class_names=spec.classes,
            num_classes=cfg.evaluation.num_classes,
            device=cfg.evaluation.device,
            output_dir=Path(cfg.evaluation.output_dir),
            save_predictions=cfg.evaluation.save_predictions,
        )
    )
    console.print("\n[bold green]Evaluation metrics:[/bold green]")

    print_metrics_table(cfg.model, metrics)

    # Save JSON summary
    metrics_path = Path(cfg.evaluation.output_dir) / "metrics.json"
    serializable: dict[str, Any] = {
        k: (float(v) if isinstance(v, (float, np.floating)) else v)
        for k, v in metrics.items()
        if k != "per_class"
    }
    if "per_class" in metrics:
        serializable["per_class"] = {
            cls: {mk: float(mv) for mk, mv in mv_dict.items()}
            for cls, mv_dict in metrics["per_class"].items()
        }
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2)
    console.print(f"\n✓ Metrics saved to [bold]{metrics_path}[/bold]")


def main() -> None:
    """Run the training CLI entrypoint."""
    seed_everything(42)  # Set a fixed seed for reproducibility
    train_and_evaluate()


if __name__ == "__main__":
    main()
