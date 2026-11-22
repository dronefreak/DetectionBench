"""
Train a YOLO/RT-DETR (Ultralytics) model on the configured dataset.

Runs evaluation afterward. Not a console-script entrypoint itself -- reached
via `detectionbench-train`, which dispatches here or to `train_rfdetr.py`
based on `model.name=...` (see `train.py`, the dispatcher).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from detectionbench.datasets import get_spec
from detectionbench.scripts.evaluate_yolo import (
    EvaluationOptions,
    evaluate_yolo,
    finalize_metrics,
)
from detectionbench.utils.trainer import YOLOTrainer
from detectionbench.utils.utils import RichConsoleManager, seed_everything

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"


@hydra.main(version_base=None, config_path=str(CONFIGS_DIR), config_name="config")
def train_and_evaluate(cfg: DictConfig) -> None:
    """Train the configured YOLO model and evaluate the best checkpoint."""
    # Ultralytics re-seeds every RNG from its own `seed` train() argument, so the
    # same value is used for both; 0 is Ultralytics' default.
    seed = int(cfg.training.get("seed", 0))
    seed_everything(seed)
    console = RichConsoleManager.get_console()
    console.print(OmegaConf.to_yaml(cfg), style="warning")
    trainer = YOLOTrainer(
        model_name=cfg.model.name,
        num_classes=cfg.model.num_classes,
        device=cfg.training.device,
    )

    # The Ultralytics engine only offers linear (default) or cosine LR decay
    # via the `cos_lr` bool -- step/exponential schedulers from config.yaml
    # have no equivalent here.
    scheduler_type = str(cfg.training.scheduler_type).lower()
    cos_lr = bool(cfg.training.use_scheduler) and scheduler_type == "cosine"
    if bool(cfg.training.use_scheduler) and scheduler_type not in ("cosine", "linear"):
        console.print(
            f"[bold yellow]training.scheduler_type='{cfg.training.scheduler_type}' "
            "has no equivalent in the Ultralytics engine (only linear/cosine "
            "are supported); falling back to linear decay.[/bold yellow]"
        )

    # Optional pass-through of extra Ultralytics train() arguments (warmup_epochs,
    # close_mosaic, scale, max_det, ...) -- empty unless the config sets them.
    extra_cfg = cfg.training.get("extra_args")
    extra_args: dict[str, Any] = (
        {str(key): value for key, value in extra_cfg.items()} if extra_cfg else {}
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
        optimizer=cfg.training.optimizer,
        cos_lr=cos_lr,
        augment=cfg.training.use_augmentation,
        resume=cfg.training.get("resume"),
        seed=seed,
        **extra_args,
    )
    if not results["model_path"]:
        raise RuntimeError("Training completed without producing a model checkpoint.")

    console.print(f"  Best model saved to: {results['model_path']}")
    console.print(f"  All artifacts saved to: {results['output_dir']}")

    console.print(
        f"\n[bold cyan]Evaluating the trained model on the "
        f"'{cfg.evaluation.split}' split...[/bold cyan]"
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
            split=cfg.evaluation.split,
        )
    )
    console.print("\n[bold green]Evaluation metrics:[/bold green]")
    finalize_metrics(str(cfg.model.name), metrics, Path(cfg.evaluation.output_dir))


def main() -> None:
    """Run the training CLI entrypoint."""
    train_and_evaluate()


if __name__ == "__main__":
    main()
