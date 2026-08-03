"""
Train RF-DETR models using Hydra configuration.

Not a console-script entrypoint itself -- reached via `detectionbench-train`,
which dispatches here or to `train_yolo.py` based on `model.name=...` (see
`train.py`, the dispatcher). Still directly runnable via
`python -m detectionbench.scripts.train_rfdetr` if ever needed.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from detectionbench.utils.rfdetr import (
    build_model_kwargs,
    build_training_kwargs,
    load_rfdetr_model_class,
    normalize_model_name,
    seed_everything,
)
from detectionbench.utils.utils import RichConsoleManager

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"


@hydra.main(version_base=None, config_path=str(CONFIGS_DIR), config_name="rfdetr")
def train_rfdetr(cfg: DictConfig) -> None:
    """Train the configured RF-DETR model."""
    seed_everything(int(cfg.training.seed))
    console = RichConsoleManager.get_console()
    console.print(OmegaConf.to_yaml(cfg), style="warning")

    model_class = load_rfdetr_model_class(str(cfg.model.name))
    model_kwargs = build_model_kwargs(cfg, device_key="training")
    training_kwargs = build_training_kwargs(cfg)

    model_name = normalize_model_name(str(cfg.model.name))
    console.print(f"[bold cyan]Starting RF-DETR training with {model_name}[/bold cyan]")
    console.print(f"  Dataset: {training_kwargs['dataset_dir']}")
    console.print(f"  Output: {training_kwargs['output_dir']}")

    model = model_class(**model_kwargs)
    model.train(**training_kwargs)

    console.print("[bold green]RF-DETR training finished.[/bold green]")
    console.print(f"  Artifacts saved to: {training_kwargs['output_dir']}")


def main() -> None:
    """Run the RF-DETR training CLI entrypoint."""
    train_rfdetr()


if __name__ == "__main__":
    main()
