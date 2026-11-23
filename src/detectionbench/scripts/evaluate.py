r"""
Unified `detectionbench-evaluate` entrypoint.

Dispatches to the Ultralytics (YOLO/RT-DETR) or RF-DETR evaluator based on
`--model`, so callers only ever need one command regardless of model family
-- mirrors `detectionbench-train`'s dispatch, and the pattern
`detectionbench-infer` already used internally.

Unlike the train dispatcher, this one can't just call the two underlying
`main()`s unmodified: `evaluate_yolo.py` is a plain argparse script,
`evaluate_rfdetr.py` is `@hydra.main`-decorated, and neither shares this
module's flag names 1:1. Instead this module owns one argparse surface and
either (a) calls `evaluate_yolo()` directly with an `EvaluationOptions`, or
(b) builds an `OmegaConf` config matching `configs/rfdetr_evaluate.yaml`'s
shape and calls `evaluate_rfdetr()` directly -- `hydra.utils.to_absolute_path`
(used internally for path resolution) is documented to fall back to
`os.getcwd()` when no Hydra run is active, so this works safely outside of
`@hydra.main`.

Both dataset path flags (`--dataset-yaml` for YOLO, `--dataset-dir` for
RF-DETR) auto-resolve from `configs/dataset/<key>.yaml` when omitted, since
that's already the source of truth both Hydra pipelines read from -- no
more manually retyping a path the config already has.

`--score-threshold` only applies to the YOLO path. RF-DETR's evaluation
requires threshold=0.0 internally to build a correct mAP curve (see
`evaluate_rfdetr.py`'s `pick_best_f1_operating_point` docstring for why);
exposing a shared threshold flag that silently broke that would reintroduce
the exact precision/recall bug fixed there.

Framework-specific tuning knobs not exposed here (Ultralytics'
`--batch-size`/`--iou-threshold`; RF-DETR's `compile_inference`/
`inference_dtype`/GPU-cleanup flags) are still reachable by calling
`evaluate_yolo.py`/`evaluate_rfdetr.py` directly -- see their own docstrings.

With ``--config-name`` the dataset, checkpoint and output locations (and, as a last
resort, the RF-DETR resolution) come from the same Hydra config used for training,
so evaluating a run needs only the model name:

  detectionbench-evaluate --config-name uavdt_rfdetr --model rfdetr-nano
  detectionbench-evaluate --config-name configs/uavdt_yolo.yaml --model yolov8n

Explicit flags always win over the config. For RF-DETR the resolution is taken from
``training_config.json`` next to the checkpoint when present (it records what the
model was really trained at, including per-model ``model.resolution`` overrides),
then from ``--resolution``/the config, then the model family default.

Usage:
  detectionbench-evaluate --model yolov8n --dataset lisa \
      --checkpoint experiments/lisa/yolov8n/yolov8n/weights/best.pt
  detectionbench-evaluate --model rfdetr-nano --dataset lisa \
      --checkpoint experiments/lisa/rfdetr-nano/checkpoint_best_total.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from detectionbench.datasets import get_spec, list_datasets
from detectionbench.utils.rfdetr import read_training_resolution
from detectionbench.utils.utils import RichConsoleManager

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / "configs"
DATASET_CONFIG_DIR = CONFIGS_DIR / "dataset"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments shared across both evaluation paths."""
    parser = argparse.ArgumentParser(
        description="Evaluate a DetectionBench model (any family, one command).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="Training Hydra config to take defaults from: a name in configs/ "
        "(e.g. uavdt_rfdetr) or a path (e.g. configs/uavdt_rfdetr.yaml). Supplies "
        "--dataset, --checkpoint, --output-dir and --split unless given explicitly.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to model checkpoint/weights (required unless --config-name)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name, e.g. yolov8n or rfdetr-nano (default with --config-name: "
        "the config's model.name)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        choices=list_datasets(),
        help="Registered dataset key (required unless --config-name)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of classes (default: full class count for --dataset)",
    )
    parser.add_argument(
        "--dataset-yaml",
        default=None,
        help="YOLO data.yaml path (YOLO/RT-DETR only; default: from "
        "configs/dataset/<key>.yaml)",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Canonical COCO root (RF-DETR only; default: from "
        "configs/dataset/<key>.yaml)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="RF-DETR input resolution (default: the one recorded in the "
        "checkpoint's training_config.json, else the model family default)",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split to evaluate (default: the config's dataset.eval_split "
        "with --config-name, else test)",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: experiments/<dataset>/<model>/evaluation, "
        "matching the Hydra configs)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Confidence threshold (YOLO/RT-DETR only; default: Ultralytics' "
        "0.001, as used for the leaderboards -- RF-DETR always evaluates at "
        "threshold=0.0 internally for a correct mAP curve; see module docstring)",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions JSON (YOLO/RT-DETR only)",
    )
    args = parser.parse_args()
    _apply_config(args, parser)
    return args


def _compose_config(config_name: str, model: str | None) -> DictConfig:
    """Compose a training Hydra config (by name or path), optionally for one model."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    path = Path(config_name)
    config_dir = path.resolve().parent if path.parent != Path(".") else CONFIGS_DIR
    name = path.stem if path.suffix in {".yaml", ".yml"} else path.name
    overrides = [f"model.name={model}"] if model else []
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name=name, overrides=overrides)


def _apply_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Fill unset arguments from ``--config-name``, then validate the required ones."""
    if args.config_name:
        cfg = _compose_config(args.config_name, args.model)
        args.model = args.model or str(cfg.model.name)
        is_yolo = args.model.lower().startswith(("yolo", "rtdetr"))
        config_is_yolo = "dataset_yaml" in cfg.training
        if is_yolo != config_is_yolo:
            parser.error(
                f"--config-name {args.config_name} is a "
                f"{'YOLO/RT-DETR' if config_is_yolo else 'RF-DETR'} config but "
                f"--model {args.model} is a {'YOLO/RT-DETR' if is_yolo else 'RF-DETR'} "
                "model."
            )
        args.dataset = args.dataset or str(cfg.dataset.name)
        args.split = args.split or str(cfg.dataset.get("eval_split", "test"))
        if is_yolo:
            args.checkpoint = args.checkpoint or str(cfg.evaluation.checkpoint)
            args.output_dir = args.output_dir or str(cfg.evaluation.output_dir)
        else:
            run_dir = Path(str(cfg.training.output_dir))
            args.checkpoint = args.checkpoint or str(
                run_dir / "checkpoint_best_total.pth"
            )
            args.output_dir = args.output_dir or str(run_dir / "evaluation")
            config_resolution = cfg.model.get("resolution")
            # training_config.json (next to the checkpoint) is the truth about what
            # the model was trained at; the config's resolution is only a default
            # that per-model overrides (e.g. model.resolution=640) may have changed.
            if (
                args.resolution is None
                and config_resolution is not None
                and read_training_resolution(args.checkpoint) is None
            ):
                args.resolution = int(config_resolution)
                RichConsoleManager.get_console().print(
                    f"[yellow]No training_config.json next to {args.checkpoint}; "
                    f"using the config's resolution {args.resolution}. Pass "
                    "--resolution if this model was trained with an override."
                    "[/yellow]"
                )
    for flag in ("model", "dataset", "checkpoint"):
        if getattr(args, flag) is None:
            parser.error(f"--{flag} is required (or pass --config-name)")
    if args.split is None:
        args.split = "test"


def _load_dataset_cfg(dataset_key: str) -> dict[str, Any]:
    """Load a dataset's Hydra config YAML (the source of truth for its paths)."""
    return yaml.safe_load((DATASET_CONFIG_DIR / f"{dataset_key}.yaml").read_text())


def _run_yolo(args: argparse.Namespace) -> None:
    """Dispatch to the Ultralytics (YOLO/RT-DETR) evaluator."""
    from detectionbench.scripts.evaluate_yolo import EvaluationOptions
    from detectionbench.scripts.evaluate_yolo import evaluate_yolo as run_evaluate_yolo
    from detectionbench.scripts.evaluate_yolo import finalize_metrics

    dataset_yaml = args.dataset_yaml or _load_dataset_cfg(args.dataset)["dataset_yaml"]
    spec = get_spec(args.dataset)
    num_classes = args.num_classes if args.num_classes is not None else spec.num_classes
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console = RichConsoleManager.get_console()
    console.print("\n[bold green]DetectionBench Evaluation (Ultralytics)[/bold green]")
    console.print(f"  Model: [bold]{args.model}[/bold]")
    console.print(f"  Dataset: {spec.display_name}")
    console.print(f"  Checkpoint: {args.checkpoint}")
    console.print(f"  Device: {args.device}\n")

    metrics = run_evaluate_yolo(
        EvaluationOptions(
            checkpoint_path=args.checkpoint,
            dataset_yaml=dataset_yaml,
            class_names=spec.classes,
            num_classes=num_classes,
            device=args.device,
            output_dir=output_dir,
            save_predictions=args.save_predictions,
            split=args.split,
            score_threshold=args.score_threshold,
        )
    )
    finalize_metrics(args.model, metrics, output_dir)


def _run_rfdetr(args: argparse.Namespace) -> None:
    """Dispatch to the RF-DETR evaluator via an equivalent Hydra config."""
    from detectionbench.scripts.evaluate_rfdetr import evaluate_rfdetr

    dataset_dir = args.dataset_dir or _load_dataset_cfg(args.dataset)["dataset_dir"]
    spec = get_spec(args.dataset)
    num_classes = args.num_classes if args.num_classes is not None else spec.num_classes

    cfg = OmegaConf.create(
        {
            "model": {
                "name": args.model,
                "num_classes": num_classes,
                "pretrain_weights": None,
                "resolution": args.resolution,
            },
            "evaluation": {
                "dataset_dir": dataset_dir,
                "checkpoint": args.checkpoint,
                "split": args.split,
                "output_dir": args.output_dir,
                "device": args.device,
                # Deliberately not args.score_threshold -- see module docstring.
                "score_threshold": 0.0,
                "optimize_for_inference": True,
                "compile_inference": True,
                "inference_batch_size": 1,
                "inference_dtype": "float32",
                "cleanup_gpu_before_load": False,
                "cleanup_gpu_after_eval": False,
                "verbose_cleanup": False,
            },
        }
    )
    evaluate_rfdetr(cfg)


def main() -> None:
    """Dispatch `detectionbench-evaluate` to the Ultralytics or RF-DETR evaluator."""
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = str(
            Path("experiments") / args.dataset / args.model / "evaluation"
        )
    if args.model.lower().startswith(("yolo", "rtdetr")):
        _run_yolo(args)
    else:
        _run_rfdetr(args)


if __name__ == "__main__":
    main()
