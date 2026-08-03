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

Usage:
  detectionbench-evaluate --model yolov8n --dataset lisa \\
      --checkpoint experiments/lisa/yolov8n/weights/best.pt
  detectionbench-evaluate --model rfdetr-nano --dataset lisa \\
      --checkpoint experiments/lisa/rfdetr-nano/checkpoint_best_total.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from omegaconf import OmegaConf

from detectionbench.datasets import get_spec, list_datasets
from detectionbench.utils.utils import RichConsoleManager

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_CONFIG_DIR = REPO_ROOT / "configs" / "dataset"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments shared across both evaluation paths."""
    parser = argparse.ArgumentParser(
        description="Evaluate a DetectionBench model (any family, one command).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint/weights"
    )
    parser.add_argument(
        "--model", required=True, help="Model name, e.g. yolov8n or rfdetr-nano"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list_datasets(),
        help="Registered dataset key",
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
    parser.add_argument("--split", default="test", help="Dataset split to evaluate")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output-dir", default="eval_outputs", help="Output directory")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.05,
        help="Score threshold (YOLO/RT-DETR only -- RF-DETR always evaluates "
        "at threshold=0.0 internally for a correct mAP curve; see module "
        "docstring)",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions JSON (YOLO/RT-DETR only)",
    )
    return parser.parse_args()


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
                "resolution": None,
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
    if args.model.lower().startswith(("yolo", "rtdetr")):
        _run_yolo(args)
    else:
        _run_rfdetr(args)


if __name__ == "__main__":
    main()
