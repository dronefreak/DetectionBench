r"""
Evaluation script for DetectionBench-registered object detection models.

Computes standard YOLO object detection metrics on validation/test sets
using the Ultralytics validation engine.

Usage examples:
  python -m detectionbench.scripts.evaluate \\
      --checkpoint outputs/yolov8n_200ep/yolov8n/weights/best.pt \\
      --model yolov8n \\
      --dataset doclaynet \\
      --dataset-yaml configs/dataset/doclaynet.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.table import Table

from detectionbench.datasets import get_spec, list_datasets
from detectionbench.utils.utils import RichConsoleManager


@dataclass(frozen=True)
class EvaluationOptions:
    """Store runtime options for YOLO evaluation."""

    checkpoint_path: str
    dataset_yaml: str | Path
    class_names: list[str]
    num_classes: int
    device: str
    output_dir: Path
    save_predictions: bool = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for YOLO evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate DetectionBench detection models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint / .pt file"
    )
    parser.add_argument("--model", default="yolov5s", help="Model name")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list_datasets(),
        help="Registered dataset key (determines class names/count)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of classes (defaults to the full class count for --dataset)",
    )

    # Dataset
    parser.add_argument("--dataset-yaml", required=True, help="Dataset YAML file")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    # Evaluation options
    parser.add_argument(
        "--score-threshold", type=float, default=0.05, help="Score threshold"
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.5, help="IoU threshold"
    )
    parser.add_argument(
        "--soft-nms", action="store_true", help="Use Soft-NMS (torchvision only)"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Output
    parser.add_argument("--output-dir", default="eval_outputs", help="Output directory")
    parser.add_argument(
        "--save-predictions", action="store_true", help="Save predictions JSON"
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# YOLO evaluation path
# ---------------------------------------------------------------------------


def evaluate_yolo(options: EvaluationOptions) -> dict[str, Any]:
    """
    Evaluate a YOLO model using the Ultralytics val engine.

    Runs ``model.val()``, and returns the standard Ultralytics metrics dict.
    """
    try:
        from ultralytics import YOLO as UltralyticsYOLO
    except ImportError as err:
        raise ImportError("pip install ultralytics>=8.0.0") from err

    console = RichConsoleManager.get_console()
    console.print(
        "\n[bold cyan]YOLO evaluation — using Ultralytics val engine[/bold cyan]"
    )

    names = options.class_names[: min(options.num_classes, len(options.class_names))]

    model = UltralyticsYOLO(str(options.checkpoint_path))
    results = model.val(
        data=str(options.dataset_yaml),
        device=options.device,
        split="test",
        save_json=options.save_predictions,
        project=str(options.output_dir.resolve()),
        name="yolo_eval",
        exist_ok=True,
    )

    # Extract metrics from Ultralytics results
    metrics: dict[str, Any] = {}
    if hasattr(results, "box"):
        metrics["mAP50"] = float(results.box.map50)
        metrics["mAP50_95"] = float(results.box.map)
        metrics["precision"] = float(results.box.mp)
        metrics["recall"] = float(results.box.mr)
        # Per-class
        if (
            hasattr(results.box, "ap_class_index")
            and results.box.ap_class_index is not None
        ):
            metrics["per_class"] = {}
            for i, cls_idx in enumerate(results.box.ap_class_index):
                cls_name = (
                    names[cls_idx] if cls_idx < len(names) else f"class_{cls_idx}"
                )
                metrics["per_class"][cls_name] = {
                    "mAP50": float(results.box.ap50[i])
                    if i < len(results.box.ap50)
                    else 0.0,
                    "mAP50_95": float(results.box.ap[i])
                    if i < len(results.box.ap)
                    else 0.0,
                }

    return metrics


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------


def print_metrics_table(model_name: str, metrics: dict[str, Any]) -> None:
    """Print a rich table of evaluation results."""
    console = RichConsoleManager.get_console()
    console.rule(f"[bold]Evaluation Results — {model_name}[/bold]")

    # Summary table
    summary = Table(title="Summary", show_header=True, header_style="bold magenta")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right")

    def fmt(v: Any) -> str:
        if v is None:
            return "[dim]N/A[/dim]"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    for key in ("mAP50", "mAP50_95", "precision", "recall", "f1"):
        if key in metrics:
            label = {"mAP50_95": "mAP@0.5:0.95", "mAP50": "mAP@0.5"}.get(
                key, key.title()
            )
            summary.add_row(label, fmt(metrics[key]))
    for key in ("fps", "avg_ms", "num_images"):
        if key in metrics:
            label = {"fps": "FPS", "avg_ms": "ms/image", "num_images": "Images"}.get(
                key, key
            )
            summary.add_row(label, fmt(metrics[key]))

    console.print(summary, style="bold green")

    # Per-class table
    per_class = metrics.get("per_class", {})
    if per_class:
        cls_table = Table(
            title="Per-Class Metrics", show_header=True, header_style="bold cyan"
        )
        cls_table.add_column("Class", style="white")
        has_map = any("mAP50" in v for v in per_class.values())
        if has_map:
            cls_table.add_column("mAP@0.5", justify="right")
            cls_table.add_column("mAP@0.5:0.95", justify="right")
        else:
            cls_table.add_column("Precision", justify="right")
            cls_table.add_column("Recall", justify="right")
            cls_table.add_column("F1", justify="right")

        for cls_name, cls_m in sorted(per_class.items()):
            if has_map:
                cls_table.add_row(
                    cls_name,
                    f"{cls_m.get('mAP50', 0):.4f}",
                    f"{cls_m.get('mAP50_95', 0):.4f}",
                )
            else:
                cls_table.add_row(
                    cls_name,
                    f"{cls_m.get('precision', 0):.4f}",
                    f"{cls_m.get('recall', 0):.4f}",
                    f"{cls_m.get('f1', 0):.4f}",
                )

        console.print(cls_table, style="bold green")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the evaluation CLI entrypoint."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.soft_nms:
        raise ValueError("Soft-NMS is not supported for Ultralytics evaluation.")

    spec = get_spec(args.dataset)
    num_classes = args.num_classes if args.num_classes is not None else spec.num_classes

    device = torch.device(args.device)
    console = RichConsoleManager.get_console()

    console.print("\n[bold green]DetectionBench Evaluation[/bold green]")
    console.print(f"  Model: [bold]{args.model}[/bold]")
    console.print(f"  Dataset: {spec.display_name}")
    console.print(f"  Checkpoint: {args.checkpoint}")
    console.print(f"  Device: {device}\n")

    metrics = evaluate_yolo(
        EvaluationOptions(
            checkpoint_path=args.checkpoint,
            dataset_yaml=args.dataset_yaml,
            class_names=spec.classes,
            num_classes=num_classes,
            device=args.device,
            output_dir=output_dir,
            save_predictions=args.save_predictions,
        )
    )

    print_metrics_table(args.model, metrics)

    # Save JSON summary
    metrics_path = output_dir / "metrics.json"
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


if __name__ == "__main__":
    main()
