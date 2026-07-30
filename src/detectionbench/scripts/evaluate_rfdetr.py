"""Evaluate RF-DETR checkpoints on a canonical COCO-format dataset split."""

from __future__ import annotations

import gc
import json
import weakref
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from detectionbench.utils.rfdetr import (
    build_model_kwargs,
    load_rfdetr_model_class,
    normalize_model_name,
    resolve_path,
)
from detectionbench.utils.utils import RichConsoleManager

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"
COCO_ANNOTATION_FILE = "_annotations.coco.json"
PROGRESS_DESCRIPTION = "[bold cyan]{task.description}[/bold cyan]"
INFERENCE_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def cleanup_gpu_memory(obj: Any = None, *, verbose: bool = False) -> None:
    """Release cached CUDA memory before or after model loading."""
    if not torch.cuda.is_available():
        if verbose:
            RichConsoleManager.get_console().print(
                "[dim]CUDA is not available. Skipping GPU cleanup.[/dim]"
            )
        return

    console = RichConsoleManager.get_console()
    torch.cuda.synchronize()

    def _memory_stats() -> tuple[float, float]:
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        return allocated, reserved

    if verbose:
        allocated, reserved = _memory_stats()
        console.print(
            "[dim]GPU memory before cleanup: "
            f"allocated={allocated:.2f} MB reserved={reserved:.2f} MB[/dim]"
        )

    if obj is not None:
        reference = weakref.ref(obj)
        del obj
        if reference() is not None and verbose:
            console.print(
                "[yellow]RF-DETR model object is still referenced after "
                "deletion.[/yellow]"
            )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()

    if verbose:
        allocated, reserved = _memory_stats()
        console.print(
            "[dim]GPU memory after cleanup: "
            f"allocated={allocated:.2f} MB reserved={reserved:.2f} MB[/dim]"
        )


def build_progress(console) -> Progress:
    """Create a standard Rich progress layout for evaluation steps."""
    return Progress(
        SpinnerColumn(),
        TextColumn(PROGRESS_DESCRIPTION),
        BarColumn(bar_width=None),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


def load_detection_dataset(dataset_dir: Path, split: str) -> tuple[Any, dict[int, str]]:
    """Load a Supervision detection dataset and class names for the requested split."""
    try:
        import supervision as sv
    except ImportError as err:
        raise ImportError(
            "Supervision is required for RF-DETR evaluation. Install it with "
            "`pip install supervision`."
        ) from err

    split_dir = dataset_dir / split
    annotation_path = split_dir / COCO_ANNOTATION_FILE
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")
    if not annotation_path.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {annotation_path}")

    dataset = sv.DetectionDataset.from_coco(
        images_directory_path=str(split_dir),
        annotations_path=str(annotation_path),
    )
    categories = json.loads(annotation_path.read_text(encoding="utf-8"))["categories"]
    ordered_categories = sorted(categories, key=lambda category: category["id"])
    class_name_by_id = {
        int(category["id"]): str(category["name"]) for category in ordered_categories
    }
    return dataset, class_name_by_id


def serialize_metric_result(result: Any) -> dict[str, Any]:
    """Convert a Supervision MeanAveragePrecisionResult into JSON-friendly data."""
    ap_per_class = []
    for class_id, ap_scores in zip(result.matched_classes, result.ap_per_class):
        ap_per_class.append(
            {
                "class_id": int(class_id),
                "ap_scores": [float(score) for score in ap_scores.tolist()],
                "mAP50_95": float(np.mean(ap_scores)),
                "mAP50": float(ap_scores[0]) if len(ap_scores) > 0 else 0.0,
            }
        )

    return {
        "map50_95": float(result.map50_95),
        "map50": float(result.map50),
        "map75": float(result.map75),
        "iou_thresholds": [float(value) for value in result.iou_thresholds.tolist()],
        "matched_classes": [
            int(class_id) for class_id in result.matched_classes.tolist()
        ],
        "ap_per_class": ap_per_class,
        "small_objects_map50_95": float(result.small_objects.map50_95)
        if result.small_objects is not None
        else None,
        "medium_objects_map50_95": float(result.medium_objects.map50_95)
        if result.medium_objects is not None
        else None,
        "large_objects_map50_95": float(result.large_objects.map50_95)
        if result.large_objects is not None
        else None,
    }


def print_metrics_table(class_names: dict[int, str], metrics: dict[str, Any]) -> None:
    """Print RF-DETR evaluation metrics using Rich tables."""
    console = RichConsoleManager.get_console()
    console.rule("[bold]RF-DETR Evaluation Results[/bold]")

    summary = Table(title="Summary", show_header=True, header_style="bold magenta")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right")
    summary.add_row("mAP@0.5:0.95", f"{metrics['map50_95']:.4f}")
    summary.add_row("mAP@0.5", f"{metrics['map50']:.4f}")
    summary.add_row("mAP@0.75", f"{metrics['map75']:.4f}")
    console.print(summary, style="bold green")

    if metrics["ap_per_class"]:
        per_class = Table(
            title="Per-Class Metrics",
            show_header=True,
            header_style="bold cyan",
        )
        per_class.add_column("Class", style="white")
        per_class.add_column("mAP@0.5", justify="right")
        per_class.add_column("mAP@0.5:0.95", justify="right")
        for class_metrics in metrics["ap_per_class"]:
            class_id = class_metrics["class_id"]
            class_name = class_names.get(class_id, str(class_id))
            per_class.add_row(
                class_name,
                f"{class_metrics['mAP50']:.4f}",
                f"{class_metrics['mAP50_95']:.4f}",
            )
        console.print(per_class, style="bold green")


def resolve_inference_dtype(dtype_name: str) -> torch.dtype:
    """Resolve a config dtype string into a torch dtype."""
    normalized = dtype_name.strip().lower()
    if normalized not in INFERENCE_DTYPE_MAP:
        supported = ", ".join(sorted(INFERENCE_DTYPE_MAP))
        raise ValueError(
            "Unsupported inference dtype "
            f"'{dtype_name}'. Supported values: {supported}."
        )
    return INFERENCE_DTYPE_MAP[normalized]


def evaluate_rfdetr(cfg: DictConfig) -> dict[str, Any]:
    """Evaluate an RF-DETR checkpoint on the configured dataset split."""
    try:
        from supervision.metrics import MeanAveragePrecision
    except ImportError as err:
        raise ImportError(
            "Supervision is required for RF-DETR evaluation. Install it with "
            "`pip install supervision`."
        ) from err

    console = RichConsoleManager.get_console()
    dataset_dir = Path(resolve_path(cfg.evaluation.dataset_dir) or "")
    output_dir = Path(resolve_path(cfg.evaluation.output_dir) or "")
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.evaluation.cleanup_gpu_before_load:
        cleanup_gpu_memory(verbose=bool(cfg.evaluation.verbose_cleanup))

    model_class = load_rfdetr_model_class(str(cfg.model.name))
    model_kwargs = build_model_kwargs(cfg, device_key="evaluation")
    checkpoint_path = resolve_path(cfg.evaluation.checkpoint)
    if checkpoint_path:
        model_kwargs["pretrain_weights"] = checkpoint_path
    model = model_class(**model_kwargs)

    if cfg.evaluation.optimize_for_inference:
        model.optimize_for_inference(
            compile=bool(cfg.evaluation.compile_inference),
            batch_size=int(cfg.evaluation.inference_batch_size),
            dtype=resolve_inference_dtype(str(cfg.evaluation.inference_dtype)),
        )

    dataset, class_names = load_detection_dataset(
        dataset_dir, str(cfg.evaluation.split)
    )
    map_metric = MeanAveragePrecision()

    console.print(
        f"[bold cyan]Evaluating {normalize_model_name(str(cfg.model.name))} on "
        f"{cfg.evaluation.split} split[/bold cyan]"
    )
    console.print(f"  Dataset: {dataset_dir}")
    console.print(f"  Checkpoint: {checkpoint_path}")
    console.print(f"  Output: {output_dir}")

    with build_progress(console) as progress:
        task_id = progress.add_task(
            "RF-DETR inference",
            total=len(dataset),
        )
        for _, image, annotations in dataset:
            detections = model.predict(
                image,
                threshold=float(cfg.evaluation.score_threshold),
                include_source_image=False,
            )
            map_metric.update(detections, annotations)
            progress.advance(task_id)

    result = map_metric.compute()
    metrics = serialize_metric_result(result)
    metrics["split"] = str(cfg.evaluation.split)
    metrics["num_images"] = len(dataset)
    metrics["score_threshold"] = float(cfg.evaluation.score_threshold)
    metrics["checkpoint"] = checkpoint_path
    metrics["dataset_dir"] = str(dataset_dir)
    metrics["class_names"] = {
        str(class_id): class_name for class_id, class_name in class_names.items()
    }

    print_metrics_table(class_names, metrics)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    console.print(f"\n✓ Metrics saved to [bold]{metrics_path}[/bold]")

    if cfg.evaluation.cleanup_gpu_after_eval:
        model_to_cleanup = model
        del model
        cleanup_gpu_memory(
            model_to_cleanup,
            verbose=bool(cfg.evaluation.verbose_cleanup),
        )

    return metrics


@hydra.main(
    version_base=None,
    config_path=str(CONFIGS_DIR),
    config_name="rfdetr_evaluate",
)
def main(cfg: DictConfig) -> None:
    """Run the RF-DETR evaluation CLI entrypoint."""
    console = RichConsoleManager.get_console()
    console.print(OmegaConf.to_yaml(cfg), style="warning")
    evaluate_rfdetr(cfg)


if __name__ == "__main__":
    main()
