r"""
Export DetectionBench models to ONNX for downstream TensorRT conversion.

Supports both Ultralytics (YOLO/RT-DETR) and RF-DETR checkpoints.
Reached via `detectionbench-export-onnx`, or directly via
`python -m detectionbench.scripts.export_onnx` (Hydra overrides). Unlike
`detectionbench-train`/`detectionbench-evaluate`, this needs no argv-peeking
dispatcher: both model families share one Hydra config schema
(`configs/export_onnx.yaml`), and `resolve_model_family` picks the export path
(`export_ultralytics_onnx`/`export_rfdetr_onnx`) from `model.family` (or
infers it from `model.name`'s prefix when left at "auto").

Usage:
  detectionbench-export-onnx dataset=lisa model.name=yolo26n \\
      export.checkpoint_path=experiments/lisa/yolo26n/weights/best.pt
  detectionbench-export-onnx dataset=exdark model.name=rfdetr-nano \\
      export.checkpoint_path=experiments/exdark/rfdetr-nano/checkpoint_best_total.pth
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import hydra
import torch
import torch.nn.functional as torch_functional
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from detectionbench.utils.rfdetr import (
    build_model_kwargs,
    load_rfdetr_model_class,
    normalize_model_name as normalize_rfdetr_model_name,
    resolve_path,
)
from detectionbench.utils.utils import RichConsoleManager

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"
ULTRALYTICS_PREFIXES = ("yolo", "rtdetr")
RFDETR_PREFIX = "rfdetr"
RECTANGULAR_IMGSZ_LENGTH = 2
YOLOV10_PREFIX = "yolov10"


@dataclass(frozen=True)
class ExportResult:
    """Store the key outputs from an ONNX export run."""

    family: str
    model_name: str
    source_checkpoint: str | None
    output_path: str


@contextmanager
def disable_bicubic_antialias_during_export() -> Any:
    """Avoid ONNX export failures from RF-DETR's antialiased bicubic interpolation."""
    original_interpolate = torch_functional.interpolate

    def wrapped_interpolate(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("mode") == "bicubic" and kwargs.get("antialias"):
            kwargs = dict(kwargs)
            kwargs["antialias"] = False
        return original_interpolate(*args, **kwargs)

    with patch("torch.nn.functional.interpolate", new=wrapped_interpolate):
        yield


def resolve_imgsz(imgsz_value: Any) -> int | tuple[int, int]:
    """Normalize Hydra imgsz input into Ultralytics/RF-DETR export dimensions."""
    if isinstance(imgsz_value, int):
        return imgsz_value
    if isinstance(imgsz_value, Sequence) and not isinstance(imgsz_value, (str, bytes)):
        values = [int(value) for value in imgsz_value]
        if len(values) == 1:
            return values[0]
        if len(values) == RECTANGULAR_IMGSZ_LENGTH:
            return (values[0], values[1])
    raise ValueError(
        "export.imgsz must be an int or a 2-item list/tuple like [768, 1472]."
    )


def resolve_export_checkpoint_path(export_cfg: DictConfig) -> str | None:
    """Resolve the explicit checkpoint path used for export, if any."""
    checkpoint_path = resolve_path(export_cfg.checkpoint_path)
    if checkpoint_path:
        return checkpoint_path
    return resolve_path(export_cfg.pretrained_ckpt_path)


def infer_model_family(model_name: str) -> str:
    """Infer the export family from the configured model name."""
    normalized = model_name.strip().lower()
    if normalized.startswith(ULTRALYTICS_PREFIXES):
        return "ultralytics"
    if normalized.startswith(RFDETR_PREFIX):
        return "rfdetr"
    raise ValueError(
        f"Could not infer model family from '{model_name}'. "
        "Set model.family explicitly to 'ultralytics' or 'rfdetr'."
    )


def resolve_model_family(cfg: DictConfig) -> str:
    """Resolve the configured or inferred model family."""
    family = str(cfg.model.family).strip().lower()
    if family == "auto":
        return infer_model_family(str(cfg.model.name))
    if family not in {"ultralytics", "rfdetr"}:
        raise ValueError(
            f"Unsupported model family '{cfg.model.family}'. "
            "Supported values: auto, ultralytics, rfdetr."
        )
    return family


def normalize_output_path(exported_path: str | Path, output_dir: Path) -> Path:
    """Move the exported artifact into the requested output directory if needed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(exported_path).resolve()
    target_path = output_dir / source_path.name
    if source_path == target_path:
        return source_path
    shutil.move(str(source_path), str(target_path))
    return target_path


def is_yolov10_model(model_name: str) -> bool:
    """Return True when the configured model is a YOLOv10 variant."""
    return model_name.strip().lower().startswith(YOLOV10_PREFIX)


def resolve_ultralytics_nms(model_name: str, nms_value: Any) -> bool:
    """Resolve ONNX NMS for YOLO-family exports."""
    if is_yolov10_model(model_name):
        return False

    if nms_value is None:
        return False

    if isinstance(nms_value, bool):
        return nms_value

    normalized_value = str(nms_value).strip().lower()
    if normalized_value in {"true", "false"}:
        return normalized_value == "true"

    raise ValueError(
        f"Unsupported export.nms value '{nms_value}'. "
        "Supported values: true, false, null."
    )


def build_rfdetr_dynamic_axes(dynamic: bool) -> dict[str, dict[int, str]] | None:
    """Return dynamic axes config for RF-DETR ONNX export."""
    if not dynamic:
        return None
    return {
        "input": {0: "batch", 2: "height", 3: "width"},
        "dets": {0: "batch"},
        "labels": {0: "batch"},
    }


def export_rfdetr_model_to_onnx(  # noqa: PLR0913
    *,
    wrapper: Any,
    output_dir: Path,
    shape: tuple[int, int],
    batch_size: int,
    dynamic: bool,
    opset: int,
    verbose: bool,
    variant_name: str,
) -> Path:
    """Export RF-DETR with dynamic spatial axes and patched bicubic interpolation."""
    from rfdetr.detr import _resolve_patch_size, _validate_shape_dims
    from rfdetr.export._onnx.exporter import export_onnx as export_rfdetr_onnx_file
    from rfdetr.export.main import make_infer_image

    wrapper.model.model = wrapper.model.model.to("cpu")
    export_model = deepcopy(wrapper.model.model)
    export_model.to("cpu")

    patch_size = _resolve_patch_size(None, wrapper.model_config, "export")
    num_windows = getattr(wrapper.model_config, "num_windows", 1)
    block_size = patch_size * num_windows
    validated_shape = _validate_shape_dims(shape, block_size, patch_size, num_windows)
    input_tensors = make_infer_image(
        None,
        validated_shape,
        batch_size,
        "cpu",
        num_channels=wrapper.model_config.num_channels,
    ).to("cpu")
    dynamic_axes = build_rfdetr_dynamic_axes(dynamic)

    export_model.eval()
    with torch.no_grad():
        export_model(input_tensors)

    with disable_bicubic_antialias_during_export():
        exported_path = export_rfdetr_onnx_file(
            output_dir=str(output_dir),
            model=export_model,
            input_names=["input"],
            input_tensors=input_tensors,
            output_names=["dets", "labels"],
            dynamic_axes=dynamic_axes,
            verbose=verbose,
            opset_version=opset,
            variant_name=variant_name,
        )
    return Path(exported_path).resolve()


def export_ultralytics_onnx(cfg: DictConfig, output_dir: Path) -> ExportResult:
    """Export a YOLO or RT-DETR model to ONNX using Ultralytics."""
    model_name = str(cfg.model.name)
    checkpoint_path = resolve_export_checkpoint_path(cfg.export)
    model_path = checkpoint_path or f"{model_name}.pt"
    model: Any
    imgsz = resolve_imgsz(cfg.export.imgsz)
    export_nms = resolve_ultralytics_nms(model_name, cfg.export.nms)

    if model_name.lower().startswith("rtdetr"):
        try:
            from ultralytics import RTDETR
        except ImportError as err:
            raise ImportError("pip install -U ultralytics") from err
        model = RTDETR(model_path)
    else:
        try:
            from ultralytics import YOLO
        except ImportError as err:
            raise ImportError("pip install -U ultralytics") from err
        model = YOLO(model_path)

    export_kwargs: dict[str, Any] = {
        "format": "onnx",
        "imgsz": imgsz,
        "batch": int(cfg.export.batch),
        "dynamic": bool(cfg.export.dynamic),
        "simplify": bool(cfg.export.simplify),
        "half": bool(cfg.export.half),
        "nms": export_nms,
        "device": cfg.export.device,
        "opset": int(cfg.export.opset),
    }
    if cfg.export.end2end is not None:
        export_kwargs["end2end"] = bool(cfg.export.end2end)

    exported_path = model.export(**export_kwargs)
    final_path = normalize_output_path(exported_path, output_dir)
    return ExportResult(
        family="ultralytics",
        model_name=model_name,
        source_checkpoint=checkpoint_path,
        output_path=str(final_path),
    )


def export_rfdetr_onnx(cfg: DictConfig, output_dir: Path) -> ExportResult:
    """Export an RF-DETR model to ONNX using the rfdetr package."""
    model_name = str(cfg.model.name)
    normalized_model_name = normalize_rfdetr_model_name(model_name)
    checkpoint_path = resolve_export_checkpoint_path(cfg.export)
    model_class = load_rfdetr_model_class(model_name)
    model_kwargs = build_model_kwargs(cfg, device_key="export")
    imgsz = resolve_imgsz(cfg.export.imgsz)
    if checkpoint_path:
        model_kwargs["pretrain_weights"] = checkpoint_path
    elif not cfg.model.pretrain_weights:
        model_kwargs.pop("num_classes", None)

    shape = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    model = model_class(**model_kwargs)
    exported_path = export_rfdetr_model_to_onnx(
        wrapper=model,
        output_dir=output_dir,
        shape=shape,
        batch_size=int(cfg.export.batch),
        dynamic=bool(cfg.export.dynamic),
        opset=int(cfg.export.opset),
        verbose=bool(cfg.export.verbose),
        variant_name=normalized_model_name,
    )
    return ExportResult(
        family="rfdetr",
        model_name=normalized_model_name,
        source_checkpoint=checkpoint_path,
        output_path=str(exported_path),
    )


def export_onnx(cfg: DictConfig) -> ExportResult:
    """Export the configured model to ONNX and return its resulting path."""
    console = RichConsoleManager.get_console()
    family = resolve_model_family(cfg)
    output_dir = Path(to_absolute_path(str(cfg.export.output_dir))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold cyan]Exporting model to ONNX[/bold cyan]")
    console.print(f"  Family: {family}")
    console.print(f"  Model: {cfg.model.name}")
    console.print(f"  Output: {output_dir}")
    if family == "ultralytics":
        resolved_nms = resolve_ultralytics_nms(str(cfg.model.name), cfg.export.nms)
        console.print(f"  NMS in graph: {resolved_nms}")
        if is_yolov10_model(str(cfg.model.name)):
            console.print(
                "  Note: YOLOv10 export is end-to-end; Ultralytics forces nms=false"
            )
    checkpoint_path = resolve_export_checkpoint_path(cfg.export)
    if checkpoint_path:
        console.print(f"  Checkpoint: {checkpoint_path}")
    else:
        console.print("  Checkpoint: pretrained default")

    if family == "ultralytics":
        result = export_ultralytics_onnx(cfg, output_dir)
    else:
        result = export_rfdetr_onnx(cfg, output_dir)

    summary_path = output_dir / "export_summary.json"
    summary_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    console.print(
        f"[bold green]ONNX export complete.[/bold green] {result.output_path}"
    )
    console.print(f"[green]Summary saved:[/green] {summary_path}")
    return result


@hydra.main(version_base=None, config_path=str(CONFIGS_DIR), config_name="export_onnx")
def main(cfg: DictConfig) -> None:
    """Run the ONNX export CLI entrypoint."""
    console = RichConsoleManager.get_console()
    console.print(OmegaConf.to_yaml(cfg), style="warning")
    export_onnx(cfg)


if __name__ == "__main__":
    main()
