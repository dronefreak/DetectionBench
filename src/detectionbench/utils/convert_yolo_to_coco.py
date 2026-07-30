#!/usr/bin/env python3
r"""
Convert a YOLO-format dataset into a COCO-format dataset for RF-DETR.

Input directory layout expected:
  dataset_root/
  ├── dataset.yaml
  ├── images/
  │   ├── train/
  │   ├── val/
  │   └── test/
  └── labels/
      ├── train/
      ├── val/
      └── test/

Output directory layout produced:
  output_root/
  ├── train/
  │   ├── _annotations.coco.json
  │   ├── image1.jpg
  │   └── ...
  ├── valid/
  │   ├── _annotations.coco.json
  │   ├── image1.jpg
  │   └── ...
  └── test/
      ├── _annotations.coco.json
      ├── image1.jpg
      └── ...

This is a generic bridge: it works for any YOLO-format dataset, regardless
of which detection dataset it came from.

Usage:
  python -m detectionbench.utils.convert_yolo_to_coco \
      /path/to/yolo_dataset \
      /path/to/coco_dataset \
      --dataset-yaml /path/to/yolo_dataset/dataset.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
import yaml

from detectionbench.utils.utils import RichConsoleManager

COCO_ANNOTATION_FILENAME = "_annotations.coco.json"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
YOLO_LABEL_FIELD_COUNT = 5
SPLIT_NAME_MAP = {
    "train": "train",
    "val": "valid",
    "valid": "valid",
    "test": "test",
}


@dataclass(frozen=True)
class SplitConversionStats:
    """Track conversion statistics for one split."""

    images: int
    annotations: int
    missing_labels: int
    skipped_invalid_labels: int


@dataclass(frozen=True)
class YoloDatasetConfig:
    """Store the parsed YOLO dataset configuration."""

    dataset_root: Path
    names: list[str]
    split_entries: dict[str, Any]


@dataclass(frozen=True)
class LabelConversionContext:
    """Store image-specific metadata needed for label conversion."""

    image_id: int
    annotation_id: int
    image_width: int
    image_height: int
    category_count: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for YOLO to COCO conversion."""
    parser = argparse.ArgumentParser(
        description="Convert a YOLO dataset into COCO train/valid/test structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_dir", help="YOLO dataset root")
    parser.add_argument("output_dir", help="Output COCO dataset root")
    parser.add_argument(
        "--dataset-yaml",
        default=None,
        help="Path to the YOLO dataset YAML (default: auto-detect inside input_dir)",
    )
    parser.add_argument(
        "--save-report",
        default=None,
        help="Optional path to save a JSON conversion report",
    )
    return parser.parse_args()


def find_dataset_yaml(input_dir: Path, dataset_yaml: str | None) -> Path:
    """Find the dataset YAML file from CLI input or common defaults."""
    if dataset_yaml is not None:
        yaml_path = Path(dataset_yaml)
        if not yaml_path.is_absolute():
            yaml_path = input_dir / yaml_path
        return yaml_path.resolve()

    candidates = sorted(input_dir.glob("*.yaml")) + sorted(input_dir.glob("*.yml"))
    preferred = [
        path
        for path in candidates
        if path.stem.startswith("data") or path.stem.startswith("dataset")
    ]
    selected = preferred[0] if preferred else (candidates[0] if candidates else None)
    if selected is None:
        raise FileNotFoundError(
            f"Could not find a dataset YAML under {input_dir}. "
            "Pass --dataset-yaml explicitly."
        )
    return selected.resolve()


def load_dataset_config(input_dir: Path, dataset_yaml: str | None) -> YoloDatasetConfig:
    """Load names and split paths from a YOLO dataset YAML file."""
    yaml_path = find_dataset_yaml(input_dir, dataset_yaml)
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if "names" not in data:
        raise KeyError(f"Dataset YAML does not contain 'names': {yaml_path}")

    names_field = data["names"]
    if isinstance(names_field, dict):
        names = [str(names_field[index]) for index in sorted(names_field)]
    else:
        names = [str(name) for name in names_field]

    dataset_root = Path(data.get("path", yaml_path.parent))
    if not dataset_root.is_absolute():
        dataset_root = (yaml_path.parent / dataset_root).resolve()

    split_entries = {
        split_name: data.get(split_name)
        for split_name in ("train", "val", "valid", "test")
        if data.get(split_name) is not None
    }
    if not split_entries:
        raise KeyError(f"Dataset YAML does not define any splits: {yaml_path}")

    return YoloDatasetConfig(
        dataset_root=dataset_root,
        names=names,
        split_entries=split_entries,
    )


def resolve_split_images(dataset_root: Path, split_entry: Any) -> list[Path]:
    """Resolve a YOLO split entry into concrete image paths."""
    if isinstance(split_entry, list):
        image_paths: list[Path] = []
        for item in split_entry:
            image_paths.extend(resolve_split_images(dataset_root, item))
        return sorted(image_paths)

    split_path = Path(str(split_entry))
    if not split_path.is_absolute():
        split_path = dataset_root / split_path
    split_path = split_path.resolve()

    if split_path.is_file():
        listed_paths = [
            Path(line.strip())
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        resolved_paths = []
        for listed_path in listed_paths:
            if not listed_path.is_absolute():
                listed_path = (split_path.parent / listed_path).resolve()
            resolved_paths.append(listed_path.resolve())
        return sorted(resolved_paths)

    if not split_path.is_dir():
        raise FileNotFoundError(f"Split path not found: {split_path}")

    return sorted(
        path
        for path in split_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def derive_label_path(image_path: Path) -> Path:
    """Derive the YOLO label path corresponding to an image path."""
    parts = list(image_path.parts)
    if "images" not in parts:
        raise ValueError(f"Image path does not contain 'images': {image_path}")
    image_index = parts.index("images")
    parts[image_index] = "labels"
    return Path(*parts).with_suffix(".txt")


def normalize_output_name(image_path: Path, seen_names: set[str]) -> str:
    """Create a stable flat filename for the output split directory."""
    candidate = image_path.name
    if candidate not in seen_names:
        seen_names.add(candidate)
        return candidate

    fallback = "__".join(image_path.parts[-3:])
    if fallback not in seen_names:
        seen_names.add(fallback)
        return fallback

    stem = image_path.stem
    suffix = image_path.suffix
    counter = 1
    while True:
        candidate = f"{stem}__{counter}{suffix}"
        if candidate not in seen_names:
            seen_names.add(candidate)
            return candidate
        counter += 1


def load_image_size(image_path: Path) -> tuple[int, int]:
    """Read image width and height without keeping the file open."""
    with Image.open(image_path) as image:
        return image.size


def convert_yolo_label_line(
    line: str,
    context: LabelConversionContext,
) -> dict[str, Any] | None:
    """Convert one YOLO label line into a COCO annotation."""
    parts = line.split()
    if len(parts) != YOLO_LABEL_FIELD_COUNT:
        return None

    try:
        category_id = int(parts[0])
        cx, cy, width, height = map(float, parts[1:])
    except ValueError:
        return None

    if not (0 <= category_id < context.category_count):
        return None

    bbox_width = width * context.image_width
    bbox_height = height * context.image_height
    x_min = (cx - width / 2.0) * context.image_width
    y_min = (cy - height / 2.0) * context.image_height

    bbox_width = max(0.0, min(float(context.image_width), bbox_width))
    bbox_height = max(0.0, min(float(context.image_height), bbox_height))
    x_min = max(0.0, min(float(context.image_width), x_min))
    y_min = max(0.0, min(float(context.image_height), y_min))

    if bbox_width <= 0.0 or bbox_height <= 0.0:
        return None

    return {
        "id": context.annotation_id,
        "image_id": context.image_id,
        "category_id": category_id,
        "bbox": [x_min, y_min, bbox_width, bbox_height],
        "area": bbox_width * bbox_height,
        "segmentation": [],
        "iscrowd": 0,
    }


def build_categories(names: list[str]) -> list[dict[str, Any]]:
    """Build COCO category entries from dataset class names."""
    return [
        {"id": index, "name": name, "supercategory": "none"}
        for index, name in enumerate(names)
    ]


def convert_split(
    *,
    split_name: str,
    output_name: str,
    image_paths: list[Path],
    class_names: list[str],
    output_root: Path,
) -> tuple[SplitConversionStats, dict[str, Any]]:
    """Convert one YOLO split into a COCO split directory and JSON file."""
    console = RichConsoleManager.get_console()
    split_output_dir = output_root / output_name
    split_output_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    annotation_id = 1
    missing_labels = 0
    skipped_invalid_labels = 0

    for image_id, image_path in enumerate(image_paths, start=1):
        output_file_name = normalize_output_name(image_path, seen_names)
        destination_path = split_output_dir / output_file_name
        shutil.copy2(image_path, destination_path)

        image_width, image_height = load_image_size(image_path)
        images.append(
            {
                "id": image_id,
                "license": 1,
                "file_name": output_file_name,
                "height": image_height,
                "width": image_width,
                "date_captured": "",
            }
        )

        label_path = derive_label_path(image_path)
        if not label_path.exists():
            missing_labels += 1
            continue

        for line in label_path.read_text(encoding="utf-8").splitlines():
            label_context = LabelConversionContext(
                image_id=image_id,
                annotation_id=annotation_id,
                image_width=image_width,
                image_height=image_height,
                category_count=len(class_names),
            )
            annotation = convert_yolo_label_line(
                line,
                label_context,
            )
            if annotation is None:
                skipped_invalid_labels += 1
                continue
            annotations.append(annotation)
            annotation_id += 1

    coco_payload = {
        "info": {
            "description": f"Converted from YOLO split '{split_name}'",
            "version": "1.0",
            "year": 2026,
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "images": images,
        "annotations": annotations,
        "categories": build_categories(class_names),
    }

    annotation_path = split_output_dir / COCO_ANNOTATION_FILENAME
    annotation_path.write_text(json.dumps(coco_payload, indent=2), encoding="utf-8")
    console.print(
        f"[green]{split_name}[/green]: {len(images)} images, "
        f"{len(annotations)} annotations -> {annotation_path}"
    )

    return (
        SplitConversionStats(
            images=len(images),
            annotations=len(annotations),
            missing_labels=missing_labels,
            skipped_invalid_labels=skipped_invalid_labels,
        ),
        coco_payload,
    )


def convert(args: argparse.Namespace) -> None:
    """Convert the configured YOLO dataset into COCO split directories."""
    console = RichConsoleManager.get_console()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = load_dataset_config(input_dir, args.dataset_yaml)
    console.print("[bold cyan]YOLO -> COCO conversion[/bold cyan]")
    console.print(f"  Input: {input_dir}")
    console.print(f"  Dataset root: {dataset_config.dataset_root}")
    console.print(f"  Output: {output_dir}")

    report: dict[str, Any] = {
        "input_dir": str(input_dir),
        "dataset_root": str(dataset_config.dataset_root),
        "output_dir": str(output_dir),
        "class_names": dataset_config.names,
        "splits": {},
    }

    processed_target_splits: set[str] = set()
    for source_split, target_split in SPLIT_NAME_MAP.items():
        split_entry = dataset_config.split_entries.get(source_split)
        if split_entry is None:
            continue
        if target_split in processed_target_splits:
            continue
        image_paths = resolve_split_images(dataset_config.dataset_root, split_entry)
        stats, _ = convert_split(
            split_name=source_split,
            output_name=target_split,
            image_paths=image_paths,
            class_names=dataset_config.names,
            output_root=output_dir,
        )
        processed_target_splits.add(target_split)
        report["splits"][target_split] = {
            "source_split": source_split,
            "images": stats.images,
            "annotations": stats.annotations,
            "missing_labels": stats.missing_labels,
            "skipped_invalid_labels": stats.skipped_invalid_labels,
        }

    if args.save_report:
        report_path = Path(args.save_report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        console.print(f"[green]Report saved:[/green] {report_path}")


def main() -> None:
    """Run the YOLO to COCO conversion CLI entrypoint."""
    convert(parse_args())


if __name__ == "__main__":
    main()
