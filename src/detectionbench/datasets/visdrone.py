"""
VisDrone2019-DET (aerial object detection) adapter.

The official VisDrone2019-DET release ships as one directory per split
(``VisDrone2019-DET-{train,val,test-dev,test-challenge}``), each containing
an ``images/`` directory of JPEGs and an ``annotations/`` directory of
per-image CSV-style ``.txt`` files with one row per box::

    bbox_left,bbox_top,bbox_width,bbox_height,score,category,truncation,occlusion

``score`` of ``0`` marks a region to ignore (not a real annotation);
``category`` of ``0`` is the special "ignored-regions" pseudo-class. Both are
dropped here, matching the standard VisDrone YOLO-conversion convention (11
usable classes, category ids 1-11 remapped to 0-10). ``test-challenge`` has
no public ground truth (a held-out competition split, same situation as
SeaDronesSee's test set) and is not converted.

This dataset has already been benchmarked end-to-end across the full YOLO
model spectrum (YOLOv8/9/10/11/26 + RT-DETR) via a separate codebase
(https://github.com/dronefreak/VisDrone-dataset-python-toolkit), with results
published at
https://huggingface.co/collections/dronefreak/visdrone-detection-model-zoo.
This adapter exists so DetectionBench itself can also train/evaluate against
VisDrone-DET (e.g. to add the RF-DETR variant that zoo doesn't have yet) --
not to reproduce or replace that existing YOLO benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from detectionbench.datasets.base import (
    COCO_ANNOTATION_FILENAME,
    DatasetAdapter,
    DatasetSpec,
    link_image,
)
from detectionbench.datasets.registry import register

_CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
    "others",
]
_IGNORED_REGION_CATEGORY_ID = 0
_ANNOTATION_FIELD_COUNT = 6  # bbox(4) + score + category (truncation/occlusion ignored)

# VisDrone's own split directory names -> DetectionBench's canonical split names.
# test-challenge is deliberately excluded (no public ground truth).
_RAW_SPLIT_DIRS = {
    "train": "VisDrone2019-DET-train",
    "valid": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
}


@register
class VisDroneDetAdapter(DatasetAdapter):
    """Adapter for the VisDrone2019-DET aerial object-detection dataset."""

    spec = DatasetSpec(
        key="visdrone",
        display_name="VisDrone-DET",
        classes=_CLASSES,
        homepage="https://github.com/VisDrone/VisDrone-Dataset",
        citation=(
            "Zhu, Pengfei and Wen, Longyin and Bian, Xiao and Ling, Haibin and "
            "Hu, Qinghua, 'Vision Meets Drones: A Challenge', arXiv:1804.07437, "
            "2018."
        ),
        license="CC BY-NC-SA 3.0 -- non-commercial research use only, see homepage.",
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert a raw VisDrone2019-DET download into the canonical COCO layout."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for split_name, raw_subdir in _RAW_SPLIT_DIRS.items():
            split_dir = raw_dir / raw_subdir
            if not split_dir.exists():
                continue
            _convert_split(
                split_dir / "images",
                split_dir / "annotations",
                output_dir / split_name,
            )


def _convert_split(
    images_dir: Path, annotations_dir: Path, split_output_dir: Path
) -> None:
    """Convert one VisDrone split directory into the canonical COCO layout."""
    split_output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for image_id, image_path in enumerate(image_paths, start=1):
        dst = split_output_dir / image_path.name
        link_image(image_path, dst)

        width, height = _image_size(image_path)
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        for left, top, box_width, box_height, category_id in _parse_annotations(
            annotations_dir / f"{image_path.stem}.txt"
        ):
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id - 1,  # raw ids 1-11 -> 0-10
                    "bbox": [left, top, box_width, box_height],
                    "area": box_width * box_height,
                    "segmentation": [],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    payload = {
        "info": {
            "description": f"Converted from VisDrone2019-DET split at {images_dir}"
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index, "name": name, "supercategory": "none"}
            for index, name in enumerate(_CLASSES)
        ],
    }
    (split_output_dir / COCO_ANNOTATION_FILENAME).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _parse_annotations(annotation_path: Path) -> list[tuple[int, int, int, int, int]]:
    """Parse one VisDrone annotation file, dropping ignored regions/scores."""
    if not annotation_path.exists():
        return []

    boxes: list[tuple[int, int, int, int, int]] = []
    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(",")
        if len(parts) < _ANNOTATION_FIELD_COUNT:
            continue
        left, top, box_width, box_height = map(int, parts[:4])
        score, category_id = int(parts[4]), int(parts[5])
        if score == 0 or category_id == _IGNORED_REGION_CATEGORY_ID:
            continue
        if box_width <= 0 or box_height <= 0:
            continue
        boxes.append((left, top, box_width, box_height, category_id))
    return boxes


def _image_size(image_path: Path) -> tuple[int, int]:
    """Read image width and height without keeping the file open."""
    with Image.open(image_path) as image:
        return image.size
