"""
Animals Detection Images Dataset adapter.

Source: a Kaggle re-export ("Animals Detection Images Dataset",
antoreepjana) of a Google Open Images V6+ subset, downloaded per class
query via the OIDv4_ToolKit (https://github.com/EscVM/OIDv4_ToolKit)
export layout: ``{train,test}/<ClassName>/*.jpg`` plus
``{train,test}/<ClassName>/Label/*.txt``, one line per box as
``<ClassName> <XMin> <YMin> <XMax> <YMax>`` in absolute pixel coordinates
(class names may themselves contain spaces, e.g. "Brown bear" or "Moths
and butterflies" -- this adapter splits off the last 4 whitespace-separated
tokens as the box and treats the remainder as the class name). 80 classes,
confirmed identical between the two splits.

**Cross-class image duplication.** Because OIDv4 downloads per class
query, the same physical image can be fetched into more than one class
folder when it contains multiple labeled animals -- confirmed on this
release (e.g. a horse/mule image is byte-identical across
``train/Horse/`` and ``train/Mule/``, each folder's label file holding
only that folder's class). Verified across the whole release: 21,649
unique filenames among 22,566 train file-copies, 6,003 unique among 6,505
test file-copies, and zero overlap between the train/test filename sets.
This adapter merges every box annotation for the same filename (within a
split) into a single COCO image entry, rather than emitting one image
entry per class-folder copy.

No official validation split -- following the pattern used for CeyMo/
NEU-DET/LLVIP/SSDD, a seeded 15% slice of the official ``train`` is
carved out as ``valid``; the official ``test`` is kept as-is.

License: **not** CC0 despite the Kaggle page's own license tag. This is
genuinely Open Images data: the images carry their original per-photo CC
BY 2.0 license (Open Images sources from Flickr), and the box annotations
are Google's own, released under CC BY 4.0. Both terms permit
redistribution with attribution -- see Google's own Open Images license
page (https://storage.googleapis.com/openimages/web/factsfigures_v7.html)
for the authoritative statement of these terms.

Stats: see docs/datasets/animalsdet/README.md (class distribution, split
summary, box geometry -- generated via detectionbench-dataset-stats).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
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
    "Bear",
    "Brown bear",
    "Bull",
    "Butterfly",
    "Camel",
    "Canary",
    "Caterpillar",
    "Cattle",
    "Centipede",
    "Cheetah",
    "Chicken",
    "Crab",
    "Crocodile",
    "Deer",
    "Duck",
    "Eagle",
    "Elephant",
    "Fish",
    "Fox",
    "Frog",
    "Giraffe",
    "Goat",
    "Goldfish",
    "Goose",
    "Hamster",
    "Harbor seal",
    "Hedgehog",
    "Hippopotamus",
    "Horse",
    "Jaguar",
    "Jellyfish",
    "Kangaroo",
    "Koala",
    "Ladybug",
    "Leopard",
    "Lion",
    "Lizard",
    "Lynx",
    "Magpie",
    "Monkey",
    "Moths and butterflies",
    "Mouse",
    "Mule",
    "Ostrich",
    "Otter",
    "Owl",
    "Panda",
    "Parrot",
    "Penguin",
    "Pig",
    "Polar bear",
    "Rabbit",
    "Raccoon",
    "Raven",
    "Red panda",
    "Rhinoceros",
    "Scorpion",
    "Seahorse",
    "Sea lion",
    "Sea turtle",
    "Shark",
    "Sheep",
    "Shrimp",
    "Snail",
    "Snake",
    "Sparrow",
    "Spider",
    "Squid",
    "Squirrel",
    "Starfish",
    "Swan",
    "Tick",
    "Tiger",
    "Tortoise",
    "Turkey",
    "Turtle",
    "Whale",
    "Woodpecker",
    "Worm",
    "Zebra",
]
_CLASS_INDEX = {name: index for index, name in enumerate(_CLASSES)}
_BOX_TOKEN_COUNT = 4

_VAL_FRACTION = 0.15
_SPLIT_SEED = 42


def _parse_label_line(line: str) -> tuple[str, float, float, float, float] | None:
    """Parse one OIDv4-format label line, handling multi-word class names."""
    parts = line.split()
    if len(parts) <= _BOX_TOKEN_COUNT:
        return None
    name = " ".join(parts[:-_BOX_TOKEN_COUNT])
    if name not in _CLASS_INDEX:
        return None
    x1, y1, x2, y2 = (float(v) for v in parts[-_BOX_TOKEN_COUNT:])
    return name, x1, y1, x2, y2


def _collect_split_images(
    split_dir: Path,
) -> dict[str, list[tuple[str, float, float, float, float]]]:
    """
    Merge every class-folder copy of each filename into one box list.

    Returns ``{filename: [(class_name, x1, y1, x2, y2), ...]}``, combining
    boxes from every class folder that happens to contain that filename
    (see the module docstring's cross-class duplication note).
    """
    boxes_by_file: dict[str, list[tuple[str, float, float, float, float]]] = (
        defaultdict(list)
    )
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        label_dir = class_dir / "Label"
        if not label_dir.is_dir():
            continue
        for image_path in sorted(class_dir.glob("*.jpg")):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parsed = _parse_label_line(line)
                if parsed is None:
                    continue
                boxes_by_file[image_path.name].append(parsed)
    return boxes_by_file


@register
class AnimalsDetAdapter(DatasetAdapter):
    """Adapter for the Animals Detection Images Dataset (Open Images subset)."""

    spec = DatasetSpec(
        key="animalsdet",
        display_name="Animals Detection",
        classes=_CLASSES,
        description=(
            "Animals Detection Images Dataset is an 80-class wildlife/animal "
            "detection benchmark drawn from Google Open Images, spanning "
            "mammals, birds, reptiles, fish, and invertebrates (e.g. lion, "
            "zebra, eagle, jellyfish, butterfly). It's used to benchmark "
            "broad-taxonomy animal detection -- the widest class vocabulary "
            "of any dataset in DetectionBench."
        ),
        homepage="https://www.kaggle.com/datasets/antoreepjana/animals-detection-images-dataset",
        citation=(
            "@misc{openimages,\n"
            "  title={OpenImages: A public dataset for large-scale multi-label "
            "and multi-class image classification.},\n"
            "  author={Krasin, Ivan and Duerig, Tom and Alldrin, Neil and "
            "Ferrari, Vittorio and Abu-El-Haija, Sami and Kuznetsova, Alina and "
            "Rom, Hassan and Uijlings, Jasper and Popov, Stefan and Veit, "
            "Andreas and others},\n"
            "  year={2017}\n"
            "}"
        ),
        license=(
            "Open Images' own terms, not the Kaggle re-upload's claimed CC0: "
            "images carry their original per-photo CC BY 2.0 license "
            "(sourced from Flickr), and the box annotations are Google's own "
            "CC BY 4.0. Both permit redistribution with attribution."
        ),
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert the OIDv4-layout Animals Detection release into canonical COCO."""
        train_dir, test_dir = raw_dir / "train", raw_dir / "test"
        if not train_dir.is_dir() or not test_dir.is_dir():
            raise FileNotFoundError(f"Expected {raw_dir}/(train|test).")
        output_dir.mkdir(parents=True, exist_ok=True)

        train_boxes = _collect_split_images(train_dir)
        test_boxes = _collect_split_images(test_dir)

        train_files = sorted(train_boxes)
        shuffled = train_files[:]
        random.Random(_SPLIT_SEED).shuffle(shuffled)  # noqa: S311  # nosec: B311
        n_val = max(1, round(len(shuffled) * _VAL_FRACTION))
        val_files = set(shuffled[:n_val])

        _write_split(
            [f for f in train_files if f not in val_files],
            train_boxes,
            train_dir,
            output_dir / "train",
        )
        _write_split(
            [f for f in train_files if f in val_files],
            train_boxes,
            train_dir,
            output_dir / "valid",
        )
        _write_split(sorted(test_boxes), test_boxes, test_dir, output_dir / "test")


def _find_image(split_dir: Path, filename: str) -> Path | None:
    """Find a filename under any class folder of ``split_dir`` (first match)."""
    for class_dir in split_dir.iterdir():
        if not class_dir.is_dir():
            continue
        candidate = class_dir / filename
        if candidate.exists():
            return candidate
    return None


def _write_split(
    filenames: list[str],
    boxes_by_file: dict[str, list[tuple[str, float, float, float, float]]],
    split_dir: Path,
    split_output_dir: Path,
) -> None:
    """Emit one canonical COCO split, merging per-class-folder duplicates."""
    split_output_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for image_id, filename in enumerate(filenames, start=1):
        src = _find_image(split_dir, filename)
        if src is None:
            continue
        with Image.open(src) as im:
            width, height = im.size

        link_image(src, split_output_dir / filename)
        images.append(
            {"id": image_id, "file_name": filename, "width": width, "height": height}
        )

        for class_name, x1, y1, x2, y2 in boxes_by_file[filename]:
            box_width, box_height = x2 - x1, y2 - y1
            if box_width <= 0 or box_height <= 0:
                continue
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": _CLASS_INDEX[class_name],
                    "bbox": [x1, y1, box_width, box_height],
                    "area": box_width * box_height,
                    "segmentation": [],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    payload = {
        "info": {
            "description": f"Animals Detection canonical COCO ({split_output_dir.name})"
        },
        "licenses": [
            {
                "id": 1,
                "name": "CC BY 2.0 (images) / CC BY 4.0 (annotations)",
                "url": "https://storage.googleapis.com/openimages/web/factsfigures_v7.html",
            }
        ],
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
    print(f"[{split_output_dir.name}] {len(images)} images, {len(annotations)} boxes")
