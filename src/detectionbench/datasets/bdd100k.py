"""
BDD100K adapter -- multi-class autonomous-driving object detection.

Source: the official Berkeley DeepDrive "100K Images" + "Labels" release,
downloaded directly from Berkeley's own distribution server (not a
third-party mirror) -- ``images_100k/100k/{train,val,test}/**/*.jpg``
(1280x720 JPEGs; 70,000/10,000/20,000, flat per split, any ``shard_NNN/``
subdirectories -- an artifact of this project's Hugging Face mirror
sharding, see below -- are walked recursively) plus
``labels/100k/{train,val,test}/<name>.json``, one JSON file per image, in
BDD100K's native per-image format: ``{"name": <id, no extension>,
"attributes": {"weather", "scene", "timeofday"}, "frames": [{"objects":
[...]}]}``. Each ``frames[0].objects`` entry mixes ``box2d`` detection boxes
(``category`` one of this adapter's 10 classes) with ``poly2d`` lane-marking
and drivable-area polygons (different tasks, e.g. ``category`` values like
``"lane/road curb"`` or ``"area/drivable"``); only ``box2d`` entries whose
category is one of the 10 detection classes are converted here.

Unlike the old Kaggle-mirror-derived version of this adapter, **all three**
official splits ship with real labels in this release -- including
``test`` (20,000 images), verified directly against sample files: it is
*not* empty or held out, contrary to the commonly repeated claim that
BDD100K's test labels are withheld for a leaderboard. This simplifies the
split handling considerably: there is no more carving a seeded validation
slice out of train. The three official splits map directly onto this
project's canonical COCO layout: official ``train`` (70,000) ->
``train``, official ``val`` (10,000) -> ``valid``, official ``test``
(20,000) -> ``test``.

License: the data and labels (as opposed to the BDD100K code repo, which is
BSD-3-Clause) are under UC Regents' own academic license, verified directly
from https://github.com/bdd100k/bdd100k/blob/master/doc/source/license.rst
-- it explicitly grants "permission to use, copy, modify, and distribute
this software and its documentation for educational, research, and
not-for-profit purposes, without fee and without a signed licensing
agreement," provided the copyright notice and license paragraphs are
carried forward into any redistribution (commercial use/distribution is
separately restricted to BDD/BAIR Commons members). The data itself is
registration-gated (manual DUA click-through on the official site), not
automatable via ``download_dataset.py``. **Mirrored on Hugging Face** as
``dronefreak/BDD100K`` under this license, with the full notice text
reproduced verbatim on the card per the license's own carry-forward
condition, in this exact raw layout (just reorganized into
``shard_NNN/`` subdirectories per split to stay under Hugging Face's
practical per-directory file-count ceiling -- this adapter's recursive
globbing is unaffected by that).

Stats: see docs/datasets/bdd100k/README.md (class distribution, split
summary, box geometry -- generated via detectionbench-dataset-stats; this
will need regenerating against the new raw layout/splits).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from detectionbench.datasets.base import (
    COCO_ANNOTATION_FILENAME,
    DatasetAdapter,
    DatasetSpec,
    link_image,
)
from detectionbench.datasets.registry import register

_CLASSES = [
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motor",
    "bike",
    "traffic light",
    "traffic sign",
]
_CLASS_TO_ID = {name: i for i, name in enumerate(_CLASSES)}

_IMAGE_WIDTH = 1280
_IMAGE_HEIGHT = 720

# Official split name -> this project's canonical COCO split name.
_SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}


@register
class BDD100KAdapter(DatasetAdapter):
    """Adapter for the BDD100K multi-class driving-scene detection dataset."""

    spec = DatasetSpec(
        key="bdd100k",
        display_name="BDD100K",
        classes=_CLASSES,
        description=(
            "BDD100K is a large-scale, diverse driving-video dataset; this "
            "adapter covers its 100K-image object-detection task: all "
            "100,000 1280x720 dashcam images (70,000 train + 10,000 val + "
            "20,000 test, all three officially released with real labels) "
            "annotated for 10 classes spanning vulnerable road users "
            "(person, rider), vehicles (car, truck, bus, train, motor, "
            "bike), and traffic control (traffic light, traffic sign), "
            "captured across diverse weather, time-of-day, and scene "
            "conditions in the US. It's used to benchmark multi-class "
            "driving-scene detection under real-world distribution shift."
        ),
        homepage="https://www.bdd100k.com/",
        citation=(
            "@inproceedings{yu2020bdd100k,\n"
            "  title={BDD100K: A Diverse Driving Dataset for Heterogeneous "
            "Multitask Learning},\n"
            "  author={Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, "
            "Wenqi and Chen, Yingying and Liu, Fangchen and Madhavan, "
            "Vashisht and Darrell, Trevor},\n"
            "  booktitle={Proceedings of the IEEE/CVF Conference on Computer "
            "Vision and Pattern Recognition},\n"
            "  pages={2633--2642},\n"
            "  year={2020}\n"
            "}"
        ),
        license=(
            "BDD100K data license (UC Regents; educational/research/"
            "not-for-profit redistribution explicitly permitted with "
            "notice; commercial use restricted to BDD/BAIR Commons)."
        ),
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert the official BDD100K 100k-images release into canonical COCO."""
        image_root = raw_dir / "images_100k" / "100k"
        label_root = raw_dir / "labels" / "100k"
        if not image_root.is_dir() or not label_root.is_dir():
            raise FileNotFoundError(
                f"Expected {raw_dir}/images_100k/100k and {raw_dir}/labels/100k."
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        for official_split, canonical_split in _SPLIT_MAP.items():
            images_by_name = _index_images(image_root / official_split)
            entries = [
                json.loads(p.read_text(encoding="utf-8"))
                for p in sorted((label_root / official_split).rglob("*.json"))
            ]
            _write_split(entries, images_by_name, output_dir / canonical_split)


def _index_images(root: Path) -> dict[str, Path]:
    """Map ``file name -> full path`` for every jpg under ``root`` (recursive)."""
    return {p.name: p for p in root.rglob("*.jpg")}


def _write_split(
    entries: list[dict[str, Any]],
    images_by_name: dict[str, Path],
    split_output_dir: Path,
) -> None:
    """Emit one canonical COCO split from a subset of BDD100K label entries."""
    split_output_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for image_id, entry in enumerate(entries):
        # ``entry["name"]`` is the bare id (no extension) in this release's
        # native per-image JSON, unlike the old Kaggle mirror's combined
        # label files where "name" already included ".jpg".
        file_name = f"{entry['name']}.jpg"
        src = images_by_name.get(file_name)
        if src is None:
            continue
        link_image(src, split_output_dir / file_name)
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": _IMAGE_WIDTH,
                "height": _IMAGE_HEIGHT,
            }
        )
        frames = entry.get("frames") or []
        objects = frames[0].get("objects", []) if frames else []
        for obj in objects:
            box = obj.get("box2d")
            category = obj.get("category")
            if box is None or category not in _CLASS_TO_ID:
                continue
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            box_width, box_height = x2 - x1, y2 - y1
            if box_width <= 0 or box_height <= 0:
                continue
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": _CLASS_TO_ID[category],
                    "bbox": [x1, y1, box_width, box_height],
                    "area": float(box_width) * float(box_height),
                    "segmentation": [],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    payload = {
        "info": {"description": f"BDD100K canonical COCO ({split_output_dir.name})"},
        "licenses": [
            {"id": 1, "name": "BDD100K License", "url": BDD100KAdapter.spec.homepage}
        ],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i, "name": name, "supercategory": "none"}
            for i, name in enumerate(_CLASSES)
        ],
    }
    (split_output_dir / COCO_ANNOTATION_FILENAME).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"[{split_output_dir.name}] {len(images)} images, {len(annotations)} boxes")
