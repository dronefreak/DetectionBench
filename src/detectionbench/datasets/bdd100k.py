"""
BDD100K adapter -- multi-class autonomous-driving object detection.

Source: the Berkeley DeepDrive "100k Images" + "Labels" release, as
downloaded via the third-party Kaggle mirror
https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k (uploaded by
Kaggle user SoleSensei, not by Berkeley DeepDrive -- the official site was
registration-gated at the time this was fetched; the Kaggle copy carries no
separate license of its own, see the License note below) --
``images/100k/{train,val}/**/*.jpg`` (1280x720 JPEGs; this mirror's zip
layout splits the 70,000 train images across four subfolders confusingly
named ``trainA``/``trainB``/``testA``/``testB`` -- all four are genuinely
part of ``train`` (verified: their combined recursive count is exactly
70,000, matching the official train total, and every one of
``train.json``'s 69,863 entries resolves to a file inside one of these
four subfolders); the loose, separate ``images/100k/test/`` top-level
directory (293 files, no matching labels) is the mirror's copy of the
*actual* unlabelled official test split and is correctly never referenced
by this adapter. ``val`` is flat. All image subfolders are walked
recursively here.) plus the original-format
``labels/bdd100k_labels_images_{train,val}.json`` (one dict per image:
``attributes`` + a ``labels`` list mixing ``box2d`` detection boxes with
``poly2d`` lane/drivable-area annotations). Only ``box2d`` entries are
converted; lane markings and drivable-area polygons are not a detection
task and are dropped. The official ``test`` split (20,000 images) ships
with no labels (held out for the leaderboard) and is not used here.

BDD100K's own labelled splits are ``train`` (69,863 labelled images) and
``val`` (10,000). This adapter keeps ``val`` as the canonical ``test``
(consistent with how ``val``'s ground truth in this release is only usable
as a held-out eval set, not for training) and carves a seeded validation
set out of ``train`` (``_VAL_FRACTION``), the same approach used for
HRSID/SSDD.

License: the data and labels (as opposed to the BDD100K code repo, which is
BSD-3-Clause) are under UC Regents' own academic license, verified directly
from https://github.com/bdd100k/bdd100k/blob/master/doc/source/license.rst
-- it explicitly grants "permission to use, copy, modify, and distribute
this software and its documentation for educational, research, and
not-for-profit purposes, without fee and without a signed licensing
agreement," provided the copyright notice and license paragraphs are
carried forward into any redistribution (commercial use/distribution is
separately restricted to BDD/BAIR Commons members). This corrects an
earlier, incorrect "no redistribution permitted" reading of this license
in this module. The data itself is registration-gated (manual DUA
click-through on the official site), not automatable via
``download_dataset.py`` -- this adapter's raw data was instead pulled from
the Kaggle mirror above for convenience; that mirror declares no license of
its own (Kaggle lists it as "Other (specified in description)"), so the
official UC Regents terms above are what actually govern this data,
regardless of download channel. **Mirrored on Hugging Face** as
``dronefreak/BDD100K`` under this license, with the full notice text
reproduced verbatim on the card per the license's own carry-forward
condition.

Stats: see docs/datasets/bdd100k/README.md (class distribution, split
summary, box geometry -- generated via detectionbench-dataset-stats).
"""

from __future__ import annotations

import json
import random
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

_VAL_FRACTION = 0.15
_SPLIT_SEED = 42


@register
class BDD100KAdapter(DatasetAdapter):
    """Adapter for the BDD100K multi-class driving-scene detection dataset."""

    spec = DatasetSpec(
        key="bdd100k",
        display_name="BDD100K",
        classes=_CLASSES,
        description=(
            "BDD100K is a large-scale, diverse driving-video dataset; this "
            "adapter covers its 100K-image object-detection task: 79,863 "
            "labelled 1280x720 dashcam images (69,863 train + 10,000 val, "
            "the official held-out test split has no released labels) "
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
        image_dir = raw_dir / "images" / "100k"
        label_dir = raw_dir / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {raw_dir}/images/100k and {raw_dir}/labels."
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        train_entries = json.loads(
            (label_dir / "bdd100k_labels_images_train.json").read_text(encoding="utf-8")
        )
        train_images_by_path = _index_images(image_dir / "train")
        shuffled = train_entries[:]
        random.Random(_SPLIT_SEED).shuffle(shuffled)  # noqa: S311  # nosec: B311
        n_val = max(1, round(len(shuffled) * _VAL_FRACTION))
        val_names = {e["name"] for e in shuffled[:n_val]}

        _write_split(
            [e for e in train_entries if e["name"] not in val_names],
            train_images_by_path,
            output_dir / "train",
        )
        _write_split(
            [e for e in train_entries if e["name"] in val_names],
            train_images_by_path,
            output_dir / "valid",
        )

        val_entries = json.loads(
            (label_dir / "bdd100k_labels_images_val.json").read_text(encoding="utf-8")
        )
        val_images_by_path = _index_images(image_dir / "val")
        _write_split(val_entries, val_images_by_path, output_dir / "test")


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
        name = entry["name"]
        src = images_by_name.get(name)
        if src is None:
            continue
        link_image(src, split_output_dir / name)
        images.append(
            {
                "id": image_id,
                "file_name": name,
                "width": _IMAGE_WIDTH,
                "height": _IMAGE_HEIGHT,
            }
        )
        for label in entry.get("labels", []):
            box = label.get("box2d")
            category = label.get("category")
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
