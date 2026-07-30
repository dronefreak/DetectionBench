"""
SeaDronesSee dataset adapter.

SeaDronesSee (the "CompressedVersion" object-detection release) ships
already-COCO-format annotations plus a flat per-split image directory::

    images/{train,val,test}/*.jpg
    annotations/instances_{train,val}.json

Note there is no ``instances_test.json`` -- ``images/test/`` is a held-out
competition test set with no public ground truth, so this adapter only
emits canonical ``train``/``valid`` splits; use ``valid`` for evaluation.

The declared category list includes an ``ignored`` class (id 0, a
"don't-care" region marker, not a real detectable object) alongside the 5
real classes. As of the 2022 "CompressedVersion" release ``ignored`` has
zero actual annotations, so this adapter drops it entirely from the
exported class list rather than remapping it into row 0 of a taxonomy
nobody should be training a detector to predict.

See https://seadronessee.cs.uni-tuebingen.de/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from detectionbench.datasets.base import (
    COCO_ANNOTATION_FILENAME,
    DatasetAdapter,
    DatasetSpec,
)
from detectionbench.datasets.registry import register

_CLASSES = [
    "swimmer",
    "boat",
    "jetski",
    "life_saving_appliances",
    "buoy",
]
_IGNORED_CATEGORY_NAME = "ignored"

# SeaDronesSee's own split names differ from the canonical roboflow-style
# {train,valid,test} convention used elsewhere in this codebase. There is
# no test-split annotation file (held-out competition test set).
_SPLIT_MAP = {"train": "train", "val": "valid"}


@register
class SeaDronesSeeAdapter(DatasetAdapter):
    """Adapter for the SeaDronesSee maritime UAV object-detection benchmark."""

    spec = DatasetSpec(
        key="seadronessee",
        display_name="SeaDronesSee",
        classes=_CLASSES,
        homepage="https://seadronessee.cs.uni-tuebingen.de/",
        citation=(
            "Varga et al., 'SeaDronesSee: A Maritime Benchmark for Detecting "
            "Humans in Maritime Environments', WACV 2022."
        ),
        license="Custom research license -- see homepage before redistributing.",
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert a raw SeaDronesSee download into canonical train/valid splits."""
        images_dir = raw_dir / "images"
        annotations_dir = raw_dir / "annotations"
        output_dir.mkdir(parents=True, exist_ok=True)

        for source_split, target_split in _SPLIT_MAP.items():
            json_path = annotations_dir / f"instances_{source_split}.json"
            if not json_path.exists():
                continue
            _convert_split(
                json_path, images_dir / source_split, output_dir / target_split
            )


def _convert_split(
    json_path: Path, split_images_dir: Path, split_output_dir: Path
) -> None:
    """Reshuffle one SeaDronesSee COCO split into the canonical layout."""
    with json_path.open(encoding="utf-8") as file:
        data = json.load(file)

    real_categories = sorted(
        (c for c in data["categories"] if c["name"] != _IGNORED_CATEGORY_NAME),
        key=lambda c: c["id"],
    )
    cat_id_to_idx = {c["id"]: index for index, c in enumerate(real_categories)}

    split_output_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    for image in data["images"]:
        src = split_images_dir / image["file_name"]
        dst = split_output_dir / image["file_name"]
        if not dst.exists() and src.exists():
            os.symlink(src, dst)
        images.append(image)

    annotations = [
        {**annotation, "category_id": cat_id_to_idx[annotation["category_id"]]}
        for annotation in data["annotations"]
        if annotation["category_id"] in cat_id_to_idx
    ]
    remapped_categories = [
        {
            "id": index,
            "name": category["name"],
            "supercategory": category.get("supercategory", "none"),
        }
        for index, category in enumerate(real_categories)
    ]

    payload = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": remapped_categories,
    }
    (split_output_dir / COCO_ANNOTATION_FILENAME).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
