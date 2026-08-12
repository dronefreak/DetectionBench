"""
DocLayNet dataset adapter.

DocLayNet ships as flat page images (``PNG/*.png``) plus per-split COCO
detection JSONs (``COCO/{train,val,test}.json``). This adapter reshuffles
that into the canonical ``{train,valid,test}/_annotations.coco.json`` layout,
remapping category ids to a contiguous 0-indexed range along the way.
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
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
]

_CLASS_COLORS = {
    0: (255, 99, 71),  # Caption - tomato
    1: (255, 165, 0),  # Footnote - orange
    2: (218, 112, 214),  # Formula - orchid
    3: (60, 179, 113),  # List-item - medium sea green
    4: (128, 128, 128),  # Page-footer - gray
    5: (169, 169, 169),  # Page-header - dark gray
    6: (30, 144, 255),  # Picture - dodger blue
    7: (255, 215, 0),  # Section-header - gold
    8: (0, 191, 255),  # Table - deep sky blue
    9: (0, 200, 0),  # Text - green
    10: (255, 0, 0),  # Title - red
}

# DocLayNet's own split names differ from the canonical roboflow-style
# {train,valid,test} convention used elsewhere in this codebase.
_SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}


@register
class DocLayNetAdapter(DatasetAdapter):
    """Adapter for IBM Research's DocLayNet document-layout dataset."""

    spec = DatasetSpec(
        key="doclaynet",
        display_name="DocLayNet",
        classes=_CLASSES,
        class_colors=_CLASS_COLORS,
        homepage="https://github.com/DS4SD/DocLayNet",
        citation=(
            "Pfitzmann et al., 'DocLayNet: A Large Human-Annotated Dataset for "
            "Document-Layout Analysis', KDD 2022."
        ),
        license="CDLA-Permissive-1.0 (dataset); Apache-2.0 (this code)",
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert a raw DocLayNet download into the canonical COCO layout."""
        png_dir = raw_dir / "PNG"
        coco_dir = raw_dir / "COCO"
        output_dir.mkdir(parents=True, exist_ok=True)

        for source_split, target_split in _SPLIT_MAP.items():
            json_path = coco_dir / f"{source_split}.json"
            if not json_path.exists():
                continue
            _convert_split(json_path, png_dir, output_dir / target_split)


def _convert_split(json_path: Path, png_dir: Path, split_output_dir: Path) -> None:
    """Reshuffle one DocLayNet COCO split into the canonical layout."""
    with json_path.open(encoding="utf-8") as file:
        data = json.load(file)

    categories = sorted(data["categories"], key=lambda category: category["id"])
    cat_id_to_idx = {category["id"]: index for index, category in enumerate(categories)}

    split_output_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    for image in data["images"]:
        src = png_dir / image["file_name"]
        dst = split_output_dir / image["file_name"]
        if src.exists():
            link_image(src, dst)
        images.append(image)

    annotations = [
        {**annotation, "category_id": cat_id_to_idx[annotation["category_id"]]}
        for annotation in data["annotations"]
    ]
    remapped_categories = [
        {
            "id": index,
            "name": category["name"],
            "supercategory": category.get("supercategory", "none"),
        }
        for index, category in enumerate(categories)
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
