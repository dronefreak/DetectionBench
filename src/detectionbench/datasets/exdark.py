"""
ExDark (Exclusively Dark Image Dataset) adapter.

ExDark is a low-light robustness benchmark. It was originally released as
an image-classification dataset organized into one subdirectory per class
(``ExDark/Bicycle/*.jpg``, ``ExDark/Boat/*.jpg``, ...); bounding-box
annotations were added later by the community, typically as one text file
per image (Pascal-VOC- or YOLO-style, not a single COCO JSON) -- see
https://github.com/cs-chan/Exclusively-Dark-Image-Dataset.

In practice, a Roboflow-exported copy already in YOLO format (train/valid/test
+ data.yaml) is a common, much simpler source -- point
``configs/dataset/exdark.yaml`` straight at its ``data.yaml`` and skip
``prepare_coco`` entirely. ``prepare_coco`` below targets the *original* raw
per-class-folder distribution, for when starting from that instead: it will
need to walk the per-class directory tree, parse each image's bounding-box
annotation file, and emit the canonical
``{train,valid,test}/_annotations.coco.json`` layout.

NOTE: the class list below has been confirmed against a real Roboflow
"Exclusively-Dark-Image" export (12 classes, same names/order).
"""

from __future__ import annotations

from pathlib import Path

from detectionbench.datasets.base import DatasetAdapter, DatasetSpec
from detectionbench.datasets.registry import register

_CLASSES = [
    "Bicycle",
    "Boat",
    "Bottle",
    "Bus",
    "Car",
    "Cat",
    "Chair",
    "Cup",
    "Dog",
    "Motorbike",
    "People",
    "Table",
]


@register
class ExDarkAdapter(DatasetAdapter):
    """Adapter for the Exclusively Dark Image Dataset (low-light robustness)."""

    spec = DatasetSpec(
        key="exdark",
        display_name="ExDark",
        classes=_CLASSES,
        homepage="https://github.com/cs-chan/Exclusively-Dark-Image-Dataset",
        citation=(
            "Loh & Chan, 'Getting to Know Low-light Images with The "
            "Exclusively Dark Dataset', CVIU 2019."
        ),
        license="BSD-3-Clause -- see homepage before redistributing.",
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert a raw ExDark download into canonical COCO splits."""
        raise NotImplementedError(
            "ExDark adapter is not implemented yet. Expected raw format: "
            "per-class image subdirectories plus per-image bounding-box "
            "annotation files (not a single COCO JSON). Implement this once "
            "a raw copy of the dataset is available."
        )
