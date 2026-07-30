"""
Aquarium dataset adapter (stub -- not yet implemented).

Aquarium is a small underwater-imagery dataset distributed via Roboflow
Universe (https://universe.roboflow.com/brad-dwyer/aquarium-combined).
Roboflow exports are already either COCO-JSON or YOLO-format, so this
adapter is likely the thinnest of the four: if exported as COCO, it is
mostly a directory reshuffle into the canonical layout (see
``detectionbench.datasets.doclaynet`` for that pattern); if exported as
YOLO, ``utils/convert_yolo_to_coco.py`` (already dataset-agnostic) can be
used directly instead of a bespoke adapter.

NOTE: the class list below matches the public Roboflow "Aquarium Combined"
dataset card and should be confirmed against the actual downloaded
``_annotations.coco.json``/``data.yaml`` before training.
"""

from __future__ import annotations

from pathlib import Path

from detectionbench.datasets.base import DatasetAdapter, DatasetSpec
from detectionbench.datasets.registry import register

_CLASSES = [
    "fish",
    "jellyfish",
    "penguin",
    "puffin",
    "shark",
    "starfish",
    "stingray",
]


@register
class AquariumAdapter(DatasetAdapter):
    """Adapter for the Roboflow Aquarium underwater-imagery dataset."""

    spec = DatasetSpec(
        key="aquarium",
        display_name="Aquarium",
        classes=_CLASSES,
        homepage="https://universe.roboflow.com/brad-dwyer/aquarium-combined",
        citation="Roboflow, 'Aquarium Combined' dataset (Roboflow Universe).",
        license="CC BY 4.0 -- see homepage before redistributing.",
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert a raw Aquarium (Roboflow) download into canonical COCO splits."""
        raise NotImplementedError(
            "Aquarium adapter is not implemented yet. Roboflow exports are "
            "typically already COCO-JSON or YOLO-format -- if YOLO, prefer "
            "utils/convert_yolo_to_coco.py directly instead of this adapter. "
            "Implement this once a raw copy of the dataset is available."
        )
