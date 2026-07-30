"""
LISA Traffic Lights dataset adapter.

The LISA Traffic Light Dataset (Jensen et al., UC San Diego) is a
day/night driving-sequence dataset for traffic-light state detection. The
original raw distribution ships as per-sequence video frame directories
plus CSV-format bounding-box annotations (not a single COCO JSON).

A pre-converted copy already in YOLO format (train/val/test + data.yaml)
is the common source here -- point ``configs/dataset/lisa.yaml`` straight
at its ``data.yaml`` and skip ``prepare_coco`` entirely. ``prepare_coco``
below targets the *original* raw CSV-annotated distribution, for when
starting from that instead.

Two annotation granularities exist upstream: "box" (one bounding box per
traffic light housing, this adapter's default) and "bulb" (finer boxes per
lit bulb region) -- both share the same 7-class taxonomy below. Pick
whichever your local conversion targets; this adapter assumes "box".

NOTE: the class list has been confirmed against a real converted "box"
export (7 classes, same names/order as both the "box" and "bulb" variants).
"""

from __future__ import annotations

from pathlib import Path

from detectionbench.datasets.base import DatasetAdapter, DatasetSpec
from detectionbench.datasets.registry import register

_CLASSES = [
    "go",
    "goForward",
    "goLeft",
    "stop",
    "stopLeft",
    "warning",
    "warningLeft",
]


@register
class LISATrafficLightsAdapter(DatasetAdapter):
    """Adapter for the LISA Traffic Light Dataset (day/night driving sequences)."""

    spec = DatasetSpec(
        key="lisa",
        display_name="LISA Traffic Lights",
        classes=_CLASSES,
        homepage="https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset",
        citation=(
            "Jensen et al., 'Vision for Looking at Traffic Lights: Issues, "
            "Survey, and Perspectives', IEEE Trans. Intelligent "
            "Transportation Systems, 2016."
        ),
        license="Research use -- see homepage before redistributing.",
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert a raw LISA Traffic Lights download into canonical COCO splits."""
        raise NotImplementedError(
            "LISA adapter is not implemented yet for the original raw "
            "distribution (per-sequence frames + CSV annotations). A "
            "pre-converted YOLO copy bypasses this entirely -- see the "
            "module docstring."
        )
