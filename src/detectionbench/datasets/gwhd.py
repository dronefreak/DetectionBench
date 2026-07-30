"""
Global Wheat Head Dataset (GWHD) adapter (stub -- not yet implemented).

GWHD is a single-class, dense-detection agriculture dataset (wheat heads
from field images across many countries/growth stages). It ships as a
CSV of ``image_id, width, height, bbox, source`` rows (one row per box,
``bbox`` a stringified ``[x, y, w, h]``) plus a flat image directory --
see https://www.global-wheat.com/. Once a raw copy is available,
``prepare_coco`` should group rows by ``image_id``, convert each row's
bbox into a COCO annotation, and emit the canonical
``{train,valid,test}/_annotations.coco.json`` layout.

NOTE: GWHD's official train/val/test split assignment and single-class
name should be confirmed against the actual downloaded CSV/competition
page before training.
"""

from __future__ import annotations

from pathlib import Path

from detectionbench.datasets.base import DatasetAdapter, DatasetSpec
from detectionbench.datasets.registry import register

_CLASSES = ["wheat_head"]


@register
class GWHDAdapter(DatasetAdapter):
    """Adapter for the Global Wheat Head Dataset."""

    spec = DatasetSpec(
        key="gwhd",
        display_name="Global Wheat Head Dataset",
        classes=_CLASSES,
        homepage="https://www.global-wheat.com/",
        citation=(
            "David et al., 'Global Wheat Head Detection (GWHD) Dataset: A Large "
            "and Diverse Dataset of High-Resolution RGB-Labelled Images to "
            "Develop and Benchmark Wheat Head Detection Methods', Plant "
            "Phenomics, 2020 (and the 2021 update)."
        ),
        license="CC BY-SA 4.0 -- see homepage before redistributing.",
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert a raw GWHD download into canonical COCO splits."""
        raise NotImplementedError(
            "GWHD adapter is not implemented yet. Expected raw format: a CSV "
            "of image_id/width/height/bbox/source rows plus a flat image "
            "directory. Implement this once a raw copy of the dataset is "
            "available."
        )
