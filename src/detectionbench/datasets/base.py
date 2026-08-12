"""
Base types for the dataset registry.

An adapter's only job is to convert a raw dataset download into the
canonical COCO split layout: ``output_dir/{train,valid,test}/_annotations.coco.json``
plus the corresponding images. Everything downstream (the generic COCO<->YOLO
bridge converters, training, evaluation, inference) operates on that canonical
layout, so an adapter never needs to know anything about YOLO, Ultralytics, or
RF-DETR -- only its own dataset's raw format.
"""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

COCO_ANNOTATION_FILENAME = "_annotations.coco.json"


def link_image(src: Path, dst: Path) -> None:
    """
    Materialize ``src`` at ``dst`` as a real directory entry (hardlink, or a copy).

    Deliberately not a symlink: a symlink here would resolve outside the
    canonical COCO split directory (back into the raw download `src` lives
    in), which downstream consumers that validate resolved paths against
    that directory -- e.g. Supervision's ``DetectionDataset.from_coco``,
    which rejects any image path escaping the declared images directory --
    refuse to load. A hardlink keeps the old symlink approach's space
    savings (same inode, no duplicated bytes) while still resolving to a
    real path inside the split directory; falls back to a copy only when
    ``src``/``dst`` are on different filesystems (hardlinks can't cross
    devices).
    """
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata describing a supported dataset."""

    key: str
    display_name: str
    classes: list[str]
    class_colors: dict[int, tuple[int, int, int]] | None = None
    homepage: str | None = None
    citation: str | None = None
    license: str | None = None

    @property
    def num_classes(self) -> int:
        """Number of detection classes in this dataset."""
        return len(self.classes)


class DatasetAdapter(ABC):
    """Converts a raw dataset download into the canonical COCO layout."""

    spec: ClassVar[DatasetSpec]

    @abstractmethod
    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert ``raw_dir`` (a raw dataset download) into canonical COCO splits."""
