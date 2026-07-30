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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

COCO_ANNOTATION_FILENAME = "_annotations.coco.json"


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
