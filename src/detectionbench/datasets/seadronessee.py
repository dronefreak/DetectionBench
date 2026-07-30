"""
SeaDronesSee dataset adapter (stub -- not yet implemented).

SeaDronesSee is a maritime UAV benchmark for detecting humans and small
objects (swimmers, boats, life-saving appliances, buoys) from drone
footage. The object-detection subset ships COCO-JSON annotations per split
(see https://seadronessee.cs.uni-tuebingen.de/). Once a raw copy is
available, ``prepare_coco`` should read those split JSONs and re-emit them
under the canonical ``{train,valid,test}/_annotations.coco.json`` layout --
see ``detectionbench.datasets.doclaynet`` for a worked "already-COCO, just
reshuffle directories and remap category ids" example.

NOTE: the class list below is a best-effort placeholder based on public
documentation and has NOT been verified against a real downloaded
annotation file. Confirm it (and ``class_colors``) against the actual
``categories`` field before training.
"""

from __future__ import annotations

from pathlib import Path

from detectionbench.datasets.base import DatasetAdapter, DatasetSpec
from detectionbench.datasets.registry import register

_CLASSES = [
    "swimmer",
    "floater",
    "boat",
    "swimmer_on_boat",
    "life_saving_appliance",
    "buoy",
]


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
        """Convert a raw SeaDronesSee download into canonical COCO splits."""
        raise NotImplementedError(
            "SeaDronesSee adapter is not implemented yet. Expected raw format: "
            "per-split COCO-JSON annotation files. Implement this once a raw "
            "copy of the dataset is available."
        )
