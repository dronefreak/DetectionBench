"""End-to-end test of the dataset statistics report on a tiny COCO dataset."""

import json
import sys
from pathlib import Path

import pytest

from detectionbench.scripts import dataset_stats


def _write_split(coco_dir: Path, split: str, n_images: int) -> None:
    split_dir = coco_dir / split
    split_dir.mkdir(parents=True)
    images = [
        {"id": i, "file_name": f"img{i}.jpg", "width": 100, "height": 100}
        for i in range(1, n_images + 1)
    ]
    annotations = [
        {
            "id": i,
            "image_id": i,
            "category_id": i % 2,
            "bbox": [10, 10, 20, 20],
            "area": 400,
            "iscrowd": 0,
        }
        for i in range(1, n_images + 1)
    ]
    categories = [{"id": 0, "name": "cat"}, {"id": 1, "name": "dog"}]
    (split_dir / "_annotations.coco.json").write_text(
        json.dumps(
            {"images": images, "annotations": annotations, "categories": categories}
        )
    )


def _run(monkeypatch: pytest.MonkeyPatch, coco_dir: Path, out_dir: Path) -> str:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset_stats",
            "--coco-dir",
            str(coco_dir),
            "--name",
            "Toy",
            "--output-dir",
            str(out_dir),
        ],
    )
    dataset_stats.main()
    return (out_dir / "README.md").read_text()


def test_report_has_split_table_and_chart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    coco_dir = tmp_path / "coco"
    _write_split(coco_dir, "train", 4)
    _write_split(coco_dir, "valid", 2)
    out_dir = tmp_path / "report"

    report = _run(monkeypatch, coco_dir, out_dir)

    assert report.startswith("# Toy: Dataset Statistics")
    assert "| train | 4 | 4 |" in report
    assert "| valid | 2 | 2 |" in report
    assert (out_dir / "class_distribution.png").is_file()
    assert "_banner.jpg" not in report


def test_report_embeds_banner_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    coco_dir = tmp_path / "coco"
    _write_split(coco_dir, "train", 2)
    out_dir = tmp_path / "report"
    out_dir.mkdir()
    (out_dir / "toy_banner.jpg").write_bytes(b"jpeg")

    report = _run(monkeypatch, coco_dir, out_dir)

    assert report.splitlines()[2] == "![Toy sample images](toy_banner.jpg)"
