"""Basic tests for the DocLayNet adapter's raw -> canonical COCO conversion."""

import json
from pathlib import Path

from detectionbench.datasets.doclaynet import DocLayNetAdapter


def _write_raw_doclaynet(raw_dir: Path) -> None:
    png_dir = raw_dir / "PNG"
    coco_dir = raw_dir / "COCO"
    png_dir.mkdir(parents=True)
    coco_dir.mkdir(parents=True)

    (png_dir / "page_1.png").write_bytes(b"fake-image-bytes")

    train_json = {
        "categories": [
            {"id": 5, "name": "Title", "supercategory": "none"},
            {"id": 2, "name": "Text", "supercategory": "none"},
        ],
        "images": [{"id": 1, "file_name": "page_1.png", "width": 100, "height": 200}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 5, "bbox": [0, 0, 10, 10]},
        ],
    }
    (coco_dir / "train.json").write_text(json.dumps(train_json))


def test_prepare_coco_writes_canonical_train_split(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "canonical"
    _write_raw_doclaynet(raw_dir)

    DocLayNetAdapter().prepare_coco(raw_dir, output_dir)

    annotation_path = output_dir / "train" / "_annotations.coco.json"
    assert annotation_path.exists()
    payload = json.loads(annotation_path.read_text())

    assert len(payload["images"]) == 1
    assert payload["images"][0]["file_name"] == "page_1.png"
    assert (output_dir / "train" / "page_1.png").exists()

    # Category ids get remapped to a contiguous 0-indexed range, ordered by
    # the original id (2 -> 0, 5 -> 1).
    categories_by_name = {c["name"]: c["id"] for c in payload["categories"]}
    assert categories_by_name == {"Text": 0, "Title": 1}
    assert payload["annotations"][0]["category_id"] == categories_by_name["Title"]


def test_prepare_coco_skips_missing_splits(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "canonical"
    _write_raw_doclaynet(raw_dir)  # only train.json exists

    DocLayNetAdapter().prepare_coco(raw_dir, output_dir)

    assert not (output_dir / "valid").exists()
    assert not (output_dir / "test").exists()
