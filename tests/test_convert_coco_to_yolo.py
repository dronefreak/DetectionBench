"""Basic tests for the canonical-COCO -> YOLO bridge converter."""

import argparse
import json
from pathlib import Path

from detectionbench.utils.convert_coco_to_yolo import convert, convert_split


def _write_canonical_split(split_dir: Path) -> None:
    split_dir.mkdir(parents=True)
    (split_dir / "img1.jpg").write_bytes(b"fake-image-bytes")
    payload = {
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 200, "height": 100}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [50, 25, 20, 10]},
        ],
        "categories": [{"id": 0, "name": "cat"}, {"id": 1, "name": "dog"}],
    }
    (split_dir / "_annotations.coco.json").write_text(json.dumps(payload))


def test_convert_split_writes_normalized_yolo_label(tmp_path: Path) -> None:
    coco_split_dir = tmp_path / "coco" / "train"
    _write_canonical_split(coco_split_dir)
    output_dir = tmp_path / "yolo"

    categories = convert_split(coco_split_dir, output_dir, "train", copy_images=True)

    assert [c["name"] for c in sorted(categories, key=lambda c: c["id"])] == [
        "cat",
        "dog",
    ]

    label_path = output_dir / "labels" / "train" / "img1.txt"
    assert label_path.exists()
    cls, xc, yc, _w, _h = label_path.read_text().split()
    assert cls == "0"
    assert float(xc) == 0.3  # (50 + 20/2) / 200
    assert float(yc) == 0.3  # (25 + 10/2) / 100

    assert (output_dir / "images" / "train" / "img1.jpg").exists()


def test_convert_writes_data_yaml_with_class_names(tmp_path: Path) -> None:
    input_dir = tmp_path / "coco"
    _write_canonical_split(input_dir / "train")
    output_dir = tmp_path / "yolo"

    args = argparse.Namespace(
        input_dir=str(input_dir), output_dir=str(output_dir), copy_images=True
    )
    convert(args)

    data_yaml_text = (output_dir / "data.yaml").read_text()
    assert "nc: 2" in data_yaml_text
    assert "cat" in data_yaml_text
    assert "dog" in data_yaml_text
