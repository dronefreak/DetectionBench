"""Basic test for the YOLO -> canonical-COCO bridge converter."""

import argparse
import json
from pathlib import Path

from PIL import Image

from detectionbench.utils.convert_yolo_to_coco import convert


def _write_yolo_dataset(root: Path) -> None:
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)

    image_path = root / "images" / "train" / "img1.jpg"
    Image.new("RGB", (100, 50), color="white").save(image_path)
    (root / "labels" / "train" / "img1.txt").write_text("0 0.5 0.5 0.2 0.4\n")

    (root / "data.yaml").write_text(
        "path: .\ntrain: images/train\nnc: 1\nnames: ['widget']\n"
    )


def test_convert_yolo_to_coco_round_trip(tmp_path: Path) -> None:
    input_dir = tmp_path / "yolo"
    _write_yolo_dataset(input_dir)
    output_dir = tmp_path / "coco"

    args = argparse.Namespace(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        dataset_yaml=None,
        save_report=None,
    )
    convert(args)

    annotation_path = output_dir / "train" / "_annotations.coco.json"
    assert annotation_path.exists()
    payload = json.loads(annotation_path.read_text())

    assert len(payload["images"]) == 1
    assert payload["images"][0]["width"] == 100
    assert payload["images"][0]["height"] == 50
    assert len(payload["annotations"]) == 1
    assert payload["categories"] == [
        {"id": 0, "name": "widget", "supercategory": "none"}
    ]
