#!/usr/bin/env python3
r"""
Convert a canonical COCO dataset into Ultralytics YOLO format.

This is a generic bridge: it works for any dataset already in the
canonical COCO split layout produced by a ``detectionbench.datasets``
adapter (via ``detectionbench-prepare-coco``) or by
``detectionbench-convert-yolo-to-coco``. It has no dataset-specific logic,
so it deliberately takes no ``--dataset`` flag -- just an input/output
directory pair, shared with its reverse-direction sibling.

Input directory layout expected (canonical COCO layout):
  input_dir/
  ├── train/
  │   ├── _annotations.coco.json
  │   └── *.jpg / *.png / ...
  ├── valid/
  │   ├── _annotations.coco.json
  │   └── ...
  └── test/
      ├── _annotations.coco.json
      └── ...

Output directory layout produced:
  output_dir/
  ├── images/
  │   ├── train/  (symlinked from input_dir/train/ by default)
  │   ├── val/
  │   └── test/
  ├── labels/
  │   ├── train/  (*.txt, one per image: "class xc yc w h", normalized)
  │   ├── val/
  │   └── test/
  └── data.yaml   (Ultralytics dataset config; nc + class names from COCO categories)

Usage:
  python -m detectionbench.utils.convert_coco_to_yolo \
      --input-dir /path/to/coco_dataset --output-dir /path/to/yolo_dataset

  # Copy image bytes instead of symlinking (e.g. across filesystems)
  python -m detectionbench.utils.convert_coco_to_yolo \
      --input-dir /path/to/coco_dataset --output-dir /path/to/yolo_dataset \
      --copy-images

  # Hardlink image files (same filesystem): no extra disk, and the output tree
  # holds real file entries, so it can be tarred/rsynced to another machine
  # without dangling absolute symlinks. Falls back to a copy across filesystems.
  python -m detectionbench.utils.convert_coco_to_yolo \
      --input-dir /path/to/coco_dataset --output-dir /path/to/yolo_dataset \
      --hardlink-images
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

from detectionbench.datasets.base import link_image

COCO_ANNOTATION_FILENAME = "_annotations.coco.json"
# Canonical COCO split directory name -> conventional Ultralytics split name.
SPLIT_DIR_MAP = {"train": "train", "valid": "val", "test": "test"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for canonical-COCO -> YOLO conversion."""
    parser = argparse.ArgumentParser(
        description="Convert a canonical COCO dataset into Ultralytics YOLO format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Canonical COCO dataset root (contains train/, valid/, test/)",
    )
    parser.add_argument("--output-dir", required=True, help="Output YOLO dataset root")
    link_mode = parser.add_mutually_exclusive_group()
    link_mode.add_argument(
        "--copy-images",
        action="store_true",
        default=False,
        help="Copy image bytes instead of symlinking (default: symlink)",
    )
    link_mode.add_argument(
        "--hardlink-images",
        action="store_true",
        default=False,
        help="Hardlink images instead of symlinking (no extra disk; falls back "
        "to a copy across filesystems). The output stays valid when moved.",
    )
    return parser.parse_args()


def _materialize_image(
    src: Path, dst: Path, copy_images: bool, hardlink_images: bool
) -> bool:
    """Place ``src`` at ``dst`` (symlink/hardlink/copy); False if ``src`` is missing."""
    if dst.exists():
        return True
    if not src.exists():
        return False
    if copy_images:
        shutil.copy2(src, dst)
    elif hardlink_images:
        link_image(src, dst)
    else:
        os.symlink(src, dst)
    return True


def convert_split(
    coco_split_dir: Path,
    output_dir: Path,
    yolo_split: str,
    copy_images: bool,
    hardlink_images: bool = False,
) -> list[dict[str, Any]]:
    """Convert one canonical COCO split directory into YOLO images/labels."""
    annotation_path = coco_split_dir / COCO_ANNOTATION_FILENAME
    with annotation_path.open(encoding="utf-8") as f:
        data = json.load(f)

    categories = sorted(data["categories"], key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(categories)}

    images_out = output_dir / "images" / yolo_split
    labels_out = output_dir / "labels" / yolo_split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    img_by_id = {img["id"]: img for img in data["images"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    n_images = 0
    n_labels = 0
    n_boxes = 0
    n_skipped_bad_box = 0
    n_missing_images = 0

    for img_id, img in img_by_id.items():
        file_name = img["file_name"]
        w, h = img["width"], img["height"]

        if not _materialize_image(
            coco_split_dir / file_name,
            images_out / file_name,
            copy_images,
            hardlink_images,
        ):
            n_missing_images += 1
            continue
        n_images += 1

        lines = []
        for ann in anns_by_image.get(img_id, []):
            x, y, bw, bh = ann["bbox"]
            if bw <= 0 or bh <= 0:
                n_skipped_bad_box += 1
                continue
            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            nbw = bw / w
            nbh = bh / h
            # clip to [0,1] to guard against any rounding artifacts
            xc, yc = min(max(xc, 0.0), 1.0), min(max(yc, 0.0), 1.0)
            nbw, nbh = min(max(nbw, 0.0), 1.0), min(max(nbh, 0.0), 1.0)
            cls = cat_id_to_idx[ann["category_id"]]
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {nbw:.6f} {nbh:.6f}")
            n_boxes += 1

        label_path = labels_out / (Path(file_name).stem + ".txt")
        label_path.write_text("\n".join(lines))
        n_labels += 1

    print(
        f"[{yolo_split}] images linked: {n_images}, label files: {n_labels}, "
        f"boxes: {n_boxes}, skipped bad boxes: {n_skipped_bad_box}"
    )
    if n_missing_images:
        print(
            f"[WARN] [{yolo_split}] {n_missing_images} images listed in "
            f"{annotation_path} were not found under {coco_split_dir} and were "
            "skipped (no image or label written)."
        )
    return categories


def write_yaml(categories: list[dict[str, Any]], output_dir: Path) -> None:
    """Write an Ultralytics dataset YAML describing the converted splits."""
    names = [c["name"] for c in sorted(categories, key=lambda c: c["id"])]
    yaml_path = output_dir / "data.yaml"
    data_yaml = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": dict(enumerate(names)),
    }
    # safe_dump quotes names that would otherwise break YAML (e.g. "a: b", "#1").
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)
    print(f"Wrote {yaml_path}")


def convert(args: argparse.Namespace) -> None:
    """Convert every split found in the configured canonical COCO dataset into YOLO."""
    # Resolve to absolute paths up front: the default (non---copy-images) path
    # writes symlinks whose target is `input_dir/<split>/<file>`, and a
    # relative target would be interpreted relative to the *link's* directory
    # (output_dir/images/<split>/) and dangle. Mirrors convert_yolo_to_coco.py.
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    categories: list[dict[str, Any]] = []
    for coco_split, yolo_split in SPLIT_DIR_MAP.items():
        coco_split_dir = input_dir / coco_split
        annotation_path = coco_split_dir / COCO_ANNOTATION_FILENAME
        if not annotation_path.exists():
            print(f"[WARN] Skipping '{coco_split}' — {annotation_path} not found.")
            continue
        categories = convert_split(
            coco_split_dir,
            output_dir,
            yolo_split,
            args.copy_images,
            getattr(args, "hardlink_images", False),
        )

    if categories:
        write_yaml(categories, output_dir)


def main() -> None:
    """Run the dataset conversion CLI entrypoint."""
    convert(parse_args())


if __name__ == "__main__":
    main()
