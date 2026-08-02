#!/usr/bin/env python3
r"""
Generate a horizontal filmstrip banner spanning every converted dataset.

For use as the main repository README's hero image. Picks one
representative, well-labeled image per dataset (reusing the
same box-drawing/letterboxing helpers as ``dataset_banner.py``), burns in
a caption with the dataset's display name, and stitches the results into
a single wide strip.

Dataset locations are read from ``configs/dataset/*.yaml``; any dataset
whose ``dataset_yaml`` doesn't exist on disk yet (e.g. an unconverted
stub dataset) is skipped with a warning rather than failing the
whole run.

Usage:
  python -m detectionbench.scripts.repo_showcase --output banners/filmstrip.jpg
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

from detectionbench.datasets import get_spec, list_datasets
from detectionbench.scripts.dataset_banner import (
    draw_yolo_boxes,
    letterbox_resize,
    load_dataset_config,
    resolve_split_images,
    select_candidates,
)

CONFIGS_DATASET_DIR = Path(__file__).resolve().parents[3] / "configs" / "dataset"
DEFAULT_TILE_SIZE = 480
CAPTION_HEIGHT = 56
CAPTION_BG_COLOR = (20, 20, 20)
CAPTION_TEXT_COLOR = (255, 255, 255)
CANVAS_BACKGROUND_COLOR = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for filmstrip generation."""
    parser = argparse.ArgumentParser(
        description="Generate a cross-dataset filmstrip banner for the main README.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--split", default="train", help="Preferred split to sample from"
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help="Each tile's width/height",
    )
    parser.add_argument(
        "--min-boxes",
        type=int,
        default=2,
        help="Only consider images with at least this many labeled boxes",
    )
    parser.add_argument(
        "--margin", type=int, default=6, help="Pixel margin between tiles"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random sampling seed")
    parser.add_argument(
        "--output", default="banners/filmstrip.jpg", help="Output banner image path"
    )
    return parser.parse_args()


def resolve_dataset_yaml(key: str) -> Path | None:
    """Look up a registered dataset's converted data.yaml path, if it exists."""
    config_path = CONFIGS_DATASET_DIR / f"{key}.yaml"
    if not config_path.exists():
        return None

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_yaml = data.get("dataset_yaml")
    if not dataset_yaml:
        return None

    path = Path(dataset_yaml)
    return path if path.exists() else None


def caption_tile(image: np.ndarray, text: str, tile_size: int) -> np.ndarray:
    """Append a caption bar with the dataset's display name below an image tile."""
    canvas = np.full(
        (tile_size + CAPTION_HEIGHT, tile_size, 3), CAPTION_BG_COLOR, dtype=np.uint8
    )
    canvas[:tile_size] = image

    font_scale = 0.9
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    x = max(0, (tile_size - text_w) // 2)
    y = tile_size + (CAPTION_HEIGHT + text_h) // 2
    cv2.putText(
        canvas,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        CAPTION_TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    return canvas


def build_tile(  # noqa: PLR0913
    key: str, split: str, min_boxes: int, tile_size: int, rng: random.Random
) -> np.ndarray | None:
    """Build one captioned, letterboxed, GT-annotated tile for a single dataset."""
    dataset_yaml = resolve_dataset_yaml(key)
    if dataset_yaml is None:
        print(f"[WARN] Skipping '{key}': no converted dataset found yet.")
        return None

    config = load_dataset_config(dataset_yaml.parent, str(dataset_yaml))
    resolved_split = (
        split if split in config.split_entries else next(iter(config.split_entries))
    )
    image_paths = resolve_split_images(
        config.dataset_root, config.split_entries[resolved_split]
    )
    if not image_paths:
        print(f"[WARN] Skipping '{key}': no images found for split '{resolved_split}'.")
        return None

    candidates = select_candidates(image_paths, min_boxes)
    image_path, boxes = rng.choice(candidates)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[WARN] Skipping '{key}': could not read {image_path}.")
        return None

    labeled_image = draw_yolo_boxes(image, boxes, config.names)
    tile = letterbox_resize(labeled_image, tile_size)
    return caption_tile(tile, get_spec(key).display_name, tile_size)


def build_filmstrip(tiles: list[np.ndarray], tile_size: int, margin: int) -> np.ndarray:
    """Stitch captioned tiles into a single horizontal strip with margins."""
    tile_h = tiles[0].shape[0]
    canvas_w = len(tiles) * tile_size + (len(tiles) + 1) * margin
    canvas = np.full(
        (tile_h + 2 * margin, canvas_w, 3), CANVAS_BACKGROUND_COLOR, dtype=np.uint8
    )

    x = margin
    for tile in tiles:
        canvas[margin : margin + tile_h, x : x + tile_size] = tile
        x += tile_size + margin
    return canvas


def main() -> None:
    """Run the filmstrip-generation CLI entrypoint."""
    args = parse_args()
    rng = random.Random(args.seed)  # noqa: S311  # nosec B311

    tiles: list[np.ndarray] = []
    for key in list_datasets():
        tile = build_tile(key, args.split, args.min_boxes, args.tile_size, rng)
        if tile is not None:
            tiles.append(tile)

    if not tiles:
        raise RuntimeError("No converted datasets were found to build a filmstrip from")

    filmstrip = build_filmstrip(tiles, args.tile_size, args.margin)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), filmstrip)
    print(
        f"Filmstrip written to {output_path} "
        f"({filmstrip.shape[1]}x{filmstrip.shape[0]}, {len(tiles)} datasets)"
    )


if __name__ == "__main__":
    main()
