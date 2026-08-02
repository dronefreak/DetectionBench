#!/usr/bin/env python3
r"""
Generate a static rows x cols mosaic banner from a YOLO-format dataset.

Randomly samples images that actually have ground-truth boxes (so the
banner is visually representative rather than blank tiles), draws every
box + class name on each, center-crops them into equal square tiles (no
letterbox padding/borders), and composes the result into a grid -- useful
as a dataset-card banner image (e.g. for a Hugging Face dataset repo).

Works on any YOLO-format dataset (images/{split} + labels/{split} +
data.yaml), not just ones registered in DetectionBench -- it only relies
on the same generic YOLO-parsing helpers used by
``detectionbench.utils.convert_yolo_to_coco``.

Usage:
  python -m detectionbench.scripts.dataset_banner \
      /path/to/yolo_dataset --split train --output banner.jpg

  python -m detectionbench.scripts.dataset_banner \
      /path/to/yolo_dataset --split train --rows 4 --cols 4 \
      --tile-size 512 --min-boxes 2 --seed 42 --output banner.jpg
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import cv2
import numpy as np

from detectionbench.utils.convert_yolo_to_coco import (
    derive_label_path,
    load_dataset_config,
    resolve_split_images,
)

YOLO_LABEL_FIELD_COUNT = 5
DEFAULT_ROWS = 4
DEFAULT_COLS = 4
DEFAULT_TILE_SIZE = 512
TILE_BACKGROUND_COLOR = (30, 30, 30)
CANVAS_BACKGROUND_COLOR = (255, 255, 255)

YoloBox = tuple[int, float, float, float, float]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for banner generation."""
    parser = argparse.ArgumentParser(
        description="Generate an NxN mosaic banner from a YOLO-format dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dataset_root", help="YOLO dataset root (contains a data.yaml)")
    parser.add_argument(
        "--dataset-yaml",
        default=None,
        help="Path to the dataset YAML (default: auto-detect inside dataset_root)",
    )
    parser.add_argument("--split", default="train", help="Split to sample images from")
    parser.add_argument(
        "--rows", type=int, default=DEFAULT_ROWS, help="Number of grid rows"
    )
    parser.add_argument(
        "--cols", type=int, default=DEFAULT_COLS, help="Number of grid columns"
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help="Each tile's width/height in pixels",
    )
    parser.add_argument(
        "--min-boxes",
        type=int,
        default=1,
        help="Only sample images with at least this many labeled boxes",
    )
    parser.add_argument(
        "--margin", type=int, default=4, help="Pixel margin between/around tiles"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random sampling seed")
    parser.add_argument(
        "--output", default="banner.jpg", help="Output banner image path"
    )
    return parser.parse_args()


def load_yolo_boxes(label_path: Path) -> list[YoloBox]:
    """Parse a YOLO label file into (class_id, cx, cy, w, h) normalized tuples."""
    if not label_path.exists():
        return []

    boxes: list[YoloBox] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != YOLO_LABEL_FIELD_COUNT:
            continue
        try:
            class_id = int(parts[0])
            cx, cy, width, height = map(float, parts[1:])
        except ValueError:
            continue
        boxes.append((class_id, cx, cy, width, height))
    return boxes


def class_color(class_id: int) -> tuple[int, int, int]:
    """Deterministically derive a BGR color for a class id."""
    hue = (class_id * 47) % 180  # spread hues across OpenCV's [0, 180) range
    hsv_pixel = np.array([[[hue, 200, 230]]], dtype=np.uint8)
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])


def draw_yolo_boxes(
    image: np.ndarray, boxes: list[YoloBox], class_names: list[str]
) -> np.ndarray:
    """Draw every YOLO-format box + class name onto a copy of a BGR image."""
    out = image.copy()
    height, width = out.shape[:2]
    scale = max(height, width) / 1000.0
    thickness = max(1, round(scale * 2))
    font_scale = max(0.4, scale * 0.5)

    for class_id, cx, cy, box_w, box_h in boxes:
        x1 = int((cx - box_w / 2) * width)
        y1 = int((cy - box_h / 2) * height)
        x2 = int((cx + box_w / 2) * width)
        y2 = int((cy + box_h / 2) * height)
        color = class_color(class_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        name = (
            class_names[class_id] if class_id < len(class_names) else f"cls{class_id}"
        )
        (text_w, text_h), baseline = cv2.getTextSize(
            name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_y1 = max(0, y1 - text_h - baseline - 2)
        cv2.rectangle(out, (x1, label_y1), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(
            out,
            name,
            (x1 + 2, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return out


def center_crop_resize(image: np.ndarray, size: int) -> np.ndarray:
    """
    Resize an image to size x size, center-cropping to fill the tile edge-to-edge.

    Deliberately crops rather than letterboxes: padding to preserve aspect
    ratio leaves a solid-color bar on non-square source images (e.g. LISA's
    1280x960 frames), which reads as a stray border in the composed mosaic.
    A small amount of the longer edge is cropped away instead.
    """
    height, width = image.shape[:2]
    scale = size / min(height, width)
    new_w, new_h = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    y_offset = max(0, (new_h - size) // 2)
    x_offset = max(0, (new_w - size) // 2)
    return resized[y_offset : y_offset + size, x_offset : x_offset + size]


def build_mosaic(
    tiles: list[np.ndarray], rows: int, cols: int, tile_size: int, margin: int
) -> np.ndarray:
    """Compose tiles into a rows x cols mosaic with margins."""
    canvas_h = rows * tile_size + (rows + 1) * margin
    canvas_w = cols * tile_size + (cols + 1) * margin
    canvas = np.full((canvas_h, canvas_w, 3), CANVAS_BACKGROUND_COLOR, dtype=np.uint8)

    for index, tile in enumerate(tiles):
        row, col = divmod(index, cols)
        y = margin + row * (tile_size + margin)
        x = margin + col * (tile_size + margin)
        canvas[y : y + tile_size, x : x + tile_size] = tile

    return canvas


def stratified_sample(
    candidates: list[tuple[Path, list[YoloBox]]], num_needed: int, rng: random.Random
) -> list[tuple[Path, list[YoloBox]]]:
    """
    Pick num_needed candidates spread evenly across the (sorted) list.

    Fallback used when sequence-based grouping (see ``diverse_sample``)
    finds no real structure to group by. Splitting into num_needed
    contiguous chunks and taking one random pick per chunk at least spreads
    selections across the dataset's sort order, rather than a plain
    uniform sample that could land every pick in the same neighborhood.
    """
    if len(candidates) <= num_needed:
        return [rng.choice(candidates) for _ in range(num_needed)]

    chunk_size = len(candidates) / num_needed
    selected = []
    for i in range(num_needed):
        start = int(i * chunk_size)
        end = int((i + 1) * chunk_size) if i < num_needed - 1 else len(candidates)
        selected.append(rng.choice(candidates[start:end]))
    return selected


def _sequence_group_key(path: Path) -> str:
    """
    Best-effort grouping key for a frame-extracted filename's source sequence.

    Strips a trailing run of digits from the filename stem, so names like
    "nightClip2--01170.jpg" or "frame_0042.png" group by their shared
    sequence prefix ("nightClip2--", "frame_"). Filenames with no trailing
    digit run (e.g. hash-named files) fall back to their full stem, which
    effectively gives each such file its own group.
    """
    stem = path.stem
    return re.sub(r"\d+$", "", stem) or stem


def diverse_sample(
    candidates: list[tuple[Path, list[YoloBox]]], num_needed: int, rng: random.Random
) -> list[tuple[Path, list[YoloBox]]]:
    """
    Sample num_needed candidates, preferring one-per-source-sequence.

    Many datasets (dashcam/video-frame extractions especially) are
    dominated by a handful of long, visually-repetitive sequences --
    plain random or index-stratified sampling can easily draw several
    frames from the very same sequence, which look like near-duplicates
    in a banner. This groups candidates by ``_sequence_group_key`` and
    round-robins across groups (shuffled) so every group contributes a
    pick before any group contributes a second one. Falls back to
    ``stratified_sample`` if grouping finds no real structure (e.g. purely
    numeric filenames collapsing to one group).
    """
    groups: dict[str, list[tuple[Path, list[YoloBox]]]] = {}
    for candidate in candidates:
        groups.setdefault(_sequence_group_key(candidate[0]), []).append(candidate)

    group_keys = list(groups.keys())
    if len(group_keys) < 2:  # noqa: PLR2004 -- grouping found no real structure
        return stratified_sample(candidates, num_needed, rng)

    rng.shuffle(group_keys)
    pools = {key: list(items) for key, items in groups.items()}
    selected: list[tuple[Path, list[YoloBox]]] = []
    index = 0
    while len(selected) < num_needed and any(pools.values()):
        key = group_keys[index % len(group_keys)]
        pool = pools[key]
        if pool:
            selected.append(pool.pop(rng.randrange(len(pool))))
        index += 1
    return selected


def select_candidates(
    image_paths: list[Path], min_boxes: int
) -> list[tuple[Path, list[YoloBox]]]:
    """Pair each image with its parsed boxes, keeping only well-labeled ones."""
    labeled = [(path, load_yolo_boxes(derive_label_path(path))) for path in image_paths]
    candidates = [(path, boxes) for path, boxes in labeled if len(boxes) >= min_boxes]
    if not candidates:
        print(
            f"[WARN] No images with >= {min_boxes} boxes found; "
            "falling back to all images (including unlabeled)."
        )
        return labeled
    return candidates


def main() -> None:  # noqa: PLR0912
    """Run the dataset-banner CLI entrypoint."""
    args = parse_args()
    dataset_root = Path(args.dataset_root)

    config = load_dataset_config(dataset_root, args.dataset_yaml)
    if args.split not in config.split_entries:
        available = ", ".join(sorted(config.split_entries))
        raise KeyError(f"Split '{args.split}' not found. Available: {available}")

    image_paths = resolve_split_images(
        config.dataset_root, config.split_entries[args.split]
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found for split '{args.split}'.")

    candidates = select_candidates(image_paths, args.min_boxes)

    num_needed = args.rows * args.cols
    rng = random.Random(args.seed)  # noqa: S311  # nosec B311
    if len(candidates) < num_needed:
        print(
            f"[WARN] Only {len(candidates)} candidate image(s) for a "
            f"{args.rows}x{args.cols} grid ({num_needed} needed); "
            "some tiles will repeat."
        )
    selected = diverse_sample(candidates, num_needed, rng)

    tiles: list[np.ndarray] = []
    for image_path, boxes in selected:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] Could not read image, skipping: {image_path}")
            continue
        labeled_image = draw_yolo_boxes(image, boxes, config.names)
        tiles.append(center_crop_resize(labeled_image, args.tile_size))

    while len(tiles) < num_needed:
        tiles.append(
            np.full(
                (args.tile_size, args.tile_size, 3),
                TILE_BACKGROUND_COLOR,
                dtype=np.uint8,
            )
        )

    mosaic = build_mosaic(tiles, args.rows, args.cols, args.tile_size, args.margin)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mosaic)
    print(f"Banner written to {output_path} ({mosaic.shape[1]}x{mosaic.shape[0]})")


if __name__ == "__main__":
    main()
