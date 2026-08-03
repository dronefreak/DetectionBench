#!/usr/bin/env python3
r"""
Prepare a raw dataset download into the canonical COCO layout.

Dispatches to the registered ``detectionbench.datasets`` adapter for the
requested dataset, which knows how to translate that dataset's raw format
into ``output_dir/{train,valid,test}/_annotations.coco.json`` (+ images).
From there, use ``detectionbench-convert-coco-to-yolo`` to bridge into
YOLO format for Ultralytics models, or train/evaluate RF-DETR directly
against the canonical COCO output.

This is the "prepare" family's COCO-output member (console script:
``detectionbench-prepare-coco``) -- a future non-COCO target (e.g. VOC)
would be a sibling ``detectionbench-prepare-voc`` sharing the same
``--dataset``/``--raw-dir``/``--output-dir`` interface.

Usage:
  python -m detectionbench.scripts.prepare_coco \\
      --dataset doclaynet \\
      --raw-dir /path/to/DocLayNet_core \\
      --output-dir /path/to/doclaynet_coco
"""

from __future__ import annotations

import argparse
from pathlib import Path

from detectionbench.datasets import get, list_datasets
from detectionbench.utils.utils import RichConsoleManager


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare a raw dataset download into the canonical COCO layout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list_datasets(),
        help="Registered dataset key",
    )
    parser.add_argument(
        "--raw-dir", required=True, help="Raw dataset download directory"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for the canonical COCO layout",
    )
    return parser.parse_args()


def main() -> None:
    """Run the dataset preparation CLI entrypoint."""
    args = parse_args()
    console = RichConsoleManager.get_console()

    adapter = get(args.dataset)
    console.print(f"[bold cyan]Preparing '{adapter.spec.display_name}'[/bold cyan]")
    console.print(f"  Raw dir:    {args.raw_dir}")
    console.print(f"  Output dir: {args.output_dir}")
    console.print(f"  Classes:    {adapter.spec.num_classes}")

    adapter.prepare_coco(Path(args.raw_dir), Path(args.output_dir))

    console.print(f"\n✓ Canonical COCO layout written to: {args.output_dir}")


if __name__ == "__main__":
    main()
