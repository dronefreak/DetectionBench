#!/usr/bin/env python3
r"""
Generate a per-dataset statistics report (markdown + charts).

Reads every split's ``_annotations.coco.json`` (as produced by
``detectionbench-prepare-coco``) and writes a markdown report covering: a short
"what is this dataset for" description, images/instances per split, per-class
instance counts (with a bar chart), boxes-per-image density, bounding-box size
relative to the image, and a References section (citation, homepage, GitHub).
Datasets whose filenames encode a natural grouping (video sequence, tile,
location -- anything before the first ``_``) can also get a per-group
breakdown via ``--group-label``.

This is a generic bridge, like ``detectionbench-convert-coco-to-yolo`` and
``detectionbench-dataset-banner``: it works on any canonical COCO dataset
layout, not just registered ones -- it only relies on the COCO json schema
every adapter already writes. Pass ``--dataset <registered_key>`` to
auto-fill the description/citation/homepage/GitHub from that dataset's
``DatasetSpec`` (see ``detectionbench.datasets.base``); for an unregistered
dataset, pass ``--name`` plus any of ``--description``/``--citation``/
``--homepage``/``--github`` directly (all optional -- sections with nothing
to show are simply omitted).

Usage:
  python -m detectionbench.scripts.dataset_stats \
      --coco-dir /path/to/uavdt_coco \
      --dataset uavdt \
      --output-dir docs/datasets/uavdt \
      --group-label sequence
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402 -- backend must be set first

from detectionbench.datasets import get_spec, list_datasets
from detectionbench.datasets.base import COCO_ANNOTATION_FILENAME
from detectionbench.utils.utils import RichConsoleManager

SPLIT_ORDER = ("train", "valid", "test")

# dataviz skill palette: slot-1 (blue), nominal-categorical single-series bars
# use one consistent hue rather than one-per-bar (color would re-encode identity
# that bar length/position already shows).
BAR_COLOR = "#2a78d6"
SURFACE_COLOR = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e3e2dd"

TOP_N_GROUPS = 20


@dataclass
class ReportMeta:
    """Display name + descriptive metadata for the report header/footer."""

    name: str
    key: str | None = None  # registered dataset key, e.g. --dataset uavdt
    description: str | None = None
    homepage: str | None = None
    github: str | None = None
    citation: str | None = None

    @property
    def cli_key(self) -> str:
        """The value for --dataset on the CLI (the registry key, or a slug of name)."""
        return self.key or self.name.lower().replace(" ", "-")


@dataclass
class SplitStats:
    """Per-split image/instance/class counts for one COCO split file."""

    split: str
    n_images: int
    n_instances: int
    class_counts: Counter[str]
    boxes_per_image: list[int]
    box_area_pct: list[float]
    group_counts: Counter[str] = field(default_factory=Counter)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset-stats generation."""
    parser = argparse.ArgumentParser(
        description="Generate a markdown + chart statistics report for a COCO dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--coco-dir",
        required=True,
        help="Canonical COCO dataset root (contains train/, valid/, test/)",
    )
    parser.add_argument(
        "--dataset",
        choices=list_datasets(),
        default=None,
        help=(
            "Registered dataset key -- auto-fills description/citation/"
            "homepage/GitHub from its DatasetSpec. Omit for an unregistered "
            "dataset and pass --name (+ optionally --description etc.) instead."
        ),
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Display name, e.g. 'UAVDT' (default: from --dataset)",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Override/supply the 'what is this dataset' blurb",
    )
    parser.add_argument(
        "--homepage", default=None, help="Override/supply the project homepage"
    )
    parser.add_argument(
        "--github", default=None, help="Override/supply the GitHub repo URL"
    )
    parser.add_argument(
        "--citation", default=None, help="Override/supply the citation text"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the report + charts into",
    )
    parser.add_argument(
        "--group-label",
        default=None,
        help=(
            "If the dataset has a natural per-file grouping (video sequence, "
            "tile, location -- the filename prefix before the first '_'), a "
            "singular label for it (e.g. 'sequence') enables a per-group "
            "breakdown table + chart. Omit if there's no meaningful grouping."
        ),
    )
    args = parser.parse_args()
    if args.dataset is None and args.name is None:
        parser.error("either --dataset or --name is required")
    return args


def resolve_meta(args: argparse.Namespace) -> ReportMeta:
    """Build a ReportMeta from --dataset's DatasetSpec, with CLI overrides applied."""
    spec = get_spec(args.dataset) if args.dataset else None
    return ReportMeta(
        name=args.name or (spec.display_name if spec else args.dataset),
        key=args.dataset,
        description=args.description or (spec.description if spec else None),
        homepage=args.homepage or (spec.homepage if spec else None),
        github=args.github or (spec.github if spec else None),
        citation=args.citation or (spec.citation if spec else None),
    )


def _group_key(file_name: str) -> str:
    """
    Best-effort grouping key: the filename tokens before a running frame id.

    Splits the stem on '_' and '--' (both seen across adapters, e.g. LISA's
    'dayClip1--00000') and joins every token up to (not including) the first
    purely-numeric token -- that numeric token is almost always the running
    frame/image id, and everything before it is the sequence/source name
    (e.g. 'M0101_img000001' -> 'M0101', 'China_Drone_000001' -> 'China_Drone').
    Falls back to the first token when no purely-numeric token exists.
    """
    stem = Path(file_name).stem
    tokens = re.split(r"_|--", stem)
    digit_idx = next((i for i, t in enumerate(tokens) if t.isdigit()), None)
    if digit_idx:
        return "_".join(tokens[:digit_idx])
    return tokens[0]


def load_split(
    coco_dir: Path, split: str, *, group_label: str | None
) -> SplitStats | None:
    """Load one split's COCO json and compute its summary statistics."""
    annotation_path = coco_dir / split / COCO_ANNOTATION_FILENAME
    if not annotation_path.exists():
        return None
    data: dict[str, Any] = json.loads(annotation_path.read_text())

    cat_names = {c["id"]: c["name"] for c in data["categories"]}
    images = {im["id"]: im for im in data["images"]}

    class_counts: Counter[str] = Counter()
    per_image: Counter[int] = Counter()
    box_area_pct: list[float] = []
    for ann in data["annotations"]:
        class_counts[cat_names[ann["category_id"]]] += 1
        per_image[ann["image_id"]] += 1
        image = images.get(ann["image_id"])
        if image and image.get("width") and image.get("height"):
            box_w, box_h = ann["bbox"][2], ann["bbox"][3]
            box_area_pct.append(
                100.0 * (box_w * box_h) / (image["width"] * image["height"])
            )

    group_counts: Counter[str] = Counter()
    if group_label:
        for im in data["images"]:
            group_counts[_group_key(im["file_name"])] += 1

    return SplitStats(
        split=split,
        n_images=len(data["images"]),
        n_instances=len(data["annotations"]),
        class_counts=class_counts,
        boxes_per_image=[per_image[img_id] for img_id in images],
        box_area_pct=box_area_pct,
        group_counts=group_counts,
    )


def render_bar_chart(
    counts: dict[str, int], *, title: str, xlabel: str, output_path: Path
) -> None:
    """Render a single-hue horizontal bar chart (sorted descending) to output_path."""
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    fig_height = max(2.2, 0.4 * len(labels) + 1.0)
    fig, ax = plt.subplots(figsize=(8, fig_height), dpi=160)
    fig.patch.set_facecolor(SURFACE_COLOR)
    ax.set_facecolor(SURFACE_COLOR)

    y_pos = range(len(labels))
    bars = ax.barh(y_pos, values, color=BAR_COLOR, height=0.6, zorder=3)
    ax.set_yticks(list(y_pos), labels=labels, color=TEXT_PRIMARY, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=12, fontweight="bold", loc="left")

    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=TEXT_SECONDARY, labelsize=8)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=1, zorder=0)
    ax.set_axisbelow(True)

    max_value = max(values) if values else 0
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_width() + max_value * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}",
            va="center",
            ha="left",
            fontsize=8,
            color=TEXT_PRIMARY,
        )
    ax.set_xlim(0, max_value * 1.15 if max_value else 1)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=SURFACE_COLOR)
    plt.close(fig)


def build_markdown(  # noqa: PLR0913
    *,
    meta: ReportMeta,
    splits: list[SplitStats],
    class_chart_path: str,
    group_label: str | None,
    group_chart_path: str | None,
    reproduce_cmd: str,
    banner_path: str | None = None,
) -> str:
    """Assemble the markdown report body from computed split statistics."""
    total_images = sum(s.n_images for s in splits)
    total_instances = sum(s.n_instances for s in splits)
    total_class_counts: Counter[str] = Counter()
    for s in splits:
        total_class_counts.update(s.class_counts)
    all_box_pct = [pct for s in splits for pct in s.box_area_pct]
    all_boxes_per_image = [n for s in splits for n in s.boxes_per_image]

    lines: list[str] = [f"# {meta.name}: Dataset Statistics", ""]
    lines += [f"![{meta.name} sample images]({banner_path})", ""] if banner_path else []
    if meta.description:
        lines += ["## About", "", meta.description, ""]
    lines += [
        f"Computed from the canonical COCO layout produced by "
        f"`detectionbench-prepare-coco --dataset {meta.cli_key}`. "
        "See the adapter and Hydra config for how these splits are built.",
        "",
        "## Split Summary",
        "",
        "| Split | Images | Instances | Instances / Image |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for s in splits:
        avg = s.n_instances / s.n_images if s.n_images else 0.0
        lines.append(f"| {s.split} | {s.n_images:,} | {s.n_instances:,} | {avg:.2f} |")
    lines.append(
        f"| **Total** | **{total_images:,}** | **{total_instances:,}** | "
        f"**{(total_instances / total_images if total_images else 0):.2f}** |"
    )

    lines += [
        "",
        "## Class Distribution",
        "",
        f"![Class distribution]({class_chart_path})",
        "",
        "| Class | Instances | Share |",
        "| :--- | ---: | ---: |",
    ]
    for cls, count in sorted(
        total_class_counts.items(), key=lambda kv: kv[1], reverse=True
    ):
        share = 100.0 * count / total_instances if total_instances else 0.0
        lines.append(f"| {cls} | {count:,} | {share:.1f}% |")

    if group_label and group_chart_path:
        total_group_counts: Counter[str] = Counter()
        for s in splits:
            total_group_counts.update(s.group_counts)
        n_groups = len(total_group_counts)
        lines += [
            "",
            f"## Per-{group_label.title()} Breakdown",
            "",
            f"{n_groups} distinct {group_label}s across all splits"
            + (
                f" (top {TOP_N_GROUPS} shown below by image count)"
                if n_groups > TOP_N_GROUPS
                else ""
            )
            + ".",
            "",
            f"![Per-{group_label} image counts]({group_chart_path})",
        ]

    if all_box_pct:
        lines += [
            "",
            "## Bounding Box Geometry",
            "",
            f"- Median box area: **{statistics.median(all_box_pct):.2f}%** of image "
            f"area (mean {statistics.mean(all_box_pct):.2f}%)",
        ]
    if all_boxes_per_image:
        mean_bpi = statistics.mean(all_boxes_per_image)
        median_bpi = statistics.median(all_boxes_per_image)
        lines += [
            f"- Instances per image: mean **{mean_bpi:.2f}**, median "
            f"**{median_bpi:.0f}**, max **{max(all_boxes_per_image)}**",
        ]

    if meta.citation or meta.homepage or meta.github:
        lines += ["", "## References", ""]
        if meta.citation:
            lines += ["**Citation:**", "", "```bibtex", meta.citation, "```", ""]
        if meta.homepage:
            lines.append(f"- Project website: <{meta.homepage}>")
        if meta.github:
            lines.append(f"- GitHub: <{meta.github}>")

    lines += [
        "",
        "---",
        "",
        f"Regenerate this report with:\n```bash\n{reproduce_cmd}\n```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Run the dataset-stats CLI entrypoint."""
    args = parse_args()
    console = RichConsoleManager.get_console()
    coco_dir = Path(args.coco_dir)
    output_dir = Path(args.output_dir)
    meta = resolve_meta(args)

    splits = [
        stats
        for split in SPLIT_ORDER
        if (stats := load_split(coco_dir, split, group_label=args.group_label))
        is not None
    ]
    if not splits:
        raise FileNotFoundError(
            f"No {COCO_ANNOTATION_FILENAME} files found under {coco_dir}"
        )

    total_class_counts: Counter[str] = Counter()
    for s in splits:
        total_class_counts.update(s.class_counts)
    class_chart_path = output_dir / "class_distribution.png"
    render_bar_chart(
        dict(total_class_counts),
        title=f"{meta.name}: instances per class",
        xlabel="instances",
        output_path=class_chart_path,
    )

    group_chart_path = None
    if args.group_label:
        total_group_counts: Counter[str] = Counter()
        for s in splits:
            total_group_counts.update(s.group_counts)
        top_groups = dict(
            sorted(total_group_counts.items(), key=lambda kv: kv[1], reverse=True)[
                :TOP_N_GROUPS
            ]
        )
        group_chart_path = output_dir / f"{args.group_label}_breakdown.png"
        render_bar_chart(
            top_groups,
            title=f"{meta.name}: images per {args.group_label}",
            xlabel="images",
            output_path=group_chart_path,
        )

    dataset_flag = (
        f"--dataset {args.dataset}" if args.dataset else f"--name {meta.name}"
    )
    reproduce_cmd = (
        f"python -m detectionbench.scripts.dataset_stats "
        f"--coco-dir <{meta.cli_key}_coco> {dataset_flag} "
        f"--output-dir {output_dir}"
        + (f" --group-label {args.group_label}" if args.group_label else "")
    )
    # A ground-truth mosaic (detectionbench-dataset-banner) placed next to the
    # report as <key>_banner.jpg is embedded under the title.
    banner = output_dir / f"{meta.cli_key}_banner.jpg"
    markdown = build_markdown(
        meta=meta,
        splits=splits,
        class_chart_path=class_chart_path.name,
        group_label=args.group_label,
        group_chart_path=group_chart_path.name if group_chart_path else None,
        reproduce_cmd=reproduce_cmd,
        banner_path=banner.name if banner.is_file() else None,
    )
    report_path = output_dir / "README.md"
    report_path.write_text(markdown)
    console.print(f"[bold green]✓[/bold green] Wrote {report_path}")


if __name__ == "__main__":
    main()
