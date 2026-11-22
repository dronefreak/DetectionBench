#!/usr/bin/env python3
r"""
Print (and optionally attempt) official download instructions for a dataset.

For a DetectionBench-registered dataset that has **no** Hugging Face mirror.
Some registered datasets (UAVDT, DUO) are distributed under terms that are
either explicitly research-only or simply undocumented, with no redistribution
grant DetectionBench can act on -- see the per-dataset license notes in
``src/detectionbench/datasets/{uavdt,duo}.py`` and the project README. Rather
than host a mirror of questionable provenance, this script is the canonical
place DetectionBench tells you **where and how to get the raw data yourself**
from the original authors' own distribution channels.

Google Drive sources are downloaded automatically via ``gdown`` (install with
`pip install "detectionbench[download]"`) when available; it transparently
handles Drive's large-file "can't scan for viruses" confirmation-token flow
that a plain ``curl``/``wget`` cannot. Baidu Netdisk sources are **never**
automated (there's no reasonable way to script past their login/captcha) --
those are always printed as manual instructions.

This script only fetches the raw archive; run ``detectionbench-prepare-coco``
afterward to convert it into the canonical layout, exactly as for any other
registered dataset.

Usage:
  detectionbench-download-dataset --dataset uavdt --list
  detectionbench-download-dataset --dataset duo --output-dir ~/Downloads/datasets/DUO
  detectionbench-download-dataset --dataset uavdt --method manual
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from detectionbench.utils.utils import RichConsoleManager


@dataclass(frozen=True)
class DownloadOption:
    """One official way to obtain a dataset (or a piece of it)."""

    label: str
    method: str  # "gdrive" | "direct" | "baidu" | "manual"
    url: str
    note: str = ""
    gdrive_id: str | None = None  # required when method == "gdrive"
    access_code: str | None = None  # Baidu Netdisk share password, if any


@dataclass(frozen=True)
class DatasetSource:
    """Everything needed to point a user at a dataset's official distribution."""

    key: str
    display_name: str
    homepage: str
    license_note: str
    citation: str
    options: tuple[DownloadOption, ...]


_SOURCES: dict[str, DatasetSource] = {
    "uavdt": DatasetSource(
        key="uavdt",
        display_name="UAVDT",
        homepage="https://sites.google.com/view/grli-uavdt",
        license_note=(
            'Distributed "for research purpose only" -- no redistribution grant. '
            "Use locally for research; do not re-host."
        ),
        citation=(
            "Du et al., 'The Unmanned Aerial Vehicle Benchmark: Object "
            "Detection and Tracking', ECCV 2018 (arXiv:1804.00518)."
        ),
        options=(
            DownloadOption(
                label="UAVDT-Benchmark-M (detection/tracking frames + annotations)",
                method="gdrive",
                url="https://drive.google.com/file/d/1m8KA6oPIRK_Iwt9TYFquC87vBc_8wRVc/view",
                gdrive_id="1m8KA6oPIRK_Iwt9TYFquC87vBc_8wRVc",
                note="This is the one you want for object detection.",
            ),
            DownloadOption(
                label="UAVDT DET/MOT toolkit",
                method="gdrive",
                url="https://drive.google.com/open?id=19498uJd7T9w4quwnQEy62nibt3uyT9pq",
                gdrive_id="19498uJd7T9w4quwnQEy62nibt3uyT9pq",
                note="Optional -- official eval scripts, not needed by this adapter.",
            ),
            DownloadOption(
                label="UAVDT-Benchmark-S (single-object tracking only)",
                method="manual",
                url="https://drive.google.com/open?id=1661_Z_zL1HxInbsA2Mll9al-Ax6Py1rG",
                note="Not detection data -- skip unless you specifically need SOT.",
            ),
        ),
    ),
    "duo": DatasetSource(
        key="duo",
        display_name="DUO (Detecting Underwater Objects)",
        homepage="https://github.com/chongweiliu/DUO",
        license_note=(
            "No explicit license from the DUO authors or from the URPC contest "
            "data it re-annotates (URPC access historically required a signed "
            "data-use commitment letter). Use locally for research; do not re-host."
        ),
        citation=(
            "Liu et al., 'A Dataset and Benchmark of Underwater Object "
            "Detection for Robot Picking', ICME Workshops 2021 "
            "(arXiv:2106.05681)."
        ),
        options=(
            DownloadOption(
                label="DUO dataset (images + COCO annotations)",
                method="gdrive",
                url="https://drive.google.com/file/d/1w-bWevH7jFs7A1bIBlAOvXOxe2OFSHHs/view",
                gdrive_id="1w-bWevH7jFs7A1bIBlAOvXOxe2OFSHHs",
            ),
            DownloadOption(
                label="DUO dataset (Baidu Netdisk mirror)",
                method="baidu",
                url="https://pan.baidu.com/s/1Be8zc9UdR_Pdsyotg_vR2Q",
                access_code="4bfl",
                note="Requires a Baidu account; not automatable from this script.",
            ),
        ),
    ),
    "llvip": DatasetSource(
        key="llvip",
        display_name="LLVIP",
        homepage="https://github.com/bupt-ai-cz/LLVIP",
        license_note=(
            "Non-commercial academic/personal use only, attribution required, "
            "no redistribution grant (see the official 'Term of Use and "
            "License.md'). Use locally for research; do not re-host."
        ),
        citation=(
            "Jia et al., 'LLVIP: A Visible-infrared Paired Dataset for "
            "Low-light Vision', ICCV Workshops 2021."
        ),
        options=(
            DownloadOption(
                label="LLVIP dataset (visible + infrared images, VOC XML annotations)",
                method="gdrive",
                url="https://drive.google.com/file/d/1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv/view",
                gdrive_id="1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv",
            ),
            DownloadOption(
                label="LLVIP dataset (Baidu Netdisk mirror)",
                method="baidu",
                url="https://pan.baidu.com/s/1eQO1Is2NPyd-mgmv1Csbfg",
                access_code="14lc",
                note="Requires a Baidu account; not automatable from this script.",
            ),
        ),
    ),
    "sku110k": DatasetSource(
        key="sku110k",
        display_name="SKU-110K",
        homepage="https://github.com/eg4000/SKU110K_CVPR19",
        license_note=(
            '"Exclusive use by the recipient... solely for academic and '
            'non-commercial purposes" -- more restrictive than a generic '
            "non-commercial clause, no redistribution implied. Use locally "
            "for research; do not re-host."
        ),
        citation=(
            "Goldman et al., 'Precise Detection in Densely Packed Scenes', CVPR 2019."
        ),
        options=(
            DownloadOption(
                label="SKU-110K dataset (images + CSV annotations)",
                method="direct",
                url=(
                    "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/"
                    "SKU110K_fixed.tar.gz"
                ),
                note="Direct S3 URL -- no confirmation flow to handle.",
            ),
        ),
    ),
    "neudet": DatasetSource(
        key="neudet",
        display_name="NEU-DET",
        homepage="http://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/index.htm",
        license_note=(
            "No license stated anywhere -- citation-requested only, no "
            "redistribution grant. The original `yunhyan` faculty URL is "
            "dead; this is the maintainer's (Kechen Song) current homepage. "
            "Use locally for research; do not re-host."
        ),
        citation=(
            "Song, K. and Yan, Y., 'A noise robust method based on completed "
            "local binary patterns for hot-rolled steel strip surface "
            "defects', Applied Surface Science, 2013; He et al., 'An "
            "End-to-end Steel Surface Defect Detection Approach via Fusing "
            "Multiple Hierarchical Features', IEEE Trans. Instrumentation "
            "and Measurement, 2020."
        ),
        options=(
            DownloadOption(
                label="NEU-DET dataset (images + annotations)",
                method="gdrive",
                url="https://drive.google.com/open?id=1qrdZlaDi272eA79b0uCwwqPrm2Q_WI3k",
                gdrive_id="1qrdZlaDi272eA79b0uCwwqPrm2Q_WI3k",
            ),
            DownloadOption(
                label="NEU-DET dataset (Baidu Netdisk mirror)",
                method="baidu",
                url="https://pan.baidu.com/s/1nBbO-jWDm1_NHDQsc1dRkg",
                access_code="pmqx",
                note="Requires a Baidu account; not automatable from this script.",
            ),
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the dataset-download helper."""
    parser = argparse.ArgumentParser(
        description=(
            "Print (and optionally attempt) the official download instructions "
            "for a dataset with no DetectionBench Hugging Face mirror."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(_SOURCES),
        help="Registered dataset key",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to save the downloaded archive (default: ./downloads/<dataset>)",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "gdrive", "direct", "manual"],
        default="auto",
        help=(
            "'auto' attempts the first gdrive/direct option, falling back to "
            "manual instructions if gdown is unavailable or the download fails; "
            "'gdrive' requires gdown to succeed; 'manual' only prints instructions."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only print the known download options; don't attempt anything.",
    )
    return parser.parse_args()


def print_source_overview(source: DatasetSource) -> None:
    """Print the dataset's homepage, citation, license note, and every option."""
    console = RichConsoleManager.get_console()
    console.print(f"\n[bold cyan]{source.display_name}[/bold cyan]")
    console.print(f"  Homepage: {source.homepage}")
    console.print(f"  Citation: {source.citation}")
    console.print(f"  [bold yellow]License:[/bold yellow] {source.license_note}\n")
    console.print("[bold]Official download options:[/bold]")
    for i, option in enumerate(source.options, start=1):
        console.print(f"  {i}. [bold]{option.label}[/bold] ({option.method})")
        console.print(f"     {option.url}")
        if option.access_code:
            console.print(f"     Access code: [bold]{option.access_code}[/bold]")
        if option.note:
            console.print(f"     [dim]{option.note}[/dim]")


def print_manual_instructions(
    option: DownloadOption, output_dir: Path, dataset_key: str
) -> None:
    """Print step-by-step manual download instructions for one option."""
    console = RichConsoleManager.get_console()
    console.print(f"\n[bold]Manual download -- {option.label}[/bold]")
    console.print(f"  1. Open {option.url} in a browser (sign in if prompted).")
    if option.access_code:
        console.print(f"  2. Enter the access code: [bold]{option.access_code}[/bold]")
        console.print("  3. Download and extract the archive.")
    elif option.method == "gdrive":
        console.print(
            "  2. Click 'Download' -- for a large file, Google Drive shows a "
            "\"can't scan for viruses\" warning; click 'Download anyway'."
        )
        console.print("  3. Extract the archive.")
    else:
        console.print("  2. Follow the page's own instructions to obtain the file(s).")
    console.print(f"  4. Place the extracted contents under: {output_dir}")
    console.print(
        "  5. Then run: detectionbench-prepare-coco --dataset "
        f"{dataset_key} --raw-dir {output_dir} --output-dir <coco_out>"
    )


def try_gdrive_download(option: DownloadOption, output_dir: Path) -> bool:
    """Attempt a gdown-based download; return True on success, False to fall back."""
    console = RichConsoleManager.get_console()
    try:
        import gdown
    except ImportError:
        console.print(
            "[yellow]gdown is not installed -- install it with "
            '`pip install "detectionbench[download]"` for automated Google Drive '
            "downloads, or use --method manual.[/yellow]"
        )
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold cyan]Downloading via gdown:[/bold cyan] {option.label}")
    try:
        downloaded = gdown.download(
            id=option.gdrive_id, output=str(output_dir) + "/", quiet=False
        )
    except Exception as exc:  # noqa: BLE001 -- surface any gdown failure, then fall back
        console.print(f"[yellow]gdown download failed: {exc}[/yellow]")
        return False
    if not downloaded:
        console.print("[yellow]gdown reported no file was downloaded.[/yellow]")
        return False
    console.print(f"[bold green]Downloaded:[/bold green] {downloaded}")
    return True


def try_direct_download(option: DownloadOption, output_dir: Path) -> bool:
    """Attempt a plain HTTP(S) streamed download; return True on success."""
    console = RichConsoleManager.get_console()
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / option.url.rsplit("/", 1)[-1]
    console.print(f"[bold cyan]Downloading:[/bold cyan] {option.label}")
    try:
        import urllib.request

        with urllib.request.urlopen(option.url) as response, dest.open("wb") as out:  # noqa: S310  # nosec: B310
            shutil.copyfileobj(response, out)
    except Exception as exc:  # noqa: BLE001 -- surface any failure, then fall back
        dest.unlink(missing_ok=True)  # don't leave a truncated archive behind
        console.print(f"[yellow]Direct download failed: {exc}[/yellow]")
        return False
    console.print(f"[bold green]Downloaded:[/bold green] {dest}")
    return True


_AUTOMATABLE_METHODS = ("gdrive", "direct")


def main() -> None:
    """Run the dataset-download-instructions CLI entrypoint."""
    args = parse_args()
    console = RichConsoleManager.get_console()
    source = _SOURCES[args.dataset]
    print_source_overview(source)

    if args.list:
        return

    output_dir = (
        Path(args.output_dir) if args.output_dir else Path("downloads") / source.key
    )

    wanted_methods = _AUTOMATABLE_METHODS if args.method == "auto" else (args.method,)
    automatable = [o for o in source.options if o.method in wanted_methods]
    if args.method in _AUTOMATABLE_METHODS and not automatable:
        raise SystemExit(
            f"'{args.dataset}' has no {args.method} download option. Use "
            "--method auto or --method manual (see the options listed above)."
        )
    if args.method != "manual" and automatable:
        primary = automatable[0]
        downloader = (
            try_gdrive_download if primary.method == "gdrive" else try_direct_download
        )
        if downloader(primary, output_dir):
            console.print(
                f"\n[bold green]Done.[/bold green] Point --raw-dir at {output_dir} "
                "for detectionbench-prepare-coco."
            )
            return
        if args.method != "auto":
            raise RuntimeError(
                f"Automated download failed for '{args.dataset}'. Re-run with "
                "--method manual for step-by-step instructions."
            )

    for option in source.options:
        print_manual_instructions(option, output_dir, source.key)


if __name__ == "__main__":
    main()
