r"""
Unified `detectionbench-train` entrypoint.

Dispatches to the Ultralytics (YOLO/RT-DETR) or RF-DETR trainer based on the
`model.name=...` (or `model=...`) Hydra override on the command line -- or,
without one, the `model.name` of the `--config-name` config -- so
callers only ever need one command regardless of model family -- adding a
new Ultralytics-registered checkpoint name (or a new RF-DETR size) never
needs a new entrypoint.

This has to be a thin argv-peeking dispatcher rather than a single
`@hydra.main`-decorated function: the Ultralytics and RF-DETR training
paths use genuinely different Hydra config schemas (`configs/config.yaml`
vs `configs/rfdetr.yaml` -- different `training`/`evaluation` trees
entirely), and Hydra's `config_name` is fixed at decoration time. This
module decides which schema applies by scanning `sys.argv` for a
`model.name=`/`model=` override *before* either underlying `@hydra.main`
function runs its own parsing -- it never touches Hydra itself.

Usage:
  detectionbench-train dataset=doclaynet model.name=yolov8n
  detectionbench-train dataset=doclaynet model.name=rfdetr-nano
  detectionbench-train --config-name bdd100k_rfdetr
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

from detectionbench.utils.rfdetr import RFDETR_MODEL_ALIASES

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"
_CONFIG_NAME_FLAGS = ("--config-name", "-cn")

# Hydra also accepts `+`/`++`-prefixed overrides and quoted values.
_MODEL_OVERRIDE_PATTERN = re.compile(r"^\+{0,2}model(?:\.name)?=(.+)$")


def _peek_model_name(argv: list[str]) -> str:
    """Scan CLI args for a `model.name=...`/`model=...` override, unparsed by Hydra."""
    for arg in argv:
        match = _MODEL_OVERRIDE_PATTERN.match(arg)
        if match:
            return match.group(1).strip("\"'")
    return ""


def _peek_config_name(argv: list[str]) -> str:
    """Return the `--config-name`/`-cn` value, if given."""
    for i, arg in enumerate(argv):
        for flag in _CONFIG_NAME_FLAGS:
            if arg == flag and i + 1 < len(argv):
                return argv[i + 1]
            if arg.startswith(f"{flag}="):
                return arg.split("=", 1)[1]
    return ""


def _config_model_name(config_name: str) -> str:
    """Read `model.name` from a `configs/<config_name>.yaml`, or "" if absent."""
    name = config_name.removesuffix(".yaml")
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.is_file():
        return ""
    model = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("model")
    return str(model.get("name") or "") if isinstance(model, dict) else ""


def _resolve_model_name(argv: list[str]) -> str:
    """CLI override first, else the selected config's `model.name`."""
    name = _peek_model_name(argv) or _config_model_name(_peek_config_name(argv))
    return name.strip().lower()


def main() -> None:
    """Dispatch `detectionbench-train` to the Ultralytics or RF-DETR trainer."""
    model_name = _resolve_model_name(sys.argv[1:])

    if model_name.startswith(("yolo", "rtdetr")) or not model_name:
        from detectionbench.scripts.train_yolo import main as yolo_main

        yolo_main()
    elif model_name.startswith("rfdetr") or model_name in RFDETR_MODEL_ALIASES:
        from detectionbench.scripts.train_rfdetr import main as rfdetr_main

        rfdetr_main()
    else:
        raise SystemExit(
            f"Unknown model '{model_name}'. Expected an Ultralytics name "
            "(yolo*, rtdetr*) or an RF-DETR name (rfdetr-nano/small/medium/large)."
        )


if __name__ == "__main__":
    main()
