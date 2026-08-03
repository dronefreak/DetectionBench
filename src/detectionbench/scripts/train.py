r"""
Unified `detectionbench-train` entrypoint.

Dispatches to the Ultralytics (YOLO/RT-DETR) or RF-DETR trainer based on the
`model.name=...` (or `model=...`) Hydra override on the command line, so
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
"""

from __future__ import annotations

import re
import sys

_MODEL_OVERRIDE_PATTERN = re.compile(r"^model(?:\.name)?=(.+)$")


def _peek_model_name(argv: list[str]) -> str:
    """Scan CLI args for a `model.name=...`/`model=...` override, unparsed by Hydra."""
    for arg in argv:
        match = _MODEL_OVERRIDE_PATTERN.match(arg)
        if match:
            return match.group(1)
    return ""


def main() -> None:
    """Dispatch `detectionbench-train` to the Ultralytics or RF-DETR trainer."""
    model_name = _peek_model_name(sys.argv[1:]).lower()

    if model_name.startswith(("yolo", "rtdetr")) or not model_name:
        from detectionbench.scripts.train_yolo import main as yolo_main

        yolo_main()
    else:
        from detectionbench.scripts.train_rfdetr import main as rfdetr_main

        rfdetr_main()


if __name__ == "__main__":
    main()
