# DetectionBench

> Reproducible benchmarks for modern object detectors on real-world datasets.

DetectionBench is a Hydra-driven framework for training, evaluating, and
benchmarking object detection models with identical recipes and metrics
across multiple datasets. It wraps two model families behind one CLI:

- **Ultralytics YOLO / RT-DETR** (`yolov8`, `yolov9`, `yolo11`, `rtdetr`, ...)
- **RF-DETR** (nano / small / medium / large)

## Datasets

| Dataset | Domain | Status |
| --- | --- | --- |
| DocLayNet | Document layout analysis | ✅ Working end-to-end |
| SeaDronesSee | Maritime UAV | 🚧 Adapter stub, raw data not yet available |
| Global Wheat Head Dataset (GWHD) | Agriculture | 🚧 Adapter stub, raw data not yet available |
| ExDark | Low-light robustness | 🚧 Adapter stub, raw data not yet available |
| Aquarium | Underwater imagery | 🚧 Adapter stub, raw data not yet available |

Each dataset is a self-contained adapter under `src/detectionbench/datasets/`
that converts its raw format into a canonical COCO layout; everything
downstream (COCO↔YOLO conversion, training, evaluation, inference,
benchmarking) is dataset-agnostic. See `src/detectionbench/datasets/doclaynet.py`
for a fully worked adapter, and `Objectives.md` for the project roadmap.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + dev tooling
pip install -e ".[rfdetr]"       # + RF-DETR training/eval
pip install -e ".[coco]"         # + pycocotools
pip install -e ".[benchmark]"    # + hardware-profiling extras (psutil, thop, fvcore, torchinfo, nvidia-ml-py)
```

## Usage

```bash
# 1. Convert a raw dataset download into the canonical COCO layout
detectionbench-prepare --dataset doclaynet --raw-dir /path/to/DocLayNet_core --output-dir /path/to/doclaynet_coco

# 2. Bridge into Ultralytics YOLO format (for YOLO/RT-DETR training)
detectionbench-convert-yolo /path/to/doclaynet_coco /path/to/doclaynet_yolo

# 3. Train + evaluate (Hydra config group `dataset=<key>` selects the dataset)
detectionbench-train dataset=doclaynet model.name=yolov8n
detectionbench-train-rfdetr dataset=doclaynet model.name=rfdetr-nano

# 4. Evaluate / infer / benchmark a checkpoint directly
detectionbench-evaluate --checkpoint experiments/yolov8n/weights/best.pt --model yolov8n --dataset doclaynet --dataset-yaml /path/to/doclaynet_yolo/data.yaml
detectionbench-infer --checkpoint experiments/yolov8n/weights/best.pt --model yolov8n --dataset doclaynet --input /path/to/images
detectionbench-benchmark --model yolov8n --checkpoint experiments/yolov8n/weights/best.pt
```

## Roadmap

DetectionBench is being built in phases (see `Objectives.md` for the full plan):

1. **Framework** (current): dataset abstraction, training/evaluation wrappers, hardware profiling.
2. **First public release**: full benchmark suite (~15 checkpoints) on SeaDronesSee, published to GitHub + Hugging Face with a leaderboard and report.
3. **Expansion**: repeat for GWHD, ExDark, and Aquarium.

Leaderboards, benchmark reports, and Hugging Face model links will be added here as each dataset's benchmark suite is published.

## License

Code is licensed under Apache-2.0 (see `LICENSE`). Each benchmarked dataset
retains its own original license — see the corresponding adapter's docstring
under `src/detectionbench/datasets/` and `CITATION.cff`.
