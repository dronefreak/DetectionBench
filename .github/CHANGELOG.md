# Changelog

All notable changes to this project will be documented in this file.

The format is based on **Keep a Changelog**,
and this project adheres to **Semantic Versioning**.

* Keep a Changelog: [https://keepachangelog.com/en/1.1.0/](https://keepachangelog.com/en/1.1.0/)
* Semantic Versioning: [https://semver.org/](https://semver.org/)

---

## [Unreleased]

### Added

* Initial project structure
* Pre-commit hooks for linting, formatting, typing, and security
* Documentation and governance templates
* Dataset statistics pages embed `<key>_banner.jpg` when present (BDD100K, CeyMo, HRP4K, KITTI, PKLot)
* CPU-only tests for `inference`, `benchmark`, `download_dataset` and `dataset_stats`

### Changed

* YOLO configs pin `training.seed: 0`, forwarded to Ultralytics `train(seed=...)` and `seed_everything` (previously a hard-coded 42 that Ultralytics overrode with its own 0)
* `evaluate_yolo` `--score-threshold` / `--iou-threshold` / `--batch-size` / `--num-workers` now reach `model.val()`; unset keeps Ultralytics defaults, matching published numbers
* `detectionbench-infer --imgsz` defaults to the checkpoint's training size (was 1280); RF-DETR inference uses the resolution in `training_config.json`
* `detectionbench-benchmark` runs RF-DETR at the checkpoint's training resolution (or family default) unless `--imgsz` is given; YOLO stays at 640
* README dataset footnotes condensed; per-split counts live in `docs/datasets/<key>/`

### Deprecated

* N/A

### Removed

* Unused `evaluation.{batch_size,num_workers,score_threshold,iou_threshold,soft_nms}` keys from YOLO configs
* `detectionbench-generate-model-cards` / `detectionbench-generate-readme-leaderboard` console scripts (maintainer-only tools, not shipped in the repo)

### Fixed

* YOLO training printed the whole model config as the results-table title
* `rfdetr_evaluate.yaml` ignored `dataset.eval_split`; RF-DETR eval now maps `val` to the canonical `valid/` dir (SeaDronesSee)
* `detectionbench-convert-coco-to-yolo` warns on missing images and writes `data.yaml` via `yaml.safe_dump`
* `--help` usage examples printed `\\` line continuations
* mypy errors in `generate_model_cards.py`
* YOLO inference on a video saved one JPEG per frame under the same name (only the last survived); now writes `<stem>_pred.mp4` and streams results
* Video output was silently skipped when OpenCV reported an unknown frame rate (-1)
* `detectionbench-benchmark` silently benchmarked RF-DETR Large for any unrecognised RF-DETR name
* `detectionbench-download-dataset`: manual instructions printed a literal `<key>`; failed direct downloads left partial files; `--method gdrive` without a Drive option fell through silently
* README Source links pointed at the untracked `dataset_cards/` directory (404 on GitHub)

### Security

* N/A

---

## [0.1.0] - YYYY-MM-DD

### Added

* Initial public release

---

## Versioning Policy

This project follows **Semantic Versioning**:

* **MAJOR** version for incompatible API changes
* **MINOR** version for backward-compatible functionality
* **PATCH** version for backward-compatible bug fixes

---

## How to Update This File

* Update **[Unreleased]** as part of each PR
* Move entries to a new version section at release time
* Keep entries concise and user-focused
* Avoid low-level implementation details

---

## Types of Changes

* **Added** – New features
* **Changed** – Changes in existing functionality
* **Deprecated** – Soon-to-be removed features
* **Removed** – Removed features
* **Fixed** – Bug fixes
* **Security** – Vulnerability-related changes
