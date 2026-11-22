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

### Changed

* YOLO configs pin `training.seed: 0`, forwarded to Ultralytics `train(seed=...)` and `seed_everything` (previously a hard-coded 42 that Ultralytics overrode with its own 0)
* `evaluate_yolo` `--score-threshold` / `--iou-threshold` / `--batch-size` / `--num-workers` now reach `model.val()`; unset keeps Ultralytics defaults, matching published numbers

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
