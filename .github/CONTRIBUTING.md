# Contributing

Thanks for your interest in contributing! DetectionBench provides Hydra-based
training, evaluation, and inference tooling for reproducibly benchmarking modern
object detectors (YOLO, RT-DETR, RF-DETR) across a growing set of real-world
detection datasets (currently DocLayNet, with SeaDronesSee, GWHD, ExDark, and
Aquarium adapters in progress -- see `Objectives.md`).

Adding a new dataset means implementing one adapter under
`src/detectionbench/datasets/` (see `doclaynet.py` for a worked example) plus a
matching `configs/dataset/<name>.yaml` entry -- it should not require touching
the training/evaluation scripts themselves.

## Getting Started

1. Fork and clone the repository.
2. Create a virtual environment and install in editable mode:

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

## Making Changes

- Create a feature branch off `main`: `git checkout -b feature/my-change`.
- Keep changes focused and scoped to a single concern.
- Follow existing code style; `pre-commit` runs linting/formatting automatically on commit.
- Add or update tests where relevant.
- Run `pre-commit run --all-files` before opening a PR.

## Submitting a Pull Request

- Describe what the change does and why.
- Reference any related issues.
- Ensure CI checks pass.
- Be responsive to review feedback.

## Reporting Issues

Please open a GitHub issue with:

- A clear description of the problem or request.
- Steps to reproduce (for bugs), including config overrides used.
- Relevant logs/stack traces.

## Note on the Datasets

This repository only contains benchmarking **code** (Apache-2.0 licensed). Every
benchmarked dataset (DocLayNet, SeaDronesSee, GWHD, ExDark, Aquarium, ...) is a
third-party dataset redistributed separately under its own original license —
see each adapter's docstring under `src/detectionbench/datasets/` and
`CITATION.cff` for details. Please do not open dataset-content issues here;
direct those to the original authors.

By contributing, you agree that your contributions will be licensed under the
Apache License, Version 2.0.
