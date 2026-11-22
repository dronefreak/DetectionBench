"""Tests for the `detectionbench-prepare-coco` CLI dispatch to dataset adapters."""

import sys
from pathlib import Path

import pytest

from detectionbench.datasets import get
from detectionbench.scripts import prepare_coco


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["detectionbench-prepare-coco", *argv])
    prepare_coco.main()


def test_dispatches_to_the_registered_adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Path, Path]] = []
    adapter = get("kitti")
    monkeypatch.setattr(
        type(adapter), "prepare_coco", lambda self, raw, out: calls.append((raw, out))
    )

    _run(
        monkeypatch,
        "--dataset",
        "kitti",
        "--raw-dir",
        str(tmp_path / "raw"),
        "--output-dir",
        str(tmp_path / "coco"),
    )
    assert calls == [(tmp_path / "raw", tmp_path / "coco")]


def test_rejects_unregistered_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        _run(
            monkeypatch,
            "--dataset",
            "imagenet",
            "--raw-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path),
        )
    assert "invalid choice: 'imagenet'" in capsys.readouterr().err
