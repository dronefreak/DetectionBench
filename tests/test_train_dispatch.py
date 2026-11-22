"""Tests for the `detectionbench-train` argv dispatcher (no training runs)."""

import sys
import types
from pathlib import Path

import pytest

from detectionbench.scripts import train


@pytest.fixture
def called(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace both trainers with recorders (imported lazily by train.main)."""
    calls: list[str] = []

    def _module(name: str) -> types.SimpleNamespace:
        return types.SimpleNamespace(main=lambda: calls.append(name))

    monkeypatch.setitem(
        sys.modules, "detectionbench.scripts.train_yolo", _module("yolo")
    )
    monkeypatch.setitem(
        sys.modules, "detectionbench.scripts.train_rfdetr", _module("rfdetr")
    )
    return calls


def _dispatch(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["detectionbench-train", *argv])
    train.main()


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], "yolo"),  # configs/config.yaml default
        (["dataset=kitti", "model.name=yolov8n"], "yolo"),
        (["model.name=RTDETR-L"], "yolo"),
        (["model.name=rfdetr-nano"], "rfdetr"),
        (["model.name=nano"], "rfdetr"),
        (["model=rfdetr-small"], "rfdetr"),
        (["++model.name=rfdetr-nano"], "rfdetr"),
        (["+model.name='rfdetr-small'"], "rfdetr"),
        (["--config-name", "bdd100k_rfdetr"], "rfdetr"),
        (["--config-name=visdrone_rfdetr", "dataset=visdrone"], "rfdetr"),
        (["-cn", "kitti_yolo"], "yolo"),
        (["--config-name", "rfdetr.yaml"], "rfdetr"),
        # an explicit override beats the config file's model.name
        (["--config-name", "bdd100k_rfdetr", "model.name=yolo11n"], "yolo"),
    ],
)
def test_dispatches_to_the_right_trainer(
    monkeypatch: pytest.MonkeyPatch, called: list[str], argv: list[str], expected: str
) -> None:
    _dispatch(monkeypatch, *argv)
    assert called == [expected]


def test_unknown_model_exits(
    monkeypatch: pytest.MonkeyPatch, called: list[str]
) -> None:
    with pytest.raises(SystemExit, match="Unknown model 'fasterrcnn'"):
        _dispatch(monkeypatch, "model.name=fasterrcnn")
    assert called == []


def test_other_overrides_starting_with_model_are_ignored() -> None:
    assert train._peek_model_name(["model.resolution=512", "models=x"]) == ""


def test_every_shipped_training_config_dispatches_to_its_family() -> None:
    names = sorted(
        p.stem
        for p in train.CONFIGS_DIR.glob("*.yaml")
        if p.stem not in {"export_onnx", "rfdetr_evaluate"}
    )
    assert names
    for name in names:
        family = "rfdetr" if "rfdetr" in name else "yolo"
        model = train._config_model_name(name)
        assert model, name
        assert model.startswith("rfdetr" if family == "rfdetr" else ("yolo", "rtdetr"))


def test_missing_config_file_has_no_model(tmp_path: Path) -> None:
    assert train._config_model_name("does_not_exist") == ""
    assert train._config_model_name("") == ""
