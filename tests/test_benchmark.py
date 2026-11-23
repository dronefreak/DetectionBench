"""CPU-only tests for detectionbench.scripts.benchmark (fake RF-DETR, no timing)."""

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

from detectionbench.scripts import benchmark

_FAMILY_DEFAULT = 384


class _FakeRFDETR:
    kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).kwargs = kwargs
        self.model_config = types.SimpleNamespace(
            resolution=kwargs.get("resolution", _FAMILY_DEFAULT)
        )
        self.model = types.SimpleNamespace(model=torch.nn.Identity())

    def optimize_for_inference(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _no_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "rfdetr",
        types.SimpleNamespace(RFDETRNano=_FakeRFDETR, RFDETRLarge=_FakeRFDETR),
    )
    monkeypatch.setattr(benchmark, "_flops_thop", lambda model, imgsz, bs: (1.0, 10))
    monkeypatch.setattr(
        benchmark,
        "_measure_latency",
        lambda fn, device, warmup, iterations: {"mean_ms": 10.0},
    )
    monkeypatch.setattr(benchmark, "_vram_info", lambda device: {})
    monkeypatch.setattr(benchmark, "_gpu_core_util_pct", lambda handle: float("nan"))
    _FakeRFDETR.kwargs = {}


def _args(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "model": "rfdetr-nano",
        "dataset": "kitti",
        "checkpoint": None,
        "no_checkpoint": False,
        "imgsz": None,
        "batch_size": 1,
        "iterations": 1,
        "warmup": 0,
        "device": "cpu",
        "summary": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_rfdetr_defaults_to_family_resolution_without_checkpoint() -> None:
    result = benchmark._benchmark_rfdetr(_args(), None)
    assert "resolution" not in _FakeRFDETR.kwargs
    assert result["imgsz"] == _FAMILY_DEFAULT


def test_rfdetr_uses_checkpoint_training_resolution(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint_best_total.pth"
    checkpoint.write_bytes(b"")
    (tmp_path / "training_config.json").write_text(
        json.dumps({"model_config": {"resolution": 576}})
    )
    result = benchmark._benchmark_rfdetr(_args(checkpoint=str(checkpoint)), None)
    assert _FakeRFDETR.kwargs["resolution"] == 576
    assert result["imgsz"] == 576


def test_rfdetr_explicit_imgsz_wins() -> None:
    result = benchmark._benchmark_rfdetr(_args(imgsz=512), None)
    assert result["imgsz"] == 512


@pytest.mark.parametrize("name", ["rfdetr_nano", "RFDETR-Nano", "nano"])
def test_rfdetr_aliases_resolve_to_the_right_class(name: str) -> None:
    benchmark._benchmark_rfdetr(_args(model=name), None)
    assert _FakeRFDETR.kwargs["num_classes"] == 8  # kitti


def test_unknown_rfdetr_name_is_an_error_not_large() -> None:
    with pytest.raises(ValueError, match="Unknown RF-DETR model"):
        benchmark._benchmark_rfdetr(_args(model="rfdetr-huge"), None)


def test_rfdetr_requires_dataset() -> None:
    with pytest.raises(ValueError, match="--dataset is required"):
        benchmark._benchmark_rfdetr(_args(dataset=None), None)


@pytest.mark.parametrize(
    ("model", "expected"),
    [("yolov8n", "ultralytics"), ("rtdetr-l", "ultralytics"), ("nano", "rfdetr")],
)
def test_main_dispatch_and_json_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model: str, expected: str
) -> None:
    called: list[str] = []
    fake_result = {"model": model, "mean_ms": 1.0, "fps": float("nan")}

    def _recorder(family: str):
        def _run(args: argparse.Namespace, handle: object) -> dict[str, Any]:
            called.append(family)
            return fake_result

        return _run

    monkeypatch.setattr(benchmark, "_benchmark_ultralytics", _recorder("ultralytics"))
    monkeypatch.setattr(benchmark, "_benchmark_rfdetr", _recorder("rfdetr"))
    monkeypatch.setattr(benchmark, "_nvml_handle", lambda idx=0: None)
    monkeypatch.setattr(benchmark, "print_benchmark_table", lambda result: None)
    out = tmp_path / "res.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["bench", "--model", model, "--device", "cpu", "--json-out", str(out)],
    )
    benchmark.main()
    assert called == [expected]
    assert json.loads(out.read_text())["fps"] is None  # NaN serialised as null


def test_main_rejects_unknown_family(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark, "_nvml_handle", lambda idx=0: None)
    monkeypatch.setattr(
        sys, "argv", ["bench", "--model", "fasterrcnn", "--device", "cpu"]
    )
    with pytest.raises(ValueError, match="Unsupported model family"):
        benchmark.main()
