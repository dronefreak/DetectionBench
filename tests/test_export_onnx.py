"""CPU-only tests for detectionbench.scripts.export_onnx (fake Ultralytics/RF-DETR)."""

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from detectionbench.scripts import export_onnx
from detectionbench.scripts.export_onnx import (
    infer_model_family,
    resolve_imgsz,
    resolve_model_family,
    resolve_ultralytics_nms,
)

_FAMILY_DEFAULT = 384


def _cfg(tmp_path: Path, **overrides: Any) -> DictConfig:
    """Compose the shipped export config for KITTI, writing into tmp_path."""
    with initialize_config_dir(
        config_dir=str(export_onnx.CONFIGS_DIR), version_base=None
    ):
        cfg = compose(
            config_name="export_onnx",
            overrides=["dataset=kitti", f"export.output_dir={tmp_path / 'export'}"],
        )
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value)
    return cfg


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), (640, 640), ([960], 960), ([768, 1472], (768, 1472))],
)
def test_resolve_imgsz(value: Any, expected: Any) -> None:
    assert resolve_imgsz(value) == expected


@pytest.mark.parametrize("value", ["640", [1, 2, 3]])
def test_resolve_imgsz_rejects_bad_shapes(value: Any) -> None:
    with pytest.raises(ValueError, match="export.imgsz"):
        resolve_imgsz(value)


@pytest.mark.parametrize(
    ("name", "family"),
    [
        ("yolov8n", "ultralytics"),
        ("YOLO26x", "ultralytics"),
        ("rtdetr-l", "ultralytics"),
        ("rfdetr-nano", "rfdetr"),
        ("nano", "rfdetr"),
        ("rfdetr_small", "rfdetr"),
    ],
)
def test_infer_model_family(name: str, family: str) -> None:
    assert infer_model_family(name) == family


def test_infer_model_family_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="model.family"):
        infer_model_family("fasterrcnn")


def test_explicit_family_overrides_name_and_is_validated(tmp_path: Path) -> None:
    assert (
        resolve_model_family(_cfg(tmp_path, **{"model.family": "RFDETR"})) == "rfdetr"
    )
    with pytest.raises(ValueError, match="Unsupported model family"):
        resolve_model_family(_cfg(tmp_path, **{"model.family": "onnx"}))


@pytest.mark.parametrize(
    ("model", "nms", "expected"),
    [
        ("yolov8n", None, False),
        ("yolov8n", True, True),
        ("yolov8n", "true", True),
        ("yolov8n", "False", False),
        ("yolov10n", True, False),  # end-to-end, NMS never baked in
    ],
)
def test_resolve_ultralytics_nms(model: str, nms: Any, expected: bool) -> None:
    assert resolve_ultralytics_nms(model, nms) is expected


def test_resolve_ultralytics_nms_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="export.nms"):
        resolve_ultralytics_nms("yolov8n", "maybe")


class _FakeUltralytics:
    """Stands in for ultralytics.YOLO/RTDETR; export() writes a dummy .onnx."""

    instances: list["_FakeUltralytics"] = []

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.export_kwargs: dict[str, Any] = {}
        type(self).instances.append(self)

    def export(self, **kwargs: Any) -> str:
        self.export_kwargs = kwargs
        out = Path(self.model_path).with_suffix(".onnx")
        out.write_bytes(b"onnx")
        return str(out)


class _FakeRTDETR(_FakeUltralytics):
    pass


@pytest.fixture
def fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> type[_FakeUltralytics]:
    _FakeUltralytics.instances = []
    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        types.SimpleNamespace(YOLO=_FakeUltralytics, RTDETR=_FakeRTDETR),
    )
    return _FakeUltralytics


def test_ultralytics_export_keeps_checkpoint_imgsz_by_default(
    fake_ultralytics: type[_FakeUltralytics], tmp_path: Path
) -> None:
    ckpt = tmp_path / "best.pt"
    result = export_onnx.export_onnx(
        _cfg(tmp_path, **{"export.checkpoint_path": str(ckpt)})
    )

    (model,) = fake_ultralytics.instances
    assert model.model_path == str(ckpt)
    assert "imgsz" not in model.export_kwargs  # Ultralytics falls back to ckpt imgsz
    assert "end2end" not in model.export_kwargs
    assert model.export_kwargs["nms"] is False
    out_dir = tmp_path / "export"
    assert Path(result.output_path) == out_dir / "best.onnx"
    assert (out_dir / "best.onnx").is_file()
    summary = json.loads((out_dir / "export_summary.json").read_text())
    assert summary["family"] == "ultralytics"
    assert summary["source_checkpoint"] == str(ckpt)


def test_ultralytics_export_passes_explicit_options(
    fake_ultralytics: type[_FakeUltralytics], tmp_path: Path
) -> None:
    cfg = _cfg(
        tmp_path,
        **{
            "model.name": "rtdetr-l",
            "export.checkpoint_path": str(tmp_path / "best.pt"),
            "export.imgsz": [736, 1280],
            "export.end2end": False,
        },
    )
    export_onnx.export_onnx(cfg)

    (model,) = fake_ultralytics.instances
    assert isinstance(model, _FakeRTDETR)
    assert model.export_kwargs["imgsz"] == (736, 1280)
    assert model.export_kwargs["end2end"] is False


class _FakeRFDETR:
    kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).kwargs = kwargs
        self.model_config = types.SimpleNamespace(
            resolution=kwargs.get("resolution", _FAMILY_DEFAULT)
        )


@pytest.fixture
def fake_rfdetr(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fake RF-DETR class; records the ONNX export call instead of tracing."""
    _FakeRFDETR.kwargs = {}
    calls: dict[str, Any] = {}

    def _export(**kwargs: Any) -> Path:
        calls.update(kwargs)
        return kwargs["output_dir"] / "inference_model.onnx"

    monkeypatch.setattr(
        export_onnx, "load_rfdetr_model_class", lambda name: _FakeRFDETR
    )
    monkeypatch.setattr(export_onnx, "export_rfdetr_model_to_onnx", _export)
    return calls


def _rfdetr_cfg(tmp_path: Path, checkpoint: Path, **overrides: Any) -> DictConfig:
    return _cfg(
        tmp_path,
        **{"model.name": "rfdetr-nano", "export.checkpoint_path": str(checkpoint)},
        **overrides,
    )


def test_rfdetr_export_uses_training_resolution(
    fake_rfdetr: dict[str, Any], tmp_path: Path
) -> None:
    ckpt = tmp_path / "checkpoint_best_total.pth"
    (tmp_path / "training_config.json").write_text(
        json.dumps({"model_config": {"resolution": 576}})
    )
    result = export_onnx.export_onnx(_rfdetr_cfg(tmp_path, ckpt))

    assert _FakeRFDETR.kwargs["resolution"] == 576
    assert _FakeRFDETR.kwargs["pretrain_weights"] == str(ckpt)
    assert _FakeRFDETR.kwargs["num_classes"] == 8  # kitti
    assert fake_rfdetr["shape"] == (576, 576)
    assert fake_rfdetr["variant_name"] == "rfdetr-nano"
    assert result.family == "rfdetr"


def test_rfdetr_export_falls_back_to_family_default(
    fake_rfdetr: dict[str, Any], tmp_path: Path
) -> None:
    export_onnx.export_onnx(_rfdetr_cfg(tmp_path, tmp_path / "ckpt.pth"))
    assert "resolution" not in _FakeRFDETR.kwargs
    assert fake_rfdetr["shape"] == (_FAMILY_DEFAULT, _FAMILY_DEFAULT)


def test_rfdetr_export_explicit_settings_win(
    fake_rfdetr: dict[str, Any], tmp_path: Path
) -> None:
    (tmp_path / "training_config.json").write_text(
        json.dumps({"model_config": {"resolution": 576}})
    )
    cfg = _rfdetr_cfg(
        tmp_path,
        tmp_path / "ckpt.pth",
        **{"model.resolution": 512, "export.imgsz": [512, 768]},
    )
    export_onnx.export_onnx(cfg)
    assert _FakeRFDETR.kwargs["resolution"] == 512
    assert fake_rfdetr["shape"] == (512, 768)
