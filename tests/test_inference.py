"""CPU-only tests for detectionbench.scripts.inference (fake Ultralytics/RF-DETR)."""

import json
import sys
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from detectionbench.scripts import inference
from detectionbench.scripts.inference import (
    DetectionOverlay,
    InferenceOptions,
    draw_detections,
)


def _options(tmp_path: Path, input_path: Path, **overrides: Any) -> InferenceOptions:
    values: dict[str, Any] = {
        "checkpoint_path": str(tmp_path / "best.pt"),
        "input_path": input_path,
        "output_dir": tmp_path / "out",
        "score_threshold": 0.5,
        "nms_threshold": 0.5,
        "device": "cpu",
        "imgsz": None,
        "save_viz": True,
        "show": False,
        "class_names": ["car", "person"],
        "class_colors": {0: (255, 0, 0)},
        "model_name": "yolov8n",
    }
    values.update(overrides)
    return InferenceOptions(**values)


class _Boxes:
    def __init__(self) -> None:
        self.xyxy = np.array([[10.0, 10.0, 40.0, 40.0]])
        self.conf = np.array([0.9])
        self.cls = np.array([0.0])

    def __len__(self) -> int:
        return 1


class _FakeUltralytics:
    """Stands in for ultralytics.YOLO; predict() yields `n_results` results."""

    predict_kwargs: dict[str, Any] = {}
    n_results = 3
    paths: list[str] = []

    def __init__(self, checkpoint: str) -> None:
        pass

    def predict(self, **kwargs: Any):
        type(self).predict_kwargs = kwargs
        for i in range(self.n_results):
            yield types.SimpleNamespace(
                orig_img=np.zeros((64, 80, 3), dtype=np.uint8),
                boxes=_Boxes() if i % 2 == 0 else None,
                path=self.paths[i] if self.paths else kwargs["source"],
            )


@pytest.fixture
def fake_yolo(monkeypatch: pytest.MonkeyPatch) -> type[_FakeUltralytics]:
    _FakeUltralytics.predict_kwargs = {}
    _FakeUltralytics.n_results = 3
    _FakeUltralytics.paths = []
    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        types.SimpleNamespace(YOLO=_FakeUltralytics, RTDETR=_FakeUltralytics),
    )
    return _FakeUltralytics


def test_yolo_video_writes_one_video_with_every_frame(
    fake_yolo: type[_FakeUltralytics], tmp_path: Path
) -> None:
    fake_yolo.n_results = 5
    inference.run_yolo(_options(tmp_path, tmp_path / "clip.mp4"))

    out_video = tmp_path / "out" / "clip_pred.mp4"
    assert out_video.is_file()
    assert not list((tmp_path / "out").glob("*.jpg"))
    cap = cv2.VideoCapture(str(out_video))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    cap.release()


def test_yolo_images_saved_one_per_input(
    fake_yolo: type[_FakeUltralytics], tmp_path: Path
) -> None:
    fake_yolo.n_results = 2
    fake_yolo.paths = [str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")]
    inference.run_yolo(_options(tmp_path, tmp_path))

    saved = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert saved == ["a_pred.jpg", "b_pred.jpg"]


def test_yolo_streams_and_keeps_checkpoint_imgsz_by_default(
    fake_yolo: type[_FakeUltralytics], tmp_path: Path
) -> None:
    inference.run_yolo(_options(tmp_path, tmp_path, save_viz=False))
    assert fake_yolo.predict_kwargs["stream"] is True
    assert "imgsz" not in fake_yolo.predict_kwargs

    inference.run_yolo(_options(tmp_path, tmp_path, save_viz=False, imgsz=960))
    assert fake_yolo.predict_kwargs["imgsz"] == 960


class _FakeRFDETR:
    kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).kwargs = kwargs

    def predict(self, source: Any, **kwargs: Any) -> Any:
        return types.SimpleNamespace(
            xyxy=np.zeros((0, 4)), confidence=np.zeros(0), class_id=np.zeros(0)
        )


@pytest.mark.parametrize(("recorded", "expected"), [(576, 576), (None, None)])
def test_rfdetr_uses_training_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    recorded: int | None,
    expected: int | None,
) -> None:
    monkeypatch.setitem(
        sys.modules, "rfdetr", types.SimpleNamespace(RFDETRNano=_FakeRFDETR)
    )
    checkpoint = tmp_path / "checkpoint_best_total.pth"
    if recorded is not None:
        (tmp_path / "training_config.json").write_text(
            json.dumps({"model_config": {"resolution": recorded}})
        )
    image = tmp_path / "img.jpg"
    cv2.imwrite(str(image), np.zeros((32, 32, 3), dtype=np.uint8))

    inference.run_rfdetr(
        _options(
            tmp_path, image, checkpoint_path=str(checkpoint), model_name="rfdetr-nano"
        )
    )

    assert _FakeRFDETR.kwargs.get("resolution") == expected
    assert (tmp_path / "out" / "img_pred.jpg").is_file()


@pytest.mark.parametrize(
    ("model", "runner"),
    [
        ("yolov8n", "run_yolo"),
        ("rtdetr-l", "run_yolo"),
        ("rfdetr-nano", "run_rfdetr"),
        ("nano", "run_rfdetr"),
    ],
)
def test_main_dispatches_by_model_family(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model: str, runner: str
) -> None:
    called: list[str] = []
    monkeypatch.setattr(inference, "run_yolo", lambda o: called.append("run_yolo"))
    monkeypatch.setattr(inference, "run_rfdetr", lambda o: called.append("run_rfdetr"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer",
            "--checkpoint",
            "x.pt",
            "--model",
            model,
            "--dataset",
            "kitti",
            "--input",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--device",
            "cpu",
        ],
    )
    inference.main()
    assert called == [runner]


def test_main_rejects_unknown_model_and_missing_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = ["infer", "--checkpoint", "x.pt", "--dataset", "kitti", "--device", "cpu"]
    out = ["--output-dir", str(tmp_path / "out")]
    monkeypatch.setattr(
        sys, "argv", [*base, "--model", "fasterrcnn", "--input", str(tmp_path), *out]
    )
    with pytest.raises(ValueError, match="Unsupported model family"):
        inference.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [*base, "--model", "yolov8n", "--input", str(tmp_path / "nope.jpg"), *out],
    )
    with pytest.raises(FileNotFoundError):
        inference.main()


def test_draw_detections_uses_bgr_colors_and_leaves_input_untouched() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_detections(
        frame,
        DetectionOverlay(
            boxes=np.array([[20.0, 50.0, 80.0, 90.0]]),
            scores=np.array([0.8]),
            labels=np.array([0]),
            class_names=["car"],
            class_colors={0: (255, 0, 0)},  # RGB red
        ),
        show_labels=False,
    )
    assert frame.sum() == 0
    assert tuple(out[70, 20]) == (0, 0, 255)  # drawn as BGR red


@pytest.mark.parametrize(
    ("reported", "expected"), [(30.0, 30.0), (0.0, 25.0), (-1.0, 25.0)]
)
def test_video_fps_falls_back_when_unknown(reported: float, expected: float) -> None:
    cap = types.SimpleNamespace(get=lambda prop: reported)
    assert inference._video_fps(cap) == expected  # type: ignore[arg-type]
