"""Wiring tests for the Ultralytics train/evaluate path (no GPU, no real model)."""

import sys
import types
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from detectionbench.scripts import evaluate_yolo as evaluate_yolo_module
from detectionbench.scripts import train_yolo as train_yolo_module
from detectionbench.scripts.evaluate_yolo import EvaluationOptions, evaluate_yolo

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
YOLO_CONFIGS = sorted(
    p.stem
    for p in CONFIGS_DIR.glob("*.yaml")
    if (p.stem == "config" or "_yolo" in p.stem)
)


class _FakeYOLO:
    calls: list[dict[str, Any]] = []

    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint

    def val(self, **kwargs: Any) -> object:
        _FakeYOLO.calls.append(kwargs)
        return object()  # no `.box` -> empty metrics dict


@pytest.fixture
def fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> type[_FakeYOLO]:
    _FakeYOLO.calls = []
    monkeypatch.setitem(
        sys.modules, "ultralytics", types.SimpleNamespace(YOLO=_FakeYOLO)
    )
    return _FakeYOLO


def _options(tmp_path: Path, **overrides: Any) -> EvaluationOptions:
    return EvaluationOptions(
        checkpoint_path="best.pt",
        dataset_yaml="data.yaml",
        class_names=["a", "b"],
        num_classes=2,
        device="cpu",
        output_dir=tmp_path,
        **overrides,
    )


def test_val_uses_ultralytics_defaults_when_knobs_unset(
    fake_ultralytics: type[_FakeYOLO], tmp_path: Path
) -> None:
    evaluate_yolo(_options(tmp_path))
    (kwargs,) = fake_ultralytics.calls
    assert not {"conf", "iou", "batch", "workers"} & kwargs.keys()


def test_val_forwards_explicit_knobs(
    fake_ultralytics: type[_FakeYOLO], tmp_path: Path
) -> None:
    evaluate_yolo(
        _options(
            tmp_path,
            score_threshold=0.25,
            iou_threshold=0.6,
            batch_size=8,
            num_workers=2,
        )
    )
    (kwargs,) = fake_ultralytics.calls
    assert (kwargs["conf"], kwargs["iou"], kwargs["batch"], kwargs["workers"]) == (
        0.25,
        0.6,
        8,
        2,
    )


def test_cli_flags_reach_val(
    fake_ultralytics: type[_FakeYOLO],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_yolo",
            "--checkpoint",
            "best.pt",
            "--dataset",
            "visdrone",
            "--dataset-yaml",
            "data.yaml",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
            "--score-threshold",
            "0.3",
            "--iou-threshold",
            "0.5",
        ],
    )
    evaluate_yolo_module.main()
    (kwargs,) = fake_ultralytics.calls
    assert (kwargs["conf"], kwargs["iou"]) == (0.3, 0.5)
    assert "batch" not in kwargs


@pytest.mark.parametrize("config_name", YOLO_CONFIGS)
def test_yolo_configs_have_seed_and_no_dead_eval_keys(config_name: str) -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        cfg = compose(config_name=config_name)
    assert cfg.training.seed == 0
    dead = {"score_threshold", "iou_threshold", "soft_nms", "batch_size", "num_workers"}
    assert not dead & set(cfg.evaluation.keys())


def test_train_passes_seed_and_reports_model_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    class _FakeTrainer:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def train(self, **kwargs: Any) -> dict[str, Any]:
            seen["train"] = kwargs
            return {"model_path": "best.pt", "output_dir": str(tmp_path)}

    def _fake_finalize(model_name: str, metrics: dict, output_dir: Path) -> Path:
        seen["finalize"] = (model_name, output_dir)
        return output_dir / "metrics.json"

    monkeypatch.setattr(train_yolo_module, "YOLOTrainer", _FakeTrainer)
    monkeypatch.setattr(train_yolo_module, "evaluate_yolo", lambda options: {})
    monkeypatch.setattr(train_yolo_module, "finalize_metrics", _fake_finalize)
    monkeypatch.setattr(
        train_yolo_module, "seed_everything", lambda s: seen.setdefault("seed", s)
    )

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "model.name=yolov8n",
                "training.seed=7",
                f"training.output_dir={tmp_path}",
            ],
        )
    train_yolo_module.train_and_evaluate.__wrapped__(cfg)

    assert seen["seed"] == 7
    assert seen["train"]["seed"] == 7
    assert seen["finalize"][0] == "yolov8n"


@pytest.mark.parametrize(
    ("dataset", "split"), [("seadronessee", "val"), ("lisa", "test")]
)
def test_rfdetr_evaluate_config_follows_dataset_eval_split(
    dataset: str, split: str
) -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        cfg = compose(config_name="rfdetr_evaluate", overrides=[f"dataset={dataset}"])
    assert cfg.evaluation.split == split


def test_rfdetr_loader_maps_val_to_canonical_valid_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from detectionbench.scripts import evaluate_rfdetr

    split_dir = tmp_path / "valid"
    split_dir.mkdir()
    (split_dir / "_annotations.coco.json").write_text(
        json.dumps(
            {"images": [], "annotations": [], "categories": [{"id": 0, "name": "boat"}]}
        )
    )
    seen: dict[str, str] = {}

    class _FakeDataset:
        @staticmethod
        def from_coco(images_directory_path: str, annotations_path: str) -> object:
            seen["dir"] = images_directory_path
            return object()

    monkeypatch.setitem(
        sys.modules, "supervision", types.SimpleNamespace(DetectionDataset=_FakeDataset)
    )
    _, names = evaluate_rfdetr.load_detection_dataset(tmp_path, "val")
    assert seen["dir"] == str(split_dir)
    assert names == {0: "boat"}
