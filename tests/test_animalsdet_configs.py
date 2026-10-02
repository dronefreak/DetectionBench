from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def _compose(config_name: str, *overrides: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        return compose(config_name=config_name, overrides=list(overrides))


def test_animalsdet_yolo_config_has_modest_imgsz_and_standard_patience() -> None:
    cfg = _compose("animalsdet_yolo")
    assert cfg.training.imgsz == 640  # boxes are large; resolution isn't the bottleneck
    assert cfg.training.epochs == 80  # medium dataset (18,402 train images)
    assert cfg.training.patience == 20
    assert cfg.training.resume is None
    assert cfg.dataset.name == "animalsdet"
    assert cfg.dataset.eval_split == "test"
    assert cfg.model.num_classes == 80  # widest vocabulary of any dataset so far


def test_animalsdet_yolo_config_accepts_model_override() -> None:
    cfg = _compose("animalsdet_yolo", "model.name=yolo26s")
    assert cfg.model.name == "yolo26s"


def test_animalsdet_rfdetr_config_uses_family_default_resolutions() -> None:
    cfg = _compose("animalsdet_rfdetr")
    assert (
        cfg.model.resolution == 512
    )  # rfdetr-small default; boxes are large, no bump needed
    assert cfg.training.epochs == 80
    assert cfg.training.early_stopping_patience == 20
    assert cfg.dataset.name == "animalsdet"
    assert cfg.model.num_classes == 80


def test_animalsdet_rfdetr_config_accepts_model_and_resolution_override() -> None:
    cfg = _compose(
        "animalsdet_rfdetr", "model.name=rfdetr-nano", "model.resolution=384"
    )
    assert cfg.model.name == "rfdetr-nano"
    assert cfg.model.resolution == 384
