"""
YOLO training via Ultralytics engine.

Delegates training to Ultralytics' native trainer, which implements the full
YOLO training pipeline (TaskAlignedAssigner, DFL loss, box/cls/dfl losses, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Ultralytics has no single "off" switch for train-time augmentation -- the
# augmentation pipeline is the sum of these hyperparameters. Zeroing every
# geometric/photometric knob and disabling the policy-based ones
# (``auto_augment``, ``erasing``) is how you actually train without
# augmentation; passed only when ``train(augment=False)``.
# Ultralytics' ``build_optimizer`` matches optimizer names case-sensitively and
# raises on anything unknown -- normalize the common lowercase spellings from
# config.yaml to its canonical casing. Unrecognized values pass through so
# Ultralytics can raise its own (clearer) error.
_OPTIMIZER_ALIASES: dict[str, str] = {
    "auto": "auto",
    "sgd": "SGD",
    "adam": "Adam",
    "adamax": "Adamax",
    "adamw": "AdamW",
    "nadam": "NAdam",
    "radam": "RAdam",
    "rmsprop": "RMSProp",
}

_NO_AUGMENTATION_HYP: dict[str, Any] = {
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "bgr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "auto_augment": None,
    "erasing": 0.0,
}


class YOLOTrainer:
    """
    Trains YOLO models using the Ultralytics training engine.

    Handles:
    - Delegating training to ultralytics.YOLO.train() using a pre-built dataset YAML
    - Saving the final model to the requested output directory

    Does NOT attempt to re-implement YOLO's internal loss or assignment logic.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 11,
        device: str = "cuda",
    ) -> None:
        """
        Initialize YOLOTrainer.

        Args:
            model_name: Registered model name, e.g. 'yolov8n', 'yolov9c', 'yolov10m'
            num_classes: Number of detection classes.
            device: Device string passed to Ultralytics ('cuda', 'cpu', '0', '0,1', ...)

        """
        try:
            from ultralytics import YOLO as UltralyticsYOLO
        except ImportError as err:
            raise ImportError(
                "Ultralytics is required for YOLO training. "
                "Install with: pip install ultralytics>=8.0.0"
            ) from err

        # Derive the .pt filename from the registered model name
        # e.g. 'yolov8n' -> 'yolov8n.pt', 'yolov10m' -> 'yolov10m.pt'
        self._pt_name = f"{model_name}.pt"
        self._model_name = model_name
        self.num_classes = num_classes
        self.device = device
        self._UltralyticsYOLO = UltralyticsYOLO

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(  # noqa: PLR0913, PLR0917
        self,
        dataset_yaml: str | Path,
        epochs: int = 100,
        batch_size: int = 16,
        lr: float = 0.001,
        imgsz: int = 640,
        use_amp: bool = True,
        output_dir: str | Path = "outputs",
        workers: int = 4,
        patience: int = 100,
        optimizer: str = "auto",
        cos_lr: bool = False,
        augment: bool = True,
        **extra_kwargs: Any,
    ) -> dict[str, Any]:
        """
        Train a YOLO model on the configured dataset.

        Args:
            dataset_yaml: Path to the Ultralytics dataset YAML file.
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Initial learning rate (lr0 in Ultralytics terminology)
            imgsz: Input image size
            use_amp: Use automatic mixed precision
            output_dir: Where to save the final model and logs
            workers: Number of DataLoader workers
            patience: Epochs with no mAP improvement before early stopping
                (Ultralytics' native early-stopping mechanism; 0 disables it)
            optimizer: Ultralytics optimizer name ('auto', 'SGD', 'Adam',
                'AdamW', ...); 'auto' lets Ultralytics pick.
            cos_lr: Use a cosine LR schedule (Ultralytics ``cos_lr``); False
                keeps Ultralytics' default linear decay.
            augment: When False, disable train-time augmentation by zeroing
                every augmentation hyperparameter (see ``_NO_AUGMENTATION_HYP``);
                explicit hyps in ``extra_kwargs`` still win.
            **extra_kwargs: Passed directly to ultralytics.YOLO.train()

        Returns:
            dict with keys: 'results', 'model_path', 'output_dir'

        """
        output_dir = Path(output_dir).resolve()  # must be absolute so Ultralytics
        output_dir.mkdir(parents=True, exist_ok=True)  # doesn't prefix runs/detect/

        model = self._UltralyticsYOLO(self._pt_name)

        optimizer = _OPTIMIZER_ALIASES.get(optimizer.strip().lower(), optimizer)

        train_kwargs: dict[str, Any] = {
            "data": str(dataset_yaml),
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": imgsz,
            "lr0": lr,
            "amp": use_amp,
            "device": self.device,
            "workers": workers,
            "patience": patience,
            "optimizer": optimizer,
            "cos_lr": cos_lr,
            "project": str(output_dir),
            "name": self._model_name,
            "exist_ok": True,
        }
        if not augment:
            train_kwargs.update(_NO_AUGMENTATION_HYP)
        train_kwargs.update(extra_kwargs)

        results = model.train(**train_kwargs)

        # Ultralytics saves best/last weights under project/name/weights/
        weights_dir = output_dir / self._model_name / "weights"
        best_model = weights_dir / "best.pt"
        last_model = weights_dir / "last.pt"
        final_path = best_model if best_model.exists() else last_model

        return {
            "results": results,
            "model_path": str(final_path) if final_path.exists() else None,
            "output_dir": str(output_dir / self._model_name),
        }
