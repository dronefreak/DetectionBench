"""
YOLO training via Ultralytics engine.

Delegates training to Ultralytics' native trainer, which implements the full
YOLO training pipeline (TaskAlignedAssigner, DFL loss, box/cls/dfl losses, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
            **extra_kwargs: Passed directly to ultralytics.YOLO.train()

        Returns:
            dict with keys: 'results', 'model_path', 'output_dir'

        """
        output_dir = Path(output_dir).resolve()  # must be absolute so Ultralytics
        output_dir.mkdir(parents=True, exist_ok=True)  # doesn't prefix runs/detect/

        model = self._UltralyticsYOLO(self._pt_name)

        results = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            lr0=lr,
            amp=use_amp,
            device=self.device,
            workers=workers,
            project=str(output_dir),
            name=self._model_name,
            exist_ok=True,
            **extra_kwargs,
        )

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
