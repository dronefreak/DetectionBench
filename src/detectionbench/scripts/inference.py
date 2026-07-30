r"""
Inference script for DetectionBench-registered object detection models.

Supports Ultralytics YOLO / RT-DETR and RF-DETR inference on:
- Single images
- Directories of images
- Video files

Usage examples:
  # YOLO - image directory
  python -m detectionbench.scripts.inference \
     --checkpoint experiments/yolov8n/weights/best.pt --model yolov8n \
     --dataset doclaynet --input yolo_dataset/images/test/ --output-dir inference/

  # YOLO - video file
  python -m detectionbench.scripts.inference \\
     --checkpoint experiments/yolov8n/weights/best.pt \\
     --model yolov8n --dataset doclaynet --input video.mp4

  # RF-DETR - image directory
  python -m detectionbench.scripts.inference \\
     --checkpoint experiments/rfdetr-nano/checkpoint_best_total.pth \\
     --model rfdetr-nano --dataset doclaynet \\
     --input yolo_dataset_coco/test/ --output-dir inference/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from detectionbench.datasets import get_spec, list_datasets
from detectionbench.utils.rfdetr import (
    RFDETR_MODEL_CLASS_MAP,
    load_rfdetr_model_class,
    normalize_model_name,
)
from detectionbench.utils.utils import RichConsoleManager

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass(frozen=True)
class InferenceOptions:
    """Store runtime options for Ultralytics inference."""

    checkpoint_path: str
    input_path: Path
    output_dir: Path
    score_threshold: float
    nms_threshold: float
    device: str
    imgsz: int
    save_viz: bool
    show: bool
    class_names: list[str]
    class_colors: dict[int, tuple[int, int, int]]
    model_name: str = ""


@dataclass(frozen=True)
class DetectionOverlay:
    """Describe detections to be rendered on an image."""

    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    class_names: list[str]
    class_colors: dict[int, tuple[int, int, int]]


def _to_numpy_array(data: Any) -> np.ndarray:
    """Convert torch or NumPy-like data to a NumPy array."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Ultralytics inference."""
    parser = argparse.ArgumentParser(
        description="Run inference on DetectionBench models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint / .pt file"
    )
    parser.add_argument("--model", default="yolov5s", help="Model name")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list_datasets(),
        help="Registered dataset key (determines class names/colors)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of classes (defaults to the full class count for --dataset)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280, help="Inference image size (YOLO only)"
    )

    # Input  (images / directory / video file)
    parser.add_argument(
        "--input", required=True, help="Input image, directory, or video file"
    )
    parser.add_argument(
        "--output-dir", default="inference_outputs", help="Output directory"
    )

    # Inference parameters
    parser.add_argument(
        "--score-threshold", type=float, default=0.5, help="Confidence threshold"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )

    # Post-processing
    parser.add_argument(
        "--soft-nms",
        action="store_true",
        help="Reserved for non-Ultralytics backends and currently unsupported here",
    )
    parser.add_argument(
        "--nms-threshold", type=float, default=0.5, help="NMS IoU threshold"
    )

    # Visualization
    parser.add_argument(
        "--no-save-viz", action="store_true", help="Don't save visualizations"
    )
    parser.add_argument(
        "--show", action="store_true", help="Display results interactively"
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# YOLO inference path
# ---------------------------------------------------------------------------


def run_yolo(options: InferenceOptions) -> None:
    """Run YOLO inference with custom visualization."""
    try:
        from ultralytics import YOLO as UltralyticsYOLO
    except ImportError as err:
        raise ImportError("pip install ultralytics>=8.0.0") from err

    # RT-DETR checkpoints must be loaded with the RTDETR class
    model: Any
    if options.model_name.lower().startswith("rtdetr"):
        try:
            from ultralytics import RTDETR
        except ImportError as err:
            raise ImportError("pip install -U ultralytics") from err
        model = RTDETR(str(options.checkpoint_path))
    else:
        model = UltralyticsYOLO(str(options.checkpoint_path))

    options.output_dir.mkdir(parents=True, exist_ok=True)
    console = RichConsoleManager.get_console()

    console.print(f"Running YOLO inference on {options.input_path} ...")

    results = model.predict(
        source=str(options.input_path),
        conf=options.score_threshold,
        iou=options.nms_threshold,
        device=options.device,
        imgsz=options.imgsz,
        save=False,
        verbose=True,
    )

    total_det = 0

    for result in results:
        if result.boxes is None:
            continue

        box_data = result.boxes
        total_det += len(box_data)

        # Original image (full resolution)
        frame = result.orig_img.copy()

        # Extract predictions
        boxes = _to_numpy_array(box_data.xyxy)
        scores = _to_numpy_array(box_data.conf)
        labels = _to_numpy_array(box_data.cls).astype(int)

        viz = draw_detections(
            frame,
            DetectionOverlay(
                boxes=boxes,
                scores=scores,
                labels=labels,
                class_names=options.class_names,
                class_colors=options.class_colors,
            ),
        )

        if options.save_viz:
            image_path = Path(result.path)
            out_path = options.output_dir / f"{image_path.stem}_pred.jpg"
            cv2.imwrite(str(out_path), viz)

        if options.show:
            cv2.imshow("YOLO Inference", viz)
            if cv2.waitKey(0) == ord("q"):
                break

    if options.show:
        cv2.destroyAllWindows()

    console.print(f"\n Processed {len(results)} image(s)")
    console.print(f"Total detections: {total_det}")
    console.print(f"Results saved to: {options.output_dir}")


# ---------------------------------------------------------------------------
# RF-DETR inference path
# ---------------------------------------------------------------------------


def run_rfdetr(options: InferenceOptions) -> None:
    """Run RF-DETR inference with custom visualization."""
    model_class = load_rfdetr_model_class(options.model_name)
    model = model_class(
        num_classes=len(options.class_names),
        pretrain_weights=str(options.checkpoint_path),
        device=options.device,
    )

    options.output_dir.mkdir(parents=True, exist_ok=True)
    console = RichConsoleManager.get_console()
    console.print(f"Running RF-DETR inference on {options.input_path} ...")

    if options.input_path.is_dir():
        image_paths = sorted(
            p
            for p in options.input_path.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        total_det = _run_rfdetr_on_images(model, image_paths, options)
        console.print(f"\n Processed {len(image_paths)} image(s)")
        console.print(f"Total detections: {total_det}")
    elif options.input_path.suffix.lower() in VIDEO_EXTENSIONS:
        total_det, num_frames = _run_rfdetr_on_video(model, options)
        console.print(f"\n Processed {num_frames} frame(s)")
        console.print(f"Total detections: {total_det}")
    else:
        total_det = _run_rfdetr_on_images(model, [options.input_path], options)
        console.print("\n Processed 1 image")
        console.print(f"Total detections: {total_det}")

    if options.show:
        cv2.destroyAllWindows()

    console.print(f"Results saved to: {options.output_dir}")


def _run_rfdetr_on_images(
    model: Any, image_paths: list[Path], options: InferenceOptions
) -> int:
    """Run RF-DETR predict() over a list of image files and save/show overlays."""
    total_det = 0
    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        detections = model.predict(
            str(image_path),
            threshold=options.score_threshold,
            include_source_image=False,
        )
        boxes = _to_numpy_array(detections.xyxy)
        scores = _to_numpy_array(detections.confidence)
        labels = _to_numpy_array(detections.class_id).astype(int)
        total_det += len(boxes)

        viz = draw_detections(
            frame,
            DetectionOverlay(
                boxes=boxes,
                scores=scores,
                labels=labels,
                class_names=options.class_names,
                class_colors=options.class_colors,
            ),
        )

        if options.save_viz:
            out_path = options.output_dir / f"{image_path.stem}_pred.jpg"
            cv2.imwrite(str(out_path), viz)

        if options.show:
            cv2.imshow("RF-DETR Inference", viz)
            if cv2.waitKey(0) == ord("q"):
                break
    return total_det


def _run_rfdetr_on_video(model: Any, options: InferenceOptions) -> tuple[int, int]:
    """Run RF-DETR predict() over every frame of a video and save/show overlays."""
    cap = cv2.VideoCapture(str(options.input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {options.input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if options.save_viz:
        out_path = options.output_dir / f"{options.input_path.stem}_pred.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, frame_h))

    total_det = 0
    num_frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            num_frames += 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = model.predict(
                rgb_frame,
                threshold=options.score_threshold,
                include_source_image=False,
            )
            boxes = _to_numpy_array(detections.xyxy)
            scores = _to_numpy_array(detections.confidence)
            labels = _to_numpy_array(detections.class_id).astype(int)
            total_det += len(boxes)

            viz = draw_detections(
                frame,
                DetectionOverlay(
                    boxes=boxes,
                    scores=scores,
                    labels=labels,
                    class_names=options.class_names,
                    class_colors=options.class_colors,
                ),
            )

            if writer is not None:
                writer.write(viz)
            if options.show:
                cv2.imshow("RF-DETR Inference", viz)
                if cv2.waitKey(1) == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    return total_det, num_frames


def draw_detections(frame: np.ndarray, overlay: DetectionOverlay) -> np.ndarray:
    """Draw bounding boxes and labels on a BGR frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    console = RichConsoleManager.get_console()
    console.print(
        f"Drawing {len(overlay.boxes)} detections on frame of size {w}x{h} ..."
    )

    # Much more conservative scaling
    scale = max(h, w) / 2000.0

    box_thickness = max(1, int(scale))
    font_scale = max(0.3, scale * 0.35)
    font_thickness = 1

    for box, score, label in zip(overlay.boxes, overlay.scores, overlay.labels):
        x1, y1, x2, y2 = box.astype(int)

        color_rgb = overlay.class_colors.get(label, (0, 255, 0))
        # class_colors are stored as RGB; OpenCV drawing functions expect BGR.
        color = (color_rgb[2], color_rgb[1], color_rgb[0])

        # Draw box
        cv2.rectangle(
            out,
            (x1, y1),
            (x2, y2),
            color,
            box_thickness,
        )

        name = (
            overlay.class_names[label]
            if label < len(overlay.class_names)
            else f"cls{label}"
        )

        text = f"{name} {score:.2f}"

        # Compute text size
        (tw, th), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness,
        )
        # Filled label background
        cv2.rectangle(
            out,
            (x1, y1 - th - baseline - 4),
            (x1 + tw + 4, y1),
            color,
            -1,
        )

        # Text
        cv2.putText(
            out,
            text,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the inference CLI entrypoint."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if args.soft_nms:
        raise ValueError("Soft-NMS is not supported for these model families.")

    is_ultralytics = args.model.lower().startswith(("yolo", "rtdetr"))
    is_rfdetr = normalize_model_name(args.model) in RFDETR_MODEL_CLASS_MAP

    spec = get_spec(args.dataset)
    num_classes = args.num_classes if args.num_classes is not None else spec.num_classes
    class_names = spec.classes[: min(num_classes, len(spec.classes))]
    class_colors = spec.class_colors or {}

    if is_ultralytics:
        run_yolo(
            InferenceOptions(
                checkpoint_path=args.checkpoint,
                input_path=input_path,
                output_dir=output_dir,
                score_threshold=args.score_threshold,
                nms_threshold=args.nms_threshold,
                device=args.device,
                imgsz=args.imgsz,
                save_viz=not args.no_save_viz,
                show=args.show,
                class_names=class_names,
                class_colors=class_colors,
                model_name=args.model,
            )
        )
        return

    if is_rfdetr:
        run_rfdetr(
            InferenceOptions(
                checkpoint_path=args.checkpoint,
                input_path=input_path,
                output_dir=output_dir,
                score_threshold=args.score_threshold,
                nms_threshold=args.nms_threshold,
                device=args.device,
                imgsz=args.imgsz,
                save_viz=not args.no_save_viz,
                show=args.show,
                class_names=class_names,
                class_colors=class_colors,
                model_name=args.model,
            )
        )
        return

    supported_rfdetr = ", ".join(sorted(RFDETR_MODEL_CLASS_MAP))
    raise ValueError(
        f"Unsupported model family '{args.model}'. "
        f"Supported: Ultralytics YOLO/RT-DETR models, or RF-DETR ({supported_rfdetr})."
    )


if __name__ == "__main__":
    main()
