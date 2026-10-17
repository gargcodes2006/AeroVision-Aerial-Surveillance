"""
AeroVision Image Processor
==========================
Handles end-to-end image processing:
  1. Decode uploaded bytes → OpenCV image.
  2. Resize if necessary.
  3. Run YOLO detection.
  4. Draw annotated bounding boxes.
  5. Return results ready for the UI.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import (
    BBOX_THICKNESS,
    BBOX_FONT_SCALE,
    BBOX_LABEL_BG_ALPHA,
    MAX_IMAGE_DIMENSION,
)
from src.detector import Detection, DetectionResult, run_detection
from src.utils import (
    bytes_to_cv2,
    resize_image,
    get_class_color,
    cv2_to_rgb,
)


# ── Annotation Drawing ──────────────────────────────────────────────────────

def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    thickness: int = BBOX_THICKNESS,
    font_scale: float = BBOX_FONT_SCALE,
    label_bg_alpha: float = BBOX_LABEL_BG_ALPHA,
) -> np.ndarray:
    """Draw bounding boxes and labels on a *copy* of *image* (BGR).

    Returns the annotated copy.
    """
    annotated = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = get_class_color(det.class_name)
        label = f"{det.class_name} {det.confidence:.0%}"

        # Draw bounding-box rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Compute label size and background rectangle
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        label_y1 = max(y1 - th - baseline - 6, 0)
        label_y2 = y1

        # Semi-transparent label background
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, label_y1), (x1 + tw + 8, label_y2), color, -1)
        cv2.addWeighted(overlay, label_bg_alpha, annotated, 1 - label_bg_alpha, 0, annotated)

        # Label text
        cv2.putText(
            annotated,
            label,
            (x1 + 4, label_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


# ── High-level Pipeline ─────────────────────────────────────────────────────

def process_image(
    file_bytes: bytes,
    model,
    confidence: float,
    model_name: str = "",
    max_dim: int = MAX_IMAGE_DIMENSION,
) -> tuple[np.ndarray | None, np.ndarray | None, DetectionResult | None, str]:
    """Full image-processing pipeline.

    Returns
    -------
    (original_rgb, annotated_rgb, result, error_message)
    If an error occurs the first three values are ``None`` and
    ``error_message`` explains the problem.
    """
    # 1. Decode
    image = bytes_to_cv2(file_bytes)
    if image is None:
        return None, None, None, "Could not decode the image. The file may be corrupt."

    # 2. Resize
    image = resize_image(image, max_dim)

    # 3. Detect
    try:
        result = run_detection(model, image, confidence, model_name)
    except Exception as exc:
        return None, None, None, f"Detection failed: {exc}"

    # 4. Annotate
    annotated = draw_detections(image, result.detections)

    # 5. Convert to RGB for display
    original_rgb = cv2_to_rgb(image)
    annotated_rgb = cv2_to_rgb(annotated)

    return original_rgb, annotated_rgb, result, ""
