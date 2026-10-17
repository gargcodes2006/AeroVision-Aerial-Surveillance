"""
AeroVision Detector
===================
Wraps the Ultralytics YOLO model for inference.
Provides a simple, cacheable interface for loading the model and running
detection on individual images (NumPy arrays).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import AVAILABLE_MODELS, DEFAULT_MODEL, DEFAULT_CONFIDENCE


# ── Detection Result Dataclass ───────────────────────────────────────────────

@dataclass
class Detection:
    """A single detected object."""
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    class_id: int = 0


@dataclass
class DetectionResult:
    """Aggregate result for one image/frame."""
    detections: list[Detection] = field(default_factory=list)
    inference_time: float = 0.0        # seconds
    image_shape: tuple[int, ...] = ()  # (H, W, C)
    model_name: str = ""

    @property
    def count(self) -> int:
        return len(self.detections)

    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts


# ── Model Loader ─────────────────────────────────────────────────────────────

def load_model(model_key: str = DEFAULT_MODEL):
    """Load and return a YOLO model.

    ``model_key`` must be one of the keys in ``AVAILABLE_MODELS``.
    Weights are automatically downloaded by Ultralytics on first use.

    Raises ``RuntimeError`` with a helpful message if loading fails.
    """
    from ultralytics import YOLO  # deferred to keep import-time fast

    weight_file = AVAILABLE_MODELS.get(model_key)
    if weight_file is None:
        raise ValueError(
            f"Unknown model key '{model_key}'. "
            f"Choose from: {list(AVAILABLE_MODELS.keys())}"
        )
    try:
        model = YOLO(weight_file)
        return model
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model '{weight_file}'. "
            f"Ensure you have internet access for first-time weight download.\n"
            f"Original error: {exc}"
        ) from exc


# ── Inference ────────────────────────────────────────────────────────────────

def run_detection(
    model,
    image: np.ndarray,
    confidence: float = DEFAULT_CONFIDENCE,
    model_name: str = "",
) -> DetectionResult:
    """Run YOLO inference on a single BGR image and return structured results.

    Parameters
    ----------
    model : YOLO
        A loaded Ultralytics YOLO model instance.
    image : np.ndarray
        BGR image (OpenCV format).
    confidence : float
        Minimum confidence threshold (0–1).
    model_name : str
        Label stored in the result for bookkeeping.

    Returns
    -------
    DetectionResult
    """
    t0 = time.perf_counter()
    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0

    detections: list[Detection] = []

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            cls_name = r.names.get(cls_id, f"class_{cls_id}")
            detections.append(
                Detection(
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    class_id=cls_id,
                )
            )

    return DetectionResult(
        detections=detections,
        inference_time=elapsed,
        image_shape=image.shape,
        model_name=model_name,
    )
