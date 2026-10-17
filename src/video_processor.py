"""
AeroVision Video Processor
==========================
Processes uploaded videos frame-by-frame through the YOLO detector.

Two modes of operation
----------------------
* **Frame preview** – yields annotated frames one-by-one so the Streamlit UI
  can display a live preview and progress bar.
* **Full export** – writes all annotated frames to an MP4 file that can be
  downloaded.

Limitations
-----------
* Streamlit cannot natively stream video from memory; the processed file must
  be written to disk first.
* On systems without H.264 / FFMPEG support the export may fail; frame-by-
  frame preview still works.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

from config import (
    MAX_VIDEO_FRAMES,
    VIDEO_FRAME_SKIP,
    MAX_IMAGE_DIMENSION,
    OUTPUTS_DIR,
)
from src.detector import DetectionResult, run_detection
from src.image_processor import draw_detections
from src.utils import resize_image


# ── Dataclass for Video Results ──────────────────────────────────────────────

@dataclass
class VideoProcessingResult:
    """Aggregate results for an entire video."""
    total_frames: int = 0
    processed_frames: int = 0
    frame_results: list[DetectionResult] = field(default_factory=list)
    total_time: float = 0.0
    output_path: str | None = None

    @property
    def total_detections(self) -> int:
        return sum(r.count for r in self.frame_results)

    @property
    def avg_confidence(self) -> float:
        confs = [d.confidence for r in self.frame_results for d in r.detections]
        return sum(confs) / len(confs) if confs else 0.0

    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.frame_results:
            for cls, c in r.class_counts().items():
                counts[cls] = counts.get(cls, 0) + c
        return counts


# ── Frame Generator ──────────────────────────────────────────────────────────

def iterate_video_frames(
    video_path: str,
    model,
    confidence: float,
    model_name: str = "",
    max_dim: int = MAX_IMAGE_DIMENSION,
    frame_skip: int = VIDEO_FRAME_SKIP,
    max_frames: int = MAX_VIDEO_FRAMES,
) -> Generator[tuple[np.ndarray, DetectionResult, int, int], None, None]:
    """Yield ``(annotated_rgb, result, frame_idx, total_frames)`` for each
    processed frame.

    The generator opens the video with OpenCV, reads frame-by-frame, runs
    detection, draws boxes, and yields the annotated RGB image.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_idx = 0
    processed = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            if processed >= max_frames:
                break

            frame = resize_image(frame, max_dim)
            result = run_detection(model, frame, confidence, model_name)
            annotated = draw_detections(frame, result.detections)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            yield annotated_rgb, result, frame_idx, total

            frame_idx += 1
            processed += 1
    finally:
        cap.release()


# ── Full Video Export ────────────────────────────────────────────────────────

def process_video_to_file(
    video_path: str,
    model,
    confidence: float,
    model_name: str = "",
    max_dim: int = MAX_IMAGE_DIMENSION,
    frame_skip: int = VIDEO_FRAME_SKIP,
    max_frames: int = MAX_VIDEO_FRAMES,
    progress_callback=None,
) -> VideoProcessingResult:
    """Process an entire video and write the annotated output to an MP4 file.

    Parameters
    ----------
    progress_callback : callable, optional
        Called with ``(processed_count, total_frames)`` after each frame.

    Returns
    -------
    VideoProcessingResult
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine output dimensions after possible resize
    dummy = np.zeros((height, width, 3), dtype=np.uint8)
    dummy = resize_image(dummy, max_dim)
    out_h, out_w = dummy.shape[:2]

    # Output file
    out_path = str(OUTPUTS_DIR / "processed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps / max(frame_skip, 1), (out_w, out_h))

    vr = VideoProcessingResult()
    t0 = time.perf_counter()
    frame_idx = 0
    processed = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            if processed >= max_frames:
                break

            frame = resize_image(frame, max_dim)
            result = run_detection(model, frame, confidence, model_name)
            annotated = draw_detections(frame, result.detections)

            # Ensure exact output size
            if annotated.shape[:2] != (out_h, out_w):
                annotated = cv2.resize(annotated, (out_w, out_h))

            writer.write(annotated)
            vr.frame_results.append(result)

            frame_idx += 1
            processed += 1

            if progress_callback:
                progress_callback(processed, total)
    finally:
        cap.release()
        writer.release()

    vr.total_frames = total
    vr.processed_frames = processed
    vr.total_time = time.perf_counter() - t0
    vr.output_path = out_path

    return vr
