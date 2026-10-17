"""
AeroVision Utilities
====================
Helper functions used across the project: file validation, image resizing,
timestamping, colour mapping, and safe temp-file cleanup.
"""

import os
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import (
    SUPPORTED_IMAGE_TYPES,
    SUPPORTED_VIDEO_TYPES,
    MAX_IMAGE_SIZE_MB,
    MAX_VIDEO_SIZE_MB,
    MAX_IMAGE_DIMENSION,
    CLASS_COLORS,
    DEFAULT_BBOX_COLOR,
    OUTPUTS_DIR,
)


# ── File Validation ─────────────────────────────────────────────────────────

def validate_image_file(uploaded_file) -> tuple[bool, str]:
    """Validate an uploaded image file's extension and size.

    Returns (is_valid, message).
    """
    name = uploaded_file.name.lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    if ext not in SUPPORTED_IMAGE_TYPES:
        return False, f"Unsupported format '.{ext}'. Use: {', '.join(SUPPORTED_IMAGE_TYPES)}"

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        return False, f"File too large ({size_mb:.1f} MB). Max: {MAX_IMAGE_SIZE_MB} MB"

    return True, "Valid image file"


def validate_video_file(uploaded_file) -> tuple[bool, str]:
    """Validate an uploaded video file's extension and size."""
    name = uploaded_file.name.lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    if ext not in SUPPORTED_VIDEO_TYPES:
        return False, f"Unsupported format '.{ext}'. Use: {', '.join(SUPPORTED_VIDEO_TYPES)}"

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_VIDEO_SIZE_MB:
        return False, f"File too large ({size_mb:.1f} MB). Max: {MAX_VIDEO_SIZE_MB} MB"

    return True, "Valid video file"


# ── Image Helpers ────────────────────────────────────────────────────────────

def resize_image(image: np.ndarray, max_dim: int = MAX_IMAGE_DIMENSION) -> np.ndarray:
    """Resize *image* so that neither side exceeds *max_dim*, preserving aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def bytes_to_cv2(file_bytes: bytes) -> Optional[np.ndarray]:
    """Decode raw bytes into a BGR OpenCV image array.

    Returns ``None`` if decoding fails.
    """
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def cv2_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR OpenCV image to RGB for display in Streamlit / Pillow."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ── Colour Mapping ──────────────────────────────────────────────────────────

def get_class_color(class_name: str) -> tuple[int, int, int]:
    """Return a deterministic BGR colour for *class_name*."""
    if class_name.lower() in CLASS_COLORS:
        return CLASS_COLORS[class_name.lower()]
    # Deterministic hash-based colour for unknown classes
    h = int(hashlib.md5(class_name.encode()).hexdigest()[:6], 16)
    return ((h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF)


# ── Timestamps ───────────────────────────────────────────────────────────────

def timestamp_str() -> str:
    """Return an ISO-8601 timestamp string in UTC."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def readable_timestamp() -> str:
    """Return a human-friendly local timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── File I/O ─────────────────────────────────────────────────────────────────

def safe_save_image(image: np.ndarray, filename: str) -> Path:
    """Save *image* (BGR) to the outputs directory and return the full path."""
    path = OUTPUTS_DIR / filename
    cv2.imwrite(str(path), image)
    return path


def cleanup_temp_files(*paths: str | Path) -> None:
    """Delete temporary files, ignoring errors."""
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


# ── Formatting ───────────────────────────────────────────────────────────────

def format_duration(seconds: float) -> str:
    """Return a human-readable duration string."""
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def truncate(text: str, max_len: int = 40) -> str:
    """Truncate *text* to *max_len* characters, adding an ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"
