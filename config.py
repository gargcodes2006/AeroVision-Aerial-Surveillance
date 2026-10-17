"""
AeroVision Configuration
========================
Central configuration for the AeroVision application.
All tuneable parameters live here so they're easy to find and modify.
"""

import os
from pathlib import Path

# ── Project Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Ensure output directory exists at import time
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Model Settings ───────────────────────────────────────────────────────────
# Available YOLO models (Ultralytics will auto-download weights on first use)
AVAILABLE_MODELS = {
    "YOLOv8 Nano  (fastest)": "yolov8n.pt",
    "YOLOv8 Small (balanced)": "yolov8s.pt",
    "YOLOv8 Medium (accurate)": "yolov8m.pt",
}
DEFAULT_MODEL = "YOLOv8 Nano  (fastest)"

DEFAULT_CONFIDENCE = 0.35
MIN_CONFIDENCE = 0.05
MAX_CONFIDENCE = 0.95
CONFIDENCE_STEP = 0.05

# ── Image Processing ────────────────────────────────────────────────────────
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]
MAX_IMAGE_SIZE_MB = 15
MAX_IMAGE_DIMENSION = 1920  # px – images larger than this are resized

# ── Video Processing ────────────────────────────────────────────────────────
SUPPORTED_VIDEO_TYPES = ["mp4", "avi", "mov"]
MAX_VIDEO_SIZE_MB = 100
MAX_VIDEO_FRAMES = 900  # safety cap for very long videos
VIDEO_FRAME_SKIP = 1     # process every Nth frame (1 = every frame)

# ── UI Colours (CSS-compatible values) ──────────────────────────────────────
THEME = {
    "bg_primary":    "#0a0e1a",
    "bg_secondary":  "#111827",
    "bg_card":       "#1a2235",
    "accent_cyan":   "#06b6d4",
    "accent_blue":   "#3b82f6",
    "accent_green":  "#10b981",
    "accent_amber":  "#f59e0b",
    "accent_red":    "#ef4444",
    "text_primary":  "#f1f5f9",
    "text_secondary":"#94a3b8",
    "border":        "#1e293b",
}

# ── Bounding-Box Drawing ────────────────────────────────────────────────────
BBOX_THICKNESS = 2
BBOX_FONT_SCALE = 0.55
BBOX_LABEL_BG_ALPHA = 0.70

# Class-name → colour mapping (BGR for OpenCV)
CLASS_COLORS = {
    "person":     (0, 200, 255),
    "car":        (255, 100, 0),
    "truck":      (255, 50, 50),
    "bus":        (50, 200, 50),
    "bicycle":    (200, 200, 0),
    "motorcycle": (200, 0, 200),
    "airplane":   (0, 255, 200),
    "boat":       (100, 200, 255),
    "bird":       (255, 255, 0),
    "dog":        (0, 150, 255),
    "cat":        (150, 0, 255),
}
DEFAULT_BBOX_COLOR = (0, 220, 220)  # cyan fallback
