"""
Tests for src.utils
===================
Covers validation helpers, image conversion, colour mapping, formatting.
"""

import os
import sys
import numpy as np
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    validate_image_file,
    validate_video_file,
    resize_image,
    bytes_to_cv2,
    cv2_to_rgb,
    get_class_color,
    format_duration,
    truncate,
    timestamp_str,
    readable_timestamp,
)


# ── Helpers to simulate Streamlit UploadedFile ──────────────────────────────

class FakeUpload:
    """Minimal stand-in for ``st.UploadedFile``."""
    def __init__(self, name: str, size_bytes: int):
        self.name = name
        self.size = size_bytes


# ── validate_image_file ─────────────────────────────────────────────────────

class TestValidateImageFile:
    def test_valid_jpg(self):
        ok, msg = validate_image_file(FakeUpload("photo.jpg", 1_000_000))
        assert ok

    def test_valid_png(self):
        ok, _ = validate_image_file(FakeUpload("photo.PNG", 500_000))
        assert ok  # case-insensitive extension check

    def test_invalid_extension(self):
        ok, msg = validate_image_file(FakeUpload("file.bmp", 100))
        assert not ok
        assert "Unsupported" in msg

    def test_too_large(self):
        ok, msg = validate_image_file(FakeUpload("big.jpg", 20 * 1024 * 1024))
        assert not ok
        assert "too large" in msg.lower()


# ── validate_video_file ─────────────────────────────────────────────────────

class TestValidateVideoFile:
    def test_valid_mp4(self):
        ok, _ = validate_video_file(FakeUpload("clip.mp4", 5_000_000))
        assert ok

    def test_invalid_extension(self):
        ok, msg = validate_video_file(FakeUpload("clip.mkv", 100))
        assert not ok

    def test_too_large(self):
        ok, msg = validate_video_file(FakeUpload("big.mp4", 200 * 1024 * 1024))
        assert not ok


# ── resize_image ─────────────────────────────────────────────────────────────

class TestResizeImage:
    def test_no_resize_needed(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = resize_image(img, max_dim=300)
        assert result.shape == (100, 200, 3)

    def test_resize_landscape(self):
        img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        result = resize_image(img, max_dim=500)
        assert result.shape[1] == 500  # width capped
        assert result.shape[0] == 250  # height proportional

    def test_resize_portrait(self):
        img = np.zeros((2000, 1000, 3), dtype=np.uint8)
        result = resize_image(img, max_dim=500)
        assert result.shape[0] == 500
        assert result.shape[1] == 250


# ── bytes_to_cv2 ────────────────────────────────────────────────────────────

class TestBytesToCv2:
    def test_valid_image(self):
        import cv2
        # Create a tiny valid PNG in memory
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".png", img)
        decoded = bytes_to_cv2(buf.tobytes())
        assert decoded is not None
        assert decoded.shape[:2] == (10, 10)

    def test_invalid_bytes(self):
        result = bytes_to_cv2(b"not an image at all")
        assert result is None


# ── cv2_to_rgb ───────────────────────────────────────────────────────────────

class TestCv2ToRgb:
    def test_channel_swap(self):
        # Pixel is pure blue in BGR → should be pure blue in RGB channel 2
        bgr = np.array([[[255, 0, 0]]], dtype=np.uint8)  # blue
        rgb = cv2_to_rgb(bgr)
        assert rgb[0, 0, 0] == 0    # R
        assert rgb[0, 0, 2] == 255  # B


# ── get_class_color ──────────────────────────────────────────────────────────

class TestGetClassColor:
    def test_known_class(self):
        color = get_class_color("person")
        assert isinstance(color, tuple) and len(color) == 3

    def test_unknown_class_is_deterministic(self):
        c1 = get_class_color("xyzzy")
        c2 = get_class_color("xyzzy")
        assert c1 == c2


# ── format_duration ──────────────────────────────────────────────────────────

class TestFormatDuration:
    def test_milliseconds(self):
        assert "ms" in format_duration(0.042)

    def test_seconds(self):
        assert "s" in format_duration(3.7)

    def test_minutes(self):
        result = format_duration(125.5)
        assert "m" in result


# ── truncate ─────────────────────────────────────────────────────────────────

class TestTruncate:
    def test_short_string(self):
        assert truncate("hi", 10) == "hi"

    def test_long_string(self):
        result = truncate("abcdefghij", 5)
        assert len(result) == 5
        assert result.endswith("…")


# ── timestamps ───────────────────────────────────────────────────────────────

class TestTimestamps:
    def test_iso_format(self):
        ts = timestamp_str()
        assert ts.endswith("Z")
        assert "T" in ts

    def test_readable(self):
        ts = readable_timestamp()
        assert len(ts) > 10
