"""
Tests for src.analytics
=======================
Covers DataFrame construction, chart generation, and summary statistics.
These tests do NOT require a GPU or YOLO model — they operate on synthetic
Detection / DetectionResult objects.
"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import Detection, DetectionResult
from src.analytics import (
    detections_to_dataframe,
    results_to_dataframe,
    class_distribution_chart,
    confidence_histogram,
    detections_per_frame_chart,
    processing_time_chart,
    compute_summary,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_detections(n: int = 5) -> list[Detection]:
    """Create *n* dummy detections."""
    names = ["person", "car", "truck", "bicycle", "dog"]
    return [
        Detection(
            class_name=names[i % len(names)],
            confidence=0.5 + i * 0.08,
            bbox=(10 * i, 20 * i, 10 * i + 50, 20 * i + 50),
            class_id=i,
        )
        for i in range(n)
    ]


def _make_result(n_dets: int = 5, time_s: float = 0.1) -> DetectionResult:
    return DetectionResult(
        detections=_make_detections(n_dets),
        inference_time=time_s,
        image_shape=(480, 640, 3),
        model_name="test",
    )


# ── detections_to_dataframe ─────────────────────────────────────────────────

class TestDetectionsToDataFrame:
    def test_columns(self):
        df = detections_to_dataframe(_make_detections(3), source_name="img.jpg")
        expected_cols = {"class_name", "confidence", "x1", "y1", "x2", "y2", "source", "timestamp"}
        assert expected_cols == set(df.columns)

    def test_row_count(self):
        df = detections_to_dataframe(_make_detections(4))
        assert len(df) == 4

    def test_empty(self):
        df = detections_to_dataframe([])
        assert len(df) == 0


# ── results_to_dataframe ────────────────────────────────────────────────────

class TestResultsToDataFrame:
    def test_merge(self):
        r1 = _make_result(3)
        r2 = _make_result(2)
        df = results_to_dataframe([r1, r2], source_names=["a.jpg", "b.jpg"])
        assert len(df) == 5

    def test_empty(self):
        df = results_to_dataframe([])
        assert len(df) == 0


# ── Charts (just check they return a Figure without error) ──────────────────

class TestCharts:
    def test_class_distribution_nonempty(self):
        fig = class_distribution_chart(_make_detections(5))
        assert fig is not None

    def test_class_distribution_empty(self):
        fig = class_distribution_chart([])
        assert fig is not None  # should show "no detections" annotation

    def test_confidence_histogram(self):
        fig = confidence_histogram(_make_detections(10))
        assert fig is not None

    def test_detections_per_frame(self):
        fig = detections_per_frame_chart([_make_result(3), _make_result(5)])
        assert fig is not None

    def test_processing_time(self):
        fig = processing_time_chart([_make_result(3, 0.05), _make_result(5, 0.12)])
        assert fig is not None


# ── compute_summary ──────────────────────────────────────────────────────────

class TestComputeSummary:
    def test_basic(self):
        r1 = _make_result(3, 0.1)
        r2 = _make_result(2, 0.2)
        s = compute_summary([r1, r2])
        assert s["total_images"] == 2
        assert s["total_detections"] == 5
        assert s["processing_time"] == pytest.approx(0.3, abs=0.01)
        assert 0 < s["avg_confidence"] < 1

    def test_empty(self):
        s = compute_summary([])
        assert s["total_detections"] == 0
        assert s["avg_confidence"] == 0.0
