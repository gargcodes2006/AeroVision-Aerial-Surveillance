"""
AeroVision Analytics
====================
Generates Plotly charts and Pandas DataFrames from detection results.
All functions take real detection data — nothing is fabricated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.detector import Detection, DetectionResult
from config import THEME


# ── DataFrame Builders ───────────────────────────────────────────────────────

def detections_to_dataframe(
    detections: list[Detection],
    source_name: str = "",
    timestamp: str | None = None,
) -> pd.DataFrame:
    """Convert a flat list of ``Detection`` objects into a tidy DataFrame.

    Columns: class_name, confidence, x1, y1, x2, y2, source, timestamp
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows = []
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        rows.append({
            "class_name": d.class_name,
            "confidence": round(d.confidence, 4),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "source": source_name,
            "timestamp": timestamp,
        })
    return pd.DataFrame(rows)


def results_to_dataframe(
    results: Sequence[DetectionResult],
    source_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Merge multiple ``DetectionResult`` objects into one DataFrame."""
    frames = []
    for i, r in enumerate(results):
        name = source_names[i] if source_names and i < len(source_names) else f"frame_{i}"
        frames.append(detections_to_dataframe(r.detections, source_name=name))
    if not frames:
        return pd.DataFrame(columns=[
            "class_name", "confidence", "x1", "y1", "x2", "y2", "source", "timestamp"
        ])
    return pd.concat(frames, ignore_index=True)


# ── Chart Helpers (common Plotly layout) ─────────────────────────────────────

def _base_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply a consistent dark-theme layout to a Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(color=THEME["text_primary"], size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME["text_secondary"], size=12),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=THEME["text_secondary"]),
        ),
    )
    fig.update_xaxes(gridcolor=THEME["border"], zerolinecolor=THEME["border"])
    fig.update_yaxes(gridcolor=THEME["border"], zerolinecolor=THEME["border"])
    return fig


# ── Charts ───────────────────────────────────────────────────────────────────

def class_distribution_chart(
    detections: list[Detection],
    title: str = "Objects Detected by Class",
) -> go.Figure:
    """Horizontal bar chart of detection counts per class."""
    if not detections:
        fig = go.Figure()
        fig.add_annotation(text="No detections to display", showarrow=False,
                           font=dict(size=14, color=THEME["text_secondary"]))
        return _base_layout(fig, title)

    counts: dict[str, int] = {}
    for d in detections:
        counts[d.class_name] = counts.get(d.class_name, 0) + 1
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    classes, values = zip(*sorted_items)

    fig = go.Figure(go.Bar(
        x=list(values),
        y=list(classes),
        orientation="h",
        marker_color=THEME["accent_cyan"],
        text=list(values),
        textposition="auto",
    ))
    return _base_layout(fig, title)


def confidence_histogram(
    detections: list[Detection],
    title: str = "Confidence Score Distribution",
    bins: int = 20,
) -> go.Figure:
    """Histogram of confidence scores."""
    if not detections:
        fig = go.Figure()
        fig.add_annotation(text="No detections to display", showarrow=False,
                           font=dict(size=14, color=THEME["text_secondary"]))
        return _base_layout(fig, title)

    confs = [d.confidence for d in detections]
    fig = go.Figure(go.Histogram(
        x=confs,
        nbinsx=bins,
        marker_color=THEME["accent_blue"],
        opacity=0.85,
    ))
    fig.update_xaxes(title_text="Confidence", range=[0, 1])
    fig.update_yaxes(title_text="Count")
    return _base_layout(fig, title)


def detections_per_frame_chart(
    results: Sequence[DetectionResult],
    title: str = "Detections per Frame",
) -> go.Figure:
    """Line chart showing the number of detections per processed frame."""
    if not results:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False,
                           font=dict(size=14, color=THEME["text_secondary"]))
        return _base_layout(fig, title)

    counts = [r.count for r in results]
    fig = go.Figure(go.Scatter(
        x=list(range(1, len(counts) + 1)),
        y=counts,
        mode="lines+markers",
        line=dict(color=THEME["accent_green"], width=2),
        marker=dict(size=5),
    ))
    fig.update_xaxes(title_text="Frame #")
    fig.update_yaxes(title_text="Detections")
    return _base_layout(fig, title)


def processing_time_chart(
    results: Sequence[DetectionResult],
    title: str = "Processing Time per Frame (ms)",
) -> go.Figure:
    """Bar chart of inference time for each frame / image."""
    if not results:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False,
                           font=dict(size=14, color=THEME["text_secondary"]))
        return _base_layout(fig, title)

    times_ms = [r.inference_time * 1000 for r in results]
    fig = go.Figure(go.Bar(
        x=list(range(1, len(times_ms) + 1)),
        y=times_ms,
        marker_color=THEME["accent_amber"],
    ))
    fig.update_xaxes(title_text="Frame #")
    fig.update_yaxes(title_text="Time (ms)")
    return _base_layout(fig, title)


# ── Summary Statistics ───────────────────────────────────────────────────────

def compute_summary(results: Sequence[DetectionResult]) -> dict:
    """Return a dict of aggregate statistics from one or more results."""
    total_dets = sum(r.count for r in results)
    total_time = sum(r.inference_time for r in results)
    confs = [d.confidence for r in results for d in r.detections]
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return {
        "total_images": len(results),
        "total_detections": total_dets,
        "processing_time": round(total_time, 3),
        "avg_confidence": round(avg_conf, 4),
    }
