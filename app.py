"""
AeroVision — AI-Powered Aerial Image & Video Analytics
=======================================================
Main Streamlit application.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ── Ensure project root is importable ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_CONFIDENCE,
    MIN_CONFIDENCE,
    MAX_CONFIDENCE,
    CONFIDENCE_STEP,
    SUPPORTED_IMAGE_TYPES,
    SUPPORTED_VIDEO_TYPES,
    MAX_IMAGE_DIMENSION,
    MAX_VIDEO_FRAMES,
    VIDEO_FRAME_SKIP,
    ASSETS_DIR,
    OUTPUTS_DIR,
    THEME,
)
from src.detector import load_model, run_detection, DetectionResult
from src.image_processor import process_image, draw_detections
from src.video_processor import iterate_video_frames, process_video_to_file
from src.analytics import (
    detections_to_dataframe,
    results_to_dataframe,
    class_distribution_chart,
    confidence_histogram,
    detections_per_frame_chart,
    processing_time_chart,
    compute_summary,
)
from src.utils import (
    validate_image_file,
    validate_video_file,
    format_duration,
    readable_timestamp,
    cleanup_temp_files,
    cv2_to_rgb,
)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AeroVision — Aerial Analytics",
    page_icon="🛩️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = f"""
<style>
/* ── Global ──────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="stApp"] {{
    font-family: 'Inter', sans-serif;
    background-color: {THEME["bg_primary"]};
    color: {THEME["text_primary"]};
}}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background-color: {THEME["bg_secondary"]};
    border-right: 1px solid {THEME["border"]};
}}
section[data-testid="stSidebar"] .stRadio label {{
    color: {THEME["text_primary"]} !important;
}}

/* ── Metric Cards ────────────────────────────────────────────────────────── */
.metric-card {{
    background: linear-gradient(135deg, {THEME["bg_card"]}cc, {THEME["bg_secondary"]}cc);
    border: 1px solid {THEME["border"]};
    border-radius: 14px;
    padding: 22px 24px;
    text-align: center;
    backdrop-filter: blur(6px);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}}
.metric-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 6px 24px rgba(6, 182, 212, 0.12);
}}
.metric-value {{
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, {THEME["accent_cyan"]}, {THEME["accent_blue"]});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.metric-label {{
    font-size: 0.82rem;
    color: {THEME["text_secondary"]};
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* ── Section Header ──────────────────────────────────────────────────────── */
.section-header {{
    font-size: 1.15rem;
    font-weight: 600;
    color: {THEME["accent_cyan"]};
    border-bottom: 2px solid {THEME["accent_cyan"]}33;
    padding-bottom: 8px;
    margin: 28px 0 16px;
}}

/* ── Hero Banner ─────────────────────────────────────────────────────────── */
.hero-banner {{
    background: linear-gradient(135deg, {THEME["bg_card"]}, {THEME["bg_secondary"]});
    border: 1px solid {THEME["border"]};
    border-radius: 16px;
    padding: 36px 32px;
    margin-bottom: 28px;
    text-align: center;
}}
.hero-title {{
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, {THEME["accent_cyan"]}, {THEME["accent_blue"]});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}}
.hero-sub {{
    color: {THEME["text_secondary"]};
    font-size: 1rem;
    margin-top: 6px;
}}

/* ── Status badge ────────────────────────────────────────────────────────── */
.status-badge {{
    display: inline-block;
    padding: 3px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
}}
.badge-success {{ background: {THEME["accent_green"]}22; color: {THEME["accent_green"]}; border: 1px solid {THEME["accent_green"]}44; }}
.badge-info    {{ background: {THEME["accent_cyan"]}22;  color: {THEME["accent_cyan"]};  border: 1px solid {THEME["accent_cyan"]}44; }}
.badge-warn    {{ background: {THEME["accent_amber"]}22; color: {THEME["accent_amber"]}; border: 1px solid {THEME["accent_amber"]}44; }}

/* ── Hide Streamlit defaults ─────────────────────────────────────────────── */
#MainMenu, footer, header {{visibility: hidden;}}

/* ── Tables ──────────────────────────────────────────────────────────────── */
.stDataFrame {{ border-radius: 10px; overflow: hidden; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

if "results" not in st.session_state:
    st.session_state.results: list[DetectionResult] = []
if "result_names" not in st.session_state:
    st.session_state.result_names: list[str] = []


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — NAVIGATION & SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center; padding:18px 0 10px;">
            <span style="font-size:2.2rem;">🛩️</span><br>
            <span style="font-size:1.3rem; font-weight:700;
                  background: linear-gradient(90deg, {THEME['accent_cyan']}, {THEME['accent_blue']});
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            AeroVision</span><br>
            <span style="font-size:0.75rem; color:{THEME['text_secondary']};">
            Aerial Analytics Dashboard</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🖼️ Image Detection", "🎬 Video Detection",
         "📊 Analytics", "⚙️ Settings", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.divider()

    # ── Model Settings (always visible) ──────────────────────────────────────
    st.markdown(f"<p class='section-header'>Model Settings</p>", unsafe_allow_html=True)
    model_key = st.selectbox("Model", list(AVAILABLE_MODELS.keys()), index=0)
    confidence = st.slider(
        "Confidence Threshold",
        min_value=MIN_CONFIDENCE,
        max_value=MAX_CONFIDENCE,
        value=DEFAULT_CONFIDENCE,
        step=CONFIDENCE_STEP,
    )
    max_dim = st.slider(
        "Max Resolution (px)",
        min_value=320,
        max_value=3840,
        value=MAX_IMAGE_DIMENSION,
        step=64,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL CACHING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_model(key: str):
    """Load and cache the YOLO model so it persists across reruns."""
    return load_model(key)


def safe_load_model(key: str):
    """Load model with a nice spinner, return (model, error_msg)."""
    try:
        with st.spinner("Loading model weights — first run downloads ~6 MB…"):
            model = get_model(key)
        return model, ""
    except Exception as exc:
        return None, str(exc)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: METRIC CARD ROW
# ══════════════════════════════════════════════════════════════════════════════

def metric_card(value, label):
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def page_dashboard():
    st.markdown(
        """<div class="hero-banner">
            <p class="hero-title">AeroVision</p>
            <p class="hero-sub">AI-Powered Aerial Image &amp; Video Analytics</p>
        </div>""",
        unsafe_allow_html=True,
    )

    results: list[DetectionResult] = st.session_state.results

    if results:
        summary = compute_summary(results)
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card(summary["total_images"], "Images / Frames")
        with c2: metric_card(summary["total_detections"], "Objects Detected")
        with c3: metric_card(format_duration(summary["processing_time"]), "Processing Time")
        with c4: metric_card(f"{summary['avg_confidence']:.0%}", "Avg Confidence")

        st.markdown("<p class='section-header'>Detection Overview</p>", unsafe_allow_html=True)
        all_dets = [d for r in results for d in r.detections]
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(class_distribution_chart(all_dets), use_container_width=True)
        with col_b:
            st.plotly_chart(confidence_histogram(all_dets), use_container_width=True)
    else:
        st.markdown(
            f"""<div style="text-align:center; padding:60px 20px; color:{THEME['text_secondary']};">
                <span style="font-size:3rem;">📡</span><br><br>
                <b>No detections yet</b><br>
                Upload an image or video from the sidebar to get started.
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""<div style="text-align:center; margin-top:24px;">
            <span class="status-badge badge-info">Session Active</span>
            &nbsp;&nbsp;
            <span style="color:{THEME['text_secondary']}; font-size:0.8rem;">
            {readable_timestamp()}</span>
        </div>""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: IMAGE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def page_image_detection():
    st.markdown("<p class='section-header'>🖼️ Image Object Detection</p>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload an aerial image",
        type=SUPPORTED_IMAGE_TYPES,
        help=f"Supported: {', '.join(SUPPORTED_IMAGE_TYPES)} — max {__import__('config').MAX_IMAGE_SIZE_MB} MB",
    )

    # ── Sample demo button ───────────────────────────────────────────────────
    sample_path = ASSETS_DIR / "sample_aerial.jpg"
    if not uploaded and sample_path.exists():
        if st.button("🎯 Run Demo with Sample Image"):
            uploaded = sample_path  # path triggers alternative branch below

    if uploaded is None:
        st.info("Upload an image above, or place `sample_aerial.jpg` in the `assets/` folder to enable the demo button.")
        return

    # Read bytes from UploadedFile or path
    if isinstance(uploaded, Path):
        file_bytes = uploaded.read_bytes()
        filename = uploaded.name
    else:
        valid, msg = validate_image_file(uploaded)
        if not valid:
            st.error(msg)
            return
        file_bytes = uploaded.getvalue()
        filename = uploaded.name

    # Load model
    model, err = safe_load_model(model_key)
    if err:
        st.error(f"Model loading failed: {err}")
        return

    # Process
    original_rgb, annotated_rgb, result, error = process_image(
        file_bytes, model, confidence, AVAILABLE_MODELS[model_key], max_dim
    )
    if error:
        st.error(error)
        return

    # Display side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Original** — `{filename}`")
        st.image(original_rgb, use_container_width=True)
    with col2:
        st.markdown(f"**Annotated** — {result.count} object(s) detected")
        st.image(annotated_rgb, use_container_width=True)

    # ── Summary badges ───────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    with m1: metric_card(result.count, "Objects Detected")
    with m2: metric_card(format_duration(result.inference_time), "Inference Time")
    with m3:
        avg_c = (sum(d.confidence for d in result.detections) / result.count) if result.count else 0
        metric_card(f"{avg_c:.0%}", "Avg Confidence")

    # ── Detections table ─────────────────────────────────────────────────────
    if result.detections:
        st.markdown("<p class='section-header'>Detection Details</p>", unsafe_allow_html=True)
        df = detections_to_dataframe(result.detections, source_name=filename)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Downloads ────────────────────────────────────────────────────────
        st.markdown("<p class='section-header'>Export</p>", unsafe_allow_html=True)
        dl1, dl2 = st.columns(2)
        with dl1:
            # Annotated image
            from PIL import Image
            pil_img = Image.fromarray(annotated_rgb)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            st.download_button(
                "⬇️ Download Annotated Image",
                buf.getvalue(),
                file_name=f"aerovision_{filename}",
                mime="image/png",
            )
        with dl2:
            csv_data = df.to_csv(index=False)
            st.download_button(
                "⬇️ Download CSV Report",
                csv_data,
                file_name=f"aerovision_{Path(filename).stem}_detections.csv",
                mime="text/csv",
            )
    else:
        st.warning("No objects detected at the current confidence threshold. Try lowering it in the sidebar.")

    # Store for analytics dashboard
    st.session_state.results.append(result)
    st.session_state.result_names.append(filename)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: VIDEO DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def page_video_detection():
    st.markdown("<p class='section-header'>🎬 Video Object Detection</p>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a video",
        type=SUPPORTED_VIDEO_TYPES,
        help=f"Supported: {', '.join(SUPPORTED_VIDEO_TYPES)} — max {__import__('config').MAX_VIDEO_SIZE_MB} MB",
    )

    if uploaded is None:
        st.info("Upload a short aerial video to begin frame-by-frame detection.")
        return

    valid, msg = validate_video_file(uploaded)
    if not valid:
        st.error(msg)
        return

    model, err = safe_load_model(model_key)
    if err:
        st.error(f"Model loading failed: {err}")
        return

    # Advanced settings
    with st.expander("⚙️ Processing Settings"):
        frame_skip = st.slider("Process every Nth frame", 1, 10, VIDEO_FRAME_SKIP)
        max_frames = st.slider("Max frames to process", 10, MAX_VIDEO_FRAMES, min(300, MAX_VIDEO_FRAMES))

    # Write uploaded bytes to a temp file so OpenCV can read it
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        # ── Mode selector ────────────────────────────────────────────────────
        mode = st.radio("Processing mode", ["🔍 Frame Preview", "💾 Full Export"], horizontal=True)

        if st.button("▶️ Start Processing", type="primary"):
            if mode.startswith("🔍"):
                _video_preview(tmp_path, model, frame_skip, max_frames)
            else:
                _video_export(tmp_path, model, frame_skip, max_frames)
    finally:
        if tmp_path:
            cleanup_temp_files(tmp_path)


def _video_preview(tmp_path, model, frame_skip, max_frames):
    """Frame-by-frame preview mode."""
    progress = st.progress(0, text="Processing frames…")
    frame_placeholder = st.empty()
    status_text = st.empty()
    t0 = time.perf_counter()
    frame_results: list[DetectionResult] = []

    for annotated_rgb, result, idx, total in iterate_video_frames(
        tmp_path, model, confidence, AVAILABLE_MODELS[model_key],
        max_dim, frame_skip, max_frames,
    ):
        frame_results.append(result)
        frame_placeholder.image(annotated_rgb, caption=f"Frame {idx} — {result.count} detections", use_container_width=True)
        pct = min(len(frame_results) / max(total // max(frame_skip, 1), 1), 1.0)
        progress.progress(pct, text=f"Frame {len(frame_results)} / ~{total // max(frame_skip, 1)}")

    elapsed = time.perf_counter() - t0
    progress.progress(1.0, text="Done!")
    status_text.success(f"Processed {len(frame_results)} frames in {format_duration(elapsed)}")

    _show_video_analytics(frame_results)

    # Store for dashboard
    st.session_state.results.extend(frame_results)
    st.session_state.result_names.extend([f"frame_{i}" for i in range(len(frame_results))])


def _video_export(tmp_path, model, frame_skip, max_frames):
    """Full export mode — writes annotated video to disk."""
    progress = st.progress(0, text="Exporting video…")

    def cb(done, total):
        pct = min(done / max(total // max(frame_skip, 1), 1), 1.0)
        progress.progress(pct, text=f"Frame {done} / ~{total // max(frame_skip, 1)}")

    vr = process_video_to_file(
        tmp_path, model, confidence, AVAILABLE_MODELS[model_key],
        max_dim, frame_skip, max_frames, progress_callback=cb,
    )
    progress.progress(1.0, text="Done!")

    st.success(f"Processed {vr.processed_frames} frames in {format_duration(vr.total_time)}")

    if vr.output_path and os.path.isfile(vr.output_path):
        with open(vr.output_path, "rb") as f:
            st.download_button(
                "⬇️ Download Processed Video",
                f.read(),
                file_name="aerovision_processed.mp4",
                mime="video/mp4",
            )

    _show_video_analytics(vr.frame_results)

    st.session_state.results.extend(vr.frame_results)
    st.session_state.result_names.extend([f"frame_{i}" for i in range(len(vr.frame_results))])


def _show_video_analytics(frame_results):
    """Charts specific to a video processing run."""
    if not frame_results:
        return
    st.markdown("<p class='section-header'>Video Analytics</p>", unsafe_allow_html=True)
    all_dets = [d for r in frame_results for d in r.detections]
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(class_distribution_chart(all_dets), use_container_width=True)
    with c2:
        st.plotly_chart(detections_per_frame_chart(frame_results), use_container_width=True)

    # CSV export
    df = results_to_dataframe(frame_results)
    if not df.empty:
        csv_data = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Video Detections CSV",
            csv_data,
            file_name="aerovision_video_detections.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def page_analytics():
    st.markdown("<p class='section-header'>📊 Detection Analytics</p>", unsafe_allow_html=True)

    results: list[DetectionResult] = st.session_state.results
    if not results:
        st.info("Run image or video detection first to populate analytics.")
        return

    summary = compute_summary(results)
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card(summary["total_images"], "Total Processed")
    with c2: metric_card(summary["total_detections"], "Total Detections")
    with c3: metric_card(format_duration(summary["processing_time"]), "Total Time")
    with c4: metric_card(f"{summary['avg_confidence']:.0%}", "Avg Confidence")

    all_dets = [d for r in results for d in r.detections]

    tab1, tab2, tab3, tab4 = st.tabs(["Class Distribution", "Confidence", "Per-Frame", "Processing Time"])
    with tab1:
        st.plotly_chart(class_distribution_chart(all_dets), use_container_width=True)
    with tab2:
        st.plotly_chart(confidence_histogram(all_dets), use_container_width=True)
    with tab3:
        st.plotly_chart(detections_per_frame_chart(results), use_container_width=True)
    with tab4:
        st.plotly_chart(processing_time_chart(results), use_container_width=True)

    # Full session CSV
    st.markdown("<p class='section-header'>Session Export</p>", unsafe_allow_html=True)
    df = results_to_dataframe(results, st.session_state.result_names)
    st.dataframe(df, use_container_width=True, hide_index=True)
    csv = df.to_csv(index=False)
    st.download_button("⬇️ Export Full Session CSV", csv, "aerovision_session.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

def page_settings():
    st.markdown("<p class='section-header'>⚙️ Model & Processing Settings</p>", unsafe_allow_html=True)

    st.markdown("#### Current Configuration")
    config_data = {
        "Model": AVAILABLE_MODELS[model_key],
        "Confidence Threshold": f"{confidence:.0%}",
        "Max Resolution": f"{max_dim} px",
        "Supported Image Types": ", ".join(SUPPORTED_IMAGE_TYPES),
        "Supported Video Types": ", ".join(SUPPORTED_VIDEO_TYPES),
        "Max Video Frames": MAX_VIDEO_FRAMES,
    }
    st.table(pd.DataFrame(config_data.items(), columns=["Setting", "Value"]))

    st.markdown("#### Model Information")
    model, err = safe_load_model(model_key)
    if err:
        st.error(err)
    else:
        st.success(f"✅ Model **{AVAILABLE_MODELS[model_key]}** loaded and ready.")
        st.markdown(
            f"""
            | Property | Value |
            |----------|-------|
            | Weight File | `{AVAILABLE_MODELS[model_key]}` |
            | Classes | 80 COCO classes |
            | Source | [Ultralytics Hub](https://github.com/ultralytics/ultralytics) |
            | Task | Object Detection |
            """
        )

    st.markdown("#### Session Data")
    if st.button("🗑️ Clear Session Results"):
        st.session_state.results = []
        st.session_state.result_names = []
        st.success("Session data cleared.")
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════

def page_about():
    st.markdown(
        """<div class="hero-banner">
            <p class="hero-title">About AeroVision</p>
            <p class="hero-sub">Educational AI/CV Research Prototype</p>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
### Purpose

**AeroVision** is a portfolio project built to demonstrate core skills in
computer vision, object detection, and data analytics — areas relevant to
AI/ML internships at research organizations.

This is an **educational prototype**. It uses:
- Publicly available pretrained models (YOLOv8 by Ultralytics)
- Open-source libraries (OpenCV, Streamlit, Plotly)
- No classified data, military hardware, or live surveillance feeds

### How It Works

1. **Upload** an aerial image or short video clip.
2. **YOLOv8** runs inference on each frame, predicting bounding boxes, class
   labels, and confidence scores.
3. Annotated results are displayed alongside interactive analytics charts.
4. Detection data can be exported as CSV for further analysis.

### Key Concepts

| Concept | Plain-English Explanation |
|---------|--------------------------|
| **Object Detection** | Finding *what* objects are in an image and *where* they are (bounding boxes). |
| **YOLO (You Only Look Once)** | A fast, single-pass neural network architecture for real-time detection. |
| **Confidence Score** | The model's estimated probability (0–1) that a detection is correct. |
| **Bounding Box** | A rectangle `(x1, y1, x2, y2)` marking the detected object's location. |
| **Pretrained Model** | A model already trained on a large dataset (COCO — 80 everyday object classes). |

### Important Disclaimer

> Model accuracy depends on the dataset, camera angle, image quality,
> object size, and lighting. Pretrained COCO models are not optimized for
> specialized aerial/satellite imagery. For production aerial detection,
> fine-tuning on a domain-specific dataset is necessary.

### Technology Stack

| Layer | Technology |
|-------|-----------|
| Web UI | Streamlit |
| Object Detection | Ultralytics YOLOv8 |
| Image / Video Processing | OpenCV, Pillow |
| Data Handling | Pandas, NumPy |
| Visualization | Plotly |
| Testing | Pytest |

---

<div style="text-align:center; color:{THEME['text_secondary']}; font-size:0.8rem; padding-top:20px;">
Built with ❤️ as an educational AI/CV research prototype.<br>
Not affiliated with any government or military organization.
</div>
""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════════════════════

PAGE_MAP = {
    "🏠 Dashboard":       page_dashboard,
    "🖼️ Image Detection": page_image_detection,
    "🎬 Video Detection":  page_video_detection,
    "📊 Analytics":        page_analytics,
    "⚙️ Settings":        page_settings,
    "ℹ️ About":           page_about,
}

PAGE_MAP[page]()
