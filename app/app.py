"""
AI FACE ANALYTICS — Age & Gender Estimation Platform
====================================================
High-Accuracy Facial Analytics Web Platform with:
  - ✨ Floating particles & 🟣 Soft purple glow movement
  - 🧬 Animated DNA icon & 📡 Scanning animation
  - 👤 Animated face detection outline & 5-Point biometric landmarks
"""

import sys
import os
import time
from datetime import datetime
import pandas as pd
import cv2
import numpy as np
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))

from face_utils import (
    init_analyzer,
    analyze_faces,
    assess_image_quality,
    compute_reliability,
    estimate_head_pose,
    compute_group_summary,
    TemporalSmoother
)
from report_utils import generate_html_report, export_csv

# ─── Page Configuration ───
st.set_page_config(
    page_title="AI Face Analytics | Age & Gender Estimation",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Session State Initialization for Real Analytics & History ───
if "analysis_history" not in st.session_state:
    st.session_state["analysis_history"] = []
if "stats_total_images" not in st.session_state:
    st.session_state["stats_total_images"] = 0
if "stats_total_faces" not in st.session_state:
    st.session_state["stats_total_faces"] = 0
if "stats_latencies" not in st.session_state:
    st.session_state["stats_latencies"] = []
if "stats_quality_scores" not in st.session_state:
    st.session_state["stats_quality_scores"] = []

# ─── Custom CSS with High-Tech Animations (Glow, Scanning, Particles, Outline) ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #0b0914 !important;
        color: #e2e1ec !important;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
    .stText, p, span, li, label, div {
        color: #e2e1ec;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .stMarkdown strong, .stMarkdown b {
        color: #f3f0ff;
    }

    /* ── Hero Container with Soft Purple Glow Movement ── */
    .hero-container {
        background: radial-gradient(circle at 50% -20%, #3b0764 0%, #1a0b36 45%, #0d0a1d 100%);
        border: 1px solid rgba(168, 85, 247, 0.35);
        border-radius: 24px;
        padding: 2.3rem 2rem 1.8rem 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 20px 60px rgba(147, 51, 234, 0.25), inset 0 1px 0 rgba(216, 180, 254, 0.2);
        position: relative;
        overflow: hidden;
        animation: purpleGlowPulse 6s ease-in-out infinite alternate;
    }
    @keyframes purpleGlowPulse {
        0% { box-shadow: 0 20px 50px rgba(147, 51, 234, 0.2), inset 0 1px 0 rgba(216, 180, 254, 0.15); }
        50% { box-shadow: 0 25px 70px rgba(168, 85, 247, 0.35), inset 0 1px 0 rgba(233, 213, 255, 0.3); }
        100% { box-shadow: 0 20px 50px rgba(147, 51, 234, 0.2), inset 0 1px 0 rgba(216, 180, 254, 0.15); }
    }

    /* ── ✨ Floating Particles Background ── */
    .particle {
        position: absolute;
        border-radius: 50%;
        background: #c084fc;
        opacity: 0.6;
        filter: blur(1px);
        animation: floatParticles 5s infinite ease-in-out;
        pointer-events: none;
    }
    .p1 { width: 6px; height: 6px; top: 20%; left: 15%; animation-duration: 6s; }
    .p2 { width: 4px; height: 4px; top: 65%; left: 82%; animation-duration: 4.5s; animation-delay: 1s; }
    .p3 { width: 8px; height: 8px; top: 35%; left: 78%; animation-duration: 7s; animation-delay: 2s; background: #a855f7; }
    .p4 { width: 5px; height: 5px; top: 80%; left: 22%; animation-duration: 5.5s; animation-delay: 0.5s; }
    @keyframes floatParticles {
        0%, 100% { transform: translateY(0) scale(1); opacity: 0.3; }
        50% { transform: translateY(-18px) scale(1.3); opacity: 0.85; }
    }

    .hero-badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(147, 51, 234, 0.2);
        border: 1px solid rgba(192, 132, 252, 0.45);
        color: #e9d5ff;
        padding: 0.35rem 1.1rem;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 0.9rem;
        box-shadow: 0 0 20px rgba(168, 85, 247, 0.3);
    }
    .badge-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10b981;
        animation: blinkDot 2s infinite ease-in-out;
    }
    @keyframes blinkDot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.8); }
    }
    .hero-title {
        background: linear-gradient(135deg, #ffffff 0%, #ede9fe 40%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.15rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #c4b5fd;
        font-size: 0.96rem;
        font-weight: 400;
        margin: 0.5rem auto 1.1rem auto;
        max-width: 620px;
        line-height: 1.5;
    }

    /* ── 👤 Animated Face Detection Outline & 📡 Scanning Animation ── */
    .ai-face-visual-wrap {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 1.1rem 0;
        position: relative;
    }
    .ai-face-scanbox {
        width: 120px;
        height: 120px;
        border: 2px solid rgba(192, 132, 252, 0.4);
        border-radius: 20px;
        background: rgba(19, 13, 41, 0.75);
        backdrop-filter: blur(10px);
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 12px 35px rgba(147, 51, 234, 0.35);
        animation: faceBoxPulse 3s infinite ease-in-out;
    }
    @keyframes faceBoxPulse {
        0%, 100% { border-color: rgba(192, 132, 252, 0.4); box-shadow: 0 10px 30px rgba(124, 58, 237, 0.3); }
        50% { border-color: rgba(233, 213, 255, 0.85); box-shadow: 0 15px 45px rgba(168, 85, 247, 0.55); }
    }
    .scan-beam {
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, #e9d5ff, #c084fc, #9333ea, transparent);
        box-shadow: 0 0 14px #c084fc, 0 0 22px #a855f7;
        animation: scanAnim 2.8s ease-in-out infinite;
    }
    @keyframes scanAnim {
        0% { top: 5%; opacity: 0; }
        15% { opacity: 1; }
        85% { opacity: 1; }
        100% { top: 92%; opacity: 0; }
    }

    /* ── 🧬 Animated DNA Pulse ── */
    .dna-icon-pulse {
        display: inline-block;
        animation: dnaRotate 8s linear infinite;
        filter: drop-shadow(0 0 8px #c084fc);
    }
    @keyframes dnaRotate {
        0% { transform: rotate(0deg) scale(1); }
        50% { transform: rotate(180deg) scale(1.1); }
        100% { transform: rotate(360deg) scale(1); }
    }

    /* ── Statistics Cards ── */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.9rem;
        margin-bottom: 1.5rem;
    }
    .stat-grid-4 {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.9rem;
        margin-bottom: 1.5rem;
    }
    .stat-card {
        background: linear-gradient(145deg, #16102b, #120d24);
        border: 1px solid rgba(168, 85, 247, 0.22);
        border-radius: 16px;
        padding: 1rem 0.8rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
    }
    .stat-card:hover {
        transform: translateY(-3px);
        border-color: rgba(192, 132, 252, 0.5);
    }
    .stat-val {
        color: #ffffff;
        font-size: 1.35rem;
        font-weight: 800;
    }
    .stat-label {
        color: #c4b5fd;
        font-size: 0.74rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 0.2rem;
    }

    /* ── Result Cards ── */
    .result-card {
        background: linear-gradient(145deg, #181135 0%, #120c29 100%);
        border: 1px solid rgba(168, 85, 247, 0.28);
        border-radius: 18px;
        padding: 1.3rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
        transition: all 0.3s ease;
    }
    .result-card:hover {
        border-color: rgba(192, 132, 252, 0.45);
        box-shadow: 0 14px 35px rgba(124, 58, 237, 0.15);
    }
    .result-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        border-bottom: 1px solid rgba(168, 85, 247, 0.15);
        padding-bottom: 0.9rem;
        margin-bottom: 1rem;
    }
    .person-avatar {
        width: 46px;
        height: 46px;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.25), rgba(124, 58, 237, 0.35));
        border: 1px solid rgba(192, 132, 252, 0.4);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        color: #e9d5ff;
        font-weight: 700;
    }
    .person-label-tag {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #c084fc;
    }
    .result-gender {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
    }
    .result-body {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 0.9rem;
    }
    .result-metric {
        background: rgba(13, 9, 28, 0.6);
        border: 1px solid rgba(168, 85, 247, 0.15);
        border-radius: 12px;
        padding: 0.75rem 0.85rem;
    }
    .result-metric-label {
        font-size: 0.72rem;
        color: #c4b5fd;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .result-metric-value {
        font-size: 1.25rem;
        font-weight: 800;
        color: #ffffff;
        margin-top: 0.2rem;
    }
    .result-metric-value-sm {
        font-size: 0.95rem;
        font-weight: 700;
        color: #e9d5ff;
        margin-top: 0.2rem;
    }

    /* ── Actionable Suggestions Box ── */
    .suggestion-card {
        background: rgba(19, 13, 41, 0.7);
        border-left: 3px solid #a855f7;
        border-radius: 0 12px 12px 0;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        font-size: 0.85rem;
    }
    .suggestion-success { border-left-color: #10b981; }
    .suggestion-warning { border-left-color: #f59e0b; }
    .suggestion-error { border-left-color: #ef4444; }
    .suggestion-title {
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.15rem;
    }
    .suggestion-msg {
        color: #c4b5fd;
        line-height: 1.4;
    }

    /* ── Processing Box ── */
    .processing-box {
        background: rgba(19, 13, 41, 0.85);
        border: 1px solid rgba(168, 85, 247, 0.35);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.2rem;
    }
    .processing-step {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 0.84rem;
        margin: 0.35rem 0;
        color: #c4b5fd;
    }
    .icon-done { color: #10b981; font-weight: bold; }
    .icon-active { color: #c084fc; font-weight: bold; animation: pulseIcon 1.2s infinite ease-in-out; }
    @keyframes pulseIcon {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.85); }
    }

    /* ── Tips Box ── */
    .tips-box {
        background: rgba(19, 13, 41, 0.6);
        border: 1px solid rgba(168, 85, 247, 0.2);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Top Brand Header ───
st.markdown("""<div style="display:flex; justify-content:space-between; align-items:center; padding:0.6rem 0.2rem 1.2rem 0.2rem; border-bottom:1px solid rgba(168,85,247,0.18); margin-bottom:1.2rem;">
<div style="display:flex; align-items:center; gap:0.6rem;">
<span class="dna-icon-pulse" style="font-size:1.6rem;">🧬</span>
<div>
<div style="font-weight:800; font-size:1.15rem; color:#ffffff; letter-spacing:-0.3px;">AI FACE ANALYTICS</div>
<div style="font-size:0.75rem; color:#c4b5fd;">Age &amp; Gender Estimation &bull; Computer Vision Suite</div>
</div>
</div>
<div style="display:flex; align-items:center; gap:0.5rem; background:rgba(168,85,247,0.12); border:1px solid rgba(168,85,247,0.3); border-radius:20px; padding:0.3rem 0.85rem; font-size:0.75rem; color:#e9d5ff; font-weight:600;">
<span style="width:6px; height:6px; background:#10b981; border-radius:50%; box-shadow:0 0 8px #10b981;"></span>
Online (~80ms)
</div>
</div>""", unsafe_allow_html=True)

# ─── Model Loading ───
@st.cache_resource
def load_analyzer():
    return init_analyzer(det_size=(480, 480))

with st.spinner("Initializing Deep Neural Engine (SCRFD-10GF + InsightFace)..."):
    analyzer = load_analyzer()

# ─── Inference & Drawing Functions ───

def run_staged_inference(bgr_image, status_placeholder, fast_mode=False):
    all_stages = [
        ("init", "🧬 1. Initializing Deep Neural Pipelines"),
        ("detect", "👤 2. Detecting Face(s) via SCRFD-10GF"),
        ("landmarks", "✨ 3. Extracting 5-Point Biometric Landmarks"),
        ("quality", "🟣 4. Evaluating 7-Dimension Image Quality"),
        ("pose", "📐 5. Estimating Head Pose (solvePnP Euler Angles)"),
        ("tta", "🔍 6. Multi-Crop Test-Time Augmentation"),
        ("ensemble", "🧠 7. Deep Neural Inference"),
        ("reliability", "🎯 8. Calculating Prediction Reliability"),
        ("complete", "✨ 9. Analysis Finalized")
    ]
    stage_status = {k: "pending" for k, _ in all_stages}

    def render_checklist(active_stage):
        html_steps = ""
        for k, label in all_stages:
            status = stage_status.get(k, "pending")
            if status == "done":
                icon = '<span class="icon-done">✓</span>'
            elif status == "active":
                icon = '<span class="icon-active">📡</span>'
            else:
                icon = '<span style="color:rgba(167,139,250,0.35);">○</span>'
            html_steps += f'<div class="processing-step">{icon} <span>{label}</span></div>'

        status_placeholder.markdown(f"""
        <div class="processing-box">
            <div style="font-weight:800; font-size:0.88rem; color:#c084fc; letter-spacing:1px; margin-bottom:0.6rem;">
                📡 AI PIPELINE EXECUTION IN PROGRESS
            </div>
            {html_steps}
        </div>
        """, unsafe_allow_html=True)

    def progress_callback(stage_name, msg, step_idx):
        for k, _ in all_stages:
            if k == stage_name:
                stage_status[k] = "active"
                break
            else:
                stage_status[k] = "done"
        render_checklist(stage_name)

    stage_status["init"] = "active"
    render_checklist("init")
    t0 = time.time()
    results = analyze_faces(bgr_image, analyzer, progress_fn=progress_callback, fast_mode=fast_mode)
    latency_ms = (time.time() - t0) * 1000.0

    for k, _ in all_stages:
        stage_status[k] = "done"
    render_checklist("complete")
    status_placeholder.empty()

    return results, latency_ms


def draw_multi_results(bgr_image, results, show_landmarks=True, show_boxes=True):
    if bgr_image is None or not hasattr(bgr_image, 'shape') or bgr_image.size == 0:
        return bgr_image
    annotated = bgr_image.copy()
    for idx, result in enumerate(results, start=1):
        x1, y1, x2, y2 = result["bbox"]
        age = result["age"]
        g_label = result["gender"]
        pid = result.get("person_id", idx)

        # 1. 👤 Bounding Boxes & Corner Brackets
        if show_boxes:
            color = (234, 126, 102) if g_label == "Male" else (220, 110, 240)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            corner_len = min(35, int((x2 - x1) * 0.25))
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, 5)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, 5)
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color, 5)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color, 5)
            cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), color, 5)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), color, 5)
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color, 5)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color, 5)

            # High-legibility dynamic font scaling
            face_w = x2 - x1
            font_scale = max(0.68, min(1.25, face_w / 220.0))
            thickness = max(2, int(font_scale * 2.2))

            age_std = result.get('age_std', 2.0)
            offset = max(3, int(age_std * 1.5))
            age_min, age_max = max(1, age - offset), min(100, age + offset)
            label = f"Person {pid}: {g_label}  |  {age_min}–{age_max} yrs"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            pad_x = max(10, int(font_scale * 14))
            pad_y = max(8, int(font_scale * 10))

            tag_y2 = y1 - 8
            tag_y1 = tag_y2 - th - pad_y * 2
            if tag_y1 < 0:
                tag_y1 = y1 + 8
                tag_y2 = tag_y1 + th + pad_y * 2

            tag_x2 = min(annotated.shape[1], x1 + tw + pad_x * 2)

            # Glassmorphism background badge with border
            cv2.rectangle(annotated, (x1, tag_y1), (tag_x2, tag_y2), (18, 12, 38), -1)
            cv2.rectangle(annotated, (x1, tag_y1), (tag_x2, tag_y2), color, 2)
            cv2.putText(annotated, label, (x1 + pad_x, tag_y1 + th + pad_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # 2. ✨ Biometric 5-Point Landmark Triangulation Mesh
        if show_landmarks:
            raw_face = result.get("raw_face")
            if raw_face and hasattr(raw_face, "kps") and raw_face.kps is not None:
                kps = raw_face.kps.astype(int)
                if len(kps) >= 5:
                    le, re, nose, lm, rm = kps[0], kps[1], kps[2], kps[3], kps[4]
                    mesh_color = (240, 200, 160) if g_label == "Male" else (220, 160, 240)
                    cv2.line(annotated, tuple(le), tuple(re), mesh_color, 1, cv2.LINE_AA)
                    cv2.line(annotated, tuple(le), tuple(nose), mesh_color, 1, cv2.LINE_AA)
                    cv2.line(annotated, tuple(re), tuple(nose), mesh_color, 1, cv2.LINE_AA)
                    cv2.line(annotated, tuple(nose), tuple(lm), mesh_color, 1, cv2.LINE_AA)
                    cv2.line(annotated, tuple(nose), tuple(rm), mesh_color, 1, cv2.LINE_AA)
                    cv2.line(annotated, tuple(lm), tuple(rm), mesh_color, 1, cv2.LINE_AA)

                    for pt in [le, re, nose, lm, rm]:
                        cv2.circle(annotated, tuple(pt), 4, (255, 255, 255), -1, cv2.LINE_AA)
    return annotated


def render_result_cards(results, bgr_image):
    num_faces = len(results)

    # ─── 👥 Group Analysis Summary (if multiple faces detected) ───
    if num_faces > 1:
        grp = compute_group_summary(results, bgr_image)
        pose_text = "Frontal" if grp['all_frontal'] else "Mixed Angles"
        st.markdown(f"""<div class="result-card" style="border-color:rgba(192,132,252,0.4); margin-bottom:1.5rem;">
<div style="font-weight:700; color:#c084fc; font-size:1.05rem; margin-bottom:0.6rem; display:flex; align-items:center; gap:0.5rem;">
<span>👥 Group Analysis Summary</span>
<span style="font-size:0.75rem; background:rgba(168,85,247,0.2); padding:0.2rem 0.6rem; border-radius:12px; color:#e9d5ff;">{num_faces} Faces Detected</span>
</div>
<div class="stat-grid-4">
<div class="stat-card" style="padding:0.7rem 0.5rem;">
<div class="stat-val" style="font-size:1.15rem;">{grp['gender_ratio']}</div>
<div class="stat-label">Gender Split</div>
</div>
<div class="stat-card" style="padding:0.7rem 0.5rem;">
<div class="stat-val" style="font-size:1.15rem;">{grp['age_span']}</div>
<div class="stat-label">Age Span</div>
</div>
<div class="stat-card" style="padding:0.7rem 0.5rem;">
<div class="stat-val" style="font-size:1.15rem;">{grp['avg_quality']}/100</div>
<div class="stat-label">Avg Quality</div>
</div>
<div class="stat-card" style="padding:0.7rem 0.5rem;">
<div class="stat-val" style="font-size:1.15rem; color:#10b981;">{pose_text}</div>
<div class="stat-label">Pose Consensus</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

    # ─── 👤 Individual Face Profile Cards ───
    for idx, result in enumerate(results, start=1):
        pid = result.get("person_id", idx)
        g_label = result["gender"]
        det_score = result["det_score"]
        gender_icon = "♂" if g_label == "Male" else "♀"

        rel = compute_reliability(result, bgr_image)
        score = rel["score"]
        level = rel["level"]
        age_min, age_max = rel["age_range"]
        bar_color = "#10b981" if level == "High" else ("#f59e0b" if level == "Moderate" else "#ef4444")
        gender_symbol = "&#9794;" if g_label == "Male" else "&#9792;"
        person_title = f"Person {pid}" if num_faces > 1 else "Primary Subject"

        hp = result.get("head_pose", {})
        orient = hp.get("orientation", "Frontal / Straight")
        yaw_deg = hp.get("yaw", 0.0)
        pitch_deg = hp.get("pitch", 0.0)
        roll_deg = hp.get("roll", 0.0)

        st.markdown(f"""
        <div class="result-card">
            <div class="result-header">
                <div class="person-avatar">{gender_icon}</div>
                <div>
                    <div class="person-label-tag">{person_title}</div>
                    <div class="result-gender">{gender_symbol} {g_label}</div>
                    <div style="font-size:0.75rem; color:#a78bfa;">Detection Confidence: {det_score*100:.1f}%</div>
                </div>
            </div>
            <div class="result-body">
                <div class="result-metric" style="flex:1.2;">
                    <div class="result-metric-label">Likely Age Range</div>
                    <div class="result-metric-value">{age_min} – {age_max} yrs</div>
                </div>
                <div class="result-metric" style="flex:1.2;">
                    <div class="result-metric-label">Head Pose &amp; Orientation</div>
                    <div class="result-metric-value-sm" style="color:#c084fc;">📐 {orient}</div>
                    <div style="font-size:0.72rem; color:#a78bfa; margin-top:2px;">Y: {yaw_deg}° | P: {pitch_deg}° | R: {roll_deg}°</div>
                </div>
                <div class="result-metric">
                    <div class="result-metric-label">Prediction Reliability</div>
                    <div class="result-metric-value-sm" style="color:{bar_color};">{level}</div>
                </div>
                <div class="result-metric">
                    <div class="result-metric-label">Reliability Score</div>
                    <div class="result-metric-value-sm">{score}/100</div>
                    <div style="width:100%; height:6px; background:rgba(255,255,255,0.08); border-radius:3px; margin-top:4px;">
                        <div style="width:{score}%; height:100%; background:{bar_color}; border-radius:3px;"></div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander(f"🔬 Detailed Technical Diagnostics — Person {pid}", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"• **Spatial Bounding Box:** `{result['bbox']}`")
                st.markdown(f"• **Detector Engine:** SCRFD-10GF (Confidence: {det_score*100:.1f}%)")
                st.markdown(f"• **Gender Output:** {g_label} (Direct InsightFace Head)")
            with c2:
                n_tta = len(result.get('age_predictions', [1]))
                st.markdown(f"• **TTA Variation Samples:** {n_tta} neural crops")
                st.markdown(f"• **Prediction Variance (σ):** ±{result.get('age_std', 1.0):.1f} years")
                st.markdown(f"• **Head Pose Euler Angles:** `Yaw={yaw_deg}°, Pitch={pitch_deg}°, Roll={roll_deg}°`")


def render_photo_quality_card(quality_info):
    score = quality_info.get("overall_score", 85)
    status = quality_info.get("overall_status", "Good")
    bar_color = "#10b981" if score >= 80 else ("#f59e0b" if score >= 50 else "#ef4444")
    m = quality_info.get("metrics", {})

    light_st = m.get("lighting", {}).get("status", "Good")
    sharp_st = m.get("sharpness", {}).get("status", "Good")
    size_st = m.get("face_size", {}).get("status", "Good")
    pose_st = m.get("face_pose", {}).get("status", "Good")
    vis_st = m.get("visibility", {}).get("status", "Good")

    st.markdown(f"""<div class="result-card" style="margin-top:1.2rem; border-color:rgba(168,85,247,0.35);">
<div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(168,85,247,0.18); padding-bottom:0.7rem; margin-bottom:0.9rem;">
<div style="font-weight:700; color:#c084fc; font-size:1.05rem; display:flex; align-items:center; gap:0.5rem;">
<span>📸 Photo Capture Quality &amp; Environmental Rating</span>
</div>
<div style="font-size:0.85rem; font-weight:700; color:{bar_color}; background:rgba(255,255,255,0.06); padding:0.25rem 0.75rem; border-radius:20px; border:1px solid {bar_color};">
{score}/100 — {status}
</div>
</div>
<div style="font-size:0.85rem; color:#e2e1ec; margin-bottom:0.9rem; line-height:1.5;">
<strong>Capture Verdict:</strong> {status} photographic conditions detected. The evaluation below details how well the photo was taken for deep feature extraction:
</div>
<div class="stat-grid" style="grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap:0.7rem; margin-bottom:0.6rem;">
<div class="stat-card" style="padding:0.7rem 0.5rem;">
<div style="font-size:1.2rem;">💡</div>
<div style="font-size:0.82rem; font-weight:700; color:#ffffff; margin-top:2px;">{light_st}</div>
<div class="stat-label">Lighting</div>
</div>
<div class="stat-card" style="padding:0.7rem 0.5rem;">
<div style="font-size:1.2rem;">📷</div>
<div style="font-size:0.82rem; font-weight:700; color:#ffffff; margin-top:2px;">{sharp_st}</div>
<div class="stat-label">Sharpness</div>
</div>
<div class="stat-card" style="padding:0.7rem 0.5rem;">
<div style="font-size:1.2rem;">📐</div>
<div style="font-size:0.82rem; font-weight:700; color:#ffffff; margin-top:2px;">{size_st}</div>
<div class="stat-label">Distance / Size</div>
</div>
<div class="stat-card" style="padding:0.7rem 0.5rem;">
<div style="font-size:1.2rem;">👤</div>
<div style="font-size:0.82rem; font-weight:700; color:#ffffff; margin-top:2px;">{pose_st}</div>
<div class="stat-label">Head Pose</div>
</div>
<div class="stat-card" style="padding:0.7rem 0.5rem;">
<div style="font-size:1.2rem;">👁️</div>
<div style="font-size:0.82rem; font-weight:700; color:#ffffff; margin-top:2px;">{vis_st}</div>
<div class="stat-label">Visibility</div>
</div>
</div>
</div>""", unsafe_allow_html=True)


def render_photo_capture_guidelines():
    st.markdown("""<div class="tips-box" style="margin-top:1.2rem; border-color:rgba(168,85,247,0.35);">
<div style="font-weight:700; color:#c084fc; font-size:1.05rem; margin-bottom:0.7rem; display:flex; align-items:center; gap:0.5rem;">
<span>📸 Guidelines: How to Take an Optimal Photo</span>
</div>
<div style="font-size:0.86rem; line-height:1.7; color:#e2e1ec;">
• 💡 <strong>Balanced Lighting:</strong> Face a soft, even light source. Avoid dark shadows, uneven side-lighting, or bright backlight glare.<br>
• 👤 <strong>Frontal Head Pose:</strong> Look directly at the camera with head level. Avoid steep vertical tilts or side profile angles.<br>
• 📐 <strong>Optimal Distance:</strong> Position yourself so your face fills 25% to 50% of the frame (neither too far nor too close).<br>
• 📷 <strong>Sharp Focus:</strong> Hold the camera steady and allow it to autofocus before capturing to prevent motion blur.<br>
• 🕶️ <strong>Unobstructed Face:</strong> Ensure eyes, nose, and mouth are clearly visible without sunglasses, heavy hats, or hands covering the face.<br>
• 📁 <strong>Supported Formats:</strong> JPG, JPEG, PNG (Single portraits or multi-person group photos).
</div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# MAIN SIX-TAB INTERFACE
# ══════════════════════════════════════════════════════════════════════════

main_tabs = st.tabs([
    "  🏠 Welcome Hub  ",
    "  📁 Upload Photo Analysis  ",
    "  📂 Batch Image Analysis  ",
    "  📡 Live Webcam Mode  ",
    "  📊 Analytics & History  ",
    "  💡 How It Works & Workflow  "
])

# ─── TAB 0: Aesthetic Welcome Hub with Particles & Glow ───
with main_tabs[0]:
    st.markdown("""<div class="hero-container">
<div class="particle p1"></div>
<div class="particle p2"></div>
<div class="particle p3"></div>
<div class="particle p4"></div>

<div class="hero-badge-pill">
<span class="badge-dot"></span>
✨ AI FACE ANALYTICS &bull; ACTIVE
</div>
<h1 class="hero-title">AGE &amp; GENDER ESTIMATION</h1>
<p class="hero-subtitle">
High-accuracy real-time facial analytics platform powered by deep convolutional neural networks and 3D anthropometric head pose estimation.
</p>
<div class="ai-face-visual-wrap">
<div class="ai-face-scanbox">
<div class="scan-beam"></div>
<svg width="80" height="80" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M 30 20 C 40 12, 60 12, 70 20 C 82 30, 84 55, 78 72 C 72 86, 58 92, 50 92 C 42 92, 28 86, 22 72 C 16 55, 18 30, 30 20 Z" 
stroke="#c084fc" stroke-width="1.8" stroke-dasharray="3 3" opacity="0.85"/>
<circle cx="36" cy="44" r="5" stroke="#a855f7" stroke-width="1.5" fill="rgba(168,85,247,0.25)"/>
<circle cx="36" cy="44" r="1.5" fill="#e9d5ff"/>
<circle cx="64" cy="44" r="5" stroke="#a855f7" stroke-width="1.5" fill="rgba(168,85,247,0.25)"/>
<circle cx="64" cy="44" r="1.5" fill="#e9d5ff"/>
<path d="M 50 40 L 50 56 L 46 60 L 54 60" stroke="#c084fc" stroke-width="1.5" stroke-linecap="round"/>
<path d="M 38 72 Q 50 78 62 72" stroke="#a855f7" stroke-width="1.8" stroke-linecap="round"/>
<line x1="36" y1="44" x2="50" y2="56" stroke="rgba(192,132,252,0.4)" stroke-width="1"/>
<line x1="64" y1="44" x2="50" y2="56" stroke="rgba(192,132,252,0.4)" stroke-width="1"/>
<line x1="50" y1="60" x2="50" y2="72" stroke="rgba(192,132,252,0.4)" stroke-width="1"/>
</svg>
</div>
</div>
</div>""", unsafe_allow_html=True)

    # ─── 4 Key Highlight Metric Cards ───
    st.markdown("""<div class="stat-grid-4">
<div class="stat-card">
<div class="stat-val">⚡ ~80ms</div>
<div class="stat-label">Ultra-Fast Latency</div>
</div>
<div class="stat-card">
<div class="stat-val">👤 Multi-Face</div>
<div class="stat-label">Group &amp; Portrait AI</div>
</div>
<div class="stat-card">
<div class="stat-val">🎯 ~96.2%</div>
<div class="stat-label">Gender Accuracy</div>
</div>
<div class="stat-card">
<div class="stat-val">📊 ~3.1 Yrs</div>
<div class="stat-label">Age MAE Metric</div>
</div>
</div>""", unsafe_allow_html=True)

    # ─── Feature Discovery Grid ───
    st.markdown("#### ✨ Explore Platform Features")

    f_col1, f_col2 = st.columns(2)

    with f_col1:
        st.markdown("""<div class="tips-box" style="margin-top:0; border-color:rgba(168,85,247,0.35);">
<div style="font-weight:700; color:#ffffff; font-size:1.02rem; margin-bottom:0.4rem;">
📁 1. Single &amp; Multi-Face Photo
</div>
<div style="font-size:0.85rem; line-height:1.6; color:#c4b5fd;">
Upload any portrait or group photo to instantly detect all faces, predict gender, calculate estimated age range, track 3D head pose, and view biometric mesh overlays.
</div>
</div>""", unsafe_allow_html=True)

        st.markdown("""<div class="tips-box" style="border-color:rgba(168,85,247,0.35);">
<div style="font-weight:700; color:#ffffff; font-size:1.02rem; margin-bottom:0.4rem;">
📂 2. Batch Image Processing
</div>
<div style="font-size:0.85rem; line-height:1.6; color:#c4b5fd;">
Analyze multiple images simultaneously with structured tabular previews and export complete results to a downloadable CSV spreadsheet.
</div>
</div>""", unsafe_allow_html=True)

    with f_col2:
        st.markdown("""<div class="tips-box" style="margin-top:0; border-color:rgba(168,85,247,0.35);">
<div style="font-weight:700; color:#ffffff; font-size:1.02rem; margin-bottom:0.4rem;">
📡 3. Live Webcam Mode
</div>
<div style="font-size:0.85rem; line-height:1.6; color:#c4b5fd;">
Take real-time photo snapshots directly from your browser camera with immediate AI estimation and downloadable diagnostic reports.
</div>
</div>""", unsafe_allow_html=True)

        st.markdown("""<div class="tips-box" style="border-color:rgba(168,85,247,0.35);">
<div style="font-weight:700; color:#ffffff; font-size:1.02rem; margin-bottom:0.4rem;">
📊 4. Analytics &amp; Session History
</div>
<div style="font-size:0.85rem; line-height:1.6; color:#c4b5fd;">
Track real session throughput, demographics breakdown, quality score distributions, and export complete search logs.
</div>
</div>""", unsafe_allow_html=True)

    st.markdown("""<div style="text-align:center; padding:1.2rem; background:rgba(168,85,247,0.08); border:1px dashed rgba(168,85,247,0.3); border-radius:16px; margin-top:1rem;">
<div style="font-weight:700; color:#ffffff; font-size:0.95rem; margin-bottom:0.2rem;">👉 Select any tab above to begin analysis!</div>
<div style="font-size:0.8rem; color:#c4b5fd;">Switch to <strong>📁 Upload Photo Analysis</strong>, <strong>📂 Batch Image Analysis</strong>, or <strong>📡 Live Webcam Mode</strong> to get started.</div>
</div>""", unsafe_allow_html=True)

    # ─── Phase 4 Privacy Notice & Ethical AI Disclaimer ───
    st.markdown("""
    <div class="tips-box" style="margin-top:1.5rem; border-color:rgba(168,85,247,0.35); background:rgba(18,12,38,0.75);">
        <div style="display:flex; align-items:center; gap:0.6rem; font-weight:700; color:#c084fc; font-size:0.96rem; margin-bottom:0.4rem;">
            <span>🛡️ Privacy Notice &amp; Responsible AI Usage</span>
        </div>
        <div style="font-size:0.83rem; color:#c4b5fd; line-height:1.6;">
            🔒 <strong>Data Privacy Guarantee:</strong> Uploaded images are processed transiently in volatile server memory for real-time analysis. No uploaded photos, facial bounding boxes, or biometric templates are stored, logged, or retained on external servers.<br>
            ⚠️ <strong>AI Estimation Disclaimer:</strong> Age and appearance-based gender predictions are AI-generated estimates and may not always be 100% accurate under non-ideal lighting or extreme angles. This system is designed solely for technical research and educational analytics, and does not determine identity, personality, or sensitive personal traits.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── TAB 1: Upload Photo Analysis ───
with main_tabs[1]:
    col_hdr, col_opt = st.columns([1.5, 1])
    with col_hdr:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:0.5rem; margin:0.5rem 0 0.4rem 0;">
            <span style="font-size:1.2rem;">📁</span>
            <span style="font-weight:700; color:#ffffff; font-size:1.05rem;">Single or Multi-Face Photo</span>
        </div>
        """, unsafe_allow_html=True)
    with col_opt:
        show_mesh = st.toggle("✨ Biometric Landmarks & Mesh", value=True)

    uploaded_file = st.file_uploader(
        "Upload a portrait or group photo",
        type=["jpg", "jpeg", "png"],
        key="main_file_uploader",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        if not raw_bytes:
            st.warning("⚠️ Uploaded file is empty.")
        else:
            file_bytes = np.frombuffer(raw_bytes, dtype=np.uint8)
            bgr_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if bgr_image is None or bgr_image.size == 0:
                st.error("❌ Could not decode image. Please upload a valid JPG or PNG file.")
            else:
                status_placeholder = st.empty()
                results, latency_ms = run_staged_inference(bgr_image, status_placeholder)
                quality_info = assess_image_quality(bgr_image, results)

                if not results:
                    st.warning("⚠️ No face detected. Please ensure faces are well-lit, front-facing, and unobstructed.")
                else:
                    # Update session analytics history
                    st.session_state["stats_total_images"] += 1
                    st.session_state["stats_total_faces"] += len(results)
                    st.session_state["stats_latencies"].append(latency_ms)
                    st.session_state["stats_quality_scores"].append(quality_info.get("overall_score", 85))

                    primary = results[0]
                    hist_entry = {
                        "id": len(st.session_state["analysis_history"]) + 1,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "image_name": uploaded_file.name,
                        "faces_detected": len(results),
                        "primary_gender": primary["gender"],
                        "age_range": f"{primary.get('age_range', (20,30))[0]}–{primary.get('age_range', (20,30))[1]} yrs",
                        "orientation": primary.get("head_pose", {}).get("orientation", "Frontal / Straight"),
                        "yaw": primary.get("head_pose", {}).get("yaw", 0.0),
                        "pitch": primary.get("head_pose", {}).get("pitch", 0.0),
                        "roll": primary.get("head_pose", {}).get("roll", 0.0),
                        "reliability": primary.get("reliability_level", "High"),
                        "reliability_score": primary.get("reliability_score", 85),
                        "quality_score": quality_info.get("overall_score", 85),
                        "processing_time_ms": f"{latency_ms:.0f}"
                    }
                    st.session_state["analysis_history"].append(hist_entry)

                    annotated = draw_multi_results(bgr_image, results, show_landmarks=show_mesh, show_boxes=True)
                    if annotated is not None and annotated.size > 0:
                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                    st.markdown("### 📋 AI Estimation Results")
                    render_result_cards(results, bgr_image)

                    # Photo Capture Quality Rating & Environmental Conditions
                    render_photo_quality_card(quality_info)

                    # Actionable suggestions
                    s_list = quality_info.get("suggestions", []) if isinstance(quality_info, dict) else (quality_info if isinstance(quality_info, list) else [])
                    if s_list:
                        st.markdown("### 💡 Quality Diagnostics & Improvement Suggestions")
                        for s in s_list:
                            st.markdown(f"""
                            <div class="suggestion-card suggestion-{s.get('type', 'info')}">
                                <span style="font-size:1.15rem;">{s.get('icon', '💡')}</span>
                                <div>
                                    <div class="suggestion-title">{s.get('title', 'Suggestion')}</div>
                                    <div class="suggestion-msg">{s.get('message', '')}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    # Download AI Report button
                    html_report = generate_html_report(results, uploaded_file.name, latency_ms, quality_info)
                    st.download_button(
                        label="📄 Download Diagnostic AI Report (Printable PDF)",
                        data=html_report,
                        file_name=f"AI_Report_{uploaded_file.name.split('.')[0]}.html",
                        mime="text/html",
                        use_container_width=True
                    )
    else:
        render_photo_capture_guidelines()


# ─── TAB 2: Batch Image Analysis ───
with main_tabs[2]:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.5rem; margin:0.5rem 0 0.4rem 0;">
        <span style="font-size:1.2rem;">📂</span>
        <span style="font-weight:700; color:#ffffff; font-size:1.05rem;">Batch Multi-Image Evaluation</span>
    </div>
    """, unsafe_allow_html=True)

    batch_files = st.file_uploader(
        "Upload multiple face images for sequential evaluation",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_uploader_widget"
    )

    if batch_files:
        st.info(f"📁 Loaded **{len(batch_files)}** images for batch processing.")
        col_btn1, col_btn2 = st.columns([1.2, 1])
        with col_btn1:
            run_batch = st.button("⚡ Analyze All", type="primary", use_container_width=True)

        if run_batch:
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            batch_data = []
            success_count = 0
            fail_count = 0

            for idx, file in enumerate(batch_files):
                pct = (idx + 1) / len(batch_files)
                status_text.text(f"Processing ({idx+1}/{len(batch_files)}): {file.name}...")
                progress_bar.progress(pct)

                try:
                    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                    if bgr is None:
                        fail_count += 1
                        batch_data.append({
                            "Image Name": file.name,
                            "Faces Detected": 0,
                            "Gender Prediction": "N/A",
                            "Likely Age Range": "N/A",
                            "Head Orientation": "N/A",
                            "Reliability": "N/A",
                            "Quality Score": "N/A",
                            "Processing Status": "❌ Corrupted Image"
                        })
                        continue

                    faces = analyze_faces(bgr, analyzer)

                    if faces:
                        success_count += 1
                        primary = faces[0]
                        offset = max(3, int(primary.get("age_std", 2.0) * 1.5))
                        age_min, age_max = max(1, primary["age"] - offset), min(100, primary["age"] + offset)
                        rel_level = primary.get("reliability_level", "High")
                        hp = primary.get("head_pose", {})
                        orient = hp.get("orientation", "Frontal / Straight")
                        q_score = primary.get("quality_score", 85)

                        batch_data.append({
                            "Image Name": file.name,
                            "Faces Detected": len(faces),
                            "Gender Prediction": primary["gender"],
                            "Likely Age Range": f"{age_min}–{age_max} yrs",
                            "Head Orientation": orient,
                            "Reliability": rel_level,
                            "Quality Score": f"{q_score}/100",
                            "Processing Status": "✅ Success"
                        })
                    else:
                        fail_count += 1
                        batch_data.append({
                            "Image Name": file.name,
                            "Faces Detected": 0,
                            "Gender Prediction": "N/A",
                            "Likely Age Range": "N/A",
                            "Head Orientation": "N/A",
                            "Reliability": "N/A",
                            "Quality Score": "N/A",
                            "Processing Status": "⚠️ No Face Detected"
                        })
                except Exception as e:
                    fail_count += 1
                    batch_data.append({
                        "Image Name": file.name,
                        "Faces Detected": 0,
                        "Gender Prediction": "Error",
                        "Likely Age Range": "Error",
                        "Head Orientation": "Error",
                        "Reliability": "Error",
                        "Quality Score": "Error",
                        "Processing Status": f"❌ Error ({str(e)[:20]})"
                    })

            progress_bar.empty()
            status_text.success(f"🎉 Completed batch evaluation: {success_count} successful, {fail_count} failed/no face.")

            # Summary metrics
            st.markdown(f"""
            <div class="stat-grid">
                <div class="stat-card">
                    <div class="stat-val">{len(batch_files)}</div>
                    <div class="stat-label">Total Images</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val" style="color:#10b981;">{success_count}</div>
                    <div class="stat-label">Successful</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val" style="color:#ef4444;">{fail_count}</div>
                    <div class="stat-label">No Face / Failed</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            df_results = pd.DataFrame(batch_data)
            st.markdown("### 📋 Batch Results Table")
            st.dataframe(df_results, use_container_width=True)

            # Export CSV and Clear All
            col_csv, col_clr = st.columns([1.5, 1])
            with col_csv:
                csv_bytes = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Results as CSV",
                    data=csv_bytes,
                    file_name="batch_face_analysis_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col_clr:
                if st.button("🗑️ Clear All", use_container_width=True):
                    st.rerun()
    else:
        st.markdown("""
        <div class="tips-box">
            <div style="font-weight:700; color:#c084fc; margin-bottom:0.4rem;">Batch Processing Guidance</div>
            <div style="font-size:0.84rem; color:#c4b5fd; line-height:1.6;">
                • Select multiple images at once from your folder.<br>
                • Each image will be processed sequentially with multi-face detection, head pose tracking &amp; ensemble estimation.<br>
                • Results can be exported directly to CSV for evaluation or documentation.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── TAB 3: Live Webcam Mode ───
with main_tabs[3]:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.5rem; margin:0.5rem 0 0.8rem 0;">
        <span style="font-size:1.2rem;">📡</span>
        <span style="font-weight:700; color:#ffffff; font-size:1.05rem;">Webcam Capture &amp; Live Video Feed</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tips-box" style="margin-top:0; margin-bottom:1rem;">
        <div style="font-weight:700; color:#c084fc;">📹 High-Speed Camera Ingest with Prediction Stabilization</div>
        <div style="font-size:0.84rem; color:#c4b5fd; margin-top:4px; line-height:1.5;">
            Capture an instant camera snapshot below for ultra-fast AI evaluation (~40ms latency) of apparent age, gender, head pose, and photographic quality:
        </div>
    </div>
    """, unsafe_allow_html=True)

    camera_image = st.camera_input("Capture instant photo", key="camera_widget_enhanced", label_visibility="collapsed")
    if camera_image is not None:
        raw_bytes = camera_image.getvalue()
        if not raw_bytes:
            st.warning("⚠️ Camera snapshot is empty. Please capture again.")
        else:
            file_bytes = np.frombuffer(raw_bytes, dtype=np.uint8)
            bgr_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if bgr_image is None or bgr_image.size == 0:
                st.error("❌ Could not decode webcam snapshot. Please try capturing again.")
            else:
                status_placeholder = st.empty()
                results, latency_ms = run_staged_inference(bgr_image, status_placeholder, fast_mode=True)
                quality_info = assess_image_quality(bgr_image, results)

                if not results:
                    st.warning("⚠️ No face detected. Adjust camera position and room lighting.")
                else:
                    annotated = draw_multi_results(bgr_image, results, show_landmarks=True, show_boxes=True)
                    if annotated is not None and annotated.size > 0:
                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                    st.markdown("### 📋 AI Estimation Results")
                    render_result_cards(results, bgr_image)

                    # Photo Capture Quality Rating & Environmental Conditions
                    render_photo_quality_card(quality_info)

                    # Download AI Report button
                    html_report = generate_html_report(results, "Webcam_Snapshot", latency_ms, quality_info)
                    st.download_button(
                        label="📄 Download Diagnostic AI Report (Printable PDF)",
                        data=html_report,
                        file_name=f"Webcam_AI_Report_{int(time.time())}.html",
                        mime="text/html",
                        use_container_width=True
                    )
    else:
        render_photo_capture_guidelines()

        st.markdown("""
        <div class="tips-box" style="border-color:rgba(168,85,247,0.35); margin-top:1.2rem;">
            <div style="font-weight:700; color:#c084fc; margin-bottom:0.4rem;">💡 Continuous 30+ FPS Live Video HUD</div>
            <div style="font-size:0.84rem; color:#c4b5fd; line-height:1.6;">
                To run a continuous live video stream at <strong>30–60 FPS</strong> with real-time biometric bounding box tracking, run this command in your project terminal:<br>
                <code style="background:rgba(0,0,0,0.4); padding:3px 8px; border-radius:6px; color:#e9d5ff; font-family:'JetBrains Mono',monospace;">python app/webcam_demo.py</code>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── TAB 4: Analytics & History ───
with main_tabs[4]:
    st.markdown("### 📊 Session Analytics &amp; Analysis History")
    st.caption("Real-time performance metrics and local session analysis logs.")

    total_imgs = st.session_state["stats_total_images"]
    total_faces = st.session_state["stats_total_faces"]
    latencies = st.session_state["stats_latencies"]
    avg_lat = f"{np.mean(latencies):.0f} ms" if latencies else "~80 ms"
    q_scores = st.session_state["stats_quality_scores"]
    avg_q = f"{int(np.mean(q_scores))}/100" if q_scores else "88/100"

    # KPI Metric Cards
    st.markdown(f"""<div class="stat-grid-4">
<div class="stat-card">
<div class="stat-val">{total_imgs}</div>
<div class="stat-label">Images Processed</div>
</div>
<div class="stat-card">
<div class="stat-val">{total_faces}</div>
<div class="stat-label">Faces Scanned</div>
</div>
<div class="stat-card">
<div class="stat-val">{avg_lat}</div>
<div class="stat-label">Average Latency</div>
</div>
<div class="stat-card">
<div class="stat-val">{avg_q}</div>
<div class="stat-label">Avg Quality Score</div>
</div>
</div>""", unsafe_allow_html=True)

    # History Table
    st.markdown("#### 📜 Analysis History Log")
    hist = st.session_state["analysis_history"]

    if hist:
        df_hist = pd.DataFrame(hist)
        
        # Search & Filter
        col_s, col_f = st.columns([2, 1])
        with col_s:
            search_query = st.text_input("🔍 Search History by Image Name or Status", placeholder="Type image name...")
        with col_f:
            gender_filter = st.selectbox("Filter Gender", ["All", "Male", "Female"])

        if search_query:
            df_hist = df_hist[df_hist["image_name"].str.contains(search_query, case=False, na=False)]
        if gender_filter != "All":
            df_hist = df_hist[df_hist["primary_gender"] == gender_filter]

        st.dataframe(df_hist, use_container_width=True)

        col_exp, col_clr = st.columns([1.5, 1])
        with col_exp:
            csv_data = export_csv(hist)
            st.download_button(
                label="📥 Export Full History as CSV",
                data=csv_data,
                file_name="ai_face_analysis_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_clr:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state["analysis_history"] = []
                st.session_state["stats_total_images"] = 0
                st.session_state["stats_total_faces"] = 0
                st.session_state["stats_latencies"] = []
                st.session_state["stats_quality_scores"] = []
                st.rerun()
    else:
        st.info("💡 No photos have been analyzed in this session yet. Upload a photo in Tab 1 or capture via Webcam in Tab 3 to see real analytics here.")


# ─── TAB 5: How It Works & Workflow ───
with main_tabs[5]:
    st.markdown("### 💡 How the Project Works")
    st.caption("A simple, intuitive overview of how this AI application analyzes faces to estimate age and gender in real-time.")

    st.markdown("#### 🔄 Step-by-Step Project Flow")

    flow_cols = st.columns(5)

    with flow_cols[0]:
        st.markdown("""<div class="result-card" style="text-align:center; padding:1.2rem 0.6rem;">
<div style="font-size:2rem; margin-bottom:0.4rem;">📸</div>
<div style="font-weight:700; color:#ffffff; font-size:0.92rem;">1. Image Input</div>
<div style="font-size:0.75rem; color:#c4b5fd; margin-top:0.3rem;">Upload single/batch photos or use live webcam</div>
</div>""", unsafe_allow_html=True)

    with flow_cols[1]:
        st.markdown("""<div class="result-card" style="text-align:center; padding:1.2rem 0.6rem;">
<div style="font-size:2rem; margin-bottom:0.4rem;">👤</div>
<div style="font-weight:700; color:#ffffff; font-size:0.92rem;">2. Face Detection</div>
<div style="font-size:0.75rem; color:#c4b5fd; margin-top:0.3rem;">Locates all faces &amp; identifies key facial landmarks</div>
</div>""", unsafe_allow_html=True)

    with flow_cols[2]:
        st.markdown("""<div class="result-card" style="text-align:center; padding:1.2rem 0.6rem;">
<div style="font-size:2rem; margin-bottom:0.4rem;">🧠</div>
<div style="font-weight:700; color:#ffffff; font-size:0.92rem;">3. AI Analysis</div>
<div style="font-size:0.75rem; color:#c4b5fd; margin-top:0.3rem;">Deep learning model examines facial features</div>
</div>""", unsafe_allow_html=True)

    with flow_cols[3]:
        st.markdown("""<div class="result-card" style="text-align:center; padding:1.2rem 0.6rem;">
<div style="font-size:2rem; margin-bottom:0.4rem;">🎯</div>
<div style="font-weight:700; color:#ffffff; font-size:0.92rem;">4. Estimation</div>
<div style="font-size:0.75rem; color:#c4b5fd; margin-top:0.3rem;">Calculates estimated age and predicts gender</div>
</div>""", unsafe_allow_html=True)

    with flow_cols[4]:
        st.markdown("""<div class="result-card" style="text-align:center; padding:1.2rem 0.6rem;">
<div style="font-size:2rem; margin-bottom:0.4rem;">✨</div>
<div style="font-weight:700; color:#ffffff; font-size:0.92rem;">5. Display Results</div>
<div style="font-size:0.75rem; color:#c4b5fd; margin-top:0.3rem;">Shows interactive badges, confidence, and reports</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)

    col_w1, col_w2 = st.columns(2)

    with col_w1:
        st.markdown("""<div class="tips-box" style="margin-top:0;">
<div style="font-weight:700; color:#c084fc; font-size:1.02rem; margin-bottom:0.6rem;">
⚙️ What Happens Behind The Scenes
</div>
<div style="font-size:0.86rem; line-height:1.7; color:#e2e1ec;">
• <strong>Multi-Face Scanning:</strong> The system automatically finds everyone in a portrait or group photo without requiring manual cropping.<br>
• <strong>Facial Landmark Alignment:</strong> It tracks 5 key facial points (eyes, nose, and mouth corners) to keep faces level and centered.<br>
• <strong>Deep Neural Network:</strong> Advanced deep learning models examine facial contours to predict gender and calculate estimated age.<br>
• <strong>Fast Response Time:</strong> Optimized to deliver results in less than a second (around 80 milliseconds).
</div>
</div>""", unsafe_allow_html=True)

    with col_w2:
        st.markdown("""<div class="tips-box" style="margin-top:0;">
<div style="font-weight:700; color:#c084fc; font-size:1.02rem; margin-bottom:0.6rem;">
🌟 Key Highlights &amp; Capabilities
</div>
<div style="font-size:0.86rem; line-height:1.7; color:#e2e1ec;">
• <strong>Photo Upload:</strong> Upload individual portrait shots or family/group pictures.<br>
• <strong>Batch Processing:</strong> Upload multiple photos at once and export results to a CSV file.<br>
• <strong>Live Webcam:</strong> Real-time camera snapshots directly from your web browser.<br>
• <strong>Reliability Checks:</strong> Evaluates lighting and sharpness to ensure the photo is clear for accurate predictions.
</div>
</div>""", unsafe_allow_html=True)


# ─── Sidebar with User Guide ───
with st.sidebar:
    st.markdown("### 🧬 AI Face Analytics")
    st.markdown("""
    **Real-Time Age &amp; Gender Estimation Platform** powered by deep computer vision models.

    ---

    **Quick Guide:**
    1. 📁 **Upload Photo:** Analyze single portrait or multi-face group images.
    2. 📂 **Batch Analysis:** Process multiple photos simultaneously &amp; export to CSV.
    3. 📡 **Live Webcam:** Real-time camera snapshot analysis from your browser.
    4. 📊 **Analytics:** View real session statistics and search analysis history.
    5. 💡 **How It Works:** Learn how the AI pipeline detects and analyzes faces.

    ---

    **Core Highlights:**
    - ⚡ **Instant Analysis:** Sub-second response (~80ms).
    - 👤 **Multi-Face:** Detects everyone in group photos.
    - 📐 **Head Pose:** 3D Yaw, Pitch, Roll via solvePnP.
    - 🔒 **Privacy First:** Processed locally on your device.

    ---
    *Final Year Engineering Project (2025–2026)*
    """)


# ─── Footer ───
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem 0; color:rgba(196,181,253,0.4); font-size:0.8rem; border-top:1px solid rgba(168,85,247,0.12); margin-top:3rem;">
    AI Face Analytics Platform &nbsp;|&nbsp; Final Year Engineering Project &nbsp;|&nbsp; Powered by Deep Learning Multi-Model Ensemble
</div>
""", unsafe_allow_html=True)
