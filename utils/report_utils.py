"""
AI Analysis Report & CSV Export Utilities
=========================================
Generates professional printable HTML reports and CSV exports for final-year project analysis.
"""

import io
import csv
from datetime import datetime


def generate_html_report(results, image_name="Uploaded_Image", processing_time_ms=250.0, quality_info=None):
    """
    Generate an elegant, printable HTML report with dark/light print styling.
    """
    timestamp = datetime.now().strftime("%B %d, %Y - %H:%M:%S")
    num_faces = len(results) if results else 0
    overall_quality_str = quality_info.get("overall_label", "Good (85/100)") if quality_info else "Good (85/100)"
    overall_quality_score = quality_info.get("overall_score", 85) if quality_info else 85

    # Face result rows
    face_rows = ""
    for idx, face in enumerate(results or [], start=1):
        pid = face.get("person_id", idx)
        gender = face.get("gender", "Unknown")
        age = face.get("age", 25)
        offset = max(3, int(face.get("age_std", 2.0) * 1.5))
        age_min, age_max = max(1, age - offset), min(100, age + offset)
        rel_level = face.get("reliability_level", "High")
        rel_score = face.get("reliability_score", 85)
        det_score = face.get("det_score", 0.95)
        hp = face.get("head_pose", {})
        orient = hp.get("orientation", "Frontal / Straight")
        yaw_val = hp.get("yaw", 0.0)
        pitch_val = hp.get("pitch", 0.0)

        face_rows += f"""
        <tr style="border-bottom: 1px solid #332d56;">
            <td style="padding: 12px; font-weight: 700; color: #c084fc;">Person {pid}</td>
            <td style="padding: 12px;">{gender}</td>
            <td style="padding: 12px; font-weight: 700; font-size: 1.05em; color: #e9d5ff;">{age_min}–{age_max} yrs</td>
            <td style="padding: 12px; color: #c4b5fd; font-size: 0.9em;">{orient} <span style="font-size:0.8em; color:#8b5cf6;">(Y:{yaw_val}°, P:{pitch_val}°)</span></td>
            <td style="padding: 12px;">
                <span style="background: rgba(168,85,247,0.2); color: #e9d5ff; padding: 3px 10px; border-radius: 12px; font-size: 0.85em; font-weight: 600;">
                    {rel_level} ({rel_score}/100)
                </span>
            </td>
            <td style="padding: 12px; color: #9ca3af;">{det_score*100:.0f}%</td>
        </tr>
        """

    # Quality breakdown rows
    q_metrics = quality_info.get("metrics", {}) if quality_info else {}
    quality_rows = ""
    for k, v in q_metrics.items():
        title = v.get("name", k.replace("_", " ").title())
        status = v.get("status", "Good")
        score = v.get("score", 80)
        quality_rows += f"""
        <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.06); font-size: 0.9em;">
            <span style="color: #c4b5fd;">{title}</span>
            <span style="font-weight: 600; color: #e9d5ff;">{status} ({score}/100)</span>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI Face Analytics - Analysis Report</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
        body {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #0b0914;
            color: #e2e1ec;
            margin: 0;
            padding: 30px;
        }}
        .report-wrapper {{
            max-width: 850px;
            margin: 0 auto;
            background: #140e2b;
            border: 1px solid #3b2d66;
            border-radius: 20px;
            padding: 35px;
            box-shadow: 0 15px 40px rgba(0,0,0,0.4);
        }}
        .report-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #6b21a8;
            padding-bottom: 20px;
            margin-bottom: 25px;
        }}
        .report-title {{
            font-size: 24px;
            font-weight: 800;
            color: #ffffff;
            letter-spacing: -0.5px;
            margin: 0;
        }}
        .report-subtitle {{
            font-size: 13px;
            color: #c4b5fd;
            margin-top: 4px;
        }}
        .meta-pill {{
            background: rgba(168,85,247,0.15);
            border: 1px solid rgba(168,85,247,0.3);
            border-radius: 20px;
            padding: 5px 14px;
            font-size: 12px;
            color: #e9d5ff;
            font-family: 'JetBrains Mono', monospace;
        }}
        .grid-summary {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
            margin-bottom: 25px;
        }}
        .summary-card {{
            background: #1c143d;
            border: 1px solid #3b2d66;
            border-radius: 12px;
            padding: 14px;
            text-align: center;
        }}
        .summary-val {{
            font-size: 20px;
            font-weight: 800;
            color: #ffffff;
        }}
        .summary-lbl {{
            font-size: 11px;
            text-transform: uppercase;
            color: #c4b5fd;
            font-weight: 600;
            margin-top: 2px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 25px;
            font-size: 14px;
        }}
        th {{
            background: #1c143d;
            color: #c4b5fd;
            text-align: left;
            padding: 10px 12px;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .disclaimer-box {{
            background: rgba(124,58,237,0.08);
            border: 1px solid rgba(168,85,247,0.25);
            border-radius: 12px;
            padding: 16px;
            margin-top: 25px;
            font-size: 12px;
            color: #a78bfa;
            line-height: 1.6;
        }}
        .print-btn {{
            background: linear-gradient(135deg, #7c3aed, #9333ea);
            color: white;
            border: none;
            padding: 10px 22px;
            border-radius: 8px;
            font-weight: 700;
            cursor: pointer;
            float: right;
            margin-bottom: 15px;
        }}
        @media print {{
            body {{ background-color: #ffffff; color: #111827; }}
            .report-wrapper {{ background: #ffffff; border: 1px solid #e5e7eb; box-shadow: none; padding: 20px; }}
            .report-title {{ color: #111827; }}
            .summary-card {{ background: #f3f4f6; border: 1px solid #e5e7eb; }}
            .summary-val {{ color: #111827; }}
            .summary-lbl {{ color: #4b5563; }}
            th {{ background: #f3f4f6; color: #374151; }}
            tr {{ border-bottom: 1px solid #e5e7eb !important; }}
            .disclaimer-box {{ background: #f9fafb; border: 1px solid #e5e7eb; color: #4b5563; }}
            .print-btn {{ display: none; }}
        }}
    </style>
</head>
<body>
    <div class="report-wrapper">
        <button class="print-btn" onclick="window.print()">🖨️ Print / Save as PDF</button>
        <div class="report-header">
            <div>
                <h1 class="report-title">AI FACE ANALYTICS REPORT</h1>
                <div class="report-subtitle">Automated Age &amp; Gender Estimation &bull; Source: {image_name}</div>
            </div>
            <div class="meta-pill">
                {timestamp}
            </div>
        </div>

        <div class="grid-summary">
            <div class="summary-card">
                <div class="summary-val">{num_faces}</div>
                <div class="summary-lbl">Faces Detected</div>
            </div>
            <div class="summary-card">
                <div class="summary-val">{processing_time_ms:.0f} ms</div>
                <div class="summary-lbl">Latency</div>
            </div>
            <div class="summary-card">
                <div class="summary-val">{overall_quality_score}/100</div>
                <div class="summary-lbl">Image Quality</div>
            </div>
            <div class="summary-card">
                <div class="summary-val">Ensemble</div>
                <div class="summary-lbl">AI Architecture</div>
            </div>
        </div>

        <h3 style="color: #ffffff; font-size: 16px; margin-bottom: 12px;">Subject Breakdown</h3>
        <table>
            <thead>
                <tr>
                    <th>Subject</th>
                    <th>Gender</th>
                    <th>Likely Age Range</th>
                    <th>Head Orientation</th>
                    <th>Reliability</th>
                    <th>Det. Conf.</th>
                </tr>
            </thead>
            <tbody>
                {face_rows}
            </tbody>
        </table>

        <h3 style="color: #ffffff; font-size: 16px; margin-bottom: 12px;">Image Quality Intelligence (7 Dimensions)</h3>
        <div style="background: #1c143d; border: 1px solid #3b2d66; border-radius: 12px; padding: 14px 18px;">
            {quality_rows}
        </div>

        <div class="disclaimer-box">
            <strong>Academic &amp; Engineering Disclaimer:</strong> Facial age estimation evaluates apparent physiological features using deep convolutional networks (InsightFace SCRFD-10GF + GenderAge ResNet-50 + DeepFace VGG-Age ensemble with Test-Time Augmentation). Head pose angles are calculated via 3D-2D anthropometric landmark perspective projection (solvePnP). Reliability scores and quality ratings are heuristic indices designed to indicate prediction stability under varying imaging conditions.
        </div>
    </div>
</body>
</html>
"""
    return html_content


def export_csv(records):
    """Export history or batch records to formatted CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID", "Timestamp", "Image_Name", "Faces_Detected", "Primary_Gender",
        "Age_Range", "Head_Pose_Orientation", "Yaw_Deg", "Pitch_Deg", "Roll_Deg",
        "Reliability", "Reliability_Score", "Quality_Score", "Processing_Time_ms"
    ])
    for r in records:
        writer.writerow([
            r.get("id", ""),
            r.get("timestamp", ""),
            r.get("image_name", ""),
            r.get("faces_detected", 1),
            r.get("primary_gender", r.get("gender", "")),
            r.get("age_range", ""),
            r.get("orientation", r.get("head_pose", {}).get("orientation", "Frontal / Straight")),
            r.get("yaw", r.get("head_pose", {}).get("yaw", 0.0)),
            r.get("pitch", r.get("head_pose", {}).get("pitch", 0.0)),
            r.get("roll", r.get("head_pose", {}).get("roll", 0.0)),
            r.get("reliability", ""),
            r.get("reliability_score", ""),
            r.get("quality_score", ""),
            r.get("processing_time_ms", "")
        ])
    return output.getvalue()
