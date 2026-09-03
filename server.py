"""
FastAPI Server for AI Face Analytics Platform
=============================================
High-performance REST API serving face detection, age & gender estimation,
biometric landmark extraction, quality assessment, and HTML/CSV reports.
"""

import sys
import os
import io
import base64
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add utils directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

from face_utils import (
    init_analyzer,
    analyze_faces,
    assess_image_quality,
    compute_reliability,
    estimate_head_pose,
    compute_group_summary
)
from report_utils import generate_html_report, export_csv

# Initialize FastAPI App
app = FastAPI(
    title="AI Face Analytics API",
    description="Age & Gender Estimation Platform REST API",
    version="2.0.0"
)

# Enable CORS for cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serve the single-page HTML5/CSS3/JS Web UI."""
    html_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>AI Face Analytics Platform</h1><p>Frontend file index.html initializing...</p>"

@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze uploaded image file for age, gender, landmarks, quality, and head pose.
    """
    global analyzer
    if analyzer is None:
        try:
            analyzer = init_analyzer(det_size=(480, 480))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analyzer error: {e}")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if bgr_img is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    t0 = datetime.now()
    results = analyze_faces(bgr_img, analyzer, fast_mode=False)
    proc_time_ms = round((datetime.now() - t0).total_seconds() * 1000, 1)

    quality_info = assess_image_quality(bgr_img)
    summary = compute_group_summary(results)

    # Convert annotated BGR image to base64 JPEG
    annotated_bgr = bgr_img.copy()

    # Encode BGR image to Base64
    _, buffer = cv2.imencode('.jpg', annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    img_str = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse({
        "success": True,
        "processing_time_ms": proc_time_ms,
        "num_faces": len(results),
        "quality": quality_info,
        "summary": summary,
        "faces": results,
        "annotated_image": f"data:image/jpeg;base64,{img_str}"
    })


class Base64Payload(BaseModel):
    image_base64: str


@app.post("/api/webcam")
async def analyze_webcam(payload: Base64Payload):
    """
    Real-time webcam snapshot analysis.
    """
    global analyzer
    if analyzer is None:
        try:
            analyzer = init_analyzer(det_size=(320, 320))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analyzer error: {e}")

    try:
        data = payload.image_base64
        if "," in data:
            data = data.split(",")[1]
        img_data = base64.b64decode(data)
        nparr = np.frombuffer(img_data, np.uint8)
        bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 payload")

    if bgr_img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    t0 = datetime.now()
    results = analyze_faces(bgr_img, analyzer, fast_mode=True)
    proc_time_ms = round((datetime.now() - t0).total_seconds() * 1000, 1)

    return JSONResponse({
        "success": True,
        "processing_time_ms": proc_time_ms,
        "num_faces": len(results),
        "faces": results
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
