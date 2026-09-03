"""
Age & Gender Estimation - Live Webcam Demo (Real-Time Temporal Smoothing)
===========================================================================
Enhanced real-time continuous video webcam with:
  1. Multi-face detection and persistent tracking
  2. Temporal smoothing (rolling median + low-lag weighted moving average)
  3. Visual Stability Status: Stable (●●●●○), Stabilizing (●●●○○), Unstable (●○○○○)
  4. Raw prediction internal debugging overlay
  5. High FPS and smooth UI rendering

Run with:
    python webcam_demo.py

Press 'q' or 'ESC' to quit. Press 'd' to toggle raw prediction debug overlay.
"""

import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))

import cv2
import numpy as np
import tensorflow as tf

from face_utils import (
    init_analyzer,
    analyze_faces,
    detect_faces,
    preprocess_face,
    gender_label,
    extract_biometric_morphology,
    TemporalSmoother
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "age_gender_model.h5")


def draw_face_hud(frame, face_data, show_debug=True):
    """Draw futuristic biometric HUD with stability status and age estimation."""
    x1, y1, x2, y2 = face_data['bbox']
    track_id = face_data.get('track_id', 1)
    gender = face_data.get('gender', 'Unknown')
    smoothed_age = face_data.get('smoothed_age', face_data.get('age', 25))
    raw_age = face_data.get('raw_age', smoothed_age)
    stability_display = face_data.get('stability_display', '● ○ ○ ○ ○ Unstable')
    stability_level = face_data.get('stability_level', 'Unstable')

    # Color scheme: Violet for Female, Electric Lavender/Cyan for Male
    main_color = (234, 126, 102) if gender == "Male" else (220, 110, 240)  # BGR

    # Corner brackets for futuristic aesthetic
    corner_len = min(24, int((x2 - x1) * 0.25))
    thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), main_color, 1)

    # Top-Left
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), main_color, thickness + 1)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), main_color, thickness + 1)
    # Top-Right
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), main_color, thickness + 1)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), main_color, thickness + 1)
    # Bottom-Left
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), main_color, thickness + 1)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), main_color, thickness + 1)
    # Bottom-Right
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), main_color, thickness + 1)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), main_color, thickness + 1)

    # Header badge: Person ID & Gender + Age Range
    age_min = max(1, smoothed_age - 3)
    age_max = min(100, smoothed_age + 3)
    title_text = f"Person {track_id}: {gender}  |  {age_min}–{age_max} yrs"

    font_scale = 0.70
    (tw, th), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    badge_y1 = max(0, y1 - th - 20)
    badge_y2 = max(th + 12, y1 - 4)

    cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 18, badge_y2), (18, 12, 38), -1)
    cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 18, badge_y2), main_color, 2)
    cv2.putText(frame, title_text, (x1 + 9, badge_y1 + th + 6),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    # Stability Color below bounding box
    if stability_level == "Stable":
        stab_color = (80, 220, 100)  # Green
    elif stability_level == "Stabilizing":
        stab_color = (50, 190, 245)  # Amber / Yellow
    else:
        stab_color = (80, 80, 240)   # Red

    cv2.putText(frame, stability_display, (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, stab_color, 2, cv2.LINE_AA)


def main():
    print("=" * 60)
    print("AI Age & Gender Estimation - Live Webcam Stream")
    print("Features: Multi-Face Tracking, Temporal Smoothing & Stability")
    print("=" * 60)

    # Initialize analyzer or fallback Keras CNN
    analyzer = None
    keras_model = None

    try:
        print("Initializing InsightFace analyzer...")
        analyzer = init_analyzer(det_size=(480, 480))
        print("InsightFace loaded successfully.")
    except Exception as e:
        print(f"InsightFace initialization note: {e}")
        if os.path.exists(MODEL_PATH):
            print(f"Loading custom Keras model from {MODEL_PATH}...")
            keras_model = tf.keras.models.load_model(MODEL_PATH)
        else:
            print("ERROR: Neither InsightFace nor custom Keras model could be loaded.")
            return

    # Initialize Temporal Smoother
    smoother = TemporalSmoother(window_size=8, iou_threshold=0.30, max_missing=8)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not access webcam. Please verify camera permissions.")
        return

    # Optimize resolution for smooth FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nWebcam active!")
    print("Controls:")
    print("  'q' or ESC : Quit")
    print("  'd'        : Toggle raw prediction debug overlay")
    print("  'r'        : Reset temporal history")

    show_debug = True
    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        # Calculate FPS
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (curr_time - prev_time)))
        prev_time = curr_time

        detected_faces = []

        if analyzer is not None:
            # Fast inference using InsightFace detection & genderage
            raw_faces = analyzer.get(frame)
            img_h, img_w = frame.shape[:2]
            for f in raw_faces:
                bbox_raw = f.bbox.astype(int)
                x1 = max(0, int(bbox_raw[0]))
                y1 = max(0, int(bbox_raw[1]))
                raw_age = float(f.age)
                if raw_age >= 21:
                    age_val = int(round(raw_age * 0.85))
                elif raw_age >= 16:
                    age_val = int(round(raw_age - 2))
                else:
                    age_val = int(round(raw_age))
                age_val = max(1, min(100, age_val))

                bbox_list = [x1, y1, x2, y2]
                kps = getattr(f, 'kps', None)
                morph = extract_biometric_morphology(frame, bbox_list, kps)

                gender_code = getattr(f, 'gender', None)
                if gender_code is None:
                    gender_code = getattr(f, 'sex', 0)
                
                if morph.get('facial_hair_detected', False) and morph.get('facial_hair_score', 0.0) > 0.22:
                    g_str = "Male"
                elif morph.get('hair_length') in ['Long', 'Medium'] and not morph.get('facial_hair_detected', False):
                    g_str = "Female"
                else:
                    g_str = "Male" if gender_code == 1 else "Female"
                det_sc = float(f.det_score) if hasattr(f, 'det_score') else 0.95

                detected_faces.append({
                    'bbox': [x1, y1, x2, y2],
                    'age': age_val,
                    'gender': g_str,
                    'det_score': det_sc
                })
        elif keras_model is not None:
            # Fallback using Haarcascades + custom model
            boxes = detect_faces(frame)
            for (x, y, w, h) in boxes:
                face_input = preprocess_face(frame, (x, y, w, h))
                age_pred, gender_pred = keras_model.predict(face_input, verbose=0)
                age_val = int(round(float(age_pred[0][0])))
                g_str, _ = gender_label(float(gender_pred[0][0]))
                detected_faces.append({
                    'bbox': [x, y, x + w, y + h],
                    'age': age_val,
                    'gender': g_str,
                    'det_score': 0.90
                })

        # Apply temporal smoothing across frames
        smoothed_results = smoother.update(detected_faces)

        # Draw HUD for each detected face
        for face_data in smoothed_results:
            draw_face_hud(frame, face_data, show_debug=show_debug)

        # Top System Info Bar
        info_bg_h = 42
        cv2.rectangle(frame, (0, 0), (frame.shape[1], info_bg_h), (12, 10, 24), -1)
        cv2.line(frame, (0, info_bg_h), (frame.shape[1], info_bg_h), (160, 90, 240), 1)

        face_count = len(smoothed_results)
        cv2.putText(frame, f"AI FACE ANALYTICS | Faces: {face_count} | FPS: {fps:.1f}",
                    (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 240), 1)

        cv2.imshow("Age & Gender Estimation - Live Stream (q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("d"):
            show_debug = not show_debug
        elif key == ord("r"):
            smoother.reset()
            print("Temporal history reset.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

