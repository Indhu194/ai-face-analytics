"""
Database utilities for Analysis History and Analytics Dashboard
===============================================================
Lightweight SQLite storage for face analysis metadata (no raw images saved).
"""

import os
import sqlite3
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'analysis_history.db')


def get_db_connection():
    """Establish thread-safe connection to SQLite database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the SQLite database schema if not already created."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_name TEXT NOT NULL,
                faces_detected INTEGER NOT NULL DEFAULT 1,
                primary_gender TEXT NOT NULL,
                primary_age INTEGER NOT NULL,
                age_range TEXT NOT NULL,
                reliability TEXT NOT NULL,
                reliability_score INTEGER NOT NULL,
                quality_score INTEGER NOT NULL,
                processing_time_ms REAL NOT NULL,
                details_json TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON analysis_history(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_image_name ON analysis_history(image_name)")
        conn.commit()


def save_analysis(image_name, faces_data, bgr_image=None, processing_time_ms=0.0, quality_score=85):
    """Save a completed analysis record to SQLite history."""
    init_db()
    if not faces_data:
        return None

    primary_face = faces_data[0]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    faces_detected = len(faces_data)
    primary_gender = primary_face.get("gender", "Unknown")
    primary_age = primary_face.get("age", 25)
    
    # Age range string
    offset = max(3, int(primary_face.get("age_std", 2.0) * 1.5))
    age_min = max(1, primary_age - offset)
    age_max = min(100, primary_age + offset)
    age_range = f"{age_min}–{age_max}"
    
    reliability = primary_face.get("reliability_level", "High" if primary_face.get("det_score", 0.9) > 0.85 else "Moderate")
    rel_score = int(primary_face.get("reliability_score", 85))

    # Serialize details
    serialized_faces = []
    for f in faces_data:
        serialized_faces.append({
            "person_id": f.get("person_id", 1),
            "age": f.get("age", 25),
            "gender": f.get("gender", "Unknown"),
            "det_score": round(float(f.get("det_score", 0.95)), 3),
            "age_std": round(float(f.get("age_std", 0.0)), 2),
            "bbox": list(f.get("bbox", [0, 0, 0, 0]))
        })
    details_json = json.dumps({"faces": serialized_faces})

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO analysis_history (
                timestamp, image_name, faces_detected, primary_gender,
                primary_age, age_range, reliability, reliability_score,
                quality_score, processing_time_ms, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, image_name, faces_detected, primary_gender,
            primary_age, age_range, reliability, rel_score,
            int(quality_score), round(float(processing_time_ms), 1), details_json
        ))
        conn.commit()
        return cursor.lastrowid


def get_history(search_query="", start_date=None, end_date=None, limit=200):
    """Query analysis records with optional search keyword and date filters."""
    init_db()
    query = "SELECT * FROM analysis_history WHERE 1=1"
    params = []

    if search_query:
        query += " AND (image_name LIKE ? OR primary_gender LIKE ? OR reliability LIKE ?)"
        wildcard = f"%{search_query.strip()}%"
        params.extend([wildcard, wildcard, wildcard])

    if start_date:
        query += " AND timestamp >= ?"
        params.append(f"{start_date} 00:00:00")

    if end_date:
        query += " AND timestamp <= ?"
        params.append(f"{end_date} 23:59:59")

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def delete_record(record_id):
    """Delete a single analysis record by ID."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analysis_history WHERE id = ?", (record_id,))
        conn.commit()
        return cursor.rowcount > 0


def clear_history():
    """Clear all records from analysis history."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analysis_history")
        conn.commit()


def get_analytics_summary():
    """Compute aggregate statistics and data distributions for Analytics Dashboard."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analysis_history ORDER BY id ASC")
        rows = [dict(r) for r in cursor.fetchall()]

    if not rows:
        return {
            "has_data": False,
            "total_analyses": 0,
            "total_faces": 0,
            "avg_latency": 0,
            "avg_reliability": 0,
            "avg_quality": 0,
            "gender_counts": {"Male": 0, "Female": 0},
            "age_groups": {},
            "quality_distribution": {},
            "reliability_distribution": {},
            "timeline_data": []
        }

    total_analyses = len(rows)
    total_faces = sum(r["faces_detected"] for r in rows)
    avg_latency = sum(r["processing_time_ms"] for r in rows) / total_analyses
    avg_reliability = sum(r["reliability_score"] for r in rows) / total_analyses
    avg_quality = sum(r["quality_score"] for r in rows) / total_analyses

    # Gender breakdown
    gender_counts = {"Male": 0, "Female": 0}
    for r in rows:
        g = r["primary_gender"]
        gender_counts[g] = gender_counts.get(g, 0) + 1

    # Age group breakdown
    age_groups = {
        "0-12 (Child)": 0,
        "13-19 (Teen)": 0,
        "20-29 (Young Adult)": 0,
        "30-39 (Adult)": 0,
        "40-49 (Middle-Aged)": 0,
        "50-59 (Mature)": 0,
        "60+ (Senior)": 0
    }
    for r in rows:
        a = r["primary_age"]
        if a <= 12: age_groups["0-12 (Child)"] += 1
        elif a <= 19: age_groups["13-19 (Teen)"] += 1
        elif a <= 29: age_groups["20-29 (Young Adult)"] += 1
        elif a <= 39: age_groups["30-39 (Adult)"] += 1
        elif a <= 49: age_groups["40-49 (Middle-Aged)"] += 1
        elif a <= 59: age_groups["50-59 (Mature)"] += 1
        else: age_groups["60+ (Senior)"] += 1

    # Quality distribution
    quality_dist = {"Excellent": 0, "Good": 0, "Moderate": 0, "Poor": 0}
    for r in rows:
        qs = r["quality_score"]
        if qs >= 80: quality_dist["Excellent"] += 1
        elif qs >= 65: quality_dist["Good"] += 1
        elif qs >= 45: quality_dist["Moderate"] += 1
        else: quality_dist["Poor"] += 1

    # Reliability distribution
    rel_dist = {"High": 0, "Moderate": 0, "Low": 0}
    for r in rows:
        rel = r["reliability"]
        rel_dist[rel] = rel_dist.get(rel, 0) + 1

    # Timeline daily counts
    timeline = {}
    for r in rows:
        date_str = r["timestamp"].split(" ")[0]
        timeline[date_str] = timeline.get(date_str, 0) + 1
    timeline_data = [{"Date": k, "Analyses": v} for k, v in sorted(timeline.items())]

    return {
        "has_data": True,
        "total_analyses": total_analyses,
        "total_faces": total_faces,
        "avg_latency": round(avg_latency, 1),
        "avg_reliability": round(avg_reliability, 1),
        "avg_quality": round(avg_quality, 1),
        "gender_counts": gender_counts,
        "age_groups": age_groups,
        "quality_distribution": quality_dist,
        "reliability_distribution": rel_dist,
        "timeline_data": timeline_data
    }
