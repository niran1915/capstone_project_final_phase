from __future__ import annotations

import csv
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import cv2

from ..database import SessionLocal
from ..entities import Inspection
from ..main_paths import storage_url
from .video_detection import video_detector, MODEL_PATH as VIDEO_MODEL_PATH

CSV_FIELDS = ["frame_index", "timestamp_seconds", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "tracklet_id"]
TRACK_FIELDS = ["tracklet_id", "class_id", "class_name", "first_timestamp_seconds", "last_timestamp_seconds", "observation_count", "peak_confidence", "peak_frame_index", "x1", "y1", "x2", "y2"]

def _iou(first: dict[str, Any], second: dict[str, Any]) -> float:
    left = max(float(first["x1"]), float(second["x1"]))
    top = max(float(first["y1"]), float(second["y1"]))
    right = min(float(first["x2"]), float(second["x2"]))
    bottom = min(float(first["y2"]), float(second["y2"]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(first["x2"]) - float(first["x1"])) * max(0.0, float(first["y2"]) - float(first["y1"]))
    area_b = max(0.0, float(second["x2"]) - float(second["x1"])) * max(0.0, float(second["y2"]) - float(second["y1"]))
    return intersection / max(1e-9, area_a + area_b - intersection)


def _assign_tracklets(rows: list[dict[str, Any]], max_gap_frames: int) -> list[dict[str, Any]]:
    """Greedy, class-aware IoU association; outputs tracklets, not verified unique defects."""
    active: dict[int, dict[str, Any]] = {}
    tracks: dict[int, dict[str, Any]] = {}
    next_id = 1
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        row["tracklet_id"] = None
        by_frame.setdefault(int(row["frame_index"]), []).append(row)
    for frame_index in sorted(by_frame):
        current = by_frame[frame_index]
        expired = [track_id for track_id, track in active.items() if frame_index - track["last_frame"] > max_gap_frames]
        for track_id in expired:
            del active[track_id]
        pairs = []
        for index, detection in enumerate(current):
            for track_id, track in active.items():
                if track["class_id"] != detection["class_id"]:
                    continue
                overlap = _iou(track["last_box"], detection)
                if overlap >= 0.25:
                    pairs.append((overlap, index, track_id))
        assigned_detections: set[int] = set()
        assigned_tracks: set[int] = set()
        for _, index, track_id in sorted(pairs, reverse=True):
            if index in assigned_detections or track_id in assigned_tracks:
                continue
            detection = current[index]
            detection["tracklet_id"] = track_id
            track = active[track_id]
            track["last_frame"] = frame_index
            track["last_box"] = detection
            track["observations"].append(detection)
            assigned_detections.add(index)
            assigned_tracks.add(track_id)
        for index, detection in enumerate(current):
            if index in assigned_detections:
                continue
            track_id = next_id
            next_id += 1
            detection["tracklet_id"] = track_id
            active[track_id] = {"class_id": detection["class_id"], "last_frame": frame_index,
                                "last_box": detection, "observations": [detection]}
            tracks[track_id] = active[track_id]
    summaries = []
    for track_id, track in tracks.items():
        observations = track["observations"]
        peak = max(observations, key=lambda row: float(row["confidence"]))
        summaries.append({"tracklet_id": track_id, "class_id": peak["class_id"], "class_name": peak["class_name"],
                          "first_timestamp_seconds": observations[0]["timestamp_seconds"],
                          "last_timestamp_seconds": observations[-1]["timestamp_seconds"],
                          "observation_count": len(observations), "peak_confidence": peak["confidence"],
                          "peak_frame_index": peak["frame_index"], **{key: peak[key] for key in ("x1", "y1", "x2", "y2")}})
    return summaries


def process_video(input_path: Path, output_dir: Path, conf: float, iou: float, imgsz: int, frame_rate: float = 2.0) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("The uploaded file could not be opened as a video.")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0.01:
        fps = 25.0
    input_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("The video has invalid frame dimensions.")
    frame_rate = float(frame_rate)
    if not 0.2 <= frame_rate <= 10.0:
        capture.release()
        raise ValueError("Video analysis frame rate must be between 0.2 and 10 frames per second.")
    frame_stride = max(1, int(round(fps / frame_rate)))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "annotated_video.mp4"
    output_csv = output_dir / "frame_detections.csv"
    tracks_csv = output_dir / "damage_tracklets.csv"
    output_json = output_dir / "video_summary.json"
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("The MP4 output writer could not be initialized on this machine.")

    frame_index = 0
    detection_rows: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    frames_with_detections = 0
    sampled_frame_count = 0
    max_confidence = 0.0
    started = time.perf_counter()
    db = None
    try:
        # Report coarse progress in the inspection record so the frontend can poll it.
        from ..entities import Inspection
        db = SessionLocal()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_stride:
                writer.write(frame)
                frame_index += 1
                if frame_index % max(12, frame_stride * 12) == 0 or frame_index == input_frame_count:
                    inspection = db.get(Inspection, output_dir.name)
                    if inspection:
                        progress = round(100.0 * frame_index / max(1, input_frame_count), 1) if input_frame_count else None
                        inspection.result_json = json.dumps({"progress_percent": progress, "processed_frames": frame_index, "total_frames": input_frame_count, "sampled_frames": sampled_frame_count})
                        db.commit()
                continue
            sampled_frame_count += 1
            result = video_detector.predict_frame(frame, conf=conf, iou=iou, imgsz=imgsz)
            if result is None:
                writer.write(frame)
                frame_index += 1
                continue
            annotated = result.plot()
            writer.write(annotated)
            boxes = result.boxes
            frame_had_detection = boxes is not None and len(boxes) > 0
            if frame_had_detection:
                frames_with_detections += 1
                names = getattr(result, "names", {}) or {}
                xyxy = boxes.xyxy.detach().cpu().numpy()
                scores = boxes.conf.detach().cpu().numpy()
                cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
                timestamp = frame_index / fps
                for box, score, cls_id in zip(xyxy, scores, cls_ids):
                    name = str(names.get(int(cls_id), cls_id)) if isinstance(names, dict) else str(names[int(cls_id)])
                    confidence = float(score)
                    class_counts[name] += 1
                    max_confidence = max(max_confidence, confidence)
                    detection_rows.append({
                        "frame_index": frame_index,
                        "timestamp_seconds": round(timestamp, 5),
                        "class_id": int(cls_id), "class_name": name,
                        "confidence": round(confidence, 6),
                        "x1": round(float(box[0]), 2), "y1": round(float(box[1]), 2),
                        "x2": round(float(box[2]), 2), "y2": round(float(box[3]), 2),
                    })
            frame_index += 1
            if frame_index % max(12, frame_stride * 12) == 0 or frame_index == input_frame_count:
                inspection = db.get(Inspection, output_dir.name)
                if inspection:
                    progress = round(100.0 * frame_index / max(1, input_frame_count), 1) if input_frame_count else None
                    inspection.result_json = json.dumps({"progress_percent": progress, "processed_frames": frame_index, "total_frames": input_frame_count, "sampled_frames": sampled_frame_count})
                    db.commit()
    finally:
        capture.release()
        writer.release()
        if db is not None:
            db.close()

    if frame_index == 0 or not output_video.exists() or output_video.stat().st_size < 1024:
        raise RuntimeError("No valid frames were processed; the output video was not produced.")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        tracklets = _assign_tracklets(detection_rows, max(1, frame_stride * 2))
        writer_csv = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer_csv.writeheader()
        writer_csv.writerows(detection_rows)
    with tracks_csv.open("w", newline="", encoding="utf-8") as handle:
        writer_tracks = csv.DictWriter(handle, fieldnames=TRACK_FIELDS)
        writer_tracks.writeheader()
        writer_tracks.writerows(tracklets)
    duration = round(frame_index / fps, 4)
    elapsed = round(time.perf_counter() - started, 3)
    summary = {
        "model_checkpoint": VIDEO_MODEL_PATH.name,
        "fps": round(fps, 5), "input_frame_count": input_frame_count,
        "processed_frame_count": sampled_frame_count, "sample_rate_fps": round(fps / frame_stride, 4), "sample_stride_frames": frame_stride,
        "width": width, "height": height,
        "duration_seconds": duration, "confidence_threshold": float(conf),
        "nms_iou_threshold": float(iou), "image_size": int(imgsz),
        "frames_with_detections": frames_with_detections,
        "total_detections": len(detection_rows), "maximum_confidence": round(max_confidence, 5),
        "tracked_damage_tracklets": len(tracklets), "tracking_method": "class-aware greedy IoU association (heuristic; not verified unique-defect counting)",
        "tracking_iou_threshold": 0.25, "tracking_max_gap_frames": max(1, frame_stride * 2),
        "class_counts": dict(class_counts), "processing_time_seconds": elapsed,
        "output_size_mb": round(output_video.stat().st_size / (1024 * 1024), 3),
        "annotated_video_url": storage_url(output_video),
        "detections_csv_url": storage_url(output_csv),
        "tracklets_csv_url": storage_url(tracks_csv),
        "summary_json_url": storage_url(output_json),
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

def run_video_job(inspection_id: str, input_path: str, output_dir: str, conf: float, iou: float, imgsz: int, frame_rate: float = 2.0) -> None:
    from ..entities import Inspection
    db = SessionLocal()
    inspection = db.get(Inspection, inspection_id)
    if inspection is None:
        db.close()
        return
    inspection.status = "processing"
    inspection.error_message = None
    db.commit()
    db.close()
    try:
        result = process_video(Path(input_path), Path(output_dir), conf, iou, imgsz, frame_rate)
        db = SessionLocal()
        inspection = db.get(Inspection, inspection_id)
        if inspection is not None:
            inspection.status = "succeeded"
            inspection.result_json = json.dumps(result)
            inspection.completed_at = datetime.now(timezone.utc)
            db.commit()
        db.close()
    except Exception as exc:
        db = SessionLocal()
        inspection = db.get(Inspection, inspection_id)
        if inspection is not None:
            inspection.status = "failed"
            inspection.error_message = f"{type(exc).__name__}: {exc}"
            inspection.completed_at = datetime.now(timezone.utc)
            db.commit()
        db.close()
