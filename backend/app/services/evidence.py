"""Normalize persisted CrackVision module results into a shared evidence contract."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from ..entities import Inspection
from ..main_paths import STORAGE_DIR


def classify_question(question: str) -> str:
    """Lightweight intent routing used to focus evidence selection, not to infer damage."""
    text = question.lower()
    if re.search(r"\b(video|when|timestamp|during|across frames|timeline)\b", text):
        return "video_summary"
    if re.search(r"\b(length|width|area|measure|measurement|how large|how long)\b", text):
        return "measurement"
    if re.search(r"\b(where|location|component|beam|column|slab|map|hotspot)\b", text):
        return "localization"
    if re.search(r"\b(visible|look|image|photo|frame|what damage|what defect)\b", text):
        return "visual_findings"
    if re.search(r"\b(priority|inspect next|next inspection|maintenance|guidance|standard)\b", text):
        return "inspection_guidance"
    return "project_summary"


def _result(item: Inspection) -> dict[str, Any]:
    try:
        value = json.loads(item.result_json or "{}")
        return value if isinstance(value, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _box(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        keys = ("x1", "y1", "x2", "y2")
        if all(key in value for key in keys):
            return [float(value[key]) for key in keys]
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return [float(part) for part in value]
    return None


def _observation(item: Inspection, *, damage_type: str, source_type: str, frame_index: int | None = None,
                 timestamp_seconds: float | None = None, confidence: float | None = None,
                 bounding_box: list[float] | None = None, segmentation: dict[str, Any] | None = None,
                 mapping: dict[str, Any] | None = None, tracklet_id: str | int | None = None,
                 observation_index: int = 0) -> dict[str, Any]:
    return {
        "observation_id": f"{item.id}:{frame_index if frame_index is not None else 'image'}:{observation_index}",
        "inspection_id": item.id, "project_id": item.project_id,
        "asset_id": None, "component_id": None,
        "source_type": source_type, "frame_id": frame_index,
        "timestamp_seconds": timestamp_seconds,
        "damage_type": damage_type, "detection_confidence": confidence,
        "bounding_box_xyxy_px": bounding_box,
        "segmentation_measurements": segmentation,
        "mapping_location": mapping,
        "tracklet_id": tracklet_id,
        "calibration": "unknown unless explicitly present in this inspection result",
        "evidence_ref": f"inspection:{item.id}",
    }


def build_damage_observations(item: Inspection, question: str, limit: int = 60) -> list[dict[str, Any]]:
    """Return normalized observations while preserving unknown asset/component/location fields."""
    result = _result(item)
    kind = result.get("kind", "image")
    source_type = "video" if item.module == "video" or (item.module == "mapping" and kind == "video") else "image"
    observations: list[dict[str, Any]] = []
    if item.module == "detection":
        for index, row in enumerate(result.get("detections", [])[:limit]):
            observations.append(_observation(item, damage_type=str(row.get("class_name", "unknown")),
                source_type="image", confidence=row.get("confidence"), bounding_box=_box(row.get("box")), observation_index=index))
    elif item.module == "video":
        path = STORAGE_DIR / item.id / "frame_detections.csv"
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    try:
                        frame = int(row.get("frame_index") or 0)
                        stamp = float(row.get("timestamp_seconds") or 0)
                        confidence = float(row.get("confidence") or 0)
                        box = _box([row.get("x1"), row.get("y1"), row.get("x2"), row.get("y2")])
                        observations.append(_observation(item, damage_type=str(row.get("class_name") or "unknown"),
                            source_type="video", frame_index=frame, timestamp_seconds=stamp, confidence=confidence,
                            bounding_box=box, tracklet_id=row.get("tracklet_id") or None, observation_index=len(observations)))
                    except (TypeError, ValueError):
                        continue
        except (OSError, csv.Error):
            pass
        target = _time_from_question(question)
        if target is not None:
            observations.sort(key=lambda row: abs((row["timestamp_seconds"] or 0) - target))
        else:
            observations.sort(key=lambda row: row["detection_confidence"] or 0, reverse=True)
    elif item.module == "segmentation":
        observations.append(_observation(item, damage_type="segmented_crack_region", source_type="image",
            segmentation={key: result[key] for key in ("mask_coverage_percent", "contour_count", "total_contour_length_px",
                "largest_contour_length_px", "largest_contour_area_px2", "estimated_median_width_px", "measurement_note") if key in result}))
    elif item.module == "mapping":
        if kind == "video":
            path = STORAGE_DIR / item.id / "mapping_frames.csv"
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    for index, row in enumerate(csv.DictReader(handle)):
                        try:
                            frame = int(row.get("frame_index") or 0)
                            stamp = float(row.get("timestamp_seconds") or 0)
                            measurements = {key: float(row[key]) for key in ("crack_area_ratio", "crack_length_px", "width_median_px", "width_p95_px") if row.get(key) not in (None, "")}
                            location = {"grid": None, "max_priority_exploratory": float(row.get("max_priority") or 0), "hotspot_cells": int(row.get("hotspot_cells") or 0)}
                            observations.append(_observation(item, damage_type="crack_mapping_summary", source_type="video",
                                frame_index=frame, timestamp_seconds=stamp, segmentation=measurements, mapping=location, observation_index=index))
                        except (TypeError, ValueError):
                            continue
            except (OSError, csv.Error):
                pass
            target = _time_from_question(question)
            if target is not None:
                observations.sort(key=lambda row: abs((row["timestamp_seconds"] or 0) - target))
            else:
                observations.sort(key=lambda row: (row["mapping_location"] or {}).get("max_priority_exploratory", 0), reverse=True)
        else:
            metrics = result.get("metrics") or {}
            for index, cell in enumerate(result.get("top_cells", [])[:limit]):
                observations.append(_observation(item, damage_type="mapped_crack_hotspot", source_type="image",
                    segmentation={key: metrics[key] for key in ("crack_length", "width_median", "width_p95", "crack_area_ratio") if key in metrics},
                    mapping={"grid_row": cell.get("grid_row"), "grid_col": cell.get("grid_col"),
                             "priority_index_exploratory": cell.get("priority_index"),
                             "map_coordinates": None}, observation_index=index))
    return observations[:limit]


def _time_from_question(question: str) -> float | None:
    clock = re.search(r"\b(\d{1,3}):(\d{2}(?:\.\d+)?)\b", question)
    if clock:
        return int(clock.group(1)) * 60 + float(clock.group(2))
    seconds = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b", question, re.I)
    return float(seconds.group(1)) if seconds else None
