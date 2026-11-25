from __future__ import annotations

import json
import hashlib
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import desc, func
from sqlalchemy.orm import Session, joinedload
from starlette.concurrency import run_in_threadpool
from PIL import Image, ImageOps, UnidentifiedImageError
from io import BytesIO
import numpy as np

from .database import Base, SessionLocal, engine, get_db
from .entities import Inspection, Project, VQAConversation
from .main_paths import STORAGE_DIR, storage_url
from .services.detection import detector
from .services.mapping import build_calibration, mapping_service, run_mapping_video_job
from .services.damage_mapping import damage_mapping_service
from .services.damage_mapping_core import CLASS_IDS as DAMAGE_CLASS_IDS, measure_damage_map
from .services.damage_segmentation import damage_segmenter
from .services.segmentation import segmenter
from .services.video import run_video_job
from .services.video_detection import video_detector
from .services.vqa import (ask as ask_vqa, delete_knowledge_document, knowledge_documents,
                           store_knowledge_document,
                           provider_status as vqa_provider_status)
from .services.evidence import build_damage_observations
from .services.vqa_screen import assessment as evidence_vqa, build_vqa_pdf_bytes, normalize_telemetry

APP_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = APP_DIR.parents[1]
REFERENCE_DIR = REPOSITORY_DIR / "assets" / "reference_outputs"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "250")) * 1024 * 1024
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
MODULES = ("detection", "segmentation", "video", "mapping")
MAPPING_SCENARIOS = {"equal", "length_emphasis", "width_emphasis", "area_emphasis", "topology_emphasis"}

app = FastAPI(title="CrackVision API", version="1.1.0", description="Four-module structural defect inspection platform: image and video YOLO detection, CRACKTO-SFNet V4 segmentation, and spatial damage mapping built on the V4 model.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/storage", StaticFiles(directory=str(STORAGE_DIR)), name="storage")
if REFERENCE_DIR.is_dir():
    app.mount("/reference", StaticFiles(directory=str(REFERENCE_DIR)), name="reference")

class ProjectCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    site_location: str = Field(default="", max_length=200)
    structure_type: str = Field(default="Building", max_length=80)
    description: str = Field(default="", max_length=3000)

class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=120)
    site_location: str | None = Field(default=None, max_length=200)
    structure_type: str | None = Field(default=None, max_length=80)
    description: str | None = Field(default=None, max_length=3000)

class VQARequest(BaseModel):
    project_id: int | None = Field(default=None, gt=0)
    question: str = Field(min_length=1, max_length=3000)
    inspection_ids: list[str] | None = Field(default=None, max_length=30)
    visual_inspection_id: str | None = None
    timestamp_seconds: float | None = Field(default=None, ge=0)
    conversation_id: str | None = None
    inspection_id: str | None = None
    structure_type: str = Field(default="Concrete Element", max_length=100)
    offline_mode: bool = True
    api_key: str | None = Field(default=None, max_length=500)

@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        if db.query(Project).count() == 0:
            db.add(Project(name="Demo inspection site", site_location="Add your site location", structure_type="Building", description="Default project for the first CrackVision run."))
            db.commit()

@app.get("/")
def root() -> dict[str, str]:
    return {"name": "CrackVision API", "docs": "/docs", "health": "/api/health"}

@app.get("/api/health")
def health(db: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        db.query(Project.id).count()
        database_ok = True
    except Exception:
        database_ok = False
    try:
        import torch
        torch_info = {"installed": True, "version": torch.__version__, "cuda_available": torch.cuda.is_available(), "device": "cuda" if torch.cuda.is_available() else "cpu"}
    except Exception as exc:
        torch_info = {"installed": False, "error": str(exc)}
    det_info = detector.model_info()
    seg_info = segmenter.model_info()
    video_info = video_detector.model_info()
    mapping_info = mapping_service.model_info()
    return {
        "status": "ok" if database_ok else "degraded", "database": database_ok,
        "torch": torch_info,
        "modules": {
            "detection": {**det_info, "available": detector.available()},
            "segmentation": {**seg_info, "available": segmenter.available()},
            "damage_segmentation": {**damage_segmenter.model_info(), "available": damage_segmenter.available()},
            "damage_mapping": {"available": damage_mapping_service.available(), "method": "20x20 grid from detector-guided FastSAM masks"},
            "video": {**video_info, "available": video_detector.available(), "depends_on": "dedicated checkpoint from the supplied video-pipeline notebook"},
            "mapping": {**mapping_info, "available": mapping_service.available(), "depends_on": "segmentation_v4_best.pth + scipy / scikit-image / pandas / matplotlib"},
        },
        "upload_limit_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "vqa": vqa_provider_status(),
    }

@app.get("/api/vqa/status")
def vqa_status() -> dict[str, Any]:
    return vqa_provider_status()

@app.get("/api/vqa/config")
def vqa_config() -> dict[str, Any]:
    return {
        "engine": "Offline Evidence Engine",
        "cloud_available": True,
        "providers": ["OpenAI Responses API", "Google Gemini generateContent"],
        "labels": ["crack", "fissures", "spalling", "rebar"],
        "unsupported_labels": ["concrete crushing"],
        "offline_default": True,
        "note": "Image uploads run the detector, crack segmenter, detector-guided FastSAM masks for non-crack labels when detections exist, and crack mapper. FastSAM masks are class-agnostic proposals, not class-trained predictions. Cloud visual screening requires a user key and offline mode disabled.",
    }

def _combined_inspection_payload(db: Session, item: Inspection) -> dict[str, Any]:
    """Merge the latest detection, segmentation, and mapping results for the same image."""
    try:
        saved_result = json.loads(item.result_json or "{}")
    except json.JSONDecodeError:
        saved_result = {}
    if not isinstance(saved_result, dict):
        saved_result = {}
    if (
        saved_result.get("source_type") == "vqa_image_pipeline"
        or item.module == "video"
        or (item.module == "mapping" and saved_result.get("kind") == "video")
    ):
        return saved_result
    try:
        saved_settings = json.loads(item.settings_json or "{}")
    except json.JSONDecodeError:
        saved_settings = {}
    fingerprint = saved_settings.get("asset_fingerprint") if isinstance(saved_settings, dict) else None

    candidates_query = (
        db.query(Inspection)
        .filter(
            Inspection.project_id == item.project_id,
            Inspection.status == "succeeded",
            Inspection.module.in_(("detection", "segmentation", "mapping")),
        )
        .order_by(desc(Inspection.created_at))
        .limit(120)
    )
    matched: list[tuple[Inspection, dict[str, Any]]] = [(item, saved_result)]
    for candidate in candidates_query.all():
        if candidate.id == item.id:
            continue
        try:
            c_settings = json.loads(candidate.settings_json or "{}")
            c_result = json.loads(candidate.result_json or "{}")
        except json.JSONDecodeError:
            continue
        if not isinstance(c_result, dict):
            continue
        if candidate.module == "mapping" and c_result.get("kind") == "video":
            continue
        if c_result.get("source_type") == "vqa_image_pipeline":
            continue
        c_fp = c_settings.get("asset_fingerprint") if isinstance(c_settings, dict) else None
        if fingerprint:
            if c_fp != fingerprint:
                continue
        elif candidate.source_filename != item.source_filename:
            continue
        matched.append((candidate, c_result))

    det_item: tuple[Inspection, dict[str, Any]] | None = None
    seg_crack_item: tuple[Inspection, dict[str, Any]] | None = None
    seg_damage_item: tuple[Inspection, dict[str, Any]] | None = None
    map_crack_item: tuple[Inspection, dict[str, Any]] | None = None
    map_damage_item: tuple[Inspection, dict[str, Any]] | None = None

    for record, res in matched:
        if record.module == "detection" and det_item is None:
            det_item = (record, res)
        elif record.module == "segmentation":
            if "crack_pixels" in res and seg_crack_item is None:
                seg_crack_item = (record, res)
            elif "class_metrics" in res and seg_damage_item is None:
                seg_damage_item = (record, res)
        elif record.module == "mapping":
            if res.get("kind") == "damage_image" and map_damage_item is None:
                map_damage_item = (record, res)
            elif map_crack_item is None:
                map_crack_item = (record, res)

    stages: dict[str, dict[str, Any]] = {}
    combined_modules: list[str] = []
    source_ids: dict[str, str] = {}
    if det_item:
        stages["detection"] = {"status": "succeeded", "result": det_item[1]}
        combined_modules.append("detection")
        source_ids["detection"] = det_item[0].id
    if seg_crack_item:
        stages["segmentation"] = {"status": "succeeded", "result": seg_crack_item[1]}
        if "segmentation" not in combined_modules:
            combined_modules.append("segmentation")
        source_ids["segmentation"] = seg_crack_item[0].id
    if seg_damage_item:
        stages["damage_segmentation"] = {"status": "succeeded", "result": seg_damage_item[1]}
        if "segmentation" not in combined_modules:
            combined_modules.append("segmentation")
        source_ids["damage_segmentation"] = seg_damage_item[0].id
    if map_crack_item:
        stages["mapping"] = {"status": "succeeded", "result": map_crack_item[1]}
        if "mapping" not in combined_modules:
            combined_modules.append("mapping")
        source_ids["mapping"] = map_crack_item[0].id
    if map_damage_item:
        stages["multi_defect_mapping"] = {"status": "succeeded", "result": map_damage_item[1]}
        if "mapping" not in combined_modules:
            combined_modules.append("mapping")
        source_ids["multi_defect_mapping"] = map_damage_item[0].id

    combined: dict[str, Any] = {
        **saved_result,
        "kind": saved_result.get("kind", "image"),
        "source_type": "combined_module_telemetry" if len(combined_modules) > 1 else item.module,
        "source_module": "combined_pipeline" if len(combined_modules) > 1 else item.module,
        "combined_modules": combined_modules,
        "source_inspection_ids": source_ids,
        "stages": stages,
    }
    if det_item:
        det_res = det_item[1]
        for k in ("detection_count", "class_counts", "detections", "average_confidence", "maximum_confidence", "annotated_image_url"):
            if k in det_res:
                combined[k] = det_res[k]
    if seg_crack_item:
        seg_res = seg_crack_item[1]
        for k in (
            "crack_pixels", "mask_coverage_percent", "contour_count",
            "total_contour_length_px", "largest_contour_length_px",
            "largest_contour_area_px2", "estimated_median_width_px",
            "mean_probability", "overlay_url", "mask_url",
        ):
            if k in seg_res:
                combined[k] = seg_res[k]
        combined.setdefault("threshold", seg_res.get("threshold"))
    if map_crack_item:
        map_res = map_crack_item[1]
        for k in (
            "metrics", "hotspot_count", "occupied_cells", "max_priority",
            "mean_priority_occupied", "top_cells", "grid", "units",
            "calibration", "physical", "test_set_percentiles", "weight_scenario",
            "report_pdf_url",
        ):
            if k in map_res:
                combined[k] = map_res[k]
        if map_res.get("overlay_url"):
            combined["mapping_overlay_url"] = map_res["overlay_url"]
            combined.setdefault("overlay_url", map_res["overlay_url"])
    merged_classes: dict[str, Any] = {}
    if seg_damage_item and isinstance(seg_damage_item[1].get("class_metrics"), dict):
        merged_classes.update(seg_damage_item[1]["class_metrics"])
    if map_damage_item and isinstance(map_damage_item[1].get("class_metrics"), dict):
        dm_res = map_damage_item[1]
        merged_classes.update(dm_res["class_metrics"])
        if "class_counts" not in combined and isinstance(dm_res.get("class_detection_counts"), dict):
            combined["class_counts"] = dm_res["class_detection_counts"]
        for k in ("units", "calibration", "physical", "grid", "composite_summary"):
            if dm_res.get(k) is not None and k not in combined:
                combined[k] = dm_res[k]
        if isinstance(dm_res.get("physical"), dict) and isinstance(combined.get("physical"), dict):
            combined["physical"] = {**dm_res["physical"], **combined["physical"]}
        if "hotspot_count" not in combined and isinstance(dm_res.get("composite_summary"), dict):
            combined["hotspot_count"] = dm_res["composite_summary"].get("hotspot_cells_count")
    if merged_classes:
        combined["class_metrics"] = merged_classes
    combined["telemetry"] = normalize_telemetry(combined)
    return combined


@app.get("/api/vqa/combined-telemetry/{inspection_id}")
def vqa_combined_telemetry(inspection_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    item = db.get(Inspection, inspection_id)
    if item is None or item.status != "succeeded":
        raise HTTPException(404, "Completed inspection not found.")
    combined = _combined_inspection_payload(db, item)
    return {
        "inspection_id": item.id,
        "module": item.module,
        "combined_modules": combined.get("combined_modules", [item.module]),
        "source_inspection_ids": combined.get("source_inspection_ids", {item.module: item.id}),
        "telemetry": combined.get("telemetry") or normalize_telemetry(combined),
    }


class VQAPdfItem(BaseModel):
    question: str = Field(default="", max_length=3000)
    answer: str = Field(default="", max_length=20000)


class VQAPdfRequest(BaseModel):
    inspection_id: str | None = None
    structure_type: str = Field(default="Concrete Element", max_length=100)
    aci_level: str = Field(default="MODERATE", max_length=40)
    action_timeline: str = Field(default="", max_length=500)
    aci_basis: str = Field(default="", max_length=1000)
    telemetry: dict[str, Any] = Field(default_factory=dict)
    qa_items: list[VQAPdfItem] = Field(default_factory=list, max_length=50)


@app.post("/api/vqa/report/pdf")
def vqa_report_pdf(payload: VQAPdfRequest, db: Session = Depends(get_db)) -> Response:
    source_filename: str | None = None
    image_path: str | None = None
    telemetry = payload.telemetry or {}
    if payload.inspection_id:
        item = db.get(Inspection, payload.inspection_id)
        if item is not None:
            source_filename = item.source_filename
            if not telemetry:
                telemetry = _combined_inspection_payload(db, item).get("telemetry", {})
            folder = STORAGE_DIR / item.id
            for candidate_name in ("mapping_overlay.png", "overlay.png", "segmentation_overlay.png", "annotated_image.jpg"):
                candidate = folder / candidate_name
                if candidate.is_file():
                    image_path = str(candidate)
                    break
            if image_path is None and folder.is_dir():
                orig = next(iter(folder.glob("original.*")), None)
                if orig and orig.is_file():
                    image_path = str(orig)
    pdf_bytes = build_vqa_pdf_bytes(
        structure_type=payload.structure_type,
        aci_level=payload.aci_level,
        action_timeline=payload.action_timeline,
        aci_basis=payload.aci_basis,
        telemetry=telemetry,
        qa_items=[item.model_dump() for item in payload.qa_items],
        source_filename=source_filename,
        inspection_id=payload.inspection_id,
        image_path=image_path,
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="crackvision-vqa-report.pdf"'},
    )


@app.post("/api/vqa/ask")
async def vqa_ask(payload: VQARequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    question = payload.question.strip()
    if payload.inspection_id:
        item = db.get(Inspection, payload.inspection_id)
        if item is None or item.status != "succeeded":
            raise HTTPException(404, "Completed inspection not found.")
        combined_payload = _combined_inspection_payload(db, item)
        source_path = next(iter((STORAGE_DIR / item.id).glob("original.*")), None)
        if source_path is None or not source_path.is_file():
            raise HTTPException(410, "The source image for this inspection is missing.")
        if not payload.offline_mode and not payload.api_key:
            raise HTTPException(422, "Cloud VQA is selected but no provider key was supplied. Enter a key or enable offline mode.")
        try:
            prepared, mime_type, _, _ = _prepare_vqa_image(source_path.read_bytes())
            result = await run_in_threadpool(
                evidence_vqa, question, combined_payload, payload.structure_type,
                payload.offline_mode, payload.api_key, prepared, mime_type,
            )
        except RuntimeError as exc:
            raise HTTPException(502, str(exc)) from exc
        normalized = result.get("telemetry") or normalize_telemetry(combined_payload)
        result["telemetry"] = normalized
        result["structure_type"] = payload.structure_type
        result["inspection_id"] = item.id
        result["stages"] = combined_payload.get("stages", {})
        result["report"].setdefault(
            "pipeline_status",
            combined_payload.get("report", {}).get(
                "pipeline_status",
                "combined_modules" if len(combined_payload.get("combined_modules", [])) > 1 else "existing_inspection",
            ),
        )
        for key, value in (
            ("detection_count", normalized.get("detection_count")),
            ("class_counts", normalized.get("class_counts")),
            ("crack_pixels", normalized.get("crack_pixels")),
            ("crack_coverage_percent", normalized.get("coverage_percent")),
            ("estimated_crack_width_px", normalized.get("mapping_width_p95", normalized.get("estimated_median_width_px"))),
            ("damage_hotspots", normalized.get("hotspot_count")),
            ("peak_damage_priority", normalized.get("max_priority")),
            ("combined_modules", combined_payload.get("combined_modules")),
        ):
            if value is not None:
                result["report"].setdefault(key, value)
        return result
    if payload.project_id is None:
        raise HTTPException(422, "Select an inspection or provide a project for evidence VQA.")
    if len(question) < 3:
        raise HTTPException(422, "Enter a question with at least 3 characters.")
    try:
        return await ask_vqa(db, payload.project_id, question, payload.inspection_ids,
                             payload.conversation_id, payload.visual_inspection_id, payload.timestamp_seconds)
    except ValueError as exc:
        status_code = 404 if str(exc) in {"Project not found.", "Conversation not found for this project."} else 422
        raise HTTPException(status_code, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc

@app.post("/api/vqa/inspect-and-ask")
async def vqa_inspect_and_ask(
    file: UploadFile = File(...),
    question: str = Form(...),
    structure_type: str = Form(default="Concrete Element"),
    offline_mode: bool = Form(default=True),
    api_key: str | None = Form(default=None),
    project_id: int | None = Form(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if not question.strip():
        raise HTTPException(422, "Question is required.")
    if not offline_mode and not api_key:
        raise HTTPException(422, "Cloud VQA is selected but no provider key was supplied. Enter a key or enable offline mode.")
    try:
        raw = await file.read(15 * 1024 * 1024 + 1)
        prepared, mime_type, dimensions, _ = _prepare_vqa_image(raw)
    except ValueError as exc:
        raise HTTPException(415, str(exc)) from exc
    file.file.seek(0)
    inspection, input_path = await create_inspection_upload(
        db, file, "detection", project_id, IMAGE_EXTENSIONS,
        {"workflow": "vqa_image_pipeline", "structure_type": structure_type},
    )
    output_dir = STORAGE_DIR / inspection.id
    stages: dict[str, dict[str, Any]] = {}
    detection_result = None
    segmentation_result = None
    damage_segmentation_result = None
    mapping_result = None
    class_mapping_result = None

    try:
        detection_result = await run_in_threadpool(
            detector.detect_image, input_path, output_dir / "annotated_image.jpg", 0.25, 0.45, 640,
        )
        stages["detection"] = {"status": "succeeded", "result": detection_result}
    except Exception as exc:
        stages["detection"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    try:
        segmentation_result = await run_in_threadpool(
            segmenter.segment_image, input_path, output_dir, None, False,
        )
        stages["segmentation"] = {"status": "succeeded", "result": segmentation_result}
    except Exception as exc:
        stages["segmentation"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    try:
        damage_segmentation_result = await run_in_threadpool(
            damage_segmenter.segment_image, input_path, output_dir, False,
            detection_result.get("detections") if detection_result else None,
        )
        stages["damage_segmentation"] = {"status": "succeeded", "result": damage_segmentation_result}
    except Exception as exc:
        stages["damage_segmentation"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    if segmentation_result is not None:
        try:
            calibration = build_calibration("none", None)
            mapping_result = await run_in_threadpool(
                mapping_service.map_image, input_path, output_dir, 0.525, "equal", calibration, False, inspection.source_filename,
            )
            stages["mapping"] = {"status": "succeeded", "result": mapping_result}
            stages["segmentation_and_mapping"] = {"status": "succeeded", "result": {**mapping_result, "peak_frame_url": segmentation_result.get("overlay_url")}}
        except Exception as exc:
            stages["mapping"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            stages["segmentation_and_mapping"] = {"status": "partial", "result": {"peak_frame_url": segmentation_result.get("overlay_url")}}
    else:
        stages["segmentation_and_mapping"] = {"status": "failed", "error": "Crack segmentation did not complete; crack mapping was skipped."}

    # Combine the already-produced V4 crack mask with the detector-guided
    # non-crack masks, then summarize each label on the same image-space grid.
    # This adds no inference pass and does not turn pixel cells into site/GIS
    # coordinates or a severity score.
    if segmentation_result is not None or damage_segmentation_result is not None:
        try:
            class_map = np.zeros((256, 256), dtype=np.uint8)
            if damage_segmentation_result is not None:
                with Image.open(output_dir / "damage_class_ids.png") as mask_image:
                    class_map = np.array(mask_image.convert("L").resize((256, 256), Image.Resampling.NEAREST), dtype=np.uint8, copy=True)
            if segmentation_result is not None:
                with Image.open(output_dir / "segmentation_mask.png") as crack_image:
                    crack_mask = np.asarray(crack_image.convert("L").resize((256, 256), Image.Resampling.NEAREST)) > 0
                class_map[crack_mask] = DAMAGE_CLASS_IDS["crack"]
            analysis = measure_damage_map(class_map)
            source_metrics = (damage_segmentation_result or {}).get("class_metrics", {})
            class_mapping_result = {
                "grid": analysis["grid"],
                "units": "analysis_px",
                "coordinate_note": "Rows and columns are 1-based from the image top-left; these are image cells, not structure, geographic, or physical coordinates.",
                "mask_provenance": (damage_segmentation_result or {}).get("model", "CRACKTO-SFNet V4 crack mask"),
                "classes": {},
                "caveats": analysis["caveats"],
            }
            for label, metrics in analysis["class_metrics"].items():
                matrix = np.asarray(analysis["matrices"][f"{label}_area_ratio"], dtype=float)
                occupied = np.argwhere(matrix > 0)
                ranked = sorted(occupied.tolist(), key=lambda cell: matrix[cell[0], cell[1]], reverse=True)[:3]
                stage_metrics = source_metrics.get(label, {})
                class_mapping_result["classes"][label] = {
                    "pixel_area": metrics["pixel_area"],
                    "coverage_percent": metrics["coverage_percent"],
                    "component_count": metrics["component_count"],
                    "mapped_cell_count": int(len(occupied)),
                    "top_cells": [
                        {"row": int(row + 1), "column": int(col + 1), "cell_coverage_percent": round(float(matrix[row, col] * 100), 2)}
                        for row, col in ranked
                    ],
                    "detector_count": int(stage_metrics.get("detector_count", 0) or 0),
                    "matched_mask_count": int(stage_metrics.get("matched_mask_count", 0) or 0),
                    "mask_method": "CRACKTO-SFNet V4" if label == "crack" else "Class-agnostic FastSAM proposal matched to detector box",
                }
            stages["label_mapping"] = {"status": "succeeded", "result": class_mapping_result}
        except Exception as exc:
            stages["label_mapping"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    if detection_result is None and segmentation_result is None:
        failure = "Neither image detection nor crack segmentation completed. Inspect individual stage errors before retrying."
        fail_inspection(db, inspection, RuntimeError(failure))
        raise HTTPException(503, failure)

    telemetry: dict[str, Any] = {
        "source_module": "vqa_image_pipeline",
        "class_counts": detection_result.get("class_counts", {}) if detection_result else {},
        "damage_classes": {},
        "class_mapping": class_mapping_result,
        "image_width_px": dimensions[0], "image_height_px": dimensions[1],
    }
    if detection_result is not None:
        telemetry["detection_count"] = detection_result["detection_count"]
    if segmentation_result is not None:
        telemetry["damage_classes"] = {
            "crack": {
                "pixel_area": segmentation_result["crack_pixels"],
                "coverage_percent": segmentation_result["mask_coverage_percent"],
                "component_count": segmentation_result["contour_count"],
                "perimeter_px": segmentation_result["total_contour_length_px"],
            },
        }
        telemetry.update({
            "crack_pixels": segmentation_result["crack_pixels"],
            "coverage_percent": segmentation_result["mask_coverage_percent"],
            "estimated_median_width_px": segmentation_result["estimated_median_width_px"],
        })
    if damage_segmentation_result is not None:
        for label, measurements in damage_segmentation_result.get("class_metrics", {}).items():
            if label != "crack":
                telemetry["damage_classes"][label] = measurements
    if mapping_result is not None:
        telemetry.update({
            "mapping_width_p95": mapping_result.get("metrics", {}).get("width_p95"),
            "hotspot_count": mapping_result.get("hotspot_count"),
            "max_priority": mapping_result.get("max_priority"),
            "mapping_junctions": mapping_result.get("metrics", {}).get("junction_clusters"),
        })
        telemetry = {key: value for key, value in telemetry.items() if value is not None}

    stage_statuses = {name: stage["status"] for name, stage in stages.items()}
    partial = any(status != "succeeded" for status in stage_statuses.values())
    report = {
        "pipeline_status": "partial" if partial else "succeeded",
        "detector_segmentation_ran": True,
        "stage_statuses": stage_statuses,
        "image_dimensions_px": list(dimensions),
        "detection_count": detection_result.get("detection_count") if detection_result else None,
        "class_counts": detection_result.get("class_counts") if detection_result else None,
        "crack_pixels": segmentation_result.get("crack_pixels") if segmentation_result else None,
        "crack_coverage_percent": segmentation_result.get("mask_coverage_percent") if segmentation_result else None,
        "non_crack_mask_method": damage_segmentation_result.get("model") if damage_segmentation_result else None,
        "estimated_crack_width_px": segmentation_result.get("estimated_median_width_px") if segmentation_result else None,
        "damage_hotspots": mapping_result.get("hotspot_count") if mapping_result else None,
        "peak_damage_priority": mapping_result.get("max_priority") if mapping_result else None,
        "class_mapping_grid": class_mapping_result.get("grid") if class_mapping_result else None,
    }
    result_record = {
        "kind": "image", "source_type": "vqa_image_pipeline", "telemetry": telemetry,
        "stages": stages, "report": report,
        "detection_count": detection_result.get("detection_count") if detection_result else None,
        "class_counts": detection_result.get("class_counts") if detection_result else {},
        "vqa_frame_url": storage_url(input_path),
        "annotated_image_url": detection_result.get("annotated_image_url") if detection_result else None,
        "overlay_url": segmentation_result.get("overlay_url") if segmentation_result else None,
        "mask_url": segmentation_result.get("mask_url") if segmentation_result else None,
        "damage_segmentation_overlay_url": damage_segmentation_result.get("overlay_url") if damage_segmentation_result else None,
        "mapping_video_url": None,
    }
    try:
        answer = await run_in_threadpool(
            evidence_vqa, question.strip(), result_record, structure_type,
            offline_mode, api_key, prepared, mime_type,
        )
    except RuntimeError as exc:
        fail_inspection(db, inspection, exc)
        raise HTTPException(502, str(exc)) from exc
    result_record.update({
        "answer": answer["answer"], "severity": answer["severity"],
        "severity_color": answer["severity_color"], "urgency": answer["urgency"],
        "assessment_type": answer["assessment_type"], "engine": answer["engine"],
        "repair_summary": None, "notice": answer["notice"], "evidence": answer["evidence"],
        "report": {**answer["report"], **report}, "structure_type": structure_type,
    })
    complete_inspection(db, inspection, result_record)
    saved_inspection = inspection_dict(inspection)
    return {**answer, **result_record, "inspection": saved_inspection, "inspection_id": inspection.id}

@app.get("/api/vqa/inspections")
def vqa_inspections(project_id: int = Query(gt=0), db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    records = (db.query(Inspection).filter(Inspection.project_id == project_id, Inspection.status == "succeeded")
               .order_by(desc(Inspection.created_at)).limit(100).all())
    output = []
    for item in records:
        try: result = json.loads(item.result_json or "{}")
        except json.JSONDecodeError: result = {}
        kind = result.get("kind", "image")
        output.append({"id": item.id, "module": item.module, "kind": kind,
                       "source_filename": item.source_filename, "created_at": _iso(item.created_at),
                       "visual_supported": item.module in {"detection", "segmentation", "mapping", "video"},
                       "duration_seconds": result.get("duration_seconds")})
    return output

@app.get("/api/inspections/{inspection_id}/evidence")
def inspection_evidence(inspection_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    item = db.get(Inspection, inspection_id)
    if item is None: raise HTTPException(404, "Inspection not found")
    if item.status != "succeeded": raise HTTPException(409, "Evidence is available after inspection processing succeeds")
    observations = build_damage_observations(item, "", limit=500)
    try: result = json.loads(item.result_json or "{}")
    except json.JSONDecodeError: result = {}
    is_video = item.module == "video" or (item.module == "mapping" and result.get("kind") == "video")
    return {"inspection_id": item.id, "project_id": item.project_id, "asset_id": None,
            "module": item.module, "source_filename": item.source_filename,
            "source_type": "video" if is_video else "image",
            "observations": observations, "truncated": len(observations) >= 500,
            "location_note": "Asset, component, and geographic coordinates are unknown unless explicitly supplied in a future inspection schema."}

@app.get("/api/inspections/{inspection_id}/timeline")
def inspection_timeline(inspection_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    evidence = inspection_evidence(inspection_id, db)
    observations = [row for row in evidence["observations"] if row.get("timestamp_seconds") is not None]
    observations.sort(key=lambda row: (row["timestamp_seconds"], row.get("frame_id") or 0))
    return {"inspection_id": inspection_id, "observations": observations, "truncated": evidence["truncated"]}

@app.get("/api/vqa/conversations")
def vqa_conversations(project_id: int = Query(gt=0), db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    records = (db.query(VQAConversation).filter(VQAConversation.project_id == project_id)
               .order_by(desc(VQAConversation.updated_at)).limit(50).all())
    return [{"id": item.id, "title": item.title, "created_at": _iso(item.created_at),
             "updated_at": _iso(item.updated_at), "message_count": len(item.messages)} for item in records]

@app.get("/api/vqa/conversations/{conversation_id}")
def vqa_conversation(conversation_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    item = db.get(VQAConversation, conversation_id)
    if item is None: raise HTTPException(404, "Conversation not found")
    messages = []
    for message in item.messages:
        try: payload = json.loads(message.sources_json or "[]")
        except json.JSONDecodeError: payload = []
        if isinstance(payload, dict):
            message_data = {key: payload[key] for key in ("findings", "evidence_refs", "source_citations", "limitations", "intent") if key in payload}
            message_data["sources"] = payload.get("sources", [])
        else:
            message_data = {"sources": payload if isinstance(payload, list) else []}
        messages.append({"role": message.role, "content": message.content, **message_data, "created_at": _iso(message.created_at)})
    return {"id": item.id, "project_id": item.project_id, "title": item.title,
            "messages": messages}

@app.delete("/api/vqa/conversations/{conversation_id}", status_code=204)
def delete_vqa_conversation(conversation_id: str, db: Session = Depends(get_db)) -> None:
    item = db.get(VQAConversation, conversation_id)
    if item is None: raise HTTPException(404, "Conversation not found")
    db.delete(item); db.commit()

@app.get("/api/vqa/knowledge")
def list_vqa_knowledge() -> list[dict[str, Any]]:
    return knowledge_documents()

@app.post("/api/vqa/knowledge", status_code=201)
def upload_vqa_knowledge(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        data = file.file.read(10 * 1024 * 1024 + 1)
        if len(data) > 10 * 1024 * 1024:
            raise HTTPException(413, "Reference documents are limited to 10 MB.")
        return store_knowledge_document(file.filename or "reference", data)
    except ValueError as exc:
        raise HTTPException(415, str(exc)) from exc

@app.delete("/api/vqa/knowledge/{document_id}", status_code=204)
def remove_vqa_knowledge(document_id: str) -> None:
    try:
        delete_knowledge_document(document_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Reference document not found") from exc

@app.get("/api/projects")
def list_projects(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    projects = db.query(Project).order_by(desc(Project.created_at)).all()
    return [project_dict(p, db) for p in projects]

@app.post("/api/projects", status_code=201)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)) -> dict[str, Any]:
    project = Project(**payload.model_dump())
    db.add(project); db.commit(); db.refresh(project)
    return project_dict(project, db)

@app.patch("/api/projects/{project_id}")
def update_project(project_id: int, payload: ProjectUpdate, db: Session = Depends(get_db)) -> dict[str, Any]:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    for key, value in payload.model_dump(exclude_unset=True).items():
        if value is not None:
            setattr(project, key, value)
    db.commit(); db.refresh(project)
    return project_dict(project, db)

@app.delete("/api/projects/{project_id}", status_code=204)
def delete_project(project_id: int, db: Session = Depends(get_db)) -> None:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    # Keep every inspection linked to the project; explicit deletion would erase history.
    if db.query(Inspection).filter(Inspection.project_id == project_id).count():
        raise HTTPException(409, "This project has inspection history. Move or delete its inspections before deleting the project.")
    for conversation in db.query(VQAConversation).filter(VQAConversation.project_id == project_id).all():
        db.delete(conversation)
    db.delete(project); db.commit()

@app.get("/api/dashboard")
def dashboard(db: Session = Depends(get_db)) -> dict[str, Any]:
    total_inspections = db.query(Inspection).count()
    completed = db.query(Inspection).filter(Inspection.status == "succeeded").count()
    active_projects = db.query(Project).count()
    by_module = {m: db.query(Inspection).filter(Inspection.module == m).count() for m in MODULES}
    mapped_hotspots = 0
    detection_total = 0
    class_counts: dict[str, int] = {}
    for inspection in db.query(Inspection).filter(Inspection.status == "succeeded").all():
        try: result = json.loads(inspection.result_json or "{}")
        except json.JSONDecodeError: result = {}
        count = result.get("detection_count", result.get("total_detections", 0))
        detection_total += int(count or 0)
        if inspection.module == "mapping" and result.get("kind") == "image":
            mapped_hotspots += int(result.get("hotspot_count", 0) or 0)
        for name, value in result.get("class_counts", {}).items():
            class_counts[name] = class_counts.get(name, 0) + int(value or 0)
    recent = db.query(Inspection).options(joinedload(Inspection.project)).order_by(desc(Inspection.created_at)).limit(7).all()
    return {
        "total_inspections": total_inspections, "completed_inspections": completed,
        "active_projects": active_projects, "defects_detected": detection_total, "mapped_hotspots": mapped_hotspots,
        "by_module": by_module, "class_counts": class_counts,
        "recent_inspections": [inspection_dict(i) for i in recent],
    }

@app.get("/api/inspections")
def list_inspections(
    module: str | None = Query(default=None), status: str | None = Query(default=None),
    project_id: int | None = Query(default=None), q: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500), db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    query = db.query(Inspection).options(joinedload(Inspection.project))
    if module in set(MODULES): query = query.filter(Inspection.module == module)
    if status: query = query.filter(Inspection.status == status)
    if project_id is not None: query = query.filter(Inspection.project_id == project_id)
    if q: query = query.filter(Inspection.source_filename.ilike(f"%{q[:100]}%"))
    return [inspection_dict(i) for i in query.order_by(desc(Inspection.created_at)).limit(limit).all()]

@app.get("/api/inspections/{inspection_id}")
def get_inspection(inspection_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    inspection = db.query(Inspection).options(joinedload(Inspection.project)).filter(Inspection.id == inspection_id).first()
    if inspection is None: raise HTTPException(404, "Inspection not found")
    return inspection_dict(inspection)

@app.delete("/api/inspections/{inspection_id}", status_code=204)
def delete_inspection(inspection_id: str, db: Session = Depends(get_db)) -> None:
    inspection = db.get(Inspection, inspection_id)
    if inspection is None: raise HTTPException(404, "Inspection not found")
    folder = STORAGE_DIR / inspection.id
    db.delete(inspection); db.commit()
    if folder.exists(): shutil.rmtree(folder, ignore_errors=True)

@app.post("/api/detect")
async def detect_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence: float = Form(default=0.25, ge=0.01, le=0.99),
    iou: float = Form(default=0.45, ge=0.05, le=0.95),
    image_size: int = Form(default=640),
    frame_rate: float = Form(default=2.0, ge=0.2, le=10.0),
    project_id: int | None = Form(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _validate_confidence(confidence, iou, image_size)
    inspection, input_path = await create_inspection_upload(db, file, "detection", project_id, IMAGE_EXTENSIONS, {"confidence": confidence, "iou": iou, "image_size": image_size})
    try:
        from PIL import Image
        with Image.open(input_path) as image: image.verify()
        output_path = STORAGE_DIR / inspection.id / "annotated_image.jpg"
        result = detector.detect_image(input_path, output_path, confidence, iou, image_size)
        complete_inspection(db, inspection, result)
        return inspection_dict(inspection)
    except Exception as exc:
        fail_inspection(db, inspection, exc)
        raise HTTPException(503 if isinstance(exc, RuntimeError) else 400, str(exc)) from exc

@app.post("/api/segment")
async def segment_image(
    file: UploadFile = File(...),
    threshold: float | None = Form(default=None, ge=0.05, le=0.95),
    tta: bool = Form(default=False),
    project_id: int | None = Form(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    inspection, input_path = await create_inspection_upload(db, file, "segmentation", project_id, IMAGE_EXTENSIONS, {"threshold": threshold, "tta": tta})
    try:
        from PIL import Image
        with Image.open(input_path) as image: image.verify()
        output_dir = STORAGE_DIR / inspection.id
        result = segmenter.segment_image(input_path, output_dir, threshold, tta)
        complete_inspection(db, inspection, result)
        return inspection_dict(inspection)
    except Exception as exc:
        fail_inspection(db, inspection, exc)
        raise HTTPException(503 if isinstance(exc, RuntimeError) else 400, str(exc)) from exc

@app.post("/api/segment/damage")
async def segment_damage_image(
    file: UploadFile = File(...),
    project_id: int | None = Form(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if not damage_segmenter.available():
        raise HTTPException(503, "Detector-guided segmentation needs detector.pt and FastSAM-s.pt in backend/app/models.")
    inspection, input_path = await create_inspection_upload(
        db, file, "segmentation", project_id, IMAGE_EXTENSIONS,
        {"model": "CRACKTO-SFNet V4 + detector-box-matched FastSAM", "class_trained_non_crack_segmenter": False},
    )
    try:
        with Image.open(input_path) as image: image.verify()
        result = await run_in_threadpool(
            damage_segmenter.segment_image, input_path, STORAGE_DIR / inspection.id, True, None,
        )
        complete_inspection(db, inspection, result)
        return inspection_dict(inspection)
    except Exception as exc:
        fail_inspection(db, inspection, exc)
        raise HTTPException(503 if isinstance(exc, RuntimeError) else 400, str(exc)) from exc

@app.post("/api/video", status_code=202)
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence: float = Form(default=0.25, ge=0.01, le=0.99),
    iou: float = Form(default=0.45, ge=0.05, le=0.95),
    image_size: int = Form(default=640),
    frame_rate: float = Form(default=2.0, ge=0.2, le=10.0),
    project_id: int | None = Form(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _validate_confidence(confidence, iou, image_size)
    inspection, input_path = await create_inspection_upload(db, file, "video", project_id, VIDEO_EXTENSIONS, {"confidence": confidence, "iou": iou, "image_size": image_size, "frame_rate": frame_rate})
    inspection.status = "queued"; db.commit(); db.refresh(inspection)
    background_tasks.add_task(run_video_job, inspection.id, str(input_path), str(STORAGE_DIR / inspection.id), confidence, iou, image_size, frame_rate)
    return inspection_dict(inspection)

def _mapping_inputs(weight_scenario: str, calibration: str, height: float | None, focal: float | None,
                    sensor_w: float | None, sensor_h: float | None, swap_axes: bool) -> dict[str, Any]:
    if weight_scenario not in MAPPING_SCENARIOS:
        raise HTTPException(422, f"Weight scenario must be one of: {', '.join(sorted(MAPPING_SCENARIOS))}")
    if not mapping_service.runtime_installed:
        raise HTTPException(503, f"Damage mapping needs Python packages that are not installed: {', '.join(mapping_service.missing_packages)}. Re-run setup to install backend requirements.")
    camera = None
    if calibration == "camera":
        if None in (height, focal, sensor_w, sensor_h):
            raise HTTPException(422, "Camera calibration needs height, focal length, sensor width and sensor height (all in mm).")
        camera = {"height_mm": height, "focal_length_mm": focal, "sensor_width_mm": sensor_w, "sensor_height_mm": sensor_h, "swap_axes": swap_axes}
    try:
        return build_calibration(calibration, camera)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

@app.post("/api/mapping/image")
async def map_damage_image(
    file: UploadFile = File(...),
    threshold: float = Form(default=0.525, ge=0.05, le=0.95),
    weight_scenario: str = Form(default="equal"),
    tta: bool = Form(default=False),
    calibration: str = Form(default="none"),
    camera_height_mm: float | None = Form(default=None, gt=0),
    focal_length_mm: float | None = Form(default=None, gt=0),
    sensor_width_mm: float | None = Form(default=None, gt=0),
    sensor_height_mm: float | None = Form(default=None, gt=0),
    swap_axes: bool = Form(default=False),
    project_id: int | None = Form(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    cal = _mapping_inputs(weight_scenario, calibration, camera_height_mm, focal_length_mm, sensor_width_mm, sensor_height_mm, swap_axes)
    settings = {"mode": "image", "threshold": threshold, "weight_scenario": weight_scenario, "tta": tta, "calibration": cal}
    inspection, input_path = await create_inspection_upload(db, file, "mapping", project_id, IMAGE_EXTENSIONS, settings)
    try:
        from PIL import Image
        with Image.open(input_path) as image: image.verify()
        result = mapping_service.map_image(input_path, STORAGE_DIR / inspection.id, threshold, weight_scenario, cal, tta, inspection.source_filename)
        complete_inspection(db, inspection, result)
        return inspection_dict(inspection)
    except Exception as exc:
        fail_inspection(db, inspection, exc)
        raise HTTPException(503 if isinstance(exc, RuntimeError) else 400, str(exc)) from exc

@app.post("/api/mapping/damage/image")
async def map_damage_classes_image(
    file: UploadFile = File(...),
    calibration: str = Form(default="none"),
    camera_height_mm: float | None = Form(default=None, gt=0),
    focal_length_mm: float | None = Form(default=None, gt=0),
    sensor_width_mm: float | None = Form(default=None, gt=0),
    sensor_height_mm: float | None = Form(default=None, gt=0),
    swap_axes: bool = Form(default=False),
    project_id: int | None = Form(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if not damage_mapping_service.available():
        raise HTTPException(503, "Detector-guided damage mapping needs detector.pt and FastSAM-s.pt in backend/app/models.")
    cal = _mapping_inputs("equal", calibration, camera_height_mm, focal_length_mm, sensor_width_mm, sensor_height_mm, swap_axes)
    settings = {"mode": "image", "model": "detector-box-matched FastSAM", "calibration": cal}
    inspection, input_path = await create_inspection_upload(db, file, "mapping", project_id, IMAGE_EXTENSIONS, settings)
    try:
        with Image.open(input_path) as image: image.verify()
        result = await run_in_threadpool(
            damage_mapping_service.map_image, input_path, STORAGE_DIR / inspection.id, cal, inspection.source_filename,
        )
        complete_inspection(db, inspection, result)
        return inspection_dict(inspection)
    except Exception as exc:
        fail_inspection(db, inspection, exc)
        raise HTTPException(503 if isinstance(exc, RuntimeError) else 400, str(exc)) from exc

@app.post("/api/mapping/video", status_code=202)
async def map_damage_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    threshold: float = Form(default=0.525, ge=0.05, le=0.95),
    weight_scenario: str = Form(default="equal"),
    frame_step: int = Form(default=5, ge=1, le=120),
    max_frames: int = Form(default=0, ge=0, le=5000),
    calibration: str = Form(default="none"),
    camera_height_mm: float | None = Form(default=None, gt=0),
    focal_length_mm: float | None = Form(default=None, gt=0),
    sensor_width_mm: float | None = Form(default=None, gt=0),
    sensor_height_mm: float | None = Form(default=None, gt=0),
    swap_axes: bool = Form(default=False),
    project_id: int | None = Form(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    cal = _mapping_inputs(weight_scenario, calibration, camera_height_mm, focal_length_mm, sensor_width_mm, sensor_height_mm, swap_axes)
    settings = {"mode": "video", "threshold": threshold, "weight_scenario": weight_scenario, "frame_step": frame_step, "max_frames": max_frames, "calibration": cal}
    inspection, input_path = await create_inspection_upload(db, file, "mapping", project_id, VIDEO_EXTENSIONS, settings)
    inspection.status = "queued"; db.commit(); db.refresh(inspection)
    background_tasks.add_task(run_mapping_video_job, inspection.id, str(input_path), str(STORAGE_DIR / inspection.id), threshold, weight_scenario, cal, frame_step, max_frames)
    return inspection_dict(inspection)

@app.get("/api/mapping/reference")
def mapping_reference() -> dict[str, Any]:
    if not mapping_service.runtime_installed:
        return {"available": False, "error": f"Missing Python packages: {', '.join(mapping_service.missing_packages)}"}
    return mapping_service.reference_summary()

async def create_inspection_upload(db: Session, file: UploadFile, module: str, project_id: int | None, allowed_extensions: set[str], settings: dict[str, Any]) -> tuple[Inspection, Path]:
    original_name = Path(file.filename or "upload").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(415, f"Unsupported file extension. Accepted: {', '.join(sorted(allowed_extensions))}")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if not content: raise HTTPException(400, "The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES: raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
    project = resolve_project(db, project_id)
    inspection_id = str(uuid.uuid4())
    workdir = STORAGE_DIR / inspection_id
    workdir.mkdir(parents=True, exist_ok=True)
    input_path = workdir / f"original{suffix}"
    input_path.write_bytes(content)
    inspection = Inspection(
        id=inspection_id, project_id=project.id if project else None, module=module,
        status="processing", source_filename=original_name,
        settings_json=json.dumps({**settings, "asset_fingerprint": hashlib.sha256(content).hexdigest()}), result_json="{}",
        created_at=datetime.now(timezone.utc),
    )
    db.add(inspection); db.commit(); db.refresh(inspection)
    return inspection, input_path

def resolve_project(db: Session, project_id: int | None) -> Project | None:
    if project_id is not None:
        project = db.get(Project, project_id)
        if project is None: raise HTTPException(404, "Selected project not found")
        return project
    return db.query(Project).order_by(Project.id.asc()).first()

def complete_inspection(db: Session, inspection: Inspection, result: dict[str, Any]) -> None:
    inspection.status = "succeeded"
    inspection.result_json = json.dumps(result)
    inspection.completed_at = datetime.now(timezone.utc)
    inspection.error_message = None
    db.commit(); db.refresh(inspection)

def fail_inspection(db: Session, inspection: Inspection, exc: Exception) -> None:
    inspection.status = "failed"
    inspection.error_message = f"{type(exc).__name__}: {exc}"
    inspection.completed_at = datetime.now(timezone.utc)
    db.commit()

def _prepare_vqa_image(raw: bytes) -> tuple[bytes, str, tuple[int, int], str]:
    if not raw:
        raise ValueError("The uploaded image is empty.")
    if len(raw) > 15 * 1024 * 1024:
        raise ValueError("Image uploads for VQA are limited to 15 MB.")
    try:
        source = Image.open(BytesIO(raw))
        image_format = (source.format or "").upper()
        if image_format not in {"JPEG", "PNG", "WEBP", "TIFF", "GIF", "BMP"}:
            raise ValueError("Use a JPEG, PNG, WEBP, TIFF, GIF, or BMP image.")
        if source.width * source.height > 60_000_000:
            raise ValueError("Image dimensions exceed the 60 megapixel limit.")
        image = ImageOps.exif_transpose(source).convert("RGB")
        size = image.size
        image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(output, format="JPEG", quality=88, optimize=True)
        return output.getvalue(), "image/jpeg", size, image_format.lower()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The uploaded file is not a readable image.") from exc

def _validate_confidence(confidence: float, iou: float, image_size: int) -> None:
    if not 0.01 <= confidence <= 0.99: raise HTTPException(422, "Confidence must be between 0.01 and 0.99")
    if not 0.05 <= iou <= 0.95: raise HTTPException(422, "IoU must be between 0.05 and 0.95")
    if image_size not in {320, 416, 512, 640, 768, 960, 1024}: raise HTTPException(422, "Image size must be one of 320, 416, 512, 640, 768, 960, or 1024")

def project_dict(project: Project, db: Session) -> dict[str, Any]:
    return {
        "id": project.id, "name": project.name, "site_location": project.site_location,
        "structure_type": project.structure_type, "description": project.description,
        "created_at": _iso(project.created_at),
        "inspection_count": db.query(Inspection).filter(Inspection.project_id == project.id).count(),
    }

def inspection_dict(inspection: Inspection) -> dict[str, Any]:
    try: result = json.loads(inspection.result_json or "{}")
    except json.JSONDecodeError: result = {}
    try: settings = json.loads(inspection.settings_json or "{}")
    except json.JSONDecodeError: settings = {}
    settings.pop("asset_fingerprint", None)
    project = getattr(inspection, "project", None)
    return {
        "id": inspection.id, "project_id": inspection.project_id,
        "project_name": project.name if project else None,
        "module": inspection.module, "status": inspection.status,
        "source_filename": inspection.source_filename, "settings": settings,
        "result": result, "error_message": inspection.error_message,
        "created_at": _iso(inspection.created_at), "completed_at": _iso(inspection.completed_at),
    }

def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None
