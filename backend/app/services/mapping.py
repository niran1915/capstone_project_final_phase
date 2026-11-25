"""Damage-mapping module (CrackVision module 04).

Wraps the V4 segmentation model (``segmenter``) with the spatial damage-mapping analysis from the
supplied research notebook. Works on a single image or, as a background job, on sampled video frames.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ..database import SessionLocal
from ..main_paths import storage_url
from .segmentation import segmenter

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
NORMALIZATION_PATH = MODELS_DIR / "mapping_normalization_reference.json"
REFERENCE_DIR = Path(__file__).resolve().parents[3] / "assets" / "reference_outputs" / "mapping"

SUT_CAMERA = {"height_mm": 672.0, "focal_length_mm": 5.4, "sensor_width_mm": 7.3, "sensor_height_mm": 5.4, "swap_axes": False}
REQUIRED_PACKAGES = ("scipy", "skimage", "pandas", "matplotlib")


def build_calibration(mode: str, camera: dict[str, float] | None) -> dict[str, Any]:
    """Resolve the calibration request into a settings dict (physical units stay OFF by default)."""
    mode = (mode or "none").lower()
    if mode == "none":
        return {"enabled": False, "mode": "none"}
    if mode == "sut_preset":
        cam = dict(SUT_CAMERA)
    elif mode == "camera":
        if not camera:
            raise ValueError("Camera geometry is required for camera calibration.")
        cam = {**camera, "swap_axes": bool(camera.get("swap_axes", False))}
    else:
        raise ValueError("Calibration must be one of: none, sut_preset, camera.")
    from .mapping_core import camera_scale
    sx, sy = camera_scale(cam)
    return {"enabled": True, "mode": mode, "camera": cam, "scale_x_mm_per_px": sx, "scale_y_mm_per_px": sy,
            "field_of_view_mm": [sx * 256, sy * 256]}


class MappingService:
    def __init__(self) -> None:
        self._scales: dict[str, float] | None = None
        self._scale_info: dict[str, Any] = {}
        self._test_stats: dict[str, np.ndarray] | None = None
        self._lock = threading.Lock()

    # ----------------------------------------------------------------------------- readiness
    @property
    def runtime_installed(self) -> bool:
        return all(importlib.util.find_spec(name) is not None for name in REQUIRED_PACKAGES)

    @property
    def missing_packages(self) -> list[str]:
        return [name for name in REQUIRED_PACKAGES if importlib.util.find_spec(name) is None]

    def available(self) -> bool:
        return segmenter.available() and self.runtime_installed

    def model_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "weights_found": segmenter.available(), "checkpoint": "segmentation_v4_best.pth (shared with Segmentation lab)",
            "runtime_installed": self.runtime_installed, "loaded": segmenter.model is not None,
            "normalization_reference_found": NORMALIZATION_PATH.is_file(),
            "reference_outputs_found": REFERENCE_DIR.is_dir(),
            "default_threshold": 0.525, "grid": [20, 20], "analysis_size": 256,
            "index_version": "fixed_validation_normalized_exploratory_index_v2",
        }
        if not self.runtime_installed:
            info["load_error"] = f"Missing Python packages: {', '.join(self.missing_packages)}. Re-run setup to install backend requirements."
        return info

    def scales(self) -> tuple[dict[str, float], dict[str, Any]]:
        if self._scales is None:
            from .mapping_core import load_scales
            self._scales, self._scale_info = load_scales(NORMALIZATION_PATH)
        return self._scales, self._scale_info

    # ----------------------------------------------------------------------------- reference data
    def _test_distribution(self) -> dict[str, np.ndarray]:
        """Per-image measurements for the 1,200 CrackVision-12K test images (from the notebook run)."""
        if self._test_stats is None:
            stats: dict[str, np.ndarray] = {}
            path = REFERENCE_DIR / "test_image_measurements.csv"
            if path.is_file():
                import pandas as pd
                frame = pd.read_csv(path)
                for column in ("crack_area_ratio", "crack_length", "width_p95", "junction_clusters"):
                    if column in frame:
                        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
                        stats[column] = values[np.isfinite(values)]
            self._test_stats = stats
        return self._test_stats

    def test_set_percentiles(self, record: dict[str, Any]) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for column, key in (("crack_area_ratio", "crack_area_ratio"), ("crack_length", "crack_length"),
                            ("width_p95", "width_p95"), ("junction_clusters", "junction_clusters")):
            reference = self._test_distribution().get(column)
            value = record.get(key)
            if reference is None or not len(reference) or value is None or not np.isfinite(value):
                out[column] = None
            else:
                out[column] = round(100.0 * float(np.mean(reference <= value)), 1)
        return out

    def reference_summary(self) -> dict[str, Any]:
        """Summary of the saved notebook research outputs for the in-app reference tab."""
        import pandas as pd
        from .mapping_core import json_safe
        if not REFERENCE_DIR.is_dir():
            return {"available": False}

        def read_csv(rel: str):
            path = REFERENCE_DIR / rel
            return pd.read_csv(path) if path.is_file() else None

        summary: dict[str, Any] = {"available": True, "base_url": "/reference/mapping"}
        metrics = read_csv("FINAL_METRICS_SUMMARY.csv")
        if metrics is not None:
            summary["segmentation_metrics"] = json_safe(metrics.to_dict(orient="records"))
        meta_path = REFERENCE_DIR / "research_mapping_metadata.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            summary["metadata"] = json_safe({k: meta.get(k) for k in (
                "method_version", "selected_validation_threshold", "analysis_size", "grid", "normalization_scales_from_validation_only",
                "normalization_quantile", "baseline_weights", "weight_scenarios", "physical_units_enabled", "caveats", "processed_test_images")})

        def med(frame, column):
            if frame is None or column not in frame:
                return None
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            return float(values.median()) if len(values) else None

        weights = read_csv("sensitivity/weight_sensitivity.csv")
        if weights is not None:
            rows = []
            for scenario, group in weights.groupby("weight_scenario", sort=False):
                rows.append({"scenario": scenario,
                             "median_spearman_vs_equal": med(group, "spearman_vs_equal_baseline"),
                             "median_top10_iou_vs_equal": med(group, "top10_region_iou_vs_equal"),
                             "images": int(group["image"].nunique())})
            summary["weight_sensitivity"] = json_safe(rows)
        thresholds = read_csv("sensitivity/threshold_sensitivity.csv")
        if thresholds is not None:
            summary["threshold_sensitivity"] = json_safe({
                "selected": float(thresholds["selected_threshold"].iloc[0]), "alternate": float(thresholds["alternate_threshold"].iloc[0]),
                "median_priority_rank_spearman": med(thresholds, "priority_rank_spearman"),
                "median_top10_iou": med(thresholds, "top10_region_iou"),
                "median_area_ratio_abs_change": med(thresholds, "area_ratio_abs_change"), "images": int(len(thresholds))})
        grid = read_csv("sensitivity/grid_resolution_sensitivity.csv")
        if grid is not None:
            rows = []
            for comparison, group in grid.groupby("comparison_grid", sort=False):
                rows.append({"comparison_grid": comparison,
                             "median_spearman": med(group, "spatial_rank_spearman_after_resampling"),
                             "median_top_decile_iou": med(group, "top_decile_spatial_iou_after_resampling"),
                             "images": int(group["image"].nunique())})
            summary["grid_sensitivity"] = json_safe(rows)
        gt = read_csv("prediction_vs_GT_geometry_validation.csv")
        if gt is not None:
            summary["prediction_vs_gt"] = json_safe({
                "images": int(len(gt)),
                "median_area_ratio_abs_error": med(gt, "area_ratio_abs_error"),
                "median_length_density_abs_error": med(gt, "length_density_abs_error"),
                "median_width_p95_abs_error": med(gt, "width_p95_abs_error"),
                "median_priority_spearman": med(gt, "priority_index_spearman_pred_vs_gt"),
                "median_top10_region_iou": med(gt, "top_10pct_priority_region_iou"),
                "note": "Predicted-mask vs ground-truth-mask geometric consistency only; not validation against physical crack measurements."})
        visuals = []
        vis_dir = REFERENCE_DIR / "visuals"
        if vis_dir.is_dir():
            for card in sorted(vis_dir.glob("*_evidence_card.png")):
                stem = card.name.split("_")[0]
                visuals.append({"id": stem, "evidence_card": f"/reference/mapping/visuals/{card.name}",
                                "surface": f"/reference/mapping/visuals/{stem}_2p5D_index.png"})
        summary["visuals"] = visuals
        summary["downloads"] = [
            {"label": "Final segmentation metrics", "url": "/reference/mapping/FINAL_METRICS_SUMMARY.csv"},
            {"label": "Validation threshold sweep", "url": "/reference/mapping/validation_threshold_sweep.csv"},
            {"label": "Test image measurements (1,200)", "url": "/reference/mapping/test_image_measurements.csv"},
            {"label": "Prediction vs GT geometry", "url": "/reference/mapping/prediction_vs_GT_geometry_validation.csv"},
            {"label": "Normalization reference", "url": "/reference/mapping/validation_normalization_reference.json"},
            {"label": "Weight sensitivity", "url": "/reference/mapping/sensitivity/weight_sensitivity.csv"},
            {"label": "Threshold sensitivity", "url": "/reference/mapping/sensitivity/threshold_sensitivity.csv"},
            {"label": "Grid-resolution sensitivity", "url": "/reference/mapping/sensitivity/grid_resolution_sensitivity.csv"},
        ]
        return summary

    # ----------------------------------------------------------------------------- shared analysis
    def _analyze(self, rgb: np.ndarray, threshold: float, scenario: str, calibration: dict[str, Any], tta: bool) -> dict[str, Any]:
        from .mapping_core import WEIGHT_SCENARIOS, analyze_probability
        if scenario not in WEIGHT_SCENARIOS:
            raise ValueError(f"Unknown weight scenario {scenario!r}. Options: {', '.join(WEIGHT_SCENARIOS)}")
        scales, _ = self.scales()
        probability = segmenter.predict_probability(rgb, tta=tta)
        scale_xy = (calibration["scale_x_mm_per_px"], calibration["scale_y_mm_per_px"]) if calibration.get("enabled") else None
        return analyze_probability(probability, threshold, scales, WEIGHT_SCENARIOS[scenario], scale_xy)

    # ----------------------------------------------------------------------------- image
    def map_image(self, input_path: Path, output_dir: Path, threshold: float, scenario: str,
                  calibration: dict[str, Any], tta: bool = False, source_name: str | None = None) -> dict[str, Any]:
        from .mapping_core import (CAVEATS, GRID_COLS, GRID_ROWS, INDEX_VERSION, WEIGHT_SCENARIOS, json_safe, overlay_on_photo,
                                   ranked_cells, render_evidence_card, render_matrix_png, render_surface_png, save_png, weight_sensitivity)
        with Image.open(input_path) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB")
        rgb = np.asarray(image, dtype=np.uint8)
        height, width = rgb.shape[:2]
        analysis = self._analyze(rgb, threshold, scenario, calibration, tta)
        regions, matrices, arrays = analysis["regions"], analysis["matrices"], analysis["arrays"]
        scales, scale_info = self.scales()

        output_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = output_dir / "mapping_overlay.png"
        heatmap_path = output_dir / "mapping_priority_heatmap.png"
        surface_path = output_dir / "mapping_surface_2p5d.png"
        card_path = output_dir / "mapping_evidence_card.png"
        regions_path = output_dir / "mapping_regions.csv"
        arrays_path = output_dir / "mapping_arrays.npz"
        summary_path = output_dir / "mapping_summary.json"
        report_path = output_dir / "mapping_report.pdf"

        save_png(overlay_on_photo(rgb, arrays["mask"], arrays["skeleton"], matrices["hotspot"], matrices["priority"]), overlay_path)
        render_matrix_png(matrices["priority"], heatmap_path, "Exploratory inspection-priority index (relative)", hotspot=matrices["hotspot"])
        render_surface_png(matrices["priority"], surface_path)
        title = source_name or input_path.name
        render_evidence_card(rgb, analysis, card_path, title, threshold)
        regions.to_csv(regions_path, index=False)
        np.savez_compressed(
            arrays_path, probability=analysis["probability"], mask=arrays["mask"].astype(np.uint8), skeleton=arrays["skeleton"].astype(np.uint8),
            width_map=arrays["width_map"], width_samples=arrays["width_samples"], length_total=arrays["length_total"],
            junction_count=arrays["junction_count"], **{f"grid_{k}": np.nan_to_num(v, nan=-1.0) for k, v in matrices.items()})

        record = analysis["image_record"]
        positive = regions.loc[regions["priority_index_exploratory"] > 0, "priority_index_exploratory"]
        hotspot_count = int(regions["relative_hotspot_top_quartile"].sum())
        occupied = int((~regions["empty_cell"]).sum())
        result = {
            "kind": "image", "model": "CRACKTO-SFNet V4", "method_version": INDEX_VERSION,
            "threshold": float(threshold), "tta_enabled": bool(tta),
            "weight_scenario": scenario, "weights": WEIGHT_SCENARIOS[scenario],
            "grid": [GRID_ROWS, GRID_COLS], "analysis_size": 256, "source_width": width, "source_height": height,
            "units": "mm" if calibration.get("enabled") else "analysis_px",
            "calibration": calibration, "physical": analysis["physical"],
            "metrics": record, "hotspot_count": hotspot_count, "occupied_cells": occupied,
            "total_cells": int(len(regions)), "max_priority": float(regions["priority_index_exploratory"].max()),
            "mean_priority_occupied": float(positive.mean()) if len(positive) else 0.0,
            "top_cells": ranked_cells(regions, 10), "matrices": matrices,
            "weight_sensitivity": weight_sensitivity(regions, scales),
            "test_set_percentiles": self.test_set_percentiles(record),
            "normalization": {"scales": scales, **scale_info},
            "overlay_url": storage_url(overlay_path), "priority_heatmap_url": storage_url(heatmap_path),
            "surface_url": storage_url(surface_path), "evidence_card_url": storage_url(card_path),
            "regions_csv_url": storage_url(regions_path), "arrays_npz_url": storage_url(arrays_path),
            "summary_json_url": storage_url(summary_path), "report_pdf_url": None,
            "caveats": CAVEATS,
            "measurement_note": ("Lengths and widths are in analysis pixels at 256x256 unless a camera calibration was supplied; "
                                 "the priority index always uses analysis-pixel features so the validation scales remain valid."),
        }
        result = json_safe(result)
        if self._write_pdf(report_path, title, result, card_path):
            result["report_pdf_url"] = storage_url(report_path)
        summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    @staticmethod
    def _write_pdf(path: Path, title: str, result: dict[str, Any], card_path: Path) -> bool:
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import mm
            from reportlab.platypus import Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        except Exception:  # noqa: BLE001 - reportlab is optional
            return False
        styles = getSampleStyleSheet()
        m = result["metrics"]

        def fmt(v, digits=3):
            return "—" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))

        doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                                title="CrackVision damage mapping report")
        story: list[Any] = [Paragraph("CrackVision — damage mapping report", styles["Title"]),
                            Paragraph(f"Source: {title}<br/>Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}<br/>"
                                      f"Model: {result['model']} · threshold {result['threshold']:.3f} · index weights: {result['weight_scenario']} · "
                                      f"grid {result['grid'][0]}×{result['grid'][1]}", styles["Normal"]), Spacer(1, 6 * mm)]
        rows = [["Measurement (analysis px @256×256)", "Value"],
                ["Crack area ratio", fmt(m["crack_area_ratio"], 4)], ["Crack length (skeleton graph)", fmt(m["crack_length"], 1)],
                ["Width median / P95 / max (EDT proxy)", f"{fmt(m['width_median'], 2)} / {fmt(m['width_p95'], 2)} / {fmt(m['width_max'], 2)}"],
                ["Endpoints / junction clusters", f"{m['endpoints']} / {m['junction_clusters']}"],
                ["Mask / skeleton components", f"{m['mask_components']} / {m['skeleton_components']}"],
                ["Relative hotspot cells", f"{result['hotspot_count']} of {result['total_cells']}"],
                ["Max exploratory index", fmt(result["max_priority"], 3)]]
        if result.get("physical"):
            p = result["physical"]
            rows += [["Calibrated crack length (mm)", fmt(p["crack_length_mm"], 1)], ["Calibrated width median / P95 (mm)", f"{fmt(p['width_median_mm'], 2)} / {fmt(p['width_p95_mm'], 2)}"]]
        table = Table(rows, colWidths=[95 * mm, 75 * mm])
        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111e2e")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                   ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c9d1da")), ("FONTSIZE", (0, 0), (-1, -1), 9),
                                   ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f8")])]))
        story += [table, Spacer(1, 5 * mm)]
        if card_path.is_file():
            story.append(RLImage(str(card_path), width=165 * mm, height=165 * mm))
        story += [Spacer(1, 3 * mm), Paragraph("<b>Top-ranked regions</b>", styles["Heading3"])]
        top_rows = [["#", "Row", "Col", "Index", "Area ratio", "Length dens.", "Width P95", "Junctions"]]
        for c in result["top_cells"]:
            top_rows.append([c["rank"], c["grid_row"], c["grid_col"], fmt(c["priority_index"], 3), fmt(c["crack_area_ratio"], 3),
                             fmt(c["length_density"], 4), fmt(c["width_p95"], 1), c["junction_count"]])
        top_table = Table(top_rows)
        top_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c9d1da")), ("FONTSIZE", (0, 0), (-1, -1), 8),
                                       ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f6"))]))
        story += [top_table, Spacer(1, 4 * mm), Paragraph("<b>Interpretation limits</b>", styles["Heading3"])]
        story += [Paragraph(f"• {c}", styles["Normal"]) for c in result["caveats"]]
        try:
            doc.build(story)
            return path.is_file()
        except Exception:  # noqa: BLE001
            return False

    # ----------------------------------------------------------------------------- video
    @staticmethod
    def _annotate_frame(frame_bgr: np.ndarray, analysis: dict[str, Any], lines: list[str]) -> np.ndarray:
        import cv2
        h, w = frame_bgr.shape[:2]
        out = frame_bgr.copy()
        mask = cv2.resize(analysis["arrays"]["mask"].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        out[mask] = (0.5 * out[mask] + 0.5 * np.array([70, 70, 226], dtype=np.float32)).astype(np.uint8)
        skeleton = cv2.resize(analysis["arrays"]["skeleton"].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        skeleton = cv2.dilate(skeleton, np.ones((3, 3), np.uint8)) > 0
        out[skeleton] = (90, 235, 255)
        hotspot = analysis["matrices"]["hotspot"]
        rows, cols = hotspot.shape
        fill = out.copy()
        thickness = max(1, int(round(max(h, w) / 600)))
        cells = [(r, c) for r, c in zip(*np.where(hotspot > 0))]
        for r, c in cells:
            p0, p1 = (int(c * w / cols), int(r * h / rows)), (int((c + 1) * w / cols), int((r + 1) * h / rows))
            cv2.rectangle(fill, p0, p1, (0, 140, 255), -1)
        out = cv2.addWeighted(fill, 0.22, out, 0.78, 0)
        for r, c in cells:
            p0, p1 = (int(c * w / cols), int(r * h / rows)), (int((c + 1) * w / cols), int((r + 1) * h / rows))
            cv2.rectangle(out, p0, p1, (0, 122, 255), thickness)
        scale = max(0.5, min(w, h) / 900)
        pad = int(8 * scale)
        line_h = int(22 * scale)
        box_w = int(max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, 1)[0][0] for t in lines)) + 2 * pad
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (box_w, line_h * len(lines) + pad), (24, 30, 46), -1)
        out = cv2.addWeighted(overlay, 0.65, out, 0.35, 0)
        for i, text in enumerate(lines):
            cv2.putText(out, text, (pad, line_h * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def map_video(self, input_path: Path, output_dir: Path, threshold: float, scenario: str, calibration: dict[str, Any],
                  frame_step: int, max_frames: int) -> dict[str, Any]:
        import cv2
        from .mapping_core import (CAVEATS, GRID_COLS, GRID_ROWS, INDEX_VERSION, WEIGHT_SCENARIOS, json_safe, render_matrix_png, save_png)
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            capture.release()
            raise ValueError("The uploaded file could not be opened as a video.")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps <= 0.01:
            fps = 25.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            capture.release()
            raise ValueError("The video has invalid frame dimensions.")
        frame_step = max(1, int(frame_step))
        max_frames = max(0, int(max_frames))
        expected = math.ceil(total_frames / frame_step) if total_frames else 0
        if max_frames and expected:
            expected = min(expected, max_frames)

        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / "mapping_annotated_video.mp4"
        csv_path = output_dir / "mapping_frames.csv"
        summary_path = output_dir / "mapping_summary.json"
        persistence_path = output_dir / "mapping_hotspot_persistence.png"
        peak_path = output_dir / "mapping_peak_frame.png"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps / frame_step), (width, height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("The MP4 output writer could not be initialized on this machine.")

        sum_priority = np.zeros((GRID_ROWS, GRID_COLS))
        max_priority = np.zeros((GRID_ROWS, GRID_COLS))
        hotspot_hits = np.zeros((GRID_ROWS, GRID_COLS))
        rows: list[dict[str, Any]] = []
        peak = {"score": -1.0, "frame": None, "index": None, "t": None}
        frame_index = -1
        started = time.perf_counter()
        db = None
        try:
            from ..entities import Inspection
            db = SessionLocal()
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                if frame_index % frame_step:
                    continue
                if max_frames and len(rows) >= max_frames:
                    break
                analysis = self._analyze(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold, scenario, calibration, False)
                record, matrices = analysis["image_record"], analysis["matrices"]
                priority = np.nan_to_num(matrices["priority"], nan=0.0)
                sum_priority += priority
                max_priority = np.maximum(max_priority, priority)
                hotspot_hits += (matrices["hotspot"] > 0)
                timestamp = frame_index / fps
                hotspots = int((matrices["hotspot"] > 0).sum())
                row = {
                    "frame_index": frame_index, "timestamp_seconds": round(timestamp, 4),
                    "crack_area_ratio": round(record["crack_area_ratio"], 6), "crack_length_px": round(record["crack_length"], 3),
                    "width_median_px": record["width_median"], "width_p95_px": record["width_p95"],
                    "endpoints": record["endpoints"], "junction_clusters": record["junction_clusters"],
                    "mask_components": record["mask_components"], "hotspot_cells": hotspots,
                    "max_priority": round(float(priority.max()), 6), "mean_priority_occupied": round(float(priority[priority > 0].mean()) if (priority > 0).any() else 0.0, 6),
                }
                if analysis["physical"]:
                    row["crack_length_mm"] = round(analysis["physical"]["crack_length_mm"], 3)
                    row["width_p95_mm"] = analysis["physical"]["width_p95_mm"]
                rows.append(row)
                lines = [f"t={timestamp:6.2f}s  frame {frame_index}", f"area {record['crack_area_ratio'] * 100:5.2f}%  length {record['crack_length']:.0f}px",
                         f"hotspots {hotspots}  max index {row['max_priority']:.2f}", "exploratory index - not a safety score"]
                annotated = self._annotate_frame(frame, analysis, lines)
                writer.write(annotated)
                if row["max_priority"] > peak["score"]:
                    peak = {"score": row["max_priority"], "frame": annotated, "index": frame_index, "t": timestamp}
                if len(rows) % 3 == 0 or (expected and len(rows) == expected):
                    inspection = db.get(Inspection, output_dir.name)
                    if inspection:
                        pct = round(100.0 * len(rows) / max(1, expected), 1) if expected else None
                        inspection.result_json = json.dumps({"progress_percent": pct, "processed_frames": len(rows), "total_frames": expected})
                        db.commit()
        finally:
            capture.release()
            writer.release()
            if db is not None:
                db.close()

        if not rows or not video_path.exists() or video_path.stat().st_size < 1024:
            raise RuntimeError("No valid frames were processed; the output video was not produced.")

        n = len(rows)
        mean_priority = sum_priority / n
        persistence = hotspot_hits / n
        render_matrix_png(persistence, persistence_path, f"Hotspot persistence across {n} sampled frames (fraction of frames)", cmap="inferno")
        if peak["frame"] is not None:
            save_png(cv2.cvtColor(peak["frame"], cv2.COLOR_BGR2RGB), peak_path)

        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows([{k: ("" if v is None or (isinstance(v, float) and not math.isfinite(v)) else v) for k, v in r.items()} for r in rows])

        stride = max(1, math.ceil(n / 240))
        series = [{k: r[k] for k in ("frame_index", "timestamp_seconds", "crack_area_ratio", "crack_length_px", "hotspot_cells", "max_priority")} for r in rows[::stride]]
        order = np.dstack(np.unravel_index(np.argsort(-persistence.ravel()), persistence.shape))[0]
        persistent = [{"grid_row": int(r), "grid_col": int(c), "persistence": float(persistence[r, c]), "mean_priority": float(mean_priority[r, c]),
                       "max_priority": float(max_priority[r, c])} for r, c in order[:10] if persistence[r, c] > 0]
        area = np.array([r["crack_area_ratio"] for r in rows])
        length = np.array([r["crack_length_px"] for r in rows])
        summary = {
            "kind": "video", "model": "CRACKTO-SFNet V4", "method_version": INDEX_VERSION,
            "threshold": float(threshold), "weight_scenario": scenario, "weights": WEIGHT_SCENARIOS[scenario],
            "grid": [GRID_ROWS, GRID_COLS], "analysis_size": 256, "calibration": calibration,
            "fps": round(fps, 4), "frame_step": frame_step, "input_frame_count": total_frames, "processed_frame_count": n,
            "width": width, "height": height, "duration_seconds": round((frame_index + 1) / fps, 3),
            "processing_time_seconds": round(time.perf_counter() - started, 2),
            "frames_with_cracks": int((area > 0).sum()), "mean_crack_area_ratio": float(area.mean()), "max_crack_area_ratio": float(area.max()),
            "mean_crack_length_px": float(length.mean()), "max_crack_length_px": float(length.max()),
            "peak_frame_index": peak["index"], "peak_timestamp_seconds": None if peak["t"] is None else round(peak["t"], 4), "peak_max_priority": max(0.0, peak["score"]),
            "matrices": {"mean_priority": mean_priority, "max_priority": max_priority, "hotspot_persistence": persistence},
            "persistent_cells": persistent, "series": series,
            "annotated_video_url": storage_url(video_path), "frames_csv_url": storage_url(csv_path), "summary_json_url": storage_url(summary_path),
            "persistence_heatmap_url": storage_url(persistence_path), "peak_frame_url": storage_url(peak_path) if peak["frame"] is not None else None,
            "caveats": CAVEATS + ["Aggregate maps are in frame coordinates; they are only comparable across frames when the camera view is steady."],
        }
        summary = json_safe(summary)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


mapping_service = MappingService()


def run_mapping_video_job(inspection_id: str, input_path: str, output_dir: str, threshold: float, scenario: str,
                          calibration: dict[str, Any], frame_step: int, max_frames: int) -> None:
    from ..entities import Inspection

    def finish(status: str, result: dict[str, Any] | None = None, error: str | None = None) -> None:
        db = SessionLocal()
        try:
            inspection = db.get(Inspection, inspection_id)
            if inspection is not None:
                inspection.status = status
                if result is not None:
                    inspection.result_json = json.dumps(result)
                inspection.error_message = error
                inspection.completed_at = datetime.now(timezone.utc)
                db.commit()
        finally:
            db.close()

    db = SessionLocal()
    try:
        inspection = db.get(Inspection, inspection_id)
        if inspection is None:
            return
        inspection.status = "processing"
        inspection.error_message = None
        db.commit()
    finally:
        db.close()
    try:
        result = mapping_service.map_video(Path(input_path), Path(output_dir), threshold, scenario, calibration, frame_step, max_frames)
        finish("succeeded", result)
    except Exception as exc:  # noqa: BLE001
        finish("failed", error=f"{type(exc).__name__}: {exc}")
