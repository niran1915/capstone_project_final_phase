"""Evidence-grounded VQA helpers for concrete defect screening.

This service deliberately separates model observations from engineering
interpretation. Pixel counts and detector boxes are never converted to a
structural severity score.
"""
from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from .class_validation import CLASS_ORDER, class_validation_summary


SOURCES = {
    "aci_224r": {
        "title": "ACI 224R-01: Control of Cracking in Concrete Structures (Table 4.1 crack-width guide)",
        "url": "https://www.concrete.org/store/productdetail.aspx?ItemID=22401",
        "scope": "Reasonable serviceability crack-width guide by exposure: dry air 0.41 mm (0.016 in.), humidity/moist air/soil 0.30 mm (0.012 in.), deicing chemicals 0.18 mm (0.007 in.), seawater/wetting-drying 0.15 mm (0.006 in.), water-retaining structures 0.10 mm (0.004 in.).",
    },
    "aci_224_1r": {
        "title": "ACI 224.1R-07: Causes, Evaluation, and Repair of Cracks in Concrete Structures",
        "url": "https://www.concrete.org/store/productdetail.aspx?ItemID=224107",
        "scope": "Establish cause and whether cracks are dormant or active before selecting repair (epoxy injection for dormant structural cracks; flexible sealing/grouting for active or leaking cracks).",
    },
    "aci_cracks": {
        "title": "ACI: Do we always need to repair cracked concrete?",
        "url": "https://www.concrete.org/frequentlyaskedquestions.aspx?faqid=809",
        "scope": "Crack significance depends on the structure and crack characteristics.",
    },
    "aci_562": {
        "title": "ACI 562: assessment, repair, and durability provisions",
        "url": "https://www.concrete.org/Portals/0/Files/PDF/562-16_Provisional.pdf",
        "scope": "Assess crack causes and movement; account for reinforcement corrosion in repair design.",
    },
    "fhwa_bridge": {
        "title": "FHWA bridge concrete spall, delamination, and exposed-steel field protocol",
        "url": "https://www.fhwa.dot.gov/publications/research/infrastructure/structures/ltbp/16007/048.cfm",
        "scope": "For bridge members: inspect suspected areas by sounding, record spall limits/dimensions/location, document exposed reinforcement, and measure section loss where applicable.",
    },
    "fhwa_cracking": {
        "title": "FHWA concrete bridge cracking field protocol",
        "url": "https://www.fhwa.dot.gov/publications/research/infrastructure/structures/ltbp/16007/049.cfm",
        "scope": "For bridge members: locate and photograph cracks, record length/orientation, measure maximum width with a crack gauge, and document moisture or related defects.",
    },
}

ACI_224R_LIMITS = (
    ("Dry air or protective membrane", 0.41, "0.016 in."),
    ("Humidity, moist air, soil", 0.30, "0.012 in."),
    ("Deicing chemicals", 0.18, "0.007 in."),
    ("Seawater and seawater spray, wetting and drying", 0.15, "0.006 in."),
    ("Water-retaining structures (excluding nonpressure pipes)", 0.10, "0.004 in."),
)

LABELS = {
    "crack": {
        "title": "Crack",
        "aliases": ("crack", "cracks", "concrete crack"),
        "sources": ("fhwa_cracking", "aci_cracks", "aci_562"),
        "mapping_action": "At the mapped cells, match the image location to the actual member, mark and photograph the crack ends, record its path and orientation, and measure its widest opening with a crack gauge. Compare later inspections at the same reference points.",
        "action": "Record the crack location, pattern, length, and field-measured width with a scale; check for leakage and change over time. Have an engineer assess cracks on load-bearing regions, cracks with displacement, or cracks associated with spalling or reinforcement corrosion. The image cannot identify cause, depth, activity, or structural significance. Crack sealing/injection is selected only after those conditions are assessed.",
    },
    "fissures": {
        "title": "Fissures / fine crack-like finding",
        "aliases": ("fissure", "fissures", "microcrack", "micro-crack", "micro cracks"),
        "sources": ("fhwa_cracking", "aci_cracks", "aci_562"),
        "mapping_action": "Use the highlighted cells to find the fine crack-like areas in the field; document their distribution, take a scale-referenced close-up, check for moisture or rust staining, and compare repeat observations. The model's fissures label is not a separate engineering severity class.",
        "action": "Treat this as a fine crack-like visual finding, not a separate severity grade. Confirm at close range with a scale, map its distribution, and check moisture and change over time. An engineer should determine the cause before a repair is selected; visual pixel width is not a physical crack width.",
    },
    "spalling": {
        "title": "Concrete spalling",
        "aliases": ("spalling", "spall", "spalls", "concrete spalling"),
        "sources": ("fhwa_bridge",),
        "mapping_action": "At the mapped cells, have a qualified inspector verify the boundaries and dimensions and sound suspected concrete for hollow or delaminated areas. Photograph and record the affected member; keep people clear if material is loose overhead.",
        "action": "Have a qualified inspector map the affected area and check for delamination/unsound concrete. If concrete is loose or overhead, keep people clear and request prompt site inspection. A repair professional must establish the cause and sound-concrete boundaries before specifying a repair; the image map is not a repair design.",
    },
    "rebar": {
        "title": "Exposed / corroded reinforcement",
        "aliases": ("rebar", "reinforcement", "exposed rebar", "corroded rebar", "exposed steel", "steel reinforcement"),
        "sources": ("fhwa_bridge", "aci_562"),
        "mapping_action": "At the mapped cells, have a qualified inspector verify that reinforcement is actually exposed, record its location and exposed length, and assess steel section loss against reliable original dimensions. The image map cannot measure remaining bar area or capacity.",
        "action": "Arrange prompt qualified engineering inspection. Record the exposed-steel location and dimensions, inspect bar condition, and measure section loss against drawings or other reliable original dimensions. A photo cannot establish remaining steel area or capacity. The corrosion mechanism and durability need to be addressed in the repair design; do not infer that a surface coating or crack injection is sufficient.",
    },
}

SEVERITY_GUIDANCE = {
    "crack": {
        "basis": "Crack significance depends on the structure and the nature of the cracking; image area does not measure physical width, movement, cause, or structural role.",
        "field_checks": "Record location, pattern, orientation, and field-measured width; check moisture and compare repeat observations.",
        "escalation": "Have an engineer assess cracks on load-bearing regions, cracks with displacement or change, and cracks associated with spalling or reinforcement corrosion.",
    },
    "fissures": {
        "basis": "The model's fissures label is a fine crack-like finding, not a separate engineering severity class; pixel coverage cannot establish physical width or cause.",
        "field_checks": "Confirm at close range with a scale, map distribution, check moisture or rust staining, and compare repeat observations.",
        "escalation": "Have an engineer assess cause when the finding changes or is associated with moisture, rust staining, displacement, or other distress.",
    },
    "spalling": {
        "basis": "Image pixel area does not establish spall depth, delamination, concrete soundness, or falling hazard.",
        "field_checks": "Have a qualified inspector verify boundaries and dimensions, sound suspected concrete for hollow or delaminated areas, and document the affected member.",
        "escalation": "If material is loose or overhead, keep people clear and request prompt site-specific inspection.",
    },
    "rebar": {
        "basis": "A photo or pixel mask cannot establish bar exposure, corrosion severity, steel section loss, remaining area, or capacity.",
        "field_checks": "Arrange qualified inspection to verify exposure, record location and dimensions, and assess section loss against drawings or other reliable original dimensions.",
        "escalation": "Confirmed exposed or corroded reinforcement warrants prompt qualified engineering inspection; the corrosion mechanism and durability need field assessment.",
    },
}

UNSUPPORTED = ("concrete crushing", "crushing")
_NUMERIC_METRICS = (
    "detection_count", "total_detections", "average_confidence", "maximum_confidence",
    "crack_pixels", "mapping_crack_pixels", "mapping_width_median", "mapping_width_p95",
    "mapping_width_max", "mapping_crack_length", "estimated_width_px",
    "estimated_median_width_px", "contour_count", "total_contour_length_px",
    "hotspot_count", "occupied_cells", "max_priority", "mapping_junctions",
    "junction_count", "endpoints", "orientation_deg", "coverage_percent",
    "mask_coverage_percent", "mapping_coverage_percent", "sampled_frame_count",
    "frames_with_cracks",
)


def _label_key(name: Any) -> str | None:
    normalized = re.sub(r"[^a-z0-9-]+", " ", str(name).lower()).strip()
    entries = [(key, alias) for key, spec in LABELS.items() for alias in spec["aliases"]]
    for key, alias in entries:
        if normalized == alias:
            return key
    for key, alias in sorted(entries, key=lambda entry: len(entry[1]), reverse=True):
        if alias in normalized:
            return key
    return None


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number == number and abs(number) != float("inf"):
        return number
    return None


def normalize_telemetry(value: Any) -> dict[str, Any]:
    """Keep only finite, typed, known evidence fields; never synthesize metrics."""
    if not isinstance(value, dict):
        return {}
    root = value
    source = root.get("telemetry") if isinstance(root.get("telemetry"), dict) else root
    stages = root.get("stages", {}) if isinstance(root.get("stages"), dict) else {}
    detect_stage = stages.get("detection", {}).get("result", {}) if isinstance(stages.get("detection"), dict) else {}
    segmentation_stage = stages.get("segmentation", {}).get("result", {}) if isinstance(stages.get("segmentation"), dict) else {}
    mapping_stage = stages.get("mapping", {}).get("result", {}) if isinstance(stages.get("mapping"), dict) else {}
    multi_mapping_stage = stages.get("multi_defect_mapping", {}).get("result", {}) if isinstance(stages.get("multi_defect_mapping"), dict) else {}
    damage_stage = stages.get("damage_segmentation", {}).get("result", {}) if isinstance(stages.get("damage_segmentation"), dict) else {}
    if not damage_stage and isinstance(stages.get("multi_defect_segmentation"), dict):
        damage_stage = stages["multi_defect_segmentation"].get("result", {})
    metrics = root.get("metrics", {}) if isinstance(root.get("metrics"), dict) else {}
    mapping_metrics = mapping_stage.get("metrics", {}) if isinstance(mapping_stage.get("metrics"), dict) else {}
    if not mapping_metrics and metrics:
        mapping_metrics = metrics
    mm_classes = multi_mapping_stage.get("class_metrics", {}) if isinstance(multi_mapping_stage.get("class_metrics"), dict) else (root.get("class_metrics", {}) if isinstance(root.get("class_metrics"), dict) else {})
    mm_crack = mm_classes.get("crack", {}) if isinstance(mm_classes.get("crack"), dict) else {}
    mm_fiss = mm_classes.get("fissures", {}) if isinstance(mm_classes.get("fissures"), dict) else {}
    mm_width_src = mm_crack if mm_crack.get("width_p95_px") is not None else mm_fiss
    result: dict[str, Any] = {}
    for key in _NUMERIC_METRICS:
        candidate = source.get(key, root.get(key))
        if candidate is None:
            candidate = metrics.get(key)
        if key == "crack_pixels" and candidate is None:
            candidate = segmentation_stage.get("crack_pixels")
            if candidate is None:
                candidate = mapping_metrics.get("crack_pixels", mm_crack.get("pixel_area"))
        if key == "mapping_crack_pixels" and candidate is None:
            candidate = mapping_metrics.get("crack_pixels", mm_crack.get("pixel_area"))
        if key == "coverage_percent" and candidate is None:
            candidate = segmentation_stage.get("mask_coverage_percent")
            if candidate is None:
                ratio = mapping_metrics.get("crack_area_ratio", mapping_metrics.get("area_ratio"))
                candidate = float(ratio) * 100 if ratio is not None else mm_crack.get("coverage_percent")
        if key == "mapping_coverage_percent" and candidate is None:
            ratio = mapping_metrics.get("crack_area_ratio", mapping_metrics.get("area_ratio"))
            candidate = float(ratio) * 100 if ratio is not None else mm_crack.get("coverage_percent")
        if key == "mask_coverage_percent" and candidate is None:
            candidate = segmentation_stage.get("mask_coverage_percent")
        if key == "estimated_median_width_px" and candidate is None:
            candidate = segmentation_stage.get("estimated_median_width_px")
        if key == "contour_count" and candidate is None:
            candidate = segmentation_stage.get("contour_count")
        if key == "total_contour_length_px" and candidate is None:
            candidate = segmentation_stage.get("total_contour_length_px")
        if key == "detection_count" and candidate is None:
            candidate = detect_stage.get("detection_count")
        if key == "average_confidence" and candidate is None:
            candidate = detect_stage.get("average_confidence")
        if key == "maximum_confidence" and candidate is None:
            candidate = detect_stage.get("maximum_confidence")
        if key == "mapping_width_median" and candidate is None:
            candidate = mapping_metrics.get("width_median", mm_width_src.get("width_median_px"))
        if key == "mapping_width_p95" and candidate is None:
            candidate = mapping_metrics.get("width_p95", mm_width_src.get("width_p95_px"))
        if key == "mapping_width_max" and candidate is None:
            candidate = mapping_metrics.get("width_max", mm_width_src.get("width_max_px"))
        if key == "mapping_crack_length" and candidate is None:
            candidate = mapping_metrics.get("crack_length", mm_crack.get("centerline_length_px"))
        if key == "mapping_junctions" and candidate is None:
            candidate = mapping_metrics.get("junction_clusters", mapping_metrics.get("junction_count"))
        if key == "endpoints" and candidate is None:
            candidate = mapping_metrics.get("endpoints")
        if key == "orientation_deg" and candidate is None:
            candidate = mapping_metrics.get("orientation_deg")
        if key == "hotspot_count" and candidate is None:
            candidate = mapping_stage.get("hotspot_count")
            if candidate is None and isinstance(multi_mapping_stage.get("composite_summary"), dict):
                candidate = multi_mapping_stage["composite_summary"].get("hotspot_cells_count")
        if key == "occupied_cells" and candidate is None:
            candidate = mapping_stage.get("occupied_cells")
        if key == "max_priority" and candidate is None:
            candidate = mapping_stage.get("max_priority")
            if candidate is None and isinstance(multi_mapping_stage.get("composite_summary"), dict):
                candidate = multi_mapping_stage["composite_summary"].get("max_exploratory_index")
        number = _finite_float(candidate)
        if number is not None:
            result[key] = number
    for key in ("source_module", "source_type", "inspection_id", "units", "weight_scenario"):
        val = source.get(key, root.get(key, mapping_stage.get(key, multi_mapping_stage.get(key))))
        if isinstance(val, str):
            result[key] = val[:120]
    combined_modules = source.get("combined_modules", root.get("combined_modules"))
    if isinstance(combined_modules, list):
        result["combined_modules"] = [str(m)[:40] for m in combined_modules if isinstance(m, str)]
    physical = source.get("physical", root.get("physical", mapping_stage.get("physical", multi_mapping_stage.get("physical"))))
    if isinstance(physical, dict) and physical:
        safe_physical = {k: _finite_float(v) for k, v in physical.items()}
        safe_physical = {k: v for k, v in safe_physical.items() if v is not None}
        if safe_physical:
            result["physical"] = safe_physical
    calibration = source.get("calibration", root.get("calibration", mapping_stage.get("calibration", multi_mapping_stage.get("calibration"))))
    if isinstance(calibration, dict) and calibration:
        result["calibration"] = {
            k: calibration[k] for k in ("enabled", "mode", "method", "mm_per_px", "scale_x_mm_per_px", "scale_y_mm_per_px")
            if k in calibration
        }
    percentiles = source.get("test_set_percentiles", root.get("test_set_percentiles", mapping_stage.get("test_set_percentiles")))
    if isinstance(percentiles, dict) and percentiles:
        safe_pct = {k: _finite_float(v) for k, v in percentiles.items()}
        safe_pct = {k: v for k, v in safe_pct.items() if v is not None}
        if safe_pct:
            result["test_set_percentiles"] = safe_pct

    counts = source.get("class_counts", source.get("class_detection_counts", root.get("class_counts", root.get("class_detection_counts", detect_stage.get("class_counts", {})))))
    if isinstance(counts, dict):
        normalized_counts: dict[str, int] = {}
        for name, count in counts.items():
            try:
                number = int(count)
                if number >= 0:
                    normalized_counts[str(name)[:80]] = number
            except (ValueError, TypeError):
                continue
        result["class_counts"] = normalized_counts

    classes: dict[str, Any] = {}
    for candidate_classes in (
        source.get("damage_classes"),
        source.get("class_metrics"),
        root.get("damage_classes"),
        root.get("class_metrics"),
        damage_stage.get("class_metrics"),
        multi_mapping_stage.get("class_metrics"),
    ):
        if isinstance(candidate_classes, dict):
            for name, val in candidate_classes.items():
                if isinstance(val, dict) and name not in classes:
                    classes[name] = dict(val)

    crack_source = segmentation_stage or (root if "crack_pixels" in root else {})
    try:
        crack_pixel_area = float(crack_source.get("crack_pixels") or 0) if isinstance(crack_source, dict) else 0
    except (TypeError, ValueError):
        crack_pixel_area = 0
    if isinstance(crack_source, dict) and crack_pixel_area > 0:
        seg_crack = {
            "pixel_area": crack_source.get("crack_pixels"),
            "coverage_percent": crack_source.get("mask_coverage_percent"),
            "component_count": crack_source.get("contour_count"),
            "perimeter_px": crack_source.get("total_contour_length_px"),
        }
        if "crack" not in classes:
            classes["crack"] = seg_crack
        else:
            classes["crack"] = {**seg_crack, **classes["crack"]}
    if mapping_metrics:
        area_ratio = mapping_metrics.get("crack_area_ratio", mapping_metrics.get("area_ratio"))
        crack_length = mapping_metrics.get("crack_length", mapping_metrics.get("centerline_length_px"))
        map_crack: dict[str, Any] = {}
        if mapping_metrics.get("crack_pixels") is not None:
            map_crack["pixel_area"] = mapping_metrics["crack_pixels"]
        if area_ratio is not None:
            map_crack["coverage_percent"] = float(area_ratio) * 100
        if crack_length is not None:
            map_crack["centerline_length_px"] = crack_length
        if mapping_metrics.get("mask_components") is not None:
            map_crack["component_count"] = mapping_metrics["mask_components"]
        if map_crack:
            if "crack" not in classes:
                classes["crack"] = map_crack
            else:
                # Keep segmentation's full-res pixel_area/coverage_percent if present, and enrich with mapping centerline_length_px
                classes["crack"] = {**map_crack, **classes["crack"], "centerline_length_px": map_crack.get("centerline_length_px", classes["crack"].get("centerline_length_px"))}

    if isinstance(classes, dict):
        normalized_classes: dict[str, dict[str, Any]] = {}
        for name, c_metrics in classes.items():
            key = _label_key(name)
            if not key or not isinstance(c_metrics, dict):
                continue
            kept: dict[str, Any] = {}
            for metric_name in (
                "pixel_area", "pixels", "coverage_percent", "component_count",
                "perimeter_px", "perimeter_mm", "area_mm2", "centerline_length_px",
                "centerline_length_mm", "equivalent_diameter_px", "equivalent_diameter_mm",
                "width_median_px", "width_p95_px", "width_max_px",
                "width_median_mm", "width_p95_mm", "width_max_mm",
            ):
                if metric_name in c_metrics:
                    number = _finite_float(c_metrics[metric_name])
                    if number is not None:
                        kept[metric_name] = number
            if isinstance(c_metrics.get("mask_method"), str):
                kept["mask_method"] = c_metrics["mask_method"][:160]
            normalized_classes[key] = kept
        result["damage_classes"] = normalized_classes

    class_mapping = source.get("class_mapping", root.get("class_mapping", {}))
    # Past damage-map inspections already carry per-class matrices or top_cells.
    # Build the same compact VQA evidence from those saved maps so users need not re-upload.
    if not isinstance(class_mapping, dict) or not class_mapping:
        map_root = mapping_stage if isinstance(mapping_stage.get("top_cells"), list) else root
        raw_matrices = multi_mapping_stage.get("matrices") if isinstance(multi_mapping_stage.get("matrices"), dict) else (root.get("matrices") if isinstance(root.get("matrices"), dict) else {})
        raw_metrics = multi_mapping_stage.get("class_metrics") if isinstance(multi_mapping_stage.get("class_metrics"), dict) else (root.get("class_metrics") if isinstance(root.get("class_metrics"), dict) else {})
        inferred_classes: dict[str, dict[str, Any]] = {}
        is_crack_only_topology_map = isinstance(map_root.get("top_cells"), list) and isinstance(map_root.get("metrics"), dict) and not raw_metrics
        if not is_crack_only_topology_map and raw_matrices:
            for label in LABELS:
                raw_matrix = raw_matrices.get(f"{label}_area_ratio", raw_matrices.get(label))
                if not isinstance(raw_matrix, list) or not raw_matrix or not isinstance(raw_matrix[0], list):
                    continue
                rows, cols = len(raw_matrix), len(raw_matrix[0])
                try:
                    occupied = [
                        (r, c, float(raw_matrix[r][c]))
                        for r in range(rows) for c in range(cols)
                        if raw_matrix[r][c] is not None and float(raw_matrix[r][c]) > 0 and float(raw_matrix[r][c]) == float(raw_matrix[r][c])
                    ]
                except (TypeError, ValueError, IndexError):
                    continue
                occupied.sort(key=lambda cell: cell[2], reverse=True)
                m_metrics = raw_metrics.get(label, {}) if isinstance(raw_metrics.get(label), dict) else {}
                inferred_classes[label] = {
                    **{name: m_metrics[name] for name in ("pixel_area", "coverage_percent", "component_count") if name in m_metrics},
                    "mapped_cell_count": len(occupied),
                    "top_cells": [{"row": r + 1, "column": c + 1, "cell_coverage_percent": round(value * 100, 2)} for r, c, value in occupied[:3]],
                    "mask_method": "CRACKTO-SFNet V4" if label == "crack" else str(root.get("model", "saved class map")),
                }
        grid = map_root.get("grid", root.get("grid", [20, 20]))
        if inferred_classes:
            class_mapping = {
                "grid": grid,
                "coordinate_note": "Rows and columns are 1-based from the image top-left; these are image cells, not structure, geographic, or physical coordinates.",
                "mask_provenance": str(root.get("model", "saved damage map")),
                "classes": inferred_classes,
            }
        elif isinstance(map_root.get("top_cells"), list) and isinstance(map_root.get("metrics"), dict):
            # The crack-only topology mapper stores ranked grid cells instead of
            # a per-class ratio matrix.
            crack_metrics = map_root["metrics"]
            coverage = crack_metrics.get("crack_area_ratio", crack_metrics.get("area_ratio"))
            try:
                coverage_percent = float(coverage) * 100 if coverage is not None else 0.0
            except (TypeError, ValueError):
                coverage_percent = 0.0
            crack_px = _finite_float(crack_metrics.get("crack_pixels"))
            comp_count = _finite_float(crack_metrics.get("mask_components"))
            top_cells = []
            for cell in map_root["top_cells"][:5]:
                if not isinstance(cell, dict):
                    continue
                try:
                    top_cells.append({
                        "row": int(cell["grid_row"]) + 1,
                        "column": int(cell["grid_col"]) + 1,
                        "cell_coverage_percent": max(0.0, float(cell.get("crack_area_ratio", 0))) * 100,
                        "priority_index": _finite_float(cell.get("priority_index")),
                        "width_p95": _finite_float(cell.get("width_p95")),
                        "junction_count": int(cell.get("junction_count") or 0),
                    })
                except (KeyError, TypeError, ValueError):
                    continue
            crack_map_entry: dict[str, Any] = {
                "coverage_percent": max(0.0, coverage_percent),
                "top_cells": top_cells,
                "mapped_cell_count": int(map_root.get("occupied_cells", len(top_cells))),
                "mask_method": str(map_root.get("model", "CRACKTO-SFNet V4 crack map")),
            }
            if crack_px is not None:
                crack_map_entry["pixel_area"] = int(crack_px)
            if comp_count is not None:
                crack_map_entry["component_count"] = int(comp_count)
            class_mapping = {
                "grid": map_root.get("grid", [20, 20]),
                "coordinate_note": "Rows and columns are 1-based from the image top-left; these are image cells, not structure, geographic, or physical coordinates.",
                "mask_provenance": str(map_root.get("model", "saved crack map")),
                "classes": {"crack": crack_map_entry},
            }
    if isinstance(class_mapping, dict):
        try:
            grid = class_mapping.get("grid", [20, 20])
            row_count, col_count = int(grid[0]), int(grid[1])
            if not (1 <= row_count <= 100 and 1 <= col_count <= 100):
                raise ValueError("invalid mapping grid")
            mapped_classes = class_mapping.get("classes", {})
            safe_classes: dict[str, dict[str, Any]] = {}
            if isinstance(mapped_classes, dict):
                for name, values in mapped_classes.items():
                    key = _label_key(name)
                    if not key or not isinstance(values, dict):
                        continue
                    safe: dict[str, Any] = {"grid": [row_count, col_count]}
                    for field in ("pixel_area", "coverage_percent", "component_count", "mapped_cell_count", "detector_count", "matched_mask_count"):
                        try:
                            number = float(values[field])
                            if number == number and abs(number) != float("inf") and number >= 0:
                                safe[field] = int(number) if field != "coverage_percent" else number
                        except (KeyError, TypeError, ValueError):
                            continue
                    safe["top_cells"] = []
                    for cell in values.get("top_cells", [])[:5] if isinstance(values.get("top_cells", []), list) else []:
                        if not isinstance(cell, dict):
                            continue
                        try:
                            cell_row, cell_col = int(cell["row"]), int(cell["column"])
                            coverage = float(cell["cell_coverage_percent"])
                            if 1 <= cell_row <= row_count and 1 <= cell_col <= col_count and coverage >= 0 and coverage == coverage and abs(coverage) != float("inf"):
                                cell_obj: dict[str, Any] = {"row": cell_row, "column": cell_col, "cell_coverage_percent": coverage}
                                if _finite_float(cell.get("priority_index")) is not None:
                                    cell_obj["priority_index"] = float(cell["priority_index"])
                                if _finite_float(cell.get("width_p95")) is not None:
                                    cell_obj["width_p95"] = float(cell["width_p95"])
                                if cell.get("junction_count") is not None:
                                    cell_obj["junction_count"] = int(cell["junction_count"])
                                safe["top_cells"].append(cell_obj)
                        except (KeyError, TypeError, ValueError):
                            continue
                    for field in ("mask_method",):
                        if isinstance(values.get(field), str):
                            safe[field] = values[field][:160]
                    safe_classes[key] = safe
            result["class_mapping"] = {
                "grid": [row_count, col_count],
                "coordinate_note": str(class_mapping.get("coordinate_note", "Image coordinates only; not physical or geographic coordinates."))[:240],
                "mask_provenance": str(class_mapping.get("mask_provenance", ""))[:240],
                "classes": safe_classes,
            }
        except (TypeError, ValueError, IndexError):
            result["class_mapping"] = {"grid": [20, 20], "classes": {}}
    return result


def present_labels(telemetry: dict[str, Any]) -> list[str]:
    found: set[str] = set()
    for name, count in telemetry.get("class_counts", {}).items():
        key = _label_key(name)
        if key and count > 0:
            found.add(key)
    for name, metrics in telemetry.get("damage_classes", {}).items():
        key = _label_key(name)
        area = metrics.get("pixel_area", metrics.get("pixels", 0)) if isinstance(metrics, dict) else 0
        other_signal = any(float(metrics.get(field, 0) or 0) > 0 for field in ("coverage_percent", "centerline_length_px")) if isinstance(metrics, dict) else False
        if key and (area > 0 or other_signal):
            found.add(key)
    return [key for key in LABELS if key in found]


def evidence_items(telemetry: dict[str, Any], vision_labels: list[str] | None = None) -> list[dict[str, Any]]:
    """Return label evidence with provenance and source-linked follow-up."""
    labels = list(dict.fromkeys([*present_labels(telemetry), *(key for key in (vision_labels or []) if key in LABELS)]))
    validation = telemetry.get("class_validation") or class_validation_summary()
    result = []
    for key in labels:
        metrics = telemetry.get("damage_classes", {}).get(key, {})
        detector_count = 0
        for name, count in telemetry.get("class_counts", {}).items():
            if _label_key(name) == key:
                detector_count += count
        evidence = []
        if metrics:
            mask_method = metrics.get("mask_method", "CRACKTO-SFNet V4")
            evidence.append({
                "type": "segmentation" if "CRACKTO-SFNet" in mask_method else "detector_guided_mask",
                "method": mask_method,
                "measurements": metrics,
                "units": "analysis pixels; not physical dimensions",
            })
        if detector_count:
            evidence.append({"type": "detector", "count": detector_count, "meaning": "model-reported boxes; not affected area or severity"})
        mapping = telemetry.get("class_mapping", {}).get("classes", {}).get(key, {}) if isinstance(telemetry.get("class_mapping"), dict) else {}
        pixel_extent = _pixel_extent(metrics, mapping)
        if mapping:
            evidence.append({
                "type": "class_mapping",
                "grid": telemetry.get("class_mapping", {}).get("grid", mapping.get("grid", [20, 20])),
                "units": "analysis pixels; image grid cells are not site or geographic coordinates",
                "mapped_pixels": mapping.get("pixel_area", 0),
                "coverage_percent": mapping.get("coverage_percent", 0),
                "mapped_cell_count": mapping.get("mapped_cell_count", 0),
                "top_cells": mapping.get("top_cells", []),
                "method": mapping.get("mask_method", ""),
            })
        if key in (vision_labels or []):
            evidence.append({"type": "visual_screening", "meaning": "VLM visual observation; confirm against the selected inspection model and field conditions"})
        result.append({
            "label": key,
            "title": LABELS[key]["title"],
            "evidence": evidence,
            "pixel_extent": pixel_extent,
            "severity_guidance": _severity_guidance(key),
            "follow_up": LABELS[key]["action"],
            "evidence_based_solution": LABELS[key]["action"],
            "mapping_suggestion": _mapping_suggestion(key, mapping, detector_count),
            "mapping_evidence": mapping,
            "sources": [SOURCES[s] for s in LABELS[key]["sources"]],
            "mapping_validation": validation.get("classes", {}).get(key),
        })
    result.sort(
        key=lambda item: (
            item["pixel_extent"]["coverage_percent"] is None,
            -(item["pixel_extent"]["coverage_percent"] or 0),
        )
    )
    return result


def _pixel_extent(metrics: dict[str, Any], mapping: dict[str, Any]) -> dict[str, float | int | str | None]:
    extent: dict[str, float | int | str | None] = {}
    for output, candidates in (
        ("pixel_area", ((mapping, "pixel_area"), (metrics, "pixel_area"), (metrics, "pixels"))),
        ("coverage_percent", ((mapping, "coverage_percent"), (metrics, "coverage_percent"))),
    ):
        value = next((source[key] for source, key in candidates if key in source and source[key] is not None), None)
        try:
            number = float(value)
            if number >= 0 and number == number and abs(number) != float("inf"):
                extent[output] = int(number) if output == "pixel_area" and number.is_integer() else number
                continue
        except (TypeError, ValueError):
            pass
        extent[output] = None
    return extent


def _format_pixel_extent(extent: dict[str, Any]) -> str:
    area = extent.get("pixel_area")
    coverage = extent.get("coverage_percent")
    area_text = f"{area:g} analysis px²" if area is not None else "unavailable"
    coverage_text = f"{coverage:g}%" if coverage is not None else "unavailable"
    return f"{area_text}; {coverage_text} of image"


def _severity_guidance(label: str) -> dict[str, Any]:
    guidance = SEVERITY_GUIDANCE[label]
    return {
        "status": "not_assessable_from_image",
        **guidance,
        "sources": [SOURCES[source] for source in LABELS[label]["sources"]],
    }


def _mapping_suggestion(label: str, mapping: dict[str, Any], detector_count: int) -> str:
    """Turn mapped image cells into sourced field checks, never a severity score."""
    area = mapping.get("pixel_area", 0) if isinstance(mapping, dict) else 0
    cells = mapping.get("top_cells", []) if isinstance(mapping, dict) else []
    note = "Image-grid rows and columns are only a way to find regions in this photo; transfer them to the actual member using field references. Map pixels are not physical dimensions or a severity grade."
    has_mapped_support = area > 0 or float(mapping.get("coverage_percent", 0) or 0) > 0 or int(mapping.get("mapped_cell_count", 0) or 0) > 0
    if has_mapped_support and cells:
        locations = ", ".join(
            f"row {cell['row']}, column {cell['column']} ({cell['cell_coverage_percent']:g}% of that cell)"
            for cell in cells[:3]
        )
        provenance = ""
        if label != "crack":
            provenance = " For this non-crack label, the mask is a class-agnostic FastSAM proposal assigned from a detector box, not a class-trained mask; confirm it directly."
        return f"Start field verification at the mapped image cells: {locations}. {LABELS[label]['mapping_action']} {note}{provenance}"
    if detector_count > 0:
        return f"The detector reported {detector_count} {LABELS[label]['title'].lower()} box(es), but the class map contains no mapped pixels. Check the annotated image and the real member directly; an empty map is not evidence that the defect is absent. {LABELS[label]['mapping_action']} {note}"
    return f"No class-specific mapped pixels are available for this visual-only finding. Verify the location in person before treating it as a model-mapped defect. {LABELS[label]['mapping_action']} {note}"


def _class_validation_section(validation: dict[str, Any]) -> str:
    lines = [
        "### Class-wise mapping validation",
        "These are external benchmark results, not validation of the uploaded image. A detector label or mask on this image does not inherit benchmark accuracy.",
    ]
    classes = validation.get("classes", {})
    for label in CLASS_ORDER:
        entry = classes.get(label, {})
        title = LABELS[label]["title"]
        status = str(entry.get("status", "validation unavailable")).replace("_", " ")
        summary = entry.get("summary", "No external class-matched validation result is available.")
        lines.append(f"- **{title} — {status}:** {summary}")
    if validation.get("dataset_url"):
        source = validation.get("dataset", "External validation dataset")
        lines.append(f"Validation dataset: [{source}]({validation['dataset_url']}).")
    return "\n".join(lines)


def evidence_based_solutions() -> list[dict[str, Any]]:
    return [
        {
            "label": label,
            "title": LABELS[label]["title"],
            "solution": LABELS[label]["action"],
            "sources": [SOURCES[source] for source in LABELS[label]["sources"]],
        }
        for label in CLASS_ORDER
    ]


def _class_guidance_section() -> str:
    lines = [
        "### Evidence-based field actions for supported labels",
        "These are general next steps for confirmed field conditions, not findings for this image or repair prescriptions. Apply only when the condition is observed; have a qualified professional determine cause and repair design.",
    ]
    for item in evidence_based_solutions():
        sources = "; ".join(f"[{source['title']}]({source['url']})" for source in item["sources"])
        lines.append(f"- **{item['title']}:** {item['solution']} **Evidence:** {sources}.")
    return "\n".join(lines)


def _severity_guidance_section() -> str:
    lines = [
        "### Class-specific severity checks",
        "Structural severity is not rated from image pixels. Use the field checks and escalation criteria below; pixel area is only image-space extent.",
    ]
    for label in CLASS_ORDER:
        guidance = SEVERITY_GUIDANCE[label]
        lines.append(
            f"- **{LABELS[label]['title']}:** {guidance['basis']} "
            f"**Field checks:** {guidance['field_checks']} "
            f"**Escalate:** {guidance['escalation']}"
        )
    return "\n".join(lines)


def label_risks(telemetry: dict[str, Any]) -> list[dict[str, str]]:
    """Return per-label ACI risk ratings (CRITICAL / MODERATE / LOW) and timelines."""
    labels = present_labels(telemetry)
    physical = telemetry.get("physical") or {}
    w_p95_mm = _finite_float(physical.get("width_p95_mm"))
    w_med_mm = _finite_float(physical.get("width_median_mm"))
    ref_mm = w_p95_mm if w_p95_mm is not None else w_med_mm
    hotspots = _finite_float(telemetry.get("hotspot_count")) or 0.0
    junctions = _finite_float(telemetry.get("mapping_junctions")) or 0.0
    max_prio = _finite_float(telemetry.get("max_priority")) or 0.0
    w_p95_px = _finite_float(telemetry.get("mapping_width_p95")) or 0.0

    items: list[dict[str, str]] = []
    for key in labels:
        title = LABELS[key]["title"]
        if key == "rebar":
            items.append({
                "label": key, "title": title, "risk": "CRITICAL", "color": "#dc2626",
                "timeline": "Take care now",
                "reason": "Exposed/corroded rebar — immediate engineering inspection & section-loss check (ACI 562).",
            })
        elif key == "spalling":
            items.append({
                "label": key, "title": title, "risk": "CRITICAL", "color": "#dc2626",
                "timeline": "Take care now",
                "reason": "Concrete spalling — check for loose/overhead concrete & delamination today (ACI 562 / FHWA).",
            })
        elif key == "crack":
            if ref_mm is not None and ref_mm > 0.41:
                items.append({
                    "label": key, "title": title, "risk": "CRITICAL", "color": "#dc2626",
                    "timeline": "Take care now",
                    "reason": f"Calibrated width ({ref_mm:.2f} mm) exceeds ACI 224R-01 max 0.41 mm limit.",
                })
            elif (ref_mm is not None and ref_mm > 0.15) or hotspots >= 5 or junctions >= 3 or max_prio >= 0.50 or w_p95_px >= 6.0:
                reason = (
                    f"Calibrated width ({ref_mm:.2f} mm) exceeds ACI 224R-01 severe-exposure limit (0.15 mm)."
                    if ref_mm is not None
                    else "Concentrated crack network/hotspots in mapping — check physical width (mm) & moisture."
                )
                items.append({
                    "label": key, "title": title, "risk": "MODERATE", "color": "#d97706",
                    "timeline": "Take care tomorrow (1–3 days)",
                    "reason": reason,
                })
            else:
                items.append({
                    "label": key, "title": title, "risk": "LOW", "color": "#16a34a",
                    "timeline": "Routine monitoring",
                    "reason": "Minor/within ACI 224R-01 limits; monitor at next scheduled inspection.",
                })
        elif key == "fissures":
            fiss_dm = (telemetry.get("damage_classes") or {}).get("fissures") or {}
            fiss_mm = _finite_float(fiss_dm.get("width_p95_mm")) or _finite_float(physical.get("fissures_width_p95_mm"))
            if fiss_mm is None and "crack" not in labels:
                fiss_mm = ref_mm
            if fiss_mm is not None and fiss_mm > 0.41:
                items.append({
                    "label": key, "title": title, "risk": "CRITICAL", "color": "#dc2626",
                    "timeline": "Take care now",
                    "reason": f"Calibrated fissure width ({fiss_mm:.2f} mm) exceeds ACI 224R-01 max 0.41 mm limit.",
                })
            elif fiss_mm is not None and fiss_mm > 0.30:
                items.append({
                    "label": key, "title": title, "risk": "MODERATE", "color": "#d97706",
                    "timeline": "Take care tomorrow (1–3 days)",
                    "reason": f"Calibrated width ({fiss_mm:.2f} mm) exceeds 0.30 mm moist-exposure limit (ACI 224R-01).",
                })
            else:
                items.append({
                    "label": key, "title": title, "risk": "LOW", "color": "#16a34a",
                    "timeline": "Routine monitoring",
                    "reason": "Fine crack-like finding; confirm with scale at next routine check (ACI 562).",
                })
    rank = {"CRITICAL": 0, "MODERATE": 1, "LOW": 2}
    items.sort(key=lambda item: rank.get(item["risk"], 3))
    return items


def triage(telemetry: dict[str, Any], image_assessed: bool = False) -> dict[str, Any]:
    labels = present_labels(telemetry)
    physical = telemetry.get("physical") or {}
    w_p95_mm = _finite_float(physical.get("width_p95_mm"))
    w_med_mm = _finite_float(physical.get("width_median_mm"))
    ref_mm = w_p95_mm if w_p95_mm is not None else w_med_mm
    hotspots = _finite_float(telemetry.get("hotspot_count")) or 0.0
    junctions = _finite_float(telemetry.get("mapping_junctions")) or 0.0
    max_prio = _finite_float(telemetry.get("max_priority")) or 0.0
    w_p95_px = _finite_float(telemetry.get("mapping_width_p95")) or 0.0
    risks = label_risks(telemetry)

    # Class presence & ACI 224R-01 / ACI 562 triage (never raw pixel area alone).
    if "rebar" in labels or "spalling" in labels:
        basis = "Exposed/corroded reinforcement or concrete spalling flagged (ACI 562 / FHWA urgent inspection)."
        return {
            "severity": "Prompt professional inspection",
            "urgency": "Arrange a qualified site inspection; isolate visibly loose overhead material.",
            "severity_color": "#dc2626",
            "aci_level": "CRITICAL",
            "aci_badge_color": "#dc2626",
            "action_timeline": "Take care now — immediate inspection & safety check today.",
            "aci_basis": basis,
            "label_risks": risks,
        }
    if ref_mm is not None and ref_mm > 0.41:
        return {
            "severity": "Field verification required",
            "urgency": "Document and confirm dimensions, movement, and context on site.",
            "severity_color": "#dc2626",
            "aci_level": "CRITICAL",
            "aci_badge_color": "#dc2626",
            "action_timeline": "Take care now — calibrated width exceeds ACI 224R-01 max 0.41 mm limit.",
            "aci_basis": f"Calibrated width ({ref_mm:.2f} mm) exceeds ACI 224R-01 maximum dry-air threshold (0.41 mm).",
            "label_risks": risks,
        }
    if labels:
        if ref_mm is not None:
            if ref_mm > 0.15:
                return {
                    "severity": "Field verification required",
                    "urgency": "Document and confirm dimensions, movement, and context on site.",
                    "severity_color": "#d97706",
                    "aci_level": "MODERATE",
                    "aci_badge_color": "#d97706",
                    "action_timeline": "Take care tomorrow / within 1–3 days — verify width & moisture (ACI 224R).",
                    "aci_basis": f"Calibrated width ({ref_mm:.2f} mm) exceeds ACI 224R-01 0.15 mm limit (within 0.41 mm).",
                    "label_risks": risks,
                }
            return {
                "severity": "Field verification required",
                "urgency": "Document and confirm dimensions, movement, and context on site.",
                "severity_color": "#16a34a",
                "aci_level": "LOW",
                "aci_badge_color": "#16a34a",
                "action_timeline": "Low priority — monitor at next scheduled inspection.",
                "aci_basis": f"Calibrated width ({ref_mm:.2f} mm) is within ACI 224R-01 thresholds (<= 0.15 mm).",
                "label_risks": risks,
            }
        if hotspots >= 5 or junctions >= 3 or max_prio >= 0.50 or w_p95_px >= 6.0:
            return {
                "severity": "Field verification required",
                "urgency": "Document and confirm dimensions, movement, and context on site.",
                "severity_color": "#d97706",
                "aci_level": "MODERATE",
                "aci_badge_color": "#d97706",
                "action_timeline": "Take care tomorrow / within 1–3 days — verify physical width with crack gauge.",
                "aci_basis": "Concentrated mapping hotspots/branching detected; confirm physical mm width against ACI 224R-01.",
                "label_risks": risks,
            }
        return {
            "severity": "Field verification required",
            "urgency": "Document and confirm dimensions, movement, and context on site.",
            "severity_color": "#16a34a",
            "aci_level": "LOW",
            "aci_badge_color": "#16a34a",
            "action_timeline": "Low priority — monitor at next scheduled inspection.",
            "aci_basis": "Isolated fine crack/fissure with low concentration; confirm during routine inspection.",
            "label_risks": risks,
        }
    if image_assessed:
        return {
            "severity": "Visual screening only",
            "urgency": "VLM observations are not saved model labels; confirm field conditions. A negative screen does not establish defect-free condition.",
            "severity_color": "#64748b",
            "aci_level": "LOW",
            "aci_badge_color": "#64748b",
            "action_timeline": "Visual screen only — verify on site if needed.",
            "aci_basis": "No CrackVision model defect labels recorded.",
            "label_risks": [],
        }
    return {
        "severity": "Insufficient image evidence",
        "urgency": "No matching model labels were present in the saved telemetry.",
        "severity_color": "#64748b",
        "aci_level": "INSUFFICIENT",
        "aci_badge_color": "#64748b",
        "action_timeline": "Run detection, segmentation, or mapping first.",
        "aci_basis": "No matching model labels present in telemetry.",
        "label_risks": [],
    }


def _format_measurements(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "No class-specific segmentation measurements are attached."
    names = (("pixel_area", "area (analysis px²)"), ("pixels", "area (analysis px²)"), ("coverage_percent", "coverage (%)"), ("component_count", "components"), ("perimeter_px", "perimeter (analysis px)"), ("centerline_length_px", "centerline (analysis px)"))
    values, seen = [], set()
    for key, label in names:
        if key in metrics and label not in seen:
            values.append(f"{label}: {metrics[key]:g}")
            seen.add(label)
    measurements = "; ".join(values) or "No class-specific segmentation measurements are attached."
    method = metrics.get("mask_method")
    if isinstance(method, str) and "FastSAM" in method:
        return f"Class-agnostic FastSAM proposal matched to a YOLO box (the class label comes from YOLO); {measurements}. This mask area is not a validated defect measurement."
    return measurements


def _knowledge_excerpts(question: str) -> list[dict[str, str]]:
    try:
        from .vqa import _retrieve_knowledge
        excerpts, _ = _retrieve_knowledge(question, limit=3)
        return excerpts
    except Exception:
        return []


def _question_facets(question: str) -> set[str]:
    text = question.lower()
    facets: set[str] = set()
    if re.search(r"\b(aci|224|562|threshold|limit|limits|width|mm|wide|allowable|tolerable|permissible|acceptable|standard|code|exceed)\b", text):
        facets.add("threshold_aci")
    if re.search(r"\b(repair|fix|inject|injection|epoxy|seal|grout|patch|rehab|remediate|treat|should i|action|recommend|solution|do we)\b", text):
        facets.add("repair")
    if re.search(r"\b(sever|safe|safety|danger|urgent|urgency|risk|priority|critical|moderate|low|when|now|tomorrow|today|timeline|serious|condition)\b", text):
        facets.add("severity")
    if re.search(r"\b(where|hotspot|hotspots|cell|cells|grid|map|mapped|region|regions|location|locate|concentrat|junction|branch|topology|orientation|angle|first)\b", text):
        facets.add("mapping")
    if re.search(r"\b(measur|pixel|pixels|coverage|area|length|count|how many|how much|detect|segment|yolo|v4|summary|summarize|combine|telemetry|result|what did|found|find|visible|show|all)\b", text):
        facets.add("telemetry")
    if re.search(r"\b(why|cause|mechanism|shrinkage|shear|flexur|corrosion|depth|active|dormant)\b", text):
        facets.add("cause")
    if not facets:
        facets.add("telemetry")
    return facets


def _combined_module_summary(telemetry: dict[str, Any]) -> list[str]:
    bullets: list[str] = []
    class_counts = telemetry.get("class_counts") or {}
    det_count = telemetry.get("detection_count")
    if det_count is not None or class_counts:
        count_val = int(det_count) if det_count is not None else sum(int(v or 0) for v in class_counts.values())
        cls_text = ", ".join(f"{name}: {cnt}" for name, cnt in class_counts.items()) if class_counts else "none"
        conf_parts = []
        if telemetry.get("maximum_confidence") is not None:
            conf_parts.append(f"max confidence {telemetry['maximum_confidence'] * 100:.1f}%")
        if telemetry.get("average_confidence") is not None:
            conf_parts.append(f"avg {telemetry['average_confidence'] * 100:.1f}%")
        conf_str = f" ({', '.join(conf_parts)})" if conf_parts else ""
        bullets.append(f"- **Detection (YOLO):** {count_val} box(es) [{cls_text}]{conf_str}.")

    seg_px = telemetry.get("crack_pixels")
    seg_cov = telemetry.get("mask_coverage_percent", telemetry.get("coverage_percent"))
    seg_w = telemetry.get("estimated_median_width_px")
    seg_len = telemetry.get("total_contour_length_px")
    seg_cnt = telemetry.get("contour_count")
    if seg_cov is not None or seg_px is not None or seg_w is not None:
        seg_parts = []
        if seg_cov is not None:
            seg_parts.append(f"coverage {seg_cov:.3g}%")
        if seg_px is not None:
            seg_parts.append(f"{int(round(seg_px)):,} px²")
        if seg_cnt is not None:
            seg_parts.append(f"{int(round(seg_cnt))} contour(s)")
        if seg_len is not None:
            seg_parts.append(f"length {seg_len:.1f} px")
        if seg_w is not None:
            seg_parts.append(f"median width {seg_w:.2f} px")
        bullets.append(f"- **Segmentation (V4):** {', '.join(seg_parts)}.")

    non_crack_lines = []
    for key in ("spalling", "rebar", "fissures"):
        dm = (telemetry.get("damage_classes") or {}).get(key)
        if dm:
            area = dm.get("pixel_area", dm.get("pixels"))
            cov = dm.get("coverage_percent")
            comp = dm.get("component_count")
            parts = []
            if cov is not None:
                parts.append(f"{cov:.3g}%")
            if area is not None:
                parts.append(f"{int(round(area)):,} px²")
            if comp is not None:
                parts.append(f"{int(round(comp))} component(s)")
            if parts:
                non_crack_lines.append(f"{key}: {', '.join(parts)}")
    if non_crack_lines:
        bullets.append(f"- **Non-crack Segmentation (YOLO + FastSAM):** {'; '.join(non_crack_lines)}.")

    map_len = telemetry.get("mapping_crack_length")
    map_w_med = telemetry.get("mapping_width_median")
    map_w_p95 = telemetry.get("mapping_width_p95")
    map_w_max = telemetry.get("mapping_width_max")
    hotspots = telemetry.get("hotspot_count")
    occupied = telemetry.get("occupied_cells")
    max_prio = telemetry.get("max_priority")
    junc = telemetry.get("mapping_junctions")
    orient = telemetry.get("orientation_deg")
    physical = telemetry.get("physical") or {}
    if any(v is not None for v in (map_len, map_w_p95, hotspots, max_prio, occupied)):
        map_parts = []
        if occupied is not None:
            map_parts.append(f"{int(round(occupied))} occupied cells (20×20)")
        if hotspots is not None:
            map_parts.append(f"{int(round(hotspots))} hotspot(s)")
        if max_prio is not None:
            map_parts.append(f"priority {max_prio:.2f}")
        if map_len is not None:
            map_parts.append(f"skeleton {map_len:.1f} px")
        if map_w_med is not None or map_w_p95 is not None or map_w_max is not None:
            w_bits = []
            if map_w_med is not None:
                w_bits.append(f"median {map_w_med:.2f} px")
            if map_w_p95 is not None:
                w_bits.append(f"P95 {map_w_p95:.2f} px")
            if map_w_max is not None:
                w_bits.append(f"max {map_w_max:.2f} px")
            map_parts.append(f"width ({', '.join(w_bits)})")
        if junc is not None:
            map_parts.append(f"{int(round(junc))} junction(s)")
        if orient is not None:
            map_parts.append(f"angle {orient:.1f}°")
        bullets.append(f"- **Damage Mapping:** {', '.join(map_parts)}.")
    if physical:
        phys_bits = []
        if physical.get("width_median_mm") is not None:
            phys_bits.append(f"crack median width {physical['width_median_mm']:.2f} mm")
        if physical.get("width_p95_mm") is not None:
            phys_bits.append(f"P95 {physical['width_p95_mm']:.2f} mm")
        if physical.get("width_max_mm") is not None:
            phys_bits.append(f"max {physical['width_max_mm']:.2f} mm")
        if physical.get("crack_length_mm") is not None:
            phys_bits.append(f"crack length {physical['crack_length_mm']:.1f} mm")
        if physical.get("crack_area_mm2") is not None:
            phys_bits.append(f"crack area {physical['crack_area_mm2']:.1f} mm²")
        if physical.get("spalling_area_mm2") is not None:
            sp_txt = f"spalling {physical['spalling_area_mm2']:.1f} mm²"
            if physical.get("spalling_equivalent_diameter_mm") is not None:
                sp_txt += f" (dia {physical['spalling_equivalent_diameter_mm']:.1f} mm)"
            phys_bits.append(sp_txt)
        if physical.get("rebar_length_mm") is not None or physical.get("rebar_area_mm2") is not None:
            rb_parts = []
            if physical.get("rebar_length_mm") is not None:
                rb_parts.append(f"length {physical['rebar_length_mm']:.1f} mm")
            if physical.get("rebar_area_mm2") is not None:
                rb_parts.append(f"area {physical['rebar_area_mm2']:.1f} mm²")
            phys_bits.append(f"rebar ({', '.join(rb_parts)})")
        if physical.get("fissures_width_p95_mm") is not None:
            phys_bits.append(f"fissures P95 {physical['fissures_width_p95_mm']:.2f} mm")
        if phys_bits:
            bullets.append(f"- **Calibrated Physical Values:** {', '.join(phys_bits)}.")
    return bullets


def _aci_threshold_response(telemetry: dict[str, Any], structure_type: str, badge: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    physical = telemetry.get("physical") or {}
    w_p95_mm = physical.get("width_p95_mm")
    w_med_mm = physical.get("width_median_mm")
    w_max_mm = physical.get("width_max_mm")
    seg_w_px = telemetry.get("estimated_median_width_px")
    map_w_med = telemetry.get("mapping_width_median")
    map_w_p95 = telemetry.get("mapping_width_p95")
    map_w_max = telemetry.get("mapping_width_max")

    aci_level = badge.get("aci_level", "MODERATE")
    action_timeline = badge.get("action_timeline", "")

    if w_p95_mm is not None or w_med_mm is not None:
        ref_mm = w_p95_mm if w_p95_mm is not None else w_med_mm
        lines.append(
            f"**ACI 224R-01 Result ({structure_type}): {aci_level}** — {action_timeline}\n"
            f"- **Measured (calibrated):** median **{w_med_mm:.2f} mm**, P95 **{w_p95_mm:.2f} mm**, max **{w_max_mm:.2f} mm**"
            if all(v is not None for v in (w_med_mm, w_p95_mm, w_max_mm))
            else f"**ACI 224R-01 Result ({structure_type}): {aci_level}** — {action_timeline}\n- **Measured (calibrated):** **{ref_mm:.2f} mm**"
        )
        for exposure, limit_mm, _ in ACI_224R_LIMITS:
            status = "EXCEEDS" if ref_mm > limit_mm else "within limit"
            lines.append(f"- **{exposure} ({limit_mm:.2f} mm):** {status}")
    else:
        px_summary = []
        if seg_w_px is not None:
            px_summary.append(f"Segmentation median {seg_w_px:.2f} px")
        if map_w_med is not None:
            px_summary.append(f"Mapping median {map_w_med:.2f} px")
        if map_w_p95 is not None:
            px_summary.append(f"P95 {map_w_p95:.2f} px")
        if map_w_max is not None:
            px_summary.append(f"max {map_w_max:.2f} px")
        measured_str = ", ".join(px_summary) if px_summary else "No width recorded"
        lines.append(
            f"**ACI 224R-01 Width Check ({structure_type}): {aci_level}** — {action_timeline}\n"
            f"- **Your measured width (pixels):** {measured_str}\n"
            "- **ACI 224R-01 limits (mm):** Dry air `0.41 mm` | Moist/soil `0.30 mm` | Deicing `0.18 mm` | Seawater `0.15 mm` | Water-retaining `0.10 mm`\n"
            "- **Note:** Pixels cannot be compared directly to `mm` — verify physically with a crack gauge or enable camera calibration in Damage Mapping."
        )
    return lines


def _direct_question_section(question: str, telemetry: dict[str, Any], structure_type: str, findings: list[dict[str, Any]], badge: dict[str, Any], excerpts: list[dict[str, str]]) -> str:
    facets = _question_facets(question)
    labels = present_labels(telemetry)
    module_bullets = _combined_module_summary(telemetry)
    lines: list[str] = []

    if not labels and not module_bullets:
        return (
            f"No detection, segmentation, or mapping results found for **{structure_type}**. "
            "Run **Defect detection**, **Segmentation lab**, or **Damage mapping** first."
        )

    aci_level = badge.get("aci_level", "MODERATE")
    action_timeline = badge.get("action_timeline", "")
    aci_basis = badge.get("aci_basis", "")
    risks = badge.get("label_risks") or []

    def risk_lines() -> str:
        if not risks:
            return ""
        return "\n".join(f"- **{r['title']}:** **{r['risk']}** ({r['timeline']}) — {r['reason']}" for r in risks)

    if "threshold_aci" in facets:
        lines.extend(_aci_threshold_response(telemetry, structure_type, badge))

    if "severity" in facets:
        lines.append(
            f"**Overall ACI Severity: {aci_level}** — {action_timeline}\n"
            f"- **ACI Basis:** {aci_basis}"
        )
        if risks:
            lines.append("**Per-Label Risk (ACI 224R / 562):**\n" + risk_lines())

    if "repair" in facets or "cause" in facets:
        repair_bullets = []
        if "rebar" in labels or "spalling" in labels:
            repair_bullets.append(
                "- **Spalling / Exposed Rebar (CRITICAL — Take care now):** Sound for delamination, remove unsound concrete, inspect steel section loss, and patch per **ACI 562** (surface sealing alone is insufficient)."
            )
        if "crack" in labels or "fissures" in labels:
            repair_bullets.append(
                "- **Cracks / Fissures (ACI 224.1R-07):** Dormant structural cracks -> **epoxy injection**; active/moving or leaking cracks -> **flexible sealant / polyurethane grout**; fine non-moving cracks -> surface sealer only if needed."
            )
        if repair_bullets:
            lines.append(f"**Recommended Action ({structure_type} — {aci_level}, {action_timeline}):**\n" + "\n".join(repair_bullets))

    if "mapping" in facets:
        mapped_classes = (telemetry.get("class_mapping") or {}).get("classes") or {}
        hotspot_lines = []
        for cls_key in ("rebar", "spalling", "crack", "fissures"):
            top_cells = (mapped_classes.get(cls_key) or {}).get("top_cells") or []
            if top_cells:
                cells_str = ", ".join(
                    f"row {c['row']} col {c['column']} ({c['cell_coverage_percent']:g}%"
                    + (f", idx {c['priority_index']:.2f}" if c.get("priority_index") is not None else "")
                    + ")"
                    for c in top_cells[:3]
                )
                hotspot_lines.append(f"- **{LABELS[cls_key]['title']}:** {cells_str}")
        if hotspot_lines:
            lines.append("**Top 20×20 Mapped Hotspots:**\n" + "\n".join(hotspot_lines))
        else:
            lines.append("**Mapped Hotspots:** None recorded — run **Damage mapping** for this image.")

    if "telemetry" in facets or not lines:
        head = f"**ACI Severity: {aci_level}** — {action_timeline}"
        if risks:
            head += "\n**Label Risk:** " + " | ".join(f"{r['title']}: **{r['risk']}**" for r in risks)
        lines.insert(0, head)
        if module_bullets:
            lines.append(f"**Combined Module Results ({structure_type}):**\n" + "\n".join(module_bullets))

    if excerpts:
        doc_lines = [f"- **{ex['source']} ({ex['section']}):** {ex['text'][:240].strip()}" for ex in excerpts[:2]]
        lines.append("**Reference Excerpt:**\n" + "\n".join(doc_lines))

    return "\n\n".join(lines)


def _offline_answer_parts(question: str, telemetry: dict[str, Any], structure_type: str) -> tuple[str, list[dict[str, Any]], dict[str, str], str]:
    validation = telemetry.get("class_validation") or class_validation_summary()
    findings = evidence_items(telemetry)
    badge = triage(telemetry)
    excerpts = _knowledge_excerpts(question)
    direct = _direct_question_section(question, telemetry, structure_type, findings, badge, excerpts)
    lines = [
        direct,
        f"### Assessment scope\nThis offline answer uses saved model telemetry only for **{structure_type}**. It does not interpret the image or establish structural safety. Findings are ordered by available pixel coverage, which is image-space extent only. Structural severity is not rated from pixels; image area is affected by framing, resolution, and mask quality.",
    ]
    if not findings:
        lines.append("### Model findings\nNo supported defect label is recorded in the selected telemetry. This means no matching saved model evidence was available for this answer; it does not prove the structure is defect-free. Upload or select a result that includes the relevant segmentation/detection stage.")
    for item in findings:
        severity_guidance = item["severity_guidance"]
        lines.append(f"### {item['title']}\n**Image pixel extent (not structural severity):** {_format_pixel_extent(item['pixel_extent'])}\n**Structural severity:** Not assessable from this image. {severity_guidance['basis']}\n**Field checks:** {severity_guidance['field_checks']}\n**Escalate:** {severity_guidance['escalation']}\n**Recorded evidence:** {_format_measurements(telemetry.get('damage_classes', {}).get(item['label'], {}))}")
        count = sum(n for name, n in telemetry.get("class_counts", {}).items() if _label_key(name) == item["label"])
        if count:
            lines[-1] += f"; detector boxes: {count} (occurrence evidence only)."
        map_item = item.get("mapping_evidence", {})
        if map_item:
            grid = map_item.get("grid", telemetry.get("class_mapping", {}).get("grid", [20, 20]))
            area_text = f"{map_item['pixel_area']} analysis pixels" if map_item.get("pixel_area") is not None else "area not available in the saved map"
            coverage_text = f"{map_item['coverage_percent']:g}%" if map_item.get("coverage_percent") is not None else "coverage not available"
            lines[-1] += f"\n**Mapped image evidence:** {area_text}; coverage {coverage_text}; {map_item.get('mapped_cell_count', 0)} occupied cells on a {grid[0]}×{grid[1]} grid. This is image-space evidence only."
        lines[-1] += f"\n**Mapping-based suggestion:** {item['mapping_suggestion']}\n**Evidence-based field action:** {item['evidence_based_solution']}"
    lines.append("### Severity and next step\n" + badge["urgency"] + " This is a triage recommendation, not a code-compliant condition rating or repair design.")
    lines.append(_class_guidance_section())
    lines.append(_severity_guidance_section())
    lines.append(_class_validation_section(validation))
    lines.append("### Evidence sources\n" + "\n".join(f"- [{s['title']}]({s['url']}) — {s['scope']}" for s in SOURCES.values()))
    return "\n\n".join(lines), findings, badge, direct


def offline_answer(question: str, telemetry: dict[str, Any], structure_type: str) -> tuple[str, list[dict[str, Any]], dict[str, str]]:
    answer, findings, badge, _ = _offline_answer_parts(question, telemetry, structure_type)
    return answer, findings, badge


SYSTEM_PROMPT = """You are a structural inspection assistant grounded in ACI (ACI 224R-01, ACI 224.1R-07, ACI 562) and FHWA field inspection protocols. Answer the user's specific question directly and concisely using the supplied image and the combined CrackVision telemetry from Detection (YOLO), Segmentation (CRACKTO-SFNet V4 / FastSAM), and Damage Mapping (20x20 grid, skeleton length, width P95/median/max, junctions, calibrated mm if present). When the user asks about ACI crack-width thresholds, cite ACI 224R-01 Table 4.1 limits (0.41 mm dry air, 0.30 mm humidity/soil, 0.18 mm deicing, 0.15 mm seawater, 0.10 mm water-retaining): compare calibrated mm values if present in `physical`, or explicitly explain that uncalibrated pixel widths cannot be compared directly to mm limits without calibration or a field crack gauge. Keep direct visual observations, model telemetry, external aggregate validation, and engineering interpretation separate. Treat class_mapping grid cells only as image-relative pointers, never as site coordinates or severity scores. Never infer structural safety, depth, or capacity from pixel area alone. Concrete crushing is NOT a supported model output. Do not repeat generic multi-class documentation walls; answer the question directly using the supplied telemetry and ACI/FHWA sources."""


def cloud_answer(image_bytes: bytes, mime_type: str, question: str, telemetry: dict[str, Any], structure_type: str, api_key: str) -> tuple[str, str]:
    """Use user-provided OpenAI or Gemini vision key; never persist or log it."""
    safe_telemetry = json.dumps(telemetry, ensure_ascii=True, separators=(",", ":"))
    excerpts = _knowledge_excerpts(question)
    excerpt_block = ""
    if excerpts:
        excerpt_block = "\n\nUploaded knowledge excerpts:\n" + "\n".join(
            f"- [{ex['source']} · {ex['section']}]: {ex['text'][:600]}" for ex in excerpts
        )
    prompt = f"Structure type: {structure_type}\nQuestion: {question}\nSaved CrackVision combined telemetry (authoritative for model outputs; empty means none): {safe_telemetry}{excerpt_block}\n\nEvidence sources:\n" + "\n".join(f"{v['title']}: {v['url']} — {v['scope']}" for v in SOURCES.values())
    encoded = base64.b64encode(image_bytes).decode("ascii")
    if api_key.startswith("AIza"):
        model = os.getenv("CRACKVISION_GEMINI_MODEL", "gemini-3.1-flash-lite")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = {"system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]}, "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": mime_type, "data": encoded}}]}], "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1200}}
        request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": api_key}, method="POST")
        engine = f"Gemini VLM ({model})"
    else:
        model = os.getenv("CRACKVISION_OPENAI_MODEL", "gpt-4o-mini")
        url = "https://api.openai.com/v1/responses"
        payload = {"model": model, "instructions": SYSTEM_PROMPT, "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}, {"type": "input_image", "image_url": f"data:{mime_type};base64,{encoded}", "detail": "high"}]}], "temperature": 0.1, "max_output_tokens": 1200}
        request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}, method="POST")
        engine = f"OpenAI VLM ({model})"
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"Vision provider returned HTTP {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"Could not reach the configured vision provider: {error}") from error
    if engine.startswith("OpenAI"):
        text = body.get("output_text") or "\n".join(part.get("text", "") for item in body.get("output", []) for part in item.get("content", []) if part.get("type") == "output_text")
    else:
        text = "\n".join(part.get("text", "") for candidate in body.get("candidates", []) for part in candidate.get("content", {}).get("parts", []) if part.get("text"))
    if not text.strip():
        raise RuntimeError("The vision provider returned no answer text.")
    return text.strip(), engine


def assessment(question: str, telemetry_value: Any, structure_type: str, offline_mode: bool, api_key: str | None, image_bytes: bytes | None, mime_type: str) -> dict[str, Any]:
    telemetry = normalize_telemetry(telemetry_value)
    validation = class_validation_summary()
    telemetry["class_validation"] = validation
    image_assessed = False
    if not offline_mode and api_key and image_bytes:
        answer, engine = cloud_answer(image_bytes, mime_type, question, telemetry, structure_type, api_key)
        direct_answer = answer
        image_assessed = True
        # add authoritative per-label evidence and fixed citations after VLM prose
        findings = evidence_items(telemetry)
        badge = triage(telemetry, image_assessed=True)
        evidence_block = "\n\n### Verified model evidence, mapped cells, and field suggestions\n" + ("Findings are ordered by pixel coverage, which is image extent only. Structural severity is not rated from pixels.\n" + "\n".join(f"- **{item['title']} — image pixel extent (not severity):** {_format_pixel_extent(item['pixel_extent'])}. {item['mapping_suggestion']} Field checks: {item['severity_guidance']['field_checks']} Escalate: {item['severity_guidance']['escalation']} {_format_measurements(telemetry.get('damage_classes', {}).get(item['label'], {}))}. Evidence-based field action: {item['evidence_based_solution']}" for item in findings) if findings else "No supported class label is present in saved telemetry. Any visual observation above is VLM screening only, not a detector or segmentation result.")
        evidence_block += "\n\n### Evidence sources\n" + "\n".join(f"- [{source['title']}]({source['url']}) — {source['scope']}" for source in SOURCES.values())
        evidence_block += "\n\n" + _class_guidance_section()
        evidence_block += "\n\n" + _severity_guidance_section()
        evidence_block += "\n\n" + _class_validation_section(validation)
        answer += evidence_block
    else:
        answer, findings, badge, direct_answer = _offline_answer_parts(question, telemetry, structure_type)
        engine = "Offline Evidence Engine"
    report = {
        "question": question,
        "structure_type": structure_type,
        "aci_level": badge.get("aci_level"),
        "action_timeline": badge.get("action_timeline"),
        "aci_basis": badge.get("aci_basis"),
        "label_risks": badge.get("label_risks", []),
        "labels": findings,
        "pixel_extent_ordering": "Descending available class-wise pixel coverage; image-space extent only, not structural severity.",
        "severity_guidance": {label: _severity_guidance(label) for label in CLASS_ORDER},
        "evidence_based_solutions": evidence_based_solutions(),
        "class_validation": validation,
        "sources": list(SOURCES.values()),
        "assessment_limit": "Triage only; image/model telemetry is not a structural condition rating, capacity assessment, or repair design.",
    }
    result = {"answer": answer, "direct_answer": direct_answer, **badge, "assessment_type": "image_screening" if image_assessed else "evidence_triage", "engine": engine, "telemetry": telemetry, "report": report, "evidence": findings, "notice": report["assessment_limit"]}
    return result


def build_vqa_pdf_bytes(
    *,
    structure_type: str,
    aci_level: str,
    action_timeline: str,
    aci_basis: str,
    telemetry: dict[str, Any],
    qa_items: list[dict[str, str]],
    source_filename: str | None = None,
    inspection_id: str | None = None,
    image_path: str | None = None,
) -> bytes:
    """Render a clean A4 PDF report for the VQA assessment and combined telemetry."""
    from datetime import datetime, timezone
    from html import escape
    from io import BytesIO
    from pathlib import Path
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="CrackVision ACI Structural VQA Report",
    )
    styles = getSampleStyleSheet()
    heading_style = ParagraphStyle("VqaHeading", parent=styles["Heading3"], spaceAfter=4, spaceBefore=8, textColor=colors.HexColor("#111e2e"))
    body_style = ParagraphStyle("VqaBody", parent=styles["Normal"], fontSize=9.5, leading=13.5, textColor=colors.HexColor("#223447"))
    small_style = ParagraphStyle("VqaSmall", parent=styles["Normal"], fontSize=8.5, leading=12, textColor=colors.HexColor("#526274"))

    level = (aci_level or "MODERATE").upper()
    badge_hex = "#dc2626" if level == "CRITICAL" else "#d97706" if level == "MODERATE" else "#16a34a" if level == "LOW" else "#64748b"

    def clean_md(text: str) -> str:
        safe = escape(text or "")
        safe = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", safe)
        safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
        safe = re.sub(r"`(.+?)`", r"\1", safe)
        safe = re.sub(r"^###+\s*", "", safe, flags=re.M)
        return safe.replace("\n", "<br/>")

    story: list[Any] = [
        Paragraph("CrackVision — ACI Structural VQA Inspection Report", styles["Title"]),
        Paragraph(
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
            + (f" &nbsp;|&nbsp; File: {escape(source_filename)}" if source_filename else "")
            + (f" &nbsp;|&nbsp; Inspection: {escape(inspection_id[:8])}" if inspection_id else "")
            + f"<br/>Element Classification: <b>{escape(structure_type)}</b>",
            small_style,
        ),
        Spacer(1, 5 * mm),
    ]

    norm = normalize_telemetry(telemetry or {})
    per_label = label_risks(norm)
    risk_summary = ", ".join(f"{r['title']}: {r['risk']} ({r['timeline']})" for r in per_label)
    triage_rows = [
        [Paragraph(f"<b>ACI RISK: {escape(level)}</b>", ParagraphStyle("TriageHead", parent=body_style, textColor=colors.white, fontSize=11))],
        [Paragraph(
            f"<b>Action Timeline:</b> {escape(action_timeline or 'Verify on site')}<br/>"
            + (f"<b>Per-Label Risk:</b> {escape(risk_summary)}<br/>" if risk_summary else "")
            + f"<b>ACI Basis:</b> {escape(aci_basis or 'Based on combined CrackVision telemetry and ACI 224R / 562 criteria.')}",
            body_style,
        )],
    ]
    triage_table = Table(triage_rows, colWidths=[176 * mm])
    triage_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(badge_hex)),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor(badge_hex)),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([triage_table, Spacer(1, 5 * mm)])

    if image_path and Path(image_path).is_file():
        try:
            from PIL import Image as PILImage
            with PILImage.open(image_path) as im:
                w, h = im.size
            max_w = 130 * mm
            max_h = 80 * mm
            scale = min(max_w / max(w, 1), max_h / max(h, 1))
            story.extend([RLImage(str(image_path), width=w * scale, height=h * scale), Spacer(1, 4 * mm)])
        except Exception:
            pass

    story.append(Paragraph("Combined Module Telemetry (Detection + Segmentation + Mapping)", heading_style))
    rows = [["Module / Metric", "Measured Value"]]
    cls_counts = norm.get("class_counts") or {}
    if norm.get("detection_count") is not None or cls_counts:
        cls_str = ", ".join(f"{k}: {v}" for k, v in cls_counts.items()) or "none"
        rows.append(["Detection (YOLO)", f"{int(round(norm.get('detection_count') or 0))} box(es) ({cls_str})"])
    if norm.get("crack_pixels") is not None or norm.get("coverage_percent") is not None:
        cov = norm.get("mask_coverage_percent", norm.get("coverage_percent"))
        px = norm.get("crack_pixels")
        w_med = norm.get("estimated_median_width_px")
        seg_bits = []
        if cov is not None:
            seg_bits.append(f"{cov:.2f}% coverage")
        if px is not None:
            seg_bits.append(f"{int(round(px)):,} px²")
        if w_med is not None:
            seg_bits.append(f"median width {w_med:.2f} px")
        rows.append(["Segmentation (V4)", ", ".join(seg_bits)])
    for cls_name in ("spalling", "rebar", "fissures"):
        dm = (norm.get("damage_classes") or {}).get(cls_name)
        if dm:
            area = dm.get("pixel_area", dm.get("pixels", 0))
            cov = dm.get("coverage_percent", 0)
            rows.append([f"Non-crack ({cls_name}, FastSAM)", f"{cov:.2f}% ({int(round(area)):,} px², {int(round(dm.get('component_count') or 0))} components)"])
    if any(norm.get(k) is not None for k in ("hotspot_count", "max_priority", "mapping_crack_length", "mapping_width_p95")):
        map_bits = []
        if norm.get("occupied_cells") is not None:
            map_bits.append(f"{int(round(norm['occupied_cells']))} cells")
        if norm.get("hotspot_count") is not None:
            map_bits.append(f"{int(round(norm['hotspot_count']))} hotspots")
        if norm.get("max_priority") is not None:
            map_bits.append(f"priority {norm['max_priority']:.2f}")
        if norm.get("mapping_crack_length") is not None:
            map_bits.append(f"length {norm['mapping_crack_length']:.1f} px")
        if norm.get("mapping_width_p95") is not None:
            map_bits.append(f"width P95 {norm['mapping_width_p95']:.2f} px")
        if norm.get("mapping_junctions") is not None:
            map_bits.append(f"{int(round(norm['mapping_junctions']))} junctions")
        rows.append(["Damage Mapping (20×20)", ", ".join(map_bits)])
    phys = norm.get("physical") or {}
    if phys:
        p_bits = [f"{k}: {v:.2f}" for k, v in phys.items() if k.endswith(("_mm", "_mm2"))]
        if p_bits:
            rows.append(["Calibrated Physical (mm/mm²)", ", ".join(p_bits)])

    if len(rows) > 1:
        tbl = Table(rows, colWidths=[56 * mm, 120 * mm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111e2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.extend([tbl, Spacer(1, 5 * mm)])

    story.append(Paragraph("VQA Questions &amp; Answers", heading_style))
    for idx, item in enumerate(qa_items, start=1):
        q = item.get("question", "")
        a = item.get("answer", "")
        if q:
            story.append(Paragraph(f"<b>Q{idx}:</b> {escape(q)}", body_style))
        if a:
            story.append(Paragraph(clean_md(a), body_style))
        story.append(Spacer(1, 3 * mm))

    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "Note: Triage is based on CrackVision model telemetry and ACI 224R-01 / ACI 224.1R-07 / ACI 562 inspection criteria; "
        "verify physical crack widths, active movement, and member condition with a qualified engineer on site.",
        small_style,
    ))
    doc.build(story)
    return buffer.getvalue()
