from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

GRID_ROWS = 20
GRID_COLS = 20
CLASS_IDS = {
    "crack": 1,
    "spalling": 2,
    "rebar": 3,
    "fissures": 4,
}
CLASS_LABELS = {
    "crack": "Crack",
    "spalling": "Concrete spalling",
    "rebar": "Exposed/corroded rebar",
    "fissures": "Fissures / Micro-cracks",
}
CAVEATS = [
    "Grid cells are image coordinates, not geographic locations.",
    "Area and perimeter are measured on the 256x256 model analysis mask.",
    "Physical units require verified camera geometry or an image scale reference.",
    "Relative hotspots are within-image visual summaries, not validated severity or failure probabilities.",
    "Predictions require engineering review and are not a structural-safety assessment.",
]


def _perimeter(mask: np.ndarray, scale_xy: tuple[float, float] | None) -> float:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if scale_xy is None:
        return float(sum(cv2.arcLength(contour, True) for contour in contours))
    scale_x, scale_y = scale_xy
    total = 0.0
    for contour in contours:
        points = contour.astype(np.float32)
        points[..., 0] *= scale_x
        points[..., 1] *= scale_y
        total += float(cv2.arcLength(points, True))
    return total


def _centerline_length(mask: np.ndarray, scale_xy: tuple[float, float] | None) -> float:
    skeleton = skeletonize(mask)
    if not skeleton.any():
        return 0.0
    scale_x, scale_y = scale_xy if scale_xy is not None else (1.0, 1.0)
    rows, cols = skeleton.shape
    length = 0.0
    for dr, dc, edge in ((0, 1, scale_x), (1, 0, scale_y), (1, 1, math.hypot(scale_x, scale_y)), (1, -1, math.hypot(scale_x, scale_y))):
        y0 = max(0, -dr)
        y1 = rows - max(0, dr)
        x0 = max(0, -dc)
        x1 = cols - max(0, dc)
        if y1 <= y0 or x1 <= x0:
            continue
        current = skeleton[y0:y1, x0:x1]
        neighbor = skeleton[y0 + dr:y1 + dr, x0 + dc:x1 + dc]
        length += float(np.count_nonzero(current & neighbor)) * edge
    return length


def _width_stats(mask: np.ndarray, scale_xy: tuple[float, float] | None) -> dict[str, float] | None:
    if not np.any(mask):
        return None
    skeleton = skeletonize(mask)
    sampling = (scale_xy[1], scale_xy[0]) if scale_xy is not None else (1.0, 1.0)
    distance = distance_transform_edt(mask, sampling=sampling).astype(np.float32)
    width_map = 2.0 * distance
    samples = width_map[skeleton & np.isfinite(width_map) & (width_map > 0)]
    if samples.size == 0:
        samples = width_map[mask & np.isfinite(width_map) & (width_map > 0)]
    if samples.size == 0:
        return None
    return {
        "median": round(float(np.median(samples)), 3),
        "p95": round(float(np.percentile(samples, 95)), 3),
        "max": round(float(np.max(samples)), 3),
    }


def _relative_hotspots(matrix: np.ndarray) -> np.ndarray:
    occupied = matrix[matrix > 0]
    if occupied.size == 0:
        return np.zeros(matrix.shape, dtype=np.uint8)
    cutoff = float(np.quantile(occupied, 0.75))
    return ((matrix > 0) & (matrix >= cutoff)).astype(np.uint8)


def measure_damage_map(
    class_map: np.ndarray,
    scale_xy: tuple[float, float] | None = None,
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
    class_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    labels = np.asarray(class_map)
    if labels.ndim != 2 or not labels.size:
        raise ValueError("A non-empty two-dimensional class map is required.")
    valid_ids = tuple(CLASS_IDS.values()) + (0,)
    if not np.isin(labels, valid_ids).all():
        raise ValueError(f"Class map contains labels outside the supported IDs {valid_ids}.")
    if rows < 1 or cols < 1:
        raise ValueError("Grid rows and columns must be positive.")
    if scale_xy is not None and (len(scale_xy) != 2 or min(scale_xy) <= 0):
        raise ValueError("Physical scales must contain positive x/y millimetres per pixel.")

    height, width = labels.shape
    matrices: dict[str, list[list[float | int]]] = {}
    class_metrics: dict[str, dict[str, float | int | None]] = {}
    regions: list[dict[str, int | float]] = []

    selected_classes = class_names or tuple(CLASS_IDS)
    if any(name not in CLASS_IDS for name in selected_classes):
        raise ValueError("One or more requested damage classes are not supported.")

    for class_name in selected_classes:
        class_id = CLASS_IDS[class_name]
        mask = labels == class_id
        pixel_count = int(mask.sum())
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        component_areas = stats[1:, cv2.CC_STAT_AREA] if component_count > 1 else np.asarray([], dtype=np.int32)
        perimeter_px = _perimeter(mask, None)
        perimeter_mm = _perimeter(mask, scale_xy) if scale_xy is not None else None

        # Linear features measured for rebar, crack, fissures
        centerline_length_px = _centerline_length(mask, None) if class_name in ("rebar", "crack", "fissures") else None
        centerline_length_mm = _centerline_length(mask, scale_xy) if class_name in ("rebar", "crack", "fissures") and scale_xy is not None else None
        width_px = _width_stats(mask, None) if class_name in ("crack", "fissures") else None
        width_mm = _width_stats(mask, scale_xy) if class_name in ("crack", "fissures") and scale_xy is not None else None

        metric: dict[str, float | int | None] = {
            "pixel_area": pixel_count,
            "coverage_percent": round(100.0 * pixel_count / labels.size, 4),
            "component_count": int(component_count - 1),
            "largest_component_area_px2": int(component_areas.max()) if component_areas.size else 0,
            "perimeter_px": round(perimeter_px, 2),
            "perimeter_mm": round(perimeter_mm, 2) if perimeter_mm is not None else None,
            "area_mm2": round(pixel_count * scale_xy[0] * scale_xy[1], 2) if scale_xy is not None else None,
        }
        if centerline_length_px is not None:
            metric["centerline_length_px"] = round(centerline_length_px, 2)
            metric["centerline_length_mm"] = round(centerline_length_mm, 2) if centerline_length_mm is not None else None
        if width_px is not None:
            metric["width_median_px"] = width_px["median"]
            metric["width_p95_px"] = width_px["p95"]
            metric["width_max_px"] = width_px["max"]
        if width_mm is not None:
            metric["width_median_mm"] = width_mm["median"]
            metric["width_p95_mm"] = width_mm["p95"]
            metric["width_max_mm"] = width_mm["max"]

        if class_name == "spalling" and pixel_count > 0:
            compactness = (perimeter_px ** 2) / (4.0 * math.pi * pixel_count)
            equiv_diam = 2.0 * math.sqrt(pixel_count / math.pi)
            metric["shape_compactness"] = round(compactness, 3)
            metric["equivalent_diameter_px"] = round(equiv_diam, 2)
            if scale_xy is not None:
                metric["equivalent_diameter_mm"] = round(equiv_diam * math.sqrt(scale_xy[0] * scale_xy[1]), 2)

        class_metrics[class_name] = metric

        matrix = np.zeros((rows, cols), dtype=np.float64)
        for grid_row in range(rows):
            y0 = int(grid_row * height / rows)
            y1 = int((grid_row + 1) * height / rows)
            for grid_col in range(cols):
                x0 = int(grid_col * width / cols)
                x1 = int((grid_col + 1) * width / cols)
                cell = mask[y0:y1, x0:x1]
                matrix[grid_row, grid_col] = float(cell.mean()) if cell.size else 0.0
                regions.append({
                    "grid_row": grid_row,
                    "grid_col": grid_col,
                    "class_id": class_id,
                    "class_name": class_name,
                    "pixel_area": int(cell.sum()),
                    "area_ratio": float(matrix[grid_row, grid_col]),
                    "cell_area_px2": int(cell.size),
                })
        matrices[f"{class_name}_area_ratio"] = matrix.tolist()
        matrices[f"{class_name}_relative_hotspot"] = _relative_hotspots(matrix).tolist()
        # Aliases for direct frontend metric switching
        matrices[class_name] = matrices[f"{class_name}_area_ratio"]

    # Normalize the active class weights so the non-crack workflow remains independent.
    base_weights = {"rebar": 0.40, "spalling": 0.30, "crack": 0.20, "fissures": 0.10}
    active_weight = sum(base_weights[name] for name in selected_classes)
    composite = sum(
        base_weights[name] / active_weight * np.asarray(matrices[f"{name}_area_ratio"])
        for name in selected_classes
    )
    composite_hotspot = _relative_hotspots(composite)

    matrices["exploratory_composite_index"] = composite.tolist()
    matrices["composite_relative_hotspot"] = composite_hotspot.tolist()
    matrices["exploratory_composite"] = matrices["exploratory_composite_index"]

    physical = None
    if scale_xy is not None:
        crack_m = class_metrics.get("crack") or {}
        fiss_m = class_metrics.get("fissures") or {}
        spall_m = class_metrics.get("spalling") or {}
        rebar_m = class_metrics.get("rebar") or {}
        width_source = crack_m if crack_m.get("width_p95_mm") is not None else fiss_m
        raw_physical = {
            "scale_x_mm_per_px": scale_xy[0],
            "scale_y_mm_per_px": scale_xy[1],
            "width_median_mm": width_source.get("width_median_mm"),
            "width_p95_mm": width_source.get("width_p95_mm"),
            "width_max_mm": width_source.get("width_max_mm"),
            "crack_length_mm": crack_m.get("centerline_length_mm"),
            "crack_area_mm2": crack_m.get("area_mm2") if crack_m.get("pixel_area", 0) > 0 else None,
            "fissures_width_p95_mm": fiss_m.get("width_p95_mm"),
            "fissures_length_mm": fiss_m.get("centerline_length_mm") if fiss_m.get("pixel_area", 0) > 0 else None,
            "fissures_area_mm2": fiss_m.get("area_mm2") if fiss_m.get("pixel_area", 0) > 0 else None,
            "spalling_area_mm2": spall_m.get("area_mm2") if spall_m.get("pixel_area", 0) > 0 else None,
            "spalling_equivalent_diameter_mm": spall_m.get("equivalent_diameter_mm") if spall_m.get("pixel_area", 0) > 0 else None,
            "spalling_perimeter_mm": spall_m.get("perimeter_mm") if spall_m.get("pixel_area", 0) > 0 else None,
            "rebar_length_mm": rebar_m.get("centerline_length_mm") if rebar_m.get("pixel_area", 0) > 0 else None,
            "rebar_area_mm2": rebar_m.get("area_mm2") if rebar_m.get("pixel_area", 0) > 0 else None,
            "rebar_perimeter_mm": rebar_m.get("perimeter_mm") if rebar_m.get("pixel_area", 0) > 0 else None,
        }
        physical = {k: v for k, v in raw_physical.items() if v is not None}

    return {
        "analysis_height": height,
        "analysis_width": width,
        "grid": [rows, cols],
        "physical": physical,
        "class_metrics": class_metrics,
        "matrices": matrices,
        "regions": regions,
        "composite_summary": {
            "max_exploratory_index": round(float(composite.max()), 4) if composite.size else 0.0,
            "mean_exploratory_index": round(float(composite.mean()), 4) if composite.size else 0.0,
            "hotspot_cells_count": int(composite_hotspot.sum()),
        },
        "caveats": CAVEATS,
    }
