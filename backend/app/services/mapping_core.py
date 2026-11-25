"""Damage-mapping analysis core (no torch / no web framework).

Ported from the supplied CRACKTO-SFNet V4 research-mapping notebook
("CORRECTED RESEARCH MAPPING CELL" and the live image + video mapping cell):

* the V4 probability map is analysed at the model's 256x256 input resolution,
* crack length comes from a skeleton graph (each undirected edge counted once),
* width is an EDT centre-line *proxy* (not an independently calibrated width),
* the image is divided into a 20x20 grid of regions,
* four regional features (area ratio, length density, width P95, junction
  density) are normalised with fixed scales estimated from VALIDATION
  predictions only, and combined into an *exploratory* priority index.

This is image-space mapping. It is NOT geographic disaster mapping and the
priority index is NOT a validated structural-safety or failure score.
"""
from __future__ import annotations

import functools
import json
import math
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr
from skimage.morphology import skeletonize

INPUT_SIZE = 256
GRID_ROWS = 20
GRID_COLS = 20
DEFAULT_THRESHOLD = 0.525  # validation-selected threshold from the notebook

FEATURES = ("crack_area_ratio", "length_density", "width_p95", "junction_density")

BASE_WEIGHTS = {
    "crack_area_ratio": 0.25,
    "length_density": 0.25,
    "width_p95": 0.25,
    "junction_density": 0.25,
}

WEIGHT_SCENARIOS: dict[str, dict[str, float]] = {
    "equal": dict(BASE_WEIGHTS),
    "length_emphasis": {"crack_area_ratio": 0.20, "length_density": 0.40, "width_p95": 0.25, "junction_density": 0.15},
    "width_emphasis": {"crack_area_ratio": 0.20, "length_density": 0.20, "width_p95": 0.40, "junction_density": 0.20},
    "area_emphasis": {"crack_area_ratio": 0.40, "length_density": 0.25, "width_p95": 0.20, "junction_density": 0.15},
    "topology_emphasis": {"crack_area_ratio": 0.20, "length_density": 0.20, "width_p95": 0.20, "junction_density": 0.40},
}

# Fixed scales from validation predictions (research_mapping_metadata.json).
FALLBACK_SCALES = {
    "crack_area_ratio": 1.0,
    "length_density": 0.11774849074918148,
    "width_p95": 48.48896789550781,
    "junction_density": 0.011834319526627219,
}

INDEX_VERSION = "fixed_validation_normalized_exploratory_index_v2"

CAVEATS = [
    "Index is exploratory, not validated damage severity or failure probability.",
    "Hotspots are relative within-image rankings, not absolute damage thresholds.",
    "Measurements are made at the V4 model-analysis resolution of 256x256 pixels.",
    "EDT centre-line width is a proxy pending OrthoBoundary and physical gauge validation.",
    "The 2.5D vertical axis is the derived index, not physical depth.",
    "This is image-space mapping, not a geographic disaster map or a structural-safety assessment.",
]


# --------------------------------------------------------------------------------------
# Normalisation scales
# --------------------------------------------------------------------------------------
def load_scales(reference_path: Path | None) -> tuple[dict[str, float], dict[str, Any]]:
    """Load the validation-derived scales; fall back to the embedded copy."""
    if reference_path is not None and reference_path.is_file():
        try:
            data = json.loads(reference_path.read_text(encoding="utf-8"))
            scales = {k: float(data["scales"][k]) for k in FEATURES}
            if all(math.isfinite(v) and v > 0 for v in scales.values()):
                return scales, {"source": reference_path.name, "reference_images": data.get("reference_images"),
                                "quantile": data.get("quantile"), "meaning": data.get("meaning")}
        except Exception:  # noqa: BLE001 - fall through to the embedded copy
            pass
    return dict(FALLBACK_SCALES), {"source": "embedded fallback (validation-derived)", "reference_images": 1200, "quantile": 0.95,
                                   "meaning": "Feature normalizers estimated from validation predictions."}


# --------------------------------------------------------------------------------------
# Skeleton graph and measurements
# --------------------------------------------------------------------------------------
def graph_degree_length(skeleton: np.ndarray, scale_x: float = 1.0, scale_y: float = 1.0):
    """Count each undirected skeleton edge once.

    Diagonal edges are suppressed when an orthogonal bridge exists (a digital
    geometry convention taken directly from the notebook).
    """
    skeleton = np.asarray(skeleton, dtype=bool)
    degree = np.zeros(skeleton.shape, dtype=np.uint16)

    horizontal = skeleton[:, :-1] & skeleton[:, 1:]
    degree[:, :-1] += horizontal
    degree[:, 1:] += horizontal
    length = float(horizontal.sum()) * scale_x

    vertical = skeleton[:-1, :] & skeleton[1:, :]
    degree[:-1, :] += vertical
    degree[1:, :] += vertical
    length += float(vertical.sum()) * scale_y

    diagonal_1 = skeleton[:-1, :-1] & skeleton[1:, 1:] & ~skeleton[:-1, 1:] & ~skeleton[1:, :-1]
    degree[:-1, :-1] += diagonal_1
    degree[1:, 1:] += diagonal_1
    length += float(diagonal_1.sum()) * math.hypot(scale_x, scale_y)

    diagonal_2 = skeleton[:-1, 1:] & skeleton[1:, :-1] & ~skeleton[:-1, :-1] & ~skeleton[1:, 1:]
    degree[:-1, 1:] += diagonal_2
    degree[1:, :-1] += diagonal_2
    length += float(diagonal_2.sum()) * math.hypot(scale_x, scale_y)

    return degree, float(length), (horizontal, vertical, diagonal_1, diagonal_2)


def _nan_stat(values: np.ndarray, fn) -> float:
    return float(fn(values)) if len(values) else float("nan")


def measure_mask_grid(mask, probability=None, scale_x: float = 1.0, scale_y: float = 1.0,
                      rows: int = GRID_ROWS, cols: int = GRID_COLS):
    """Return (image_record, regions DataFrame, arrays) for a binary crack mask."""
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape

    skeleton = skeletonize(mask) if mask.any() else np.zeros(mask.shape, dtype=bool)
    degree, total_length, edges = graph_degree_length(skeleton, scale_x, scale_y)

    eight = np.ones((3, 3), dtype=np.uint8)
    branch_pixels = skeleton & (degree >= 3)
    branch_labels, branch_count = ndimage.label(branch_pixels, structure=eight)
    endpoint_count = int(np.count_nonzero(skeleton & (degree == 1)))
    mask_components = int(ndimage.label(mask, structure=eight)[1]) if mask.any() else 0
    skeleton_components = int(ndimage.label(skeleton, structure=eight)[1]) if skeleton.any() else 0

    # Width proxy: sample the EDT at centre-line points (NOT a calibrated physical width).
    if mask.any() and skeleton.any():
        distance = distance_transform_edt(mask, sampling=(scale_y, scale_x)).astype(np.float32)
        width_map = (2.0 * distance).astype(np.float32)
        width_sample_mask = skeleton & (degree == 2) & np.isfinite(width_map) & (width_map > 0)
        if not width_sample_mask.any():
            width_sample_mask = skeleton & (degree < 3) & (width_map > 0)
        width_y, width_x = np.where(width_sample_mask)
        width_values = width_map[width_y, width_x].astype(np.float32)
    else:
        width_map = np.zeros(mask.shape, dtype=np.float32)
        width_sample_mask = np.zeros(mask.shape, dtype=bool)
        width_y = np.array([], dtype=int)
        width_x = np.array([], dtype=int)
        width_values = np.array([], dtype=np.float32)

    # Prediction-uncertainty proxies (not epistemic model uncertainty).
    entropy = ambiguous = None
    if probability is not None:
        p = np.clip(np.asarray(probability, dtype=np.float32), 1e-7, 1.0 - 1e-7)
        entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        ambiguous = (p >= 0.40) & (p <= 0.60)

    pixel_area = scale_x * scale_y
    length_grid = np.zeros((rows, cols), dtype=np.float64)

    def accumulate(edge_mask, row_offset, col_offset, dr, dc, edge_length):
        rr, cc = np.nonzero(edge_mask)
        if len(rr) == 0:
            return
        mid_r = rr.astype(float) + row_offset + dr / 2.0
        mid_c = cc.astype(float) + col_offset + dc / 2.0
        gr = np.clip((mid_r * rows / max(height, 1)).astype(int), 0, rows - 1)
        gc = np.clip((mid_c * cols / max(width, 1)).astype(int), 0, cols - 1)
        np.add.at(length_grid, (gr, gc), edge_length)

    horizontal, vertical, diagonal_1, diagonal_2 = edges
    accumulate(horizontal, 0, 0, 0, 1, scale_x)
    accumulate(vertical, 0, 0, 1, 0, scale_y)
    accumulate(diagonal_1, 0, 0, 1, 1, math.hypot(scale_x, scale_y))
    accumulate(diagonal_2, 0, 1, 1, -1, math.hypot(scale_x, scale_y))

    width_groups: list[list[list[float]]] = [[[] for _ in range(cols)] for _ in range(rows)]
    if len(width_values):
        gr = np.clip((width_y * rows / max(height, 1)).astype(int), 0, rows - 1)
        gc = np.clip((width_x * cols / max(width, 1)).astype(int), 0, cols - 1)
        for r, c, value in zip(gr, gc, width_values):
            width_groups[int(r)][int(c)].append(float(value))

    # Each clustered junction is assigned to one grid cell.
    junction_grid = np.zeros((rows, cols), dtype=np.float64)
    for label_id in range(1, branch_count + 1):
        yy, xx = np.where(branch_labels == label_id)
        if len(xx) == 0:
            continue
        gr = min(int(np.mean(yy) * rows / max(height, 1)), rows - 1)
        gc = min(int(np.mean(xx) * cols / max(width, 1)), cols - 1)
        junction_grid[gr, gc] += 1

    region_rows: list[dict[str, Any]] = []
    for grid_row in range(rows):
        y0 = int(grid_row * height / rows)
        y1 = int((grid_row + 1) * height / rows)
        for grid_col in range(cols):
            x0 = int(grid_col * width / cols)
            x1 = int((grid_col + 1) * width / cols)
            cell = mask[y0:y1, x0:x1]
            if cell.size == 0:
                continue
            crack_pixels = int(cell.sum())
            cell_area = float(cell.size * pixel_area)
            cell_widths = np.asarray(width_groups[grid_row][grid_col], dtype=np.float32)
            region_rows.append({
                "grid_row": grid_row, "grid_col": grid_col,
                "cell_area_px2": int(cell.size), "crack_area_px2": crack_pixels,
                "crack_area_ratio": float(crack_pixels / cell.size),
                "total_crack_length": float(length_grid[grid_row, grid_col]),
                "length_density": float(length_grid[grid_row, grid_col] / max(cell_area, 1e-12)),
                "width_median": _nan_stat(cell_widths, np.median),
                "width_p90": _nan_stat(cell_widths, lambda v: np.percentile(v, 90)),
                "width_p95": _nan_stat(cell_widths, lambda v: np.percentile(v, 95)),
                "width_max": _nan_stat(cell_widths, np.max),
                "width_sample_count": int(len(cell_widths)),
                "junction_count": int(junction_grid[grid_row, grid_col]),
                "junction_density": float(junction_grid[grid_row, grid_col] / max(cell_area, 1e-12)),
                "mean_probability_entropy": float(np.mean(entropy[y0:y1, x0:x1])) if entropy is not None else float("nan"),
                "ambiguous_probability_fraction": float(np.mean(ambiguous[y0:y1, x0:x1])) if ambiguous is not None else float("nan"),
                "empty_cell": bool(crack_pixels == 0),
            })

    # Image-level orientation from the skeleton's principal axis.
    orientation = float("nan")
    yy, xx = np.where(skeleton)
    if len(xx) >= 2:
        coords = np.column_stack([xx.astype(float) * scale_x, yy.astype(float) * scale_y])
        try:
            cov = np.cov((coords - coords.mean(axis=0)).T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            vector = eigenvectors[:, int(np.argmax(eigenvalues))]
            orientation = float(np.degrees(np.arctan2(vector[1], vector[0])) % 180.0)
        except Exception:  # noqa: BLE001
            orientation = float("nan")

    pixel_count = int(mask.sum())
    image_record = {
        "crack_pixels": pixel_count,
        "crack_area_ratio": float(pixel_count / max(mask.size, 1)),
        "crack_area": float(pixel_count * pixel_area),
        "surface_area": float(mask.size * pixel_area),
        "crack_length": float(total_length),
        "length_density": float(total_length / max(mask.size * pixel_area, 1e-12)),
        "width_mean": _nan_stat(width_values, np.mean),
        "width_median": _nan_stat(width_values, np.median),
        "width_p90": _nan_stat(width_values, lambda v: np.percentile(v, 90)),
        "width_p95": _nan_stat(width_values, lambda v: np.percentile(v, 95)),
        "width_max": _nan_stat(width_values, np.max),
        "width_sample_count": int(len(width_values)),
        "endpoints": endpoint_count,
        "junction_clusters": int(branch_count),
        "branch_pixels_raw": int(branch_pixels.sum()),
        "skeleton_pixels": int(skeleton.sum()),
        "skeleton_components": skeleton_components,
        "mask_components": mask_components,
        "orientation_deg": orientation,
        "analysis_height": int(height),
        "analysis_width": int(width),
        "mean_probability_entropy": float(np.mean(entropy)) if entropy is not None else float("nan"),
        "ambiguous_probability_fraction": float(np.mean(ambiguous)) if ambiguous is not None else float("nan"),
    }

    arrays = {
        "mask": mask,
        "skeleton": skeleton,
        "degree": degree,
        "width_map": width_map.astype(np.float32),
        "width_samples": width_sample_mask.astype(np.uint8),
        # NOTE: the notebook's saved ``length_density`` / ``junction_density`` arrays hold the
        # per-cell totals (length, junction count). They are named honestly here.
        "length_total": length_grid.astype(np.float32),
        "junction_count": junction_grid.astype(np.float32),
    }
    return image_record, pd.DataFrame(region_rows), arrays


# --------------------------------------------------------------------------------------
# Priority index
# --------------------------------------------------------------------------------------
def add_priority_index(regions: pd.DataFrame, scales: dict[str, float], weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Exploratory relative inspection-priority index (not a severity or failure score)."""
    weights = BASE_WEIGHTS if weights is None else weights
    out = regions.copy()
    normalized_columns = []
    for column in FEATURES:
        values = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float)
        out["norm_" + column] = np.clip(values / max(float(scales[column]), 1e-12), 0.0, 1.0)
        normalized_columns.append("norm_" + column)

    values = out[normalized_columns].to_numpy(dtype=float)
    weight_values = np.asarray([weights[c] for c in FEATURES], dtype=float)
    valid = np.isfinite(values)
    weighted_sum = np.nansum(values * weight_values[None, :], axis=1)
    available = (valid * weight_values[None, :]).sum(axis=1)
    score = np.divide(weighted_sum, available, out=np.zeros_like(weighted_sum), where=available > 0)

    out["priority_index_exploratory"] = np.clip(score, 0.0, 1.0)
    out["index_feature_coverage"] = available
    out["relative_within_image_rank"] = out["priority_index_exploratory"].rank(method="average", pct=True)
    positive = out.loc[out["priority_index_exploratory"] > 0, "priority_index_exploratory"]
    cutoff = float(positive.quantile(0.75)) if len(positive) else np.inf
    out["relative_hotspot_top_quartile"] = (out["priority_index_exploratory"] > 0) & (out["priority_index_exploratory"] >= cutoff)
    return out


def safe_spearman(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
        return float("nan")
    return float(spearmanr(a[valid], b[valid]).statistic)


def top_cells(frame: pd.DataFrame, score_column: str = "priority_index_exploratory", fraction: float = 0.10) -> set[tuple[int, int]]:
    positive = frame.loc[pd.to_numeric(frame[score_column], errors="coerce") > 0]
    if positive.empty:
        return set()
    count = max(1, int(np.ceil(fraction * len(positive))))
    top = positive.nlargest(count, score_column)
    return set(zip(top["grid_row"].astype(int), top["grid_col"].astype(int)))


def grid_matrix(frame: pd.DataFrame, column: str, rows: int = GRID_ROWS, cols: int = GRID_COLS) -> np.ndarray:
    return (frame.pivot(index="grid_row", columns="grid_col", values=column)
            .reindex(index=range(rows), columns=range(cols)).to_numpy(dtype=float))


def weight_sensitivity(regions: pd.DataFrame, scales: dict[str, float], baseline_scenario: str = "equal") -> list[dict[str, Any]]:
    """How stable the top-10 % ranking is when the index weights change."""
    baseline = add_priority_index(regions, scales, WEIGHT_SCENARIOS[baseline_scenario])
    base_top = top_cells(baseline)
    rows = []
    for name, weights in WEIGHT_SCENARIOS.items():
        scored = add_priority_index(regions, scales, weights)
        scenario_top = top_cells(scored)
        union = base_top | scenario_top
        iou = float(len(base_top & scenario_top) / len(union)) if union else float("nan")
        rho = safe_spearman(baseline["priority_index_exploratory"], scored["priority_index_exploratory"])
        rows.append({"scenario": name, "weights": weights, "top10pct_overlap_iou": iou, "spearman_vs_equal": rho,
                     "top10pct_cells": len(scenario_top)})
    return rows


# --------------------------------------------------------------------------------------
# High-level analysis of one probability map
# --------------------------------------------------------------------------------------
def camera_scale(camera: dict[str, float], size: int = INPUT_SIZE) -> tuple[float, float]:
    """mm per analysis pixel from camera geometry (the notebook's GSD formula).

    The field of view is  height * sensor / focal_length  and is independent of image width, so
    dividing it by the analysis size gives the per-pixel scale at the model resolution.
    """
    h, f = float(camera["height_mm"]), float(camera["focal_length_mm"])
    sw, sh = float(camera["sensor_width_mm"]), float(camera["sensor_height_mm"])
    if min(h, f, sw, sh) <= 0:
        raise ValueError("Camera height, focal length and sensor dimensions must all be positive.")
    if camera.get("swap_axes"):
        sw, sh = sh, sw
    return (h * sw / f) / size, (h * sh / f) / size


def analyze_probability(probability: np.ndarray, threshold: float, scales: dict[str, float],
                        weights: dict[str, float] | None = None, scale_xy: tuple[float, float] | None = None) -> dict[str, Any]:
    """Full mapping analysis of a 256x256 crack-probability map."""
    probability = np.asarray(probability, dtype=np.float32)
    if probability.shape != (INPUT_SIZE, INPUT_SIZE):
        probability = np.asarray(Image.fromarray(probability, mode="F").resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR), dtype=np.float32)
    probability = np.clip(probability, 0.0, 1.0)
    mask = probability >= float(threshold)

    # The priority index always works in analysis-pixel units so the validation scales stay valid.
    image_record, regions, arrays = measure_mask_grid(mask, probability)
    regions = add_priority_index(regions, scales, weights)

    physical = None
    if scale_xy is not None:
        phys_record, _, _ = measure_mask_grid(mask, probability, scale_xy[0], scale_xy[1])
        physical = {
            "scale_x_mm_per_px": scale_xy[0], "scale_y_mm_per_px": scale_xy[1],
            "crack_area_mm2": phys_record["crack_area"], "field_area_mm2": phys_record["surface_area"],
            "crack_length_mm": phys_record["crack_length"],
            "width_mean_mm": phys_record["width_mean"], "width_median_mm": phys_record["width_median"],
            "width_p95_mm": phys_record["width_p95"], "width_max_mm": phys_record["width_max"],
        }

    matrices = {
        "priority": grid_matrix(regions, "priority_index_exploratory"),
        "crack_area_ratio": grid_matrix(regions, "crack_area_ratio"),
        "length_density": grid_matrix(regions, "length_density"),
        "width_p95": grid_matrix(regions, "width_p95"),
        "junction_density": grid_matrix(regions, "junction_density"),
        "hotspot": grid_matrix(regions, "relative_hotspot_top_quartile"),
    }
    return {"probability": probability, "mask": mask, "image_record": image_record, "regions": regions,
            "arrays": arrays, "matrices": matrices, "physical": physical}


def ranked_cells(regions: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    positive = regions.loc[regions["priority_index_exploratory"] > 0].nlargest(limit, "priority_index_exploratory")
    out = []
    for rank, (_, row) in enumerate(positive.iterrows(), start=1):
        out.append({
            "rank": rank, "grid_row": int(row["grid_row"]), "grid_col": int(row["grid_col"]),
            "priority_index": float(row["priority_index_exploratory"]),
            "crack_area_ratio": float(row["crack_area_ratio"]), "length_density": float(row["length_density"]),
            "width_p95": None if not np.isfinite(row["width_p95"]) else float(row["width_p95"]),
            "junction_count": int(row["junction_count"]),
            "hotspot": bool(row["relative_hotspot_top_quartile"]),
        })
    return out


def json_safe(value: Any) -> Any:
    """Recursively convert numpy / NaN values into strict-JSON-safe Python objects."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


# --------------------------------------------------------------------------------------
# Rendering (matplotlib, Agg backend)
# --------------------------------------------------------------------------------------
# pyplot keeps global state, so every figure render is serialised (video jobs run in a worker thread).
PLOT_LOCK = threading.RLock()


def _locked(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with PLOT_LOCK:
            return fn(*args, **kwargs)
    return wrapper


def _plt():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    return plt


def overlay_on_photo(rgb: np.ndarray, mask256: np.ndarray, skeleton256: np.ndarray, hotspot: np.ndarray,
                     priority: np.ndarray, max_side: int = 1600, grid: bool = True) -> np.ndarray:
    """Segmentation tint, skeleton, grid and hotspot cells drawn on the original photograph."""
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    scale = min(1.0, max_side / max(image.size))
    if scale < 1.0:
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.BILINEAR)
    base = np.asarray(image, dtype=np.uint8).copy()
    h, w = base.shape[:2]

    def up(arr):
        return np.asarray(Image.fromarray((np.asarray(arr) > 0).astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 0

    mask = up(mask256)
    tint = np.array([226, 70, 70], dtype=np.float32)
    base[mask] = (0.5 * base[mask].astype(np.float32) + 0.5 * tint).astype(np.uint8)
    skel = up(skeleton256)
    if skel.any():
        thick = int(round(max(h, w) / 1400))  # upscaled skeleton is already ~2px wide; only widen on large photos
        if thick:
            skel = ndimage.binary_dilation(skel, iterations=thick)
        base[skel] = (255, 235, 90)

    rows, cols = hotspot.shape
    canvas = Image.fromarray(base).convert("RGBA")
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(layer)
    if grid:
        for r in range(1, rows):
            y = int(r * h / rows)
            draw.line([(0, y), (w, y)], fill=(255, 255, 255, 38), width=1)
        for c in range(1, cols):
            x = int(c * w / cols)
            draw.line([(x, 0), (x, h)], fill=(255, 255, 255, 38), width=1)
    line = max(2, int(round(max(h, w) / 480)))
    for r in range(rows):
        for c in range(cols):
            if hotspot[r, c] > 0:
                x0, x1 = int(c * w / cols), int((c + 1) * w / cols)
                y0, y1 = int(r * h / rows), int((r + 1) * h / rows)
                draw.rectangle([x0, y0, x1, y1], fill=(255, 140, 0, 46), outline=(255, 122, 0, 235), width=line)
    return np.asarray(Image.alpha_composite(canvas, layer).convert("RGB"))


def save_png(array: np.ndarray, path: Path) -> None:
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)


@_locked
def render_matrix_png(matrix: np.ndarray, path: Path, title: str, cmap: str = "viridis", hotspot: np.ndarray | None = None) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 4.9), dpi=130)
    values = np.ma.masked_invalid(np.asarray(matrix, dtype=float))
    image = ax.imshow(values, cmap=cmap, interpolation="nearest", vmin=0 if np.nanmax(matrix) > 0 else None)
    if hotspot is not None:
        for r, c in zip(*np.where(hotspot > 0)):
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="#ff3b30", linewidth=1.3))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    ax.set_xticks(range(0, matrix.shape[1], 4))
    ax.set_yticks(range(0, matrix.shape[0], 4))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


@_locked
def render_surface_png(priority: np.ndarray, path: Path, title: str = "2.5D inspection-index surface (not physical depth)") -> None:
    plt = _plt()
    z = np.nan_to_num(np.asarray(priority, dtype=np.float32), nan=0.0)
    fig = plt.figure(figsize=(7.2, 5.4), dpi=130)
    ax = fig.add_subplot(111, projection="3d")
    x, y = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
    surface = ax.plot_surface(x, y, z, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    ax.set_zlabel("Exploratory index")
    ax.set_zlim(0, max(0.1, float(z.max()) * 1.1))
    ax.invert_yaxis()
    fig.colorbar(surface, ax=ax, shrink=0.6, pad=0.1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


@_locked
def render_evidence_card(rgb: np.ndarray, analysis: dict[str, Any], path: Path, title: str, threshold: float) -> None:
    """3x3 evidence card matching the layout of the notebook's research evidence cards."""
    plt = _plt()
    arrays, matrices = analysis["arrays"], analysis["matrices"]
    mask, skeleton = arrays["mask"], arrays["skeleton"]
    small = np.asarray(Image.fromarray(np.asarray(rgb, dtype=np.uint8)).resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR))

    over = small.copy()
    over[mask] = (0.45 * over[mask] + 0.55 * np.array([210, 40, 40])).astype(np.uint8)
    skel_view = small.copy()
    skel_view[ndimage.binary_dilation(skeleton)] = (0, 255, 255)

    width_view = np.ma.masked_where(arrays["width_samples"] == 0, arrays["width_map"])

    fig, axes = plt.subplots(3, 3, figsize=(13, 13), dpi=110)
    panels = [
        ("Image in analysis coordinates", small, {}),
        (f"V4 mask overlay (t={threshold:.3f})", over, {}),
        ("Probability map", analysis["probability"], {"cmap": "magma", "vmin": 0, "vmax": 1}),
        ("Skeleton support", skel_view, {}),
        ("EDT centre-line width proxy", width_view, {"cmap": "turbo"}),
        ("20x20 crack area ratio", matrices["crack_area_ratio"], {"cmap": "viridis"}),
        ("20x20 length density", matrices["length_density"], {"cmap": "viridis"}),
        ("20x20 width P95 proxy", np.ma.masked_invalid(matrices["width_p95"]), {"cmap": "viridis"}),
        ("Exploratory index + relative hotspots", matrices["priority"], {"cmap": "viridis"}),
    ]
    for ax, (name, data, kw) in zip(axes.ravel(), panels):
        ax.imshow(data, interpolation="nearest", **kw)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    hot = matrices["hotspot"]
    for r, c in zip(*np.where(hot > 0)):
        axes[2, 2].add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="#ff2d2d", linewidth=1.4))
    fig.suptitle(f"Research evidence card — {title}\nNot a structural safety assessment", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)
