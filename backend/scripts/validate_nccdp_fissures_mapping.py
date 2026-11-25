"""Run a small, repeatable external fissure/mapping validation on NCCD-PF.

NCCD-PF contains engineer-annotated narrow concrete crack masks. Those labels
are a useful fissure-like proxy, but they are not an explicit fissure class.
This script keeps strict detector fissure hits separate from generic crack hits.

Example:
  python scripts/validate_nccdp_fissures_mapping.py \
    --dataset-root <NCCD-PF_Dataset/Dataset_for_semantic_segmentation> \
    --api http://127.0.0.1:8001 --positive-count 30 --negative-count 20
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _mask_ids(root: Path) -> tuple[list[tuple[int, Path]], list[tuple[int, Path]]]:
    positives: list[tuple[int, Path]] = []
    negatives: list[tuple[int, Path]] = []
    for path in sorted((root / "Masks").glob("mask_*.png"), key=lambda p: int(p.stem.split("_")[-1])):
        image_id = int(path.stem.split("_")[-1])
        mask = np.asarray(Image.open(path).convert("L")) > 0
        (positives if mask.any() else negatives).append((image_id, path))
    return positives, negatives


def _even_sample(items: list[tuple[int, Path]], count: int) -> list[tuple[int, Path]]:
    if count <= 0 or not items:
        return []
    positions = np.linspace(0, len(items) - 1, min(count, len(items)), dtype=int)
    return [items[int(i)] for i in positions]


def _inspection_request(api: str, route: str, image_path: Path, timeout: int = 300) -> tuple[dict[str, Any], str]:
    with image_path.open("rb") as stream:
        response = requests.post(
            f"{api}{route}", files={"file": (image_path.name, stream, "image/jpeg")}, timeout=timeout,
        )
    response.raise_for_status()
    body = response.json()
    result = body.get("result") or {}
    inspection_id = body.get("id")
    if body.get("status") != "succeeded" or not inspection_id:
        raise RuntimeError(f"{route} did not complete for {image_path.name}: {body.get('error_message')}")
    return result, inspection_id


def _top_k_iou(a: np.ndarray, b: np.ndarray, fraction: float = 0.10) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    k = max(1, int(np.ceil(len(a) * fraction)))
    aa = np.zeros(len(a), dtype=bool)
    bb = np.zeros(len(b), dtype=bool)
    aa[np.argsort(a)[-k:]] = True
    bb[np.argsort(b)[-k:]] = True
    union = np.count_nonzero(aa | bb)
    return float(np.count_nonzero(aa & bb) / union) if union else 1.0


def run(dataset_root: Path, api: str, positive_count: int, negative_count: int, multiclass_count: int) -> dict[str, Any]:
    images = dataset_root / "Images"
    masks = dataset_root / "Masks"
    if not images.is_dir() or not masks.is_dir():
        raise FileNotFoundError("Expected NCCD-PF semantic segmentation Images/ and Masks/ folders.")
    health = requests.get(f"{api}/api/health", timeout=30).json()
    if health.get("status") != "ok" or not health.get("modules", {}).get("mapping", {}).get("available"):
        raise RuntimeError("The API or crack-mapping module is unavailable.")
    groups = _mask_ids(dataset_root)
    selected = [("narrow_crack_positive", *item) for item in _even_sample(groups[0], positive_count)]
    selected += [("negative_control", *item) for item in _even_sample(groups[1], negative_count)]
    multiclass_ids = {
        ("narrow_crack_positive", image_id)
        for image_id, _ in _even_sample([(i, p) for label, i, p in selected if label == "narrow_crack_positive"], multiclass_count)
    } | {
        ("negative_control", image_id)
        for image_id, _ in _even_sample([(i, p) for label, i, p in selected if label == "negative_control"], max(1, multiclass_count // 2))
    }

    rows: list[dict[str, Any]] = []
    prediction_priorities: list[np.ndarray] = []
    truth_priorities: list[np.ndarray] = []
    class_map_rows: list[dict[str, Any]] = []
    from app.services.mapping import mapping_service
    from app.services.mapping_core import analyze_probability

    scales, _ = mapping_service.scales()
    try:
        for ground_truth, image_id, mask_path in selected:
            image_path = images / f"image_{image_id}.jpg"
            gt = np.asarray(Image.open(mask_path).convert("L")) > 0
            row: dict[str, Any] = {"image_id": image_id, "ground_truth": ground_truth}

            detected, inspection_id = _inspection_request(api, "/api/detect", image_path)
            try:
                counts = detected.get("class_counts", {})
                row["detector_class_counts"] = counts
                row["any_crack_detection"] = bool(counts.get("crack", 0) or counts.get("fissures", 0))
                row["strict_fissures_detection"] = bool(counts.get("fissures", 0))
            finally:
                requests.delete(f"{api}/api/inspections/{inspection_id}", timeout=30).raise_for_status()

            if (ground_truth, image_id) in multiclass_ids:
                damage_map, inspection_id = _inspection_request(api, "/api/mapping/damage/image", image_path)
                try:
                    class_response = requests.get(f"{api}{damage_map['class_ids_url']}", timeout=60)
                    class_response.raise_for_status()
                    class_ids = np.asarray(Image.open(io.BytesIO(class_response.content)).convert("L"))
                    gt_native = np.asarray(Image.fromarray(gt.astype(np.uint8) * 255).resize(
                        (class_ids.shape[1], class_ids.shape[0]), Image.Resampling.NEAREST
                    )) > 0
                    fissures_mask = class_ids == 4
                    crack_family_mask = np.isin(class_ids, (1, 4))
                    def overlap(pred: np.ndarray) -> tuple[float, float]:
                        inter = int(np.count_nonzero(pred & gt_native))
                        union = int(np.count_nonzero(pred | gt_native))
                        denom = int(pred.sum()) + int(gt_native.sum())
                        return (inter / union if union else 1.0,
                                2 * inter / denom if denom else 1.0)
                    fiss_iou, fiss_dice = overlap(fissures_mask)
                    union_iou, union_dice = overlap(crack_family_mask)
                    cls_row = {
                        "image_id": image_id, "ground_truth": ground_truth,
                        "class_detection_counts": damage_map.get("class_detection_counts", {}),
                        "matched_mask_counts": damage_map.get("matched_mask_counts", {}),
                        "strict_fissures_mask_iou": fiss_iou, "strict_fissures_mask_dice": fiss_dice,
                        "crack_or_fissures_mask_iou": union_iou, "crack_or_fissures_mask_dice": union_dice,
                    }
                    class_map_rows.append(cls_row)
                finally:
                    requests.delete(f"{api}/api/inspections/{inspection_id}", timeout=30).raise_for_status()

            mapped, inspection_id = _inspection_request(api, "/api/mapping/image", image_path)
            try:
                npz_response = requests.get(f"{api}{mapped['arrays_npz_url']}", timeout=60)
                npz_response.raise_for_status()
                with np.load(io.BytesIO(npz_response.content)) as archive:
                    predicted = archive["mask"].astype(bool)
                gt_256 = np.asarray(Image.fromarray(gt.astype(np.uint8) * 255).resize((256, 256), Image.Resampling.NEAREST)) > 0
                intersection = int(np.count_nonzero(predicted & gt_256))
                union = int(np.count_nonzero(predicted | gt_256))
                predicted_count = int(predicted.sum())
                gt_count = int(gt_256.sum())
                row["mapping_mask_iou"] = float(intersection / union) if union else 1.0
                row["mapping_mask_dice"] = float(2 * intersection / (predicted_count + gt_count)) if predicted_count + gt_count else 1.0
                row["gt_crack_pixels_256"] = gt_count
                row["predicted_crack_pixels_256"] = predicted_count
                row["mapping_max_exploratory_priority"] = mapped.get("max_priority")
                truth_map = analyze_probability(gt_256.astype(np.float32), 0.525, scales)["matrices"]["priority"]
                pred_map = np.asarray(mapped["matrices"]["priority"], dtype=float)
                prediction_priorities.append(pred_map.ravel())
                truth_priorities.append(np.asarray(truth_map).ravel())
                row["top_10pct_priority_cells_iou"] = _top_k_iou(pred_map, truth_map)
            finally:
                requests.delete(f"{api}/api/inspections/{inspection_id}", timeout=30).raise_for_status()
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("image_id", "ground_truth", "detector_class_counts", "mapping_mask_iou", "mapping_mask_dice", "top_10pct_priority_cells_iou")}, sort_keys=True), flush=True)
    finally:
        pass

    positives = [r for r in rows if r["ground_truth"] == "narrow_crack_positive"]
    negatives = [r for r in rows if r["ground_truth"] == "negative_control"]
    detected_positive = sum(r["any_crack_detection"] for r in positives)
    detected_fissures = sum(r["strict_fissures_detection"] for r in positives)
    false_positive = sum(r["any_crack_detection"] for r in negatives)
    spatial_rho = float(spearmanr(np.concatenate(prediction_priorities), np.concatenate(truth_priorities)).statistic) if prediction_priorities else float("nan")
    positive_maps = [r for r in positives]
    report = {
        "dataset": "NCCD-PF narrow concrete crack image/mask set",
        "dataset_doi": "10.5281/zenodo.8215100",
        "dataset_license": "CC BY 4.0",
        "dataset_paper_doi": "10.1038/s41597-023-02839-z",
        "dataset_scope": "External fissure-like proxy: engineer-annotated narrow cracks (<0.3 mm); labels do not explicitly distinguish fissures from other cracks.",
        "sampled_positive_count": len(positives), "sampled_negative_count": len(negatives),
        "positive_crack_detection_recall": detected_positive / len(positives) if positives else None,
        "strict_fissures_class_recall_on_narrow_cracks": detected_fissures / len(positives) if positives else None,
        "negative_control_specificity_any_crack_label": 1 - false_positive / len(negatives) if negatives else None,
        "positive_mapping_mean_iou": float(np.mean([r["mapping_mask_iou"] for r in positive_maps])) if positive_maps else None,
        "positive_mapping_mean_dice": float(np.mean([r["mapping_mask_dice"] for r in positive_maps])) if positive_maps else None,
        "all_sample_mapping_mean_iou": float(np.mean([r["mapping_mask_iou"] for r in rows])) if rows else None,
        "priority_grid_global_spearman": spatial_rho,
        "mean_top_10pct_priority_cell_iou": float(np.mean([r["top_10pct_priority_cells_iou"] for r in rows])) if rows else None,
        "multi_class_damage_map_sample_count": len(class_map_rows),
        "positive_strict_fissures_map_mean_iou": float(np.mean([r["strict_fissures_mask_iou"] for r in class_map_rows if r["ground_truth"] == "narrow_crack_positive"])) if any(r["ground_truth"] == "narrow_crack_positive" for r in class_map_rows) else None,
        "positive_crack_or_fissures_map_mean_iou": float(np.mean([r["crack_or_fissures_mask_iou"] for r in class_map_rows if r["ground_truth"] == "narrow_crack_positive"])) if any(r["ground_truth"] == "narrow_crack_positive" for r in class_map_rows) else None,
        # A zero-filled class-count dictionary is truthy, so bool(dict) incorrectly
        # marks every control as a false positive. Use actual matched mask counts
        # when reporting false damage *maps* (detector FP rate is reported separately).
        "negative_multi_class_false_damage_map_rate": float(np.mean([
            any(int(count) > 0 for count in r.get("matched_mask_counts", {}).values())
            for r in class_map_rows if r["ground_truth"] == "negative_control"
        ])) if any(r["ground_truth"] == "negative_control" for r in class_map_rows) else None,
        "mapping_limit": "The priority-grid agreement compares model-derived geometry with engineer pixel masks; it does not validate physical severity, structural safety, or geographic disaster mapping.",
        "samples": rows, "multi_class_damage_map_samples": class_map_rows,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8001")
    parser.add_argument("--positive-count", type=int, default=30)
    parser.add_argument("--negative-count", type=int, default=20)
    parser.add_argument("--multiclass-count", type=int, default=12, help="Positive class-map cases; half as many negative controls")
    parser.add_argument("--output", type=Path, default=Path("validation_reports/nccdp_fissures_mapping.json"))
    args = parser.parse_args()
    report = run(args.dataset_root, args.api, args.positive_count, args.negative_count, args.multiclass_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
