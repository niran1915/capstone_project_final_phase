"""Compare the multi-class damage-map endpoint with external NCCD-PF masks.

NCCD-PF has engineer-prepared binary narrow-crack masks, not multiclass
spalling/rebar/fissures masks. This check measures fissure/crack-family overlap
only and reports detector-guided proposal counts separately.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sample_masks(root: Path, positive_count: int, negative_count: int) -> list[tuple[str, int]]:
    positives, negatives = [], []
    for path in sorted((root / "Masks").glob("mask_*.png"), key=lambda p: int(p.stem.split("_")[-1])):
        image_id = int(path.stem.split("_")[-1])
        (positives if np.asarray(Image.open(path).convert("L")).any() else negatives).append(image_id)
    result = [("narrow_crack_positive", int(positives[i])) for i in np.linspace(0, len(positives)-1, min(positive_count, len(positives)), dtype=int)]
    result += [("negative_control", int(negatives[i])) for i in np.linspace(0, len(negatives)-1, min(negative_count, len(negatives)), dtype=int)]
    return result


def overlap(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    inter = int(np.count_nonzero(pred & truth))
    union = int(np.count_nonzero(pred | truth))
    total = int(pred.sum()) + int(truth.sum())
    return (inter / union if union else 1.0, 2 * inter / total if total else 1.0)


def run(root: Path, api: str, positive_count: int, negative_count: int) -> dict[str, Any]:
    images, masks = root / "Images", root / "Masks"
    if not images.is_dir() or not masks.is_dir():
        raise FileNotFoundError("Expected NCCD-PF semantic segmentation Images/ and Masks/ folders.")
    results = []
    for gt_label, image_id in sample_masks(root, positive_count, negative_count):
        image_path = images / f"image_{image_id}.jpg"
        gt = np.asarray(Image.open(masks / f"mask_{image_id}.png").convert("L")) > 0
        with image_path.open("rb") as stream:
            response = requests.post(f"{api}/api/mapping/damage/image", files={"file": (image_path.name, stream, "image/jpeg")}, timeout=300)
        response.raise_for_status()
        body = response.json()
        if body.get("status") != "succeeded":
            raise RuntimeError(f"Damage mapping failed for {image_path.name}: {body.get('error_message')}")
        inspection_id = body["id"]
        try:
            result = body["result"]
            mapped_response = requests.get(f"{api}{result['class_ids_url']}", timeout=60)
            mapped_response.raise_for_status()
            ids = np.asarray(Image.open(io.BytesIO(mapped_response.content)).convert("L"))
            gt_resized = np.asarray(Image.fromarray(gt.astype(np.uint8) * 255).resize((ids.shape[1], ids.shape[0]), Image.Resampling.NEAREST)) > 0
            fissures_iou, fissures_dice = overlap(ids == 4, gt_resized)
            crack_family_iou, crack_family_dice = overlap(np.isin(ids, (1, 4)), gt_resized)
            row = {
                "image_id": image_id, "ground_truth": gt_label,
                "class_detection_counts": result.get("class_detection_counts", {}),
                "matched_mask_counts": result.get("matched_mask_counts", {}),
                "fissures_class_mask_iou": fissures_iou,
                "fissures_class_mask_dice": fissures_dice,
                "crack_or_fissures_mask_iou": crack_family_iou,
                "crack_or_fissures_mask_dice": crack_family_dice,
            }
            results.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        finally:
            requests.delete(f"{api}/api/inspections/{inspection_id}", timeout=30).raise_for_status()

    positive = [r for r in results if r["ground_truth"] == "narrow_crack_positive"]
    negative = [r for r in results if r["ground_truth"] == "negative_control"]
    return {
        "dataset": "NCCD-PF; genuine bridge/tunnel concrete narrow-crack images with expert masks",
        "dataset_doi": "10.5281/zenodo.8215100",
        "dataset_license": "CC BY 4.0",
        "dataset_paper_doi": "10.1038/s41597-023-02839-z",
        "class_scope": "Masks identify narrow cracks, not a distinct fissures semantic class. No spalling or rebar GT is supplied by this dataset.",
        "sample_counts": {"narrow_crack_positive": len(positive), "negative_control": len(negative)},
        "mean_fissures_class_iou_positive": float(np.mean([r["fissures_class_mask_iou"] for r in positive])) if positive else None,
        "mean_crack_or_fissures_iou_positive": float(np.mean([r["crack_or_fissures_mask_iou"] for r in positive])) if positive else None,
        "false_crack_or_fissures_map_rate_negative": float(np.mean([r["crack_or_fissures_mask_iou"] < 1 for r in negative])) if negative else None,
        "samples": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8001")
    parser.add_argument("--positive-count", type=int, default=8)
    parser.add_argument("--negative-count", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("validation_reports/nccdp_damage_map.json"))
    args = parser.parse_args()
    report = run(args.dataset_root, args.api, args.positive_count, args.negative_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
