from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from .damage_mapping_core import CLASS_IDS, CLASS_LABELS, measure_damage_map
from .damage_segmentation import CLASS_COLORS, IMAGE_SIZE, damage_segmenter
from .class_validation import class_validation_summary
from ..main_paths import storage_url


class DamageMappingService:
    def available(self) -> bool:
        return damage_segmenter.available()

    def map_image(
        self,
        input_path: Path,
        output_dir: Path,
        calibration: dict[str, Any],
        source_name: str | None = None,
    ) -> dict[str, Any]:
        with Image.open(input_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        rgb = np.asarray(image, dtype=np.uint8)
        probabilities = damage_segmenter.predict_probabilities(rgb, include_cracks=True, include_non_cracks=True)
        class_map = probabilities.argmax(axis=0).astype(np.uint8)
        support_map = probabilities.max(axis=0)
        scale_xy = None
        if calibration.get("enabled"):
            scale_xy = (
                float(calibration["scale_x_mm_per_px"]),
                float(calibration["scale_y_mm_per_px"]),
            )
        analysis = measure_damage_map(class_map, scale_xy=scale_xy)
        for class_name, class_id in CLASS_IDS.items():
            metric = analysis["class_metrics"].get(class_name)
            if metric is None:
                continue

        height, width = rgb.shape[:2]
        full_map = cv2.resize(class_map, (width, height), interpolation=cv2.INTER_NEAREST)
        full_support = cv2.resize(support_map, (width, height), interpolation=cv2.INTER_LINEAR)
        color_mask = np.zeros_like(rgb)
        overlay = rgb.copy()
        contours_image = rgb.copy()
        for class_id in CLASS_IDS.values():
            active = full_map == class_id
            color = CLASS_COLORS[class_id]
            color_mask[active] = color
            overlay[active] = (0.48 * rgb[active] + 0.52 * np.array(color, dtype=np.uint8)).astype(np.uint8)
            contours, _ = cv2.findContours(active.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                cv2.drawContours(contours_image, contours, -1, color, 2)

        output_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = output_dir / "damage_mapping_overlay.png"
        mask_path = output_dir / "damage_mapping_mask.png"
        ids_path = output_dir / "damage_mapping_class_ids.png"
        confidence_path = output_dir / "damage_mapping_confidence.png"
        contours_path = output_dir / "damage_mapping_contours.png"
        regions_path = output_dir / "damage_mapping_regions.csv"
        summary_path = output_dir / "damage_mapping_summary.json"
        Image.fromarray(overlay).save(overlay_path)
        Image.fromarray(color_mask).save(mask_path)
        Image.fromarray(full_map).save(ids_path)
        Image.fromarray(np.clip(full_support * 255.0, 0, 255).astype(np.uint8)).save(confidence_path)
        Image.fromarray(contours_image).save(contours_path)

        with regions_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("grid_row", "grid_col", "class_id", "class_name", "pixel_area", "area_ratio", "cell_area_px2"),
            )
            writer.writeheader()
            writer.writerows(analysis["regions"])

        result = {
            "kind": "damage_image",
            "model": "CRACKTO-SFNet V4 + detector-box-matched FastSAM proposals",
            "inference_backend": getattr(damage_segmenter._context, "non_crack_backend", "unknown"),
            "class_detection_counts": getattr(damage_segmenter._context, "detection_counts", {}),
            "matched_mask_counts": getattr(damage_segmenter._context, "matched_counts", {}),
            "source_name": source_name or input_path.name,
            "source_width": width,
            "source_height": height,
            "analysis_size": IMAGE_SIZE,
            "grid": analysis["grid"],
            "units": "mm" if scale_xy is not None else "analysis_px",
            "calibration": calibration,
            "physical": analysis.get("physical"),
            "classes": [
                {"id": class_id, "name": class_name, "label": CLASS_LABELS[class_name], "color": CLASS_COLORS[class_id]}
                for class_name, class_id in CLASS_IDS.items()
            ],
            "class_metrics": analysis["class_metrics"],
            "class_validation": class_validation_summary(),
            "matrices": analysis["matrices"],
            "composite_summary": analysis.get("composite_summary", {}),
            "caveats": analysis["caveats"],
            "measurement_note": (
                "Crack masks use CRACKTO-SFNet V4; non-crack masks are class-agnostic FastSAM proposals assigned from detector boxes. "
                "The 20x20 grid measures image-space regions. Non-crack mask proposals are class-agnostic and unvalidated; composite values are exploratory, not severity. Physical measurements require a verified image scale."
            ),
            "overlay_url": storage_url(overlay_path),
            "mask_url": storage_url(mask_path),
            "class_ids_url": storage_url(ids_path),
            "probability_url": storage_url(confidence_path),
            "contours_url": storage_url(contours_path),
            "regions_csv_url": storage_url(regions_path),
            "summary_json_url": storage_url(summary_path),
        }
        summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result


damage_mapping_service = DamageMappingService()
