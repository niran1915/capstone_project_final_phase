from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from ..main_paths import storage_url
from .detection import detector
from .segmentation import segmenter

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "FastSAM-s.pt"
IMAGE_SIZE = 256
CLASS_IDS = {"crack": 1, "spalling": 2, "rebar": 3, "fissures": 4}
CLASS_LABELS = {
    "crack": "Crack",
    "spalling": "Concrete spalling",
    "rebar": "Rebar",
    "fissures": "Fissures",
}
CLASS_COLORS = {
    1: (236, 82, 91),
    2: (255, 170, 45),
    3: (48, 165, 235),
    4: (186, 85, 211),
}
MASK_METHOD = "CRACKTO-SFNet V4 for cracks; detector-box-matched FastSAM proposals for non-crack labels"


class DamageSegmentationService:
    """Build per-label masks from the supplied detector and available mask models.

    Crack masks come from the crack-trained V4 network. For the other labels, the
    detector supplies the label and FastSAM supplies a class-agnostic object mask
    selected by overlap with that detector box. Those masks are proposals, not
    outputs of a class-trained non-crack segmentation network.
    """

    def __init__(self) -> None:
        self._model = None
        self._model_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._context = threading.local()
        self.load_error: str | None = None

    def available(self) -> bool:
        return MODEL_PATH.is_file() and detector.available() and segmenter.available()

    def model_info(self) -> dict[str, Any]:
        return {
            "available": self.available(),
            "weights_found": MODEL_PATH.is_file(),
            "detector_available": detector.available(),
            "crack_checkpoint_available": segmenter.available(),
            "loaded": self._model is not None,
            "checkpoint": MODEL_PATH.name,
            "method": MASK_METHOD,
            "supervised_multiclass_checkpoint": False,
            "load_error": self.load_error,
        }

    def _get_model(self):
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                if not MODEL_PATH.is_file():
                    raise RuntimeError(f"FastSAM checkpoint is missing: {MODEL_PATH.name}")
                try:
                    from ultralytics import FastSAM

                    self._model = FastSAM(str(MODEL_PATH))
                except Exception as exc:
                    self.load_error = f"Could not load FastSAM checkpoint: {type(exc).__name__}: {exc}"
                    raise RuntimeError(self.load_error) from exc
        return self._model

    @staticmethod
    def _label_id(name: str) -> int | None:
        value = name.lower().replace("_", " ").replace("-", " ").strip()
        if "fissure" in value or "microcrack" in value or "micro crack" in value:
            return CLASS_IDS["fissures"]
        if "spall" in value:
            return CLASS_IDS["spalling"]
        if "rebar" in value or "reinforcement" in value:
            return CLASS_IDS["rebar"]
        return None

    @staticmethod
    def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
        x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
        x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
        area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
        return intersection / max(1e-6, area_a + area_b - intersection)

    @staticmethod
    def _detections_from_result(result: dict[str, Any] | None) -> list[dict[str, Any]]:
        rows = result.get("detections", []) if isinstance(result, dict) else []
        return [row for row in rows if isinstance(row, dict)]

    def _run_detector(self, rgb: np.ndarray) -> list[dict[str, Any]]:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        result = detector.predict_frame(bgr, conf=0.25, iou=0.45, imgsz=640)
        if result is None or result.boxes is None:
            return []
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        scores = result.boxes.conf.detach().cpu().numpy()
        names = result.names
        return [
            {
                "class_name": str(names.get(int(class_id), class_id)),
                "confidence": float(score),
                "box": {"x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3])},
            }
            for box, class_id, score in zip(boxes, class_ids, scores)
        ]

    @torch.inference_mode()
    def predict_probabilities(
        self,
        rgb: np.ndarray,
        include_cracks: bool = True,
        include_non_cracks: bool = True,
        detections: list[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        """Return class support maps at 256 px; values are not calibrated probabilities."""
        image = np.asarray(rgb, dtype=np.uint8)
        height, width = image.shape[:2]
        support = np.zeros((5, height, width), dtype=np.float32)

        if include_cracks:
            crack_probability = segmenter.predict_probability(image)
            crack_probability = cv2.resize(crack_probability, (width, height), interpolation=cv2.INTER_LINEAR)
            support[CLASS_IDS["crack"]] = crack_probability

        self._context.non_crack_backend = "not_run"
        self._context.detection_counts = {name: 0 for name in CLASS_IDS}
        self._context.matched_counts = {name: 0 for name in CLASS_IDS}
        if include_non_cracks:
            rows = detections if detections is not None else self._run_detector(image)
            targets: list[tuple[np.ndarray, int, float]] = []
            for row in rows:
                class_id = self._label_id(str(row.get("class_name", "")))
                box = row.get("box")
                if class_id is None or not isinstance(box, dict):
                    continue
                try:
                    xyxy = np.asarray([box["x1"], box["y1"], box["x2"], box["y2"]], dtype=np.float32)
                    confidence = float(row.get("confidence", 0.0))
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(xyxy).all() and np.isfinite(confidence) and xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]:
                    targets.append((xyxy, class_id, float(np.clip(confidence, 0.0, 1.0))))
                    label = next(key for key, value in CLASS_IDS.items() if value == class_id)
                    self._context.detection_counts[label] += 1

            if targets:
                try:
                    model = self._get_model()
                    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    with self._inference_lock:
                        results = model.predict(source=bgr, imgsz=640, conf=0.25, iou=0.7, max_det=250, verbose=False)
                    result = results[0] if results else None
                    if result is None or result.masks is None or len(result.masks.data) == 0:
                        self._context.non_crack_backend = "no_fast_sam_proposals"
                    else:
                        raw_masks = result.masks.data.detach().float().cpu().numpy()
                        proposal_boxes = result.boxes.xyxy.detach().float().cpu().numpy()
                        masks = [
                            cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                            if mask.shape != (height, width) else mask.astype(bool)
                            for mask in raw_masks
                        ]
                        for det_box, class_id, confidence in targets:
                            best_index, best_score = None, 0.0
                            for index, (mask, proposal_box) in enumerate(zip(masks, proposal_boxes)):
                                iou = self._box_iou(det_box, proposal_box)
                                dx1, dy1 = max(float(det_box[0]), float(proposal_box[0])), max(float(det_box[1]), float(proposal_box[1]))
                                dx2, dy2 = min(float(det_box[2]), float(proposal_box[2])), min(float(det_box[3]), float(proposal_box[3]))
                                overlap = max(0.0, dx2 - dx1) * max(0.0, dy2 - dy1)
                                det_area = max(1.0, float(det_box[2] - det_box[0]) * float(det_box[3] - det_box[1]))
                                containment = overlap / det_area
                                score = 0.7 * iou + 0.3 * containment
                                if score > best_score:
                                    best_index, best_score = index, score
                            if best_index is None or best_score < 0.04:
                                continue
                            mask = masks[best_index]
                            mask_support = 0.5 + 0.5 * confidence
                            support[class_id, mask] = np.maximum(support[class_id, mask], mask_support)
                            label = next(key for key, value in CLASS_IDS.items() if value == class_id)
                            self._context.matched_counts[label] += 1
                        self._context.non_crack_backend = "detector_box_matched_fastsam_proposals"
                except Exception as exc:
                    self.load_error = f"Detector-guided FastSAM inference failed: {type(exc).__name__}: {exc}"
                    raise RuntimeError(self.load_error) from exc
            else:
                self._context.non_crack_backend = "no_non_crack_detections"

        # Resolve overlapping masks using the strongest class support. Background
        # remains zero in this support tensor; callers use argmax and separately
        # retain class-specific masks where overlap metrics matter.
        target_size = (IMAGE_SIZE, IMAGE_SIZE)
        small = np.zeros((5, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        for class_id in range(1, 5):
            small[class_id] = cv2.resize(support[class_id], target_size, interpolation=cv2.INTER_LINEAR)
        background = np.maximum(0.01, 1.0 - small[1:].max(axis=0))
        return np.concatenate([background[None], small[1:]], axis=0)

    def segment_image(
        self,
        input_path: Path,
        output_dir: Path,
        include_cracks: bool = True,
        detections: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        with Image.open(input_path) as source:
            image = source.convert("RGB")
        rgb = np.asarray(image, dtype=np.uint8)
        scores = self.predict_probabilities(rgb, include_cracks, True, detections=detections)
        class_map = scores.argmax(axis=0).astype(np.uint8)
        confidence = scores.max(axis=0)
        height, width = rgb.shape[:2]
        full_map = cv2.resize(class_map, (width, height), interpolation=cv2.INTER_NEAREST)
        full_confidence = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_LINEAR)

        colors = np.zeros_like(rgb)
        overlay = rgb.copy()
        contours_image = rgb.copy()
        class_metrics: dict[str, dict[str, Any]] = {}
        for name, class_id in CLASS_IDS.items():
            if not include_cracks and name == "crack":
                continue
            mask = class_map == class_id
            count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
            pixels = int(mask.sum())
            if pixels == 0:
                continue
            class_metrics[name] = {
                "pixels": pixels,
                "coverage_percent": round(100 * pixels / max(mask.size, 1), 4),
                "component_count": int(count - 1),
                "largest_component_area_px2": int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0,
                "perimeter_px": round(perimeter, 2),
                "mask_method": "CRACKTO-SFNet V4" if name == "crack" else "FastSAM proposal matched to detector box",
                "detector_count": getattr(self._context, "detection_counts", {}).get(name, 0),
                "matched_mask_count": getattr(self._context, "matched_counts", {}).get(name, 0),
            }
            colors[full_map == class_id] = CLASS_COLORS[class_id]
            if contours:
                cv2.drawContours(contours_image, contours, -1, CLASS_COLORS[class_id], 1)
        active = full_map > 0
        overlay[active] = (0.48 * rgb[active] + 0.52 * colors[active]).astype(np.uint8)

        output_dir.mkdir(parents=True, exist_ok=True)
        ids_path = output_dir / "damage_class_ids.png"
        mask_path = output_dir / "damage_class_mask.png"
        overlay_path = output_dir / "damage_segmentation_overlay.png"
        confidence_path = output_dir / "damage_segmentation_confidence.png"
        contours_path = output_dir / "damage_segmentation_contours.png"
        Image.fromarray(full_map).save(ids_path)
        Image.fromarray(colors).save(mask_path)
        Image.fromarray(overlay).save(overlay_path)
        Image.fromarray(np.clip(full_confidence * 255, 0, 255).astype(np.uint8)).save(confidence_path)
        Image.fromarray(contours_image).save(contours_path)

        return {
            "model": MASK_METHOD,
            "analysis_size": IMAGE_SIZE,
            "source_width": width,
            "source_height": height,
            "class_names": ["background", *CLASS_IDS] if include_cracks else ["background", "spalling", "rebar", "fissures"],
            "class_ids": {str(class_id): name for name, class_id in CLASS_IDS.items() if include_cracks or name != "crack"},
            "class_metrics": class_metrics,
            "inference_backend": getattr(self._context, "non_crack_backend", "unknown"),
            "mask_url": storage_url(mask_path),
            "class_ids_url": storage_url(ids_path),
            "overlay_url": storage_url(overlay_path),
            "probability_url": storage_url(confidence_path),
            "contours_url": storage_url(contours_path),
            "measurement_note": (
                "Crack pixels use CRACKTO-SFNet V4. Non-crack label masks use class-agnostic FastSAM proposals matched to boxes from the supplied YOLO detector. Review the overlay; these are not outputs of a class-trained non-crack segmentation model. All geometry is image-space, not severity."
                if include_cracks else
                "Spalling, rebar, and fissure labels come from the supplied YOLO detector; FastSAM proposals are matched to those boxes. Review the overlay; these are not outputs of a class-trained non-crack segmentation model. All geometry is image-space, not severity."
            ),
        }


damage_segmenter = DamageSegmentationService()
