from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path
from typing import Any
import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "detector.pt"

class DetectorService:
    """Lazy loader for the exact Ultralytics checkpoint used by the supplied video notebook."""
    def __init__(self):
        self._model = None
        self._load_error: str | None = None
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    @property
    def runtime_installed(self) -> bool:
        try:
            import ultralytics  # noqa: F401
            return True
        except ImportError:
            return False

    def available(self) -> bool:
        return MODEL_PATH.is_file() and self.runtime_installed

    def get_model(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            if not MODEL_PATH.is_file():
                raise RuntimeError(f"Detector checkpoint is missing: {MODEL_PATH.name}")
            try:
                from ultralytics import YOLO
                self._model = YOLO(str(MODEL_PATH))
            except Exception as exc:
                self._load_error = f"Could not load the supplied detector checkpoint: {type(exc).__name__}: {exc}"
                raise RuntimeError(self._load_error) from exc
        return self._model

    def model_info(self) -> dict[str, Any]:
        info = {"weights_found": MODEL_PATH.is_file(), "runtime_installed": self.runtime_installed, "loaded": self._model is not None}
        if self._model is not None:
            info["classes"] = self._model.names
            info["task"] = getattr(self._model, "task", "detect")
        elif self._load_error:
            info["load_error"] = self._load_error
        return info

    def predict_frame(self, frame: np.ndarray, conf: float = 0.25, iou: float = 0.45, imgsz: int = 640):
        model = self.get_model()
        with self._predict_lock:
            results = model.predict(source=frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        if not results:
            return None
        return results[0]

    def detect_image(self, input_path: Path, output_path: Path, conf: float = 0.25, iou: float = 0.45, imgsz: int = 640) -> dict[str, Any]:
        frame = cv2.imread(str(input_path))
        if frame is None:
            raise ValueError("The uploaded file could not be decoded as an image.")
        result = self.predict_frame(frame, conf, iou, imgsz)
        if result is None:
            raise RuntimeError("The detector returned no inference result.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        annotated = result.plot()
        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError("Could not save the annotated image.")
        names = getattr(result, "names", {}) or {}
        detections: list[dict[str, Any]] = []
        classes: list[str] = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            cls_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
            for box, score, cls_id in zip(xyxy, scores, cls_ids):
                label = str(names.get(int(cls_id), cls_id)) if isinstance(names, dict) else str(names[int(cls_id)])
                classes.append(label)
                detections.append({
                    "class_id": int(cls_id), "class_name": label, "confidence": round(float(score), 5),
                    "box": {"x1": round(float(box[0]), 2), "y1": round(float(box[1]), 2), "x2": round(float(box[2]), 2), "y2": round(float(box[3]), 2)},
                })
        confidences = [item["confidence"] for item in detections]
        return {
            "model_checkpoint": MODEL_PATH.name,
            "annotated_image_url": _storage_url(output_path),
            "source_width": int(frame.shape[1]), "source_height": int(frame.shape[0]),
            "detection_count": len(detections), "class_counts": dict(Counter(classes)),
            "average_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0,
            "maximum_confidence": round(max(confidences), 4) if confidences else 0,
            "confidence_threshold": float(conf), "nms_iou_threshold": float(iou), "image_size": int(imgsz),
            "detections": detections,
        }

def _storage_url(path: Path) -> str:
    from ..main_paths import storage_url
    return storage_url(path)

detector = DetectorService()
