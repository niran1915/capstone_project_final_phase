from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

# This is the dedicated checkpoint exported by the user's video-detection notebook.
# Keep it separate from detector.pt: the two supplied files are not byte-identical.
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "video_detector_best_reconstructed.pt"


class VideoDetectorService:
    """Lazy loader for the exact YOLO checkpoint supplied with the video pipeline."""

    def __init__(self) -> None:
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
                raise RuntimeError(f"Video detector checkpoint is missing: {MODEL_PATH}")
            try:
                from ultralytics import YOLO
                self._model = YOLO(str(MODEL_PATH))
            except Exception as exc:
                self._load_error = (
                    "Could not load the video-pipeline checkpoint: "
                    f"{type(exc).__name__}: {exc}"
                )
                raise RuntimeError(self._load_error) from exc
        return self._model

    def model_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "weights_found": MODEL_PATH.is_file(),
            "checkpoint": MODEL_PATH.name,
            "checkpoint_bytes": MODEL_PATH.stat().st_size if MODEL_PATH.is_file() else None,
            "runtime_installed": self.runtime_installed,
            "loaded": self._model is not None,
        }
        if self._model is not None:
            info["classes"] = self._model.names
            info["task"] = getattr(self._model, "task", "detect")
        elif self._load_error:
            info["load_error"] = self._load_error
        return info

    def predict_frame(self, frame: np.ndarray, conf: float = 0.25, iou: float = 0.45, imgsz: int = 640):
        model = self.get_model()
        with self._predict_lock:
            results = model.predict(
                source=frame,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                verbose=False,
            )
        if not results:
            return None
        return results[0]


video_detector = VideoDetectorService()
