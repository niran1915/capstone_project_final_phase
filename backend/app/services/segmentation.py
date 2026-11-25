from __future__ import annotations

import threading
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from PIL import Image
import torch

from .segmentation_arch import CrackTopoSFNetV4
from ..main_paths import storage_url

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "segmentation_v4_best.pth"
IMAGE_SIZE = 256
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

class SegmentationService:
    """Inference wrapper for the checkpoint-matched CRACKTO-SFNet V4 model."""
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_threshold = 0.45
        self.checkpoint_info: dict[str, Any] = {}
        self.load_error: str | None = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        return MODEL_PATH.is_file()

    def get_model(self):
        if self.model is not None:
            return self.model
        with self._lock:
            if self.model is not None:
                return self.model
            if not MODEL_PATH.is_file():
                raise RuntimeError(f"Segmentation checkpoint is missing: {MODEL_PATH.name}")
            try:
                checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
                expected = "CRACKTO_SFNET_V4_FROM_V3_BEST_24E_FINAL2"
                marker = checkpoint.get("runtime_marker")
                if marker != expected:
                    raise RuntimeError(f"Checkpoint architecture marker mismatch: {marker!r}")
                model = CrackTopoSFNetV4()
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                model.to(self.device)
                model.eval()
                self.best_threshold = float(checkpoint.get("best_threshold", 0.45))
                self.checkpoint_info = {
                    "runtime_marker": marker,
                    "best_iou": checkpoint.get("best_iou"),
                    "best_dice": checkpoint.get("best_dice"),
                    "best_threshold": self.best_threshold,
                    "global_epoch": checkpoint.get("global_epoch"),
                }
                self.model = model
            except Exception as exc:
                self.load_error = f"Could not load the provided V4 segmentation checkpoint: {type(exc).__name__}: {exc}"
                raise RuntimeError(self.load_error) from exc
        return self.model

    def model_info(self) -> dict[str, Any]:
        return {
            "weights_found": MODEL_PATH.is_file(), "loaded": self.model is not None,
            "device": str(self.device), "input_size": IMAGE_SIZE,
            "best_threshold": self.best_threshold, "checkpoint": self.checkpoint_info,
            "load_error": self.load_error,
        }

    @torch.inference_mode()
    def segment_image(self, input_path: Path, output_dir: Path, threshold: float | None = None, tta: bool = False) -> dict[str, Any]:
        model = self.get_model()
        image = Image.open(input_path).convert("RGB")
        original_w, original_h = image.size
        resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = (arr - MEAN) / STD
        tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy()).unsqueeze(0).to(self.device)
        with self._lock:
            if tta:
                probability = self._predict_tta(model, tensor)
            else:
                probability = model(tensor)["mask"].float()
        prob_small = probability[0, 0].detach().float().cpu().numpy()
        # Restore predictions to the original photograph dimensions for useful overlays and measurements.
        prob = cv2.resize(prob_small, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        used_threshold = self.best_threshold if threshold is None else float(threshold)
        binary = (prob >= used_threshold).astype(np.uint8) * 255
        boundary = np.zeros_like(binary)
        contours, _ = cv2.findContours(binary.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if contours:
            cv2.drawContours(boundary, contours, -1, 255, 1)
        rgb = np.asarray(image, dtype=np.uint8)
        overlay = rgb.copy()
        crack_pixels = binary > 0
        tint = np.zeros_like(rgb); tint[:, :, 1] = 195; tint[:, :, 0] = 30; tint[:, :, 2] = 110
        overlay[crack_pixels] = (0.45 * overlay[crack_pixels] + 0.55 * tint[crack_pixels]).astype(np.uint8)
        # Outline the predicted regions to help distinguish thin cracks from texture.
        contour_rgb = (255, 205, 75)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.drawContours(overlay_bgr, contours, -1, contour_rgb, 1)
        overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_path = output_dir / "segmentation_mask.png"
        overlay_path = output_dir / "segmentation_overlay.png"
        probability_path = output_dir / "segmentation_probability.png"
        boundary_path = output_dir / "segmentation_contours.png"
        Image.fromarray(binary).save(mask_path)
        Image.fromarray(overlay).save(overlay_path)
        Image.fromarray(np.clip(prob * 255, 0, 255).astype(np.uint8)).save(probability_path)
        Image.fromarray(boundary).save(boundary_path)
        areas = [float(cv2.contourArea(c)) for c in contours]
        perimeters = [float(cv2.arcLength(c, True)) for c in contours]
        foreground = int(crack_pixels.sum())
        distance = cv2.distanceTransform((binary > 0).astype(np.uint8), cv2.DIST_L2, 3)
        width_samples = distance[crack_pixels]
        # Without a known physical calibration, report only pixel-space estimates.
        return {
            "mask_url": storage_url(mask_path), "overlay_url": storage_url(overlay_path),
            "probability_url": storage_url(probability_path), "contours_url": storage_url(boundary_path),
            "source_width": original_w, "source_height": original_h,
            "threshold": used_threshold, "tta_enabled": bool(tta),
            "crack_pixels": foreground,
            "mask_coverage_percent": round(100.0 * foreground / max(1, original_w * original_h), 3),
            "contour_count": len(contours),
            "total_contour_length_px": round(sum(perimeters), 2),
            "largest_contour_length_px": round(max(perimeters, default=0.0), 2),
            "largest_contour_area_px2": round(max(areas, default=0.0), 2),
            "estimated_median_width_px": round(float(np.median(width_samples) * 2.0), 2) if width_samples.size else 0.0,
            "mean_probability": round(float(prob.mean()), 4),
            "model": "CRACKTO-SFNet V4",
            "measurement_note": "Geometry is reported in pixels. Physical mm/cm measurements require a scale/calibration reference.",
        }

    @torch.inference_mode()
    def predict_probability(self, rgb: np.ndarray, tta: bool = False) -> np.ndarray:
        """Return the 256x256 V4 crack-probability map for an RGB uint8 array.

        Uses exactly the same resize / normalisation as ``segment_image`` and is used by the
        damage-mapping module, which analyses the map at the model's native input resolution.
        """
        model = self.get_model()
        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
        resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = (arr - MEAN) / STD
        tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy()).unsqueeze(0).to(self.device)
        with self._lock:
            probability = self._predict_tta(model, tensor) if tta else model(tensor)["mask"].float()
        return probability[0, 0].detach().float().cpu().numpy()

    @staticmethod
    def _predict_tta(model, tensor):
        predictions = []
        for flip_horizontal, flip_vertical in [(False, False), (True, False), (False, True), (True, True)]:
            transformed = tensor
            if flip_horizontal:
                transformed = transformed.flip(-1)
            if flip_vertical:
                transformed = transformed.flip(-2)
            predicted = model(transformed)["mask"].float()
            if flip_horizontal:
                predicted = predicted.flip(-1)
            if flip_vertical:
                predicted = predicted.flip(-2)
            predictions.append(predicted)
        return torch.stack(predictions, dim=0).mean(0)

segmenter = SegmentationService()
