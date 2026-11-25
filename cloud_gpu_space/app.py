from pathlib import Path

import gradio as gr
import numpy as np
import spaces
from PIL import Image
from ultralytics import FastSAM, YOLO

ROOT = Path(__file__).parent
detector = YOLO(str(ROOT / "detector.pt"))
fastsam = FastSAM(str(ROOT / "FastSAM-s.pt"))
LABEL_CHANNELS = {"spall": 2, "rebar": 3, "fissure": 4}


@spaces.GPU(duration=120)
def predict_non_crack(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    detections = detector.predict(rgb, conf=0.15, iou=0.50, imgsz=640, device=0, verbose=False)[0]
    prompts: list[tuple[list[float], int]] = []
    if detections.boxes is not None:
        names = detections.names
        for box, cls_id in zip(detections.boxes.xyxy.cpu().numpy(), detections.boxes.cls.cpu().numpy().astype(int)):
            label = str(names.get(cls_id, cls_id)).lower()
            if "crack" in label and "fissure" not in label:
                continue
            channel = next((value for key, value in LABEL_CHANNELS.items() if key in label), None)
            if channel is not None:
                prompts.append((box.tolist(), channel))

    class_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    prompts = prompts[:10]
    if prompts:
        masks_result = fastsam(rgb, bboxes=[box for box, _ in prompts], device=0, verbose=False)[0]
        if masks_result.masks is not None:
            masks = masks_result.masks.data.cpu().numpy()
            for index, (_, channel) in enumerate(prompts):
                if index >= len(masks):
                    break
                mask = Image.fromarray((masks[index] > 0.5).astype(np.uint8) * 255).resize(
                    (rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST
                )
                class_mask[np.asarray(mask) > 0] = channel
    return Image.fromarray(class_mask, mode="L")


demo = gr.Interface(
    fn=predict_non_crack,
    inputs=gr.Image(type="pil", label="Inspection image"),
    outputs=gr.Image(type="pil", image_mode="L", format="png", label="Class mask (0 background, 2 spalling, 3 rebar, 4 fissures)"),
    api_name="predict_non_crack",
    title="CrackVision non-crack segmentation (ZeroGPU)",
    description="YOLO locates spalling, rebar, and fissures; FastSAM returns masks. Crack segmentation remains in the local CrackVision app.",
)

if __name__ == "__main__":
    demo.launch()
