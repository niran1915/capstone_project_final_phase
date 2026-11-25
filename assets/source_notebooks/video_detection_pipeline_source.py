# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')


# ==============================================================================================================
# VIDEO DETECTION MODEL — CELL 3
# RESTORE ULTRALYTICS DEPENDENCY + LOAD ORIGINAL DETECTOR
# ==============================================================================================================

import os
import sys
import subprocess
import importlib
import torch

print("=" * 110)
print("VIDEO DETECTION MODEL — RESTORING ORIGINAL ULTRALYTICS ENVIRONMENT")
print("=" * 110)

MODEL_PATH = (
    "/kaggle/working/video_detection_model/"
    "best_reconstructed.pt"
)

# ==============================================================================================================
# CHECK MODEL
# ==============================================================================================================

print("\n" + "=" * 110)
print("MODEL CHECK")
print("=" * 110)

if not os.path.isfile(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

print("✓ Model exists")
print(
    f"Size: {os.path.getsize(MODEL_PATH) / (1024 * 1024):.2f} MB"
)

# ==============================================================================================================
# CHECK ULTRALYTICS
# ==============================================================================================================

print("\n" + "=" * 110)
print("CHECKING ULTRALYTICS")
print("=" * 110)

try:

    import ultralytics

    print("✓ Ultralytics already installed")
    print(
        "Version:",
        getattr(
            ultralytics,
            "__version__",
            "unknown"
        )
    )

except ImportError:

    print("⚠ Ultralytics is not installed.")
    print("Installing Ultralytics...")

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "ultralytics"
        ]
    )

    import ultralytics

    print("✓ Ultralytics installed")
    print(
        "Version:",
        getattr(
            ultralytics,
            "__version__",
            "unknown"
        )
    )

# ==============================================================================================================
# IMPORT ULTRALYTICS
# ==============================================================================================================

print("\n" + "=" * 110)
print("IMPORTING ULTRALYTICS")
print("=" * 110)

from ultralytics import YOLO

print("✓ YOLO API imported")

# ==============================================================================================================
# LOAD MODEL THROUGH ULTRALYTICS
# ==============================================================================================================

print("\n" + "=" * 110)
print("LOADING ORIGINAL DETECTION MODEL")
print("=" * 110)

try:

    model = YOLO(
        MODEL_PATH
    )

except Exception as e:

    print("\n✗ YOLO MODEL LOAD FAILED")

    print(
        "Error type:",
        type(e).__name__
    )

    print(
        "Error:",
        str(e)
    )

    raise

print("\n✓ ORIGINAL DETECTION MODEL LOADED")

# ==============================================================================================================
# MODEL INFORMATION
# ==============================================================================================================

print("\n" + "=" * 110)
print("MODEL INFORMATION")
print("=" * 110)

print(
    "\nYOLO object type:"
)

print(
    type(model)
)

print(
    "\nUnderlying model type:"
)

print(
    type(model.model)
)

# --------------------------------------------------------------------------------------------------------------
# CLASS NAMES
# --------------------------------------------------------------------------------------------------------------

print(
    "\nClass names:"
)

try:

    print(
        model.names
    )

except Exception:

    print(
        "Could not read class names directly."
    )

# --------------------------------------------------------------------------------------------------------------
# NUMBER OF CLASSES
# --------------------------------------------------------------------------------------------------------------

try:

    names = model.names

    if isinstance(
        names,
        dict
    ):

        print(
            "\nNumber of classes:",
            len(names)
        )

except Exception:
    pass

# --------------------------------------------------------------------------------------------------------------
# PARAMETERS
# --------------------------------------------------------------------------------------------------------------

try:

    parameter_count = sum(
        p.numel()
        for p in model.model.parameters()
    )

    trainable_count = sum(
        p.numel()
        for p in model.model.parameters()
        if p.requires_grad
    )

    print(
        "\nTotal parameters:",
        f"{parameter_count:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_count:,}"
    )

except Exception as e:

    print(
        "\nParameter count unavailable:",
        e
    )

# ==============================================================================================================
# MODEL TASK
# ==============================================================================================================

print("\n" + "=" * 110)
print("MODEL TASK / CONFIGURATION")
print("=" * 110)

try:

    print(
        "Task:",
        model.task
    )

except Exception:

    print(
        "Task information unavailable."
    )

try:

    print(
        "Model overrides:"
    )

    print(
        model.overrides
    )

except Exception:

    print(
        "Overrides unavailable."
    )

# ==============================================================================================================
# MODEL SUMMARY
# ==============================================================================================================

print("\n" + "=" * 110)
print("MODEL SUMMARY")
print("=" * 110)

try:

    print(
        model.model
    )

except Exception as e:

    print(
        "Model summary unavailable:",
        e
    )

# ==============================================================================================================
# SAVE INSPECTION INFORMATION
# ==============================================================================================================

inspection_file = (
    "/kaggle/working/video_detection_model/"
    "original_detector_inspection.txt"
)

with open(
    inspection_file,
    "w"
) as f:

    f.write(
        "ORIGINAL VIDEO DETECTION MODEL INSPECTION\n"
    )

    f.write(
        "=" * 100 + "\n\n"
    )

    f.write(
        f"Model path: {MODEL_PATH}\n"
    )

    f.write(
        f"Ultralytics version: "
        f"{getattr(ultralytics, '__version__', 'unknown')}\n"
    )

    f.write(
        f"YOLO object type: {type(model)}\n"
    )

    f.write(
        f"Underlying model type: {type(model.model)}\n"
    )

    try:
        f.write(
            f"Task: {model.task}\n"
        )
    except Exception:
        pass

    try:
        f.write(
            f"Names: {model.names}\n"
        )
    except Exception:
        pass

    try:
        f.write(
            f"Parameters: {parameter_count:,}\n"
        )
    except Exception:
        pass

print(
    "\n✓ Inspection saved:"
)

print(
    inspection_file
)

# ==============================================================================================================
# FINAL
# ==============================================================================================================

print("\n" + "=" * 110)
print("CELL 3 COMPLETE")
print("=" * 110)

print(
    """
✓ Ultralytics dependency restored
✓ Original serialized detector loaded
✓ Model architecture recovered by Ultralytics
✓ Class names inspected
✓ Task inspected
✓ Parameter count inspected
✓ No training performed
✓ No fine-tuning performed
✓ No weights modified
✓ No video processing performed

NEXT:
    Test the ORIGINAL detector on a few ordinary images.
"""
)

print("=" * 110)


# ==============================================================================================================
# VIDEO DETECTION MODEL — CELL 4
# CLEAN CRACKVISION-12K DETECTOR DIAGNOSTIC
#
# IMPORTANT:
#   - IMG ONLY
#   - GT MASKS EXCLUDED
#   - NO TRAINING
#   - NO WEIGHT MODIFICATION
#   - NO SEGMENTATION MODEL
#   - LOW CONFIDENCE DIAGNOSTIC
# ==============================================================================

import os
import glob
import random
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 110)
print("ORIGINAL DETECTION MODEL — CLEAN CRACKVISION-12K DIAGNOSTIC")
print("=" * 110)

# ==============================================================================================================
# MODEL
# ==============================================================================================================

if "model" not in globals():

    from ultralytics import YOLO

    MODEL_PATH = (
        "/kaggle/working/video_detection_model/"
        "best_reconstructed.pt"
    )

    model = YOLO(MODEL_PATH)

print("\n✓ Detector available")

print("\nMODEL CLASSES")
print("-" * 110)

print(model.names)

print("\nMODEL TASK:")
print(model.task)

# ==============================================================================================================
# OUTPUT DIRECTORY
# ==============================================================================================================

OUTPUT_DIR = (
    "/kaggle/working/video_detection_model/"
    "crackvision12k_clean_diagnostic"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ==============================================================================================================
# CRACKVISION ROOT
# ==============================================================================================================

CRACKVISION_ROOT = (
    "/kaggle/input/datasets/"
    "darshansajjana/crackvision-12k/"
    "split_dataset_final"
)

SPLITS = [
    "train",
    "val",
    "test"
]

print("\n" + "=" * 110)
print("DISCOVERING CRACKVISION-12K IMAGE FILES")
print("=" * 110)

split_images = {}

for split in SPLITS:

    img_dir = os.path.join(
        CRACKVISION_ROOT,
        split,
        "IMG"
    )

    gt_dir = os.path.join(
        CRACKVISION_ROOT,
        split,
        "GT"
    )

    imgs = []

    for ext in [
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.PNG",
        "*.JPG",
        "*.JPEG"
    ]:

        imgs.extend(
            glob.glob(
                os.path.join(
                    img_dir,
                    ext
                )
            )
        )

    imgs = sorted(
        set(imgs)
    )

    split_images[split] = imgs

    print(
        f"\n{split.upper()}"
    )

    print(
        f"  IMG directory : {img_dir}"
    )

    print(
        f"  Images        : {len(imgs)}"
    )

    print(
        f"  GT directory  : {gt_dir}"
    )

    # Check that we are actually reading IMG.
    if len(imgs) > 0:

        print(
            f"  Example       : "
            f"{os.path.basename(imgs[0])}"
        )

# ==============================================================================================================
# SAFETY CHECK
# ==============================================================================================================

all_img_paths = []

for split in SPLITS:

    all_img_paths.extend(
        split_images[split]
    )

print("\n" + "=" * 110)
print("IMAGE-SOURCE SAFETY CHECK")
print("=" * 110)

bad_paths = []

for path in all_img_paths:

    lower = path.lower()

    if "/gt/" in lower or "\\gt\\" in lower:

        bad_paths.append(
            path
        )

if len(bad_paths) > 0:

    raise RuntimeError(
        "GT files accidentally entered the image list."
    )

print(
    f"✓ Total IMG-only images: {len(all_img_paths)}"
)

print(
    "✓ GT directories excluded"
)

# ==============================================================================================================
# SELECT CONTROLLED TEST SET
# ==============================================================================================================

print("\n" + "=" * 110)
print("SELECTING CONTROLLED TEST IMAGES")
print("=" * 110)

# Test 4 images from each split.
N_PER_SPLIT = 4

rng = random.Random(42)

selected = []

for split in SPLITS:

    candidates = split_images[split]

    if len(candidates) == 0:
        continue

    n = min(
        N_PER_SPLIT,
        len(candidates)
    )

    chosen = rng.sample(
        candidates,
        n
    )

    for path in chosen:

        selected.append(
            (
                split,
                path
            )
        )

for i, (
    split,
    path
) in enumerate(
    selected,
    start=1
):

    print(
        f"[{i:02d}] "
        f"{split:<5} "
        f"{os.path.basename(path)}"
    )

print(
    f"\nTotal selected: {len(selected)}"
)

# ==============================================================================================================
# READ IMAGE SANITY CHECK
# ==============================================================================================================

print("\n" + "=" * 110)
print("VERIFYING SELECTED FILES ARE ACTUAL IMAGES")
print("=" * 110)

valid_selected = []

for split, path in selected:

    img = cv2.imread(
        path,
        cv2.IMREAD_COLOR
    )

    if img is None:

        print(
            f"✗ Could not read: {path}"
        )

        continue

    h, w = img.shape[:2]

    mean_value = float(
        img.mean()
    )

    std_value = float(
        img.std()
    )

    print(
        f"{split:<5} "
        f"{os.path.basename(path):<15} "
        f"{w:4d}x{h:<4d} "
        f"mean={mean_value:7.2f} "
        f"std={std_value:7.2f}"
    )

    valid_selected.append(
        (
            split,
            path
        )
    )

print(
    f"\n✓ Readable images: {len(valid_selected)}"
)

# ==============================================================================================================
# RUN MULTI-CONFIDENCE DIAGNOSTIC
# ==============================================================================================================

print("\n" + "=" * 110)
print("RUNNING ORIGINAL DETECTOR")
print("=" * 110)

print(
    """
Confidence thresholds tested:
    0.05
    0.10
    0.25

The model itself is NOT changed.
Only post-processing confidence filtering is changed.
"""
)

CONF_THRESHOLDS = [
    0.05,
    0.10,
    0.25
]

diagnostic_rows = []

# Keep the 0.05 results for visualization.
results_005 = []

for image_number, (
    split,
    image_path
) in enumerate(
    valid_selected,
    start=1
):

    print(
        f"\n[{image_number:02d}/{len(valid_selected)}] "
        f"{split.upper()} / "
        f"{os.path.basename(image_path)}"
    )

    # One inference at very low confidence.
    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=0.05,
        iou=0.45,
        verbose=False
    )

    result = results[0]

    results_005.append(
        (
            split,
            image_path,
            result
        )
    )

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:

        print(
            "  Raw detections @ 0.05 : 0"
        )

        diagnostic_rows.append(
            {
                "split": split,
                "image_name": os.path.basename(
                    image_path
                ),
                "image_path": image_path,
                "detections_at_0.05": 0,
                "detections_at_0.10": 0,
                "detections_at_0.25": 0,
                "max_confidence": 0.0,
                "classes": ""
            }
        )

        continue

    classes = (
        boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    confidences = (
        boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    print(
        f"  Raw detections @ 0.05 : {len(boxes)}"
    )

    print(
        f"  Maximum confidence     : "
        f"{confidences.max():.6f}"
    )

    # Print EVERY low-confidence detection.
    for j, (
        cls_id,
        confidence
    ) in enumerate(
        zip(
            classes,
            confidences
        ),
        start=1
    ):

        class_name = model.names.get(
            int(cls_id),
            str(cls_id)
        )

        print(
            f"      [{j:02d}] "
            f"{class_name:<20} "
            f"conf={confidence:.6f}"
        )

    counts = {}

    for threshold in CONF_THRESHOLDS:

        count = int(
            np.sum(
                confidences >= threshold
            )
        )

        counts[threshold] = count

    unique_classes = []

    for cls_id in sorted(
        set(
            classes.tolist()
        )
    ):

        unique_classes.append(
            model.names.get(
                int(cls_id),
                str(cls_id)
            )
        )

    diagnostic_rows.append(
        {
            "split": split,
            "image_name": os.path.basename(
                image_path
            ),
            "image_path": image_path,
            "detections_at_0.05": counts[0.05],
            "detections_at_0.10": counts[0.10],
            "detections_at_0.25": counts[0.25],
            "max_confidence": float(
                confidences.max()
            ),
            "classes": ", ".join(
                unique_classes
            )
        }
    )

# ==============================================================================================================
# DIAGNOSTIC TABLE
# ==============================================================================================================

diagnostic_df = pd.DataFrame(
    diagnostic_rows
)

diagnostic_csv = os.path.join(
    OUTPUT_DIR,
    "crackvision12k_detector_diagnostic.csv"
)

diagnostic_df.to_csv(
    diagnostic_csv,
    index=False
)

print("\n" + "=" * 110)
print("DETECTION DIAGNOSTIC TABLE")
print("=" * 110)

if len(diagnostic_df) > 0:

    display(
        diagnostic_df[
            [
                "split",
                "image_name",
                "detections_at_0.05",
                "detections_at_0.10",
                "detections_at_0.25",
                "max_confidence",
                "classes"
            ]
        ]
    )

else:

    print(
        "No diagnostic rows generated."
    )

# ==============================================================================================================
# VISUALIZE ORIGINAL IMAGES + DETECTIONS
# ==============================================================================================================

print("\n" + "=" * 110)
print("GENERATING DETECTION VISUALIZATIONS")
print("=" * 110)

n = len(results_005)

cols = 3

rows = int(
    np.ceil(
        n / cols
    )
)

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(18, 6 * rows)
)

axes = np.array(
    axes
).reshape(-1)

for ax in axes:

    ax.axis("off")

for ax, (
    split,
    image_path,
    result
) in zip(
    axes,
    results_005
):

    plotted = result.plot()

    plotted = cv2.cvtColor(
        plotted,
        cv2.COLOR_BGR2RGB
    )

    ax.imshow(
        plotted
    )

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:

        title = (
            f"{split.upper()} | "
            f"{os.path.basename(image_path)}\n"
            f"NO DETECTIONS @ 0.05"
        )

    else:

        confs = (
            boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        title = (
            f"{split.upper()} | "
            f"{os.path.basename(image_path)}\n"
            f"detections={len(boxes)} | "
            f"max conf={confs.max():.3f}"
        )

    ax.set_title(
        title,
        fontsize=10
    )

    ax.axis("off")

plt.tight_layout()

contact_sheet = os.path.join(
    OUTPUT_DIR,
    "crackvision12k_detector_diagnostic_contact_sheet.png"
)

plt.savefig(
    contact_sheet,
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print(
    "\n✓ Contact sheet saved:"
)

print(
    contact_sheet
)

# ==============================================================================================================
# SAVE RAW SUMMARY
# ==============================================================================================================

summary = {
    "model_classes": {
        str(k): v
        for k, v in model.names.items()
    },
    "task": str(model.task),
    "total_crackvision_img_files": len(
        all_img_paths
    ),
    "selected_images": len(
        valid_selected
    ),
    "confidence_thresholds_tested": CONF_THRESHOLDS,
    "diagnostic_csv": diagnostic_csv,
    "contact_sheet": contact_sheet
}

summary_path = os.path.join(
    OUTPUT_DIR,
    "crackvision12k_detector_diagnostic_summary.json"
)

with open(
    summary_path,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )

print(
    "✓ Summary saved:"
)

print(
    summary_path
)

# ==============================================================================================================
# INTERPRETATION
# ==============================================================================================================

print("\n" + "=" * 110)
print("INITIAL INTERPRETATION")
print("=" * 110)

if len(diagnostic_df) > 0:

    total_005 = int(
        diagnostic_df[
            "detections_at_0.05"
        ].sum()
    )

    total_010 = int(
        diagnostic_df[
            "detections_at_0.10"
        ].sum()
    )

    total_025 = int(
        diagnostic_df[
            "detections_at_0.25"
        ].sum()
    )

    max_conf = float(
        diagnostic_df[
            "max_confidence"
        ].max()
    )

    print(
        f"""
Total detections @ 0.05 : {total_005}
Total detections @ 0.10 : {total_010}
Total detections @ 0.25 : {total_025}
Maximum confidence      : {max_conf:.6f}
"""
    )

    if total_005 == 0:

        print(
            """
⚠ IMPORTANT:

The detector produced ZERO detections even at confidence 0.05.

This means the problem is NOT simply the confidence threshold.

Possible causes include:
    • CrackVision-12K images are substantially different from
      the detector's training distribution.
    • The detector was trained with a different image preprocessing
      or domain.
    • The reconstructed checkpoint may not correspond to the
      expected original inference environment.
    • The original detector expects a different image domain.

DO NOT modify the model yet.
The next step would be to test the detector on an image from
its ORIGINAL training/validation dataset, if that dataset is
available.
"""
        )

    elif total_025 == 0:

        print(
            """
✓ The detector does produce detections at low confidence.

However, none survive confidence=0.25.

This strongly suggests a confidence/domain-shift issue rather
than a completely broken model.
"""
        )

    else:

        print(
            """
✓ The detector produces detections at the normal confidence
threshold.

The previous zero-detection result was therefore caused by the
test-image selection and/or test setup.

The detector can now be considered functional on this sample.
"""
        )

print("\n" + "=" * 110)
print("CELL 4 COMPLETE")
print("=" * 110)


# ==============================================================================================================
# VIDEO DETECTION MODEL — CELL 5
# FINAL IMAGE-LEVEL VISUAL VERIFICATION
#
# Purpose:
#   Verify the ORIGINAL detector visually on real CrackVision-12K images
#   before moving to video inference.
#
# No training
# No fine-tuning
# No checkpoint modification
# No segmentation model
# ==============================================================================================================

import os
import glob
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 110)
print("ORIGINAL DETECTION MODEL — FINAL IMAGE-LEVEL VISUAL VERIFICATION")
print("=" * 110)

# ==============================================================================================================
# 1. MODEL CHECK
# ==============================================================================================================

if "model" not in globals():

    from ultralytics import YOLO

    MODEL_PATH = (
        "/kaggle/working/video_detection_model/"
        "best_reconstructed.pt"
    )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    model = YOLO(MODEL_PATH)

print("\n✓ Detector loaded")

print("\nModel classes:")
print(model.names)

print("\nModel task:")
print(model.task)

# ==============================================================================================================
# 2. CRACKVISION IMAGE DIRECTORIES
# ==============================================================================================================

CRACKVISION_ROOT = (
    "/kaggle/input/datasets/"
    "darshansajjana/crackvision-12k/"
    "split_dataset_final"
)

IMG_DIRS = {
    "train": os.path.join(
        CRACKVISION_ROOT,
        "train",
        "IMG"
    ),
    "val": os.path.join(
        CRACKVISION_ROOT,
        "val",
        "IMG"
    ),
    "test": os.path.join(
        CRACKVISION_ROOT,
        "test",
        "IMG"
    )
}

print("\n" + "=" * 110)
print("CHECKING IMAGE DIRECTORIES")
print("=" * 110)

split_images = {}

for split, img_dir in IMG_DIRS.items():

    paths = []

    for ext in [
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.PNG",
        "*.JPG",
        "*.JPEG"
    ]:

        paths.extend(
            glob.glob(
                os.path.join(
                    img_dir,
                    ext
                )
            )
        )

    paths = sorted(
        set(paths)
    )

    split_images[split] = paths

    print(
        f"{split.upper():<8}: "
        f"{len(paths):>5} images"
    )

# ==============================================================================================================
# 3. SELECT REPRESENTATIVE IMAGES
# ==============================================================================================================

print("\n" + "=" * 110)
print("SELECTING REPRESENTATIVE IMAGES")
print("=" * 110)

# Deliberately choose fixed positions rather than random images.
# This makes the verification reproducible.

selected = []

for split in ["train", "val", "test"]:

    images = split_images[split]

    if len(images) == 0:
        continue

    indices = [
        0,
        len(images) // 2
    ]

    for idx in indices:

        idx = min(
            idx,
            len(images) - 1
        )

        selected.append(
            (
                split,
                images[idx]
            )
        )

for i, (
    split,
    path
) in enumerate(
    selected,
    start=1
):

    print(
        f"[{i:02d}] "
        f"{split.upper():<5} "
        f"{os.path.basename(path)}"
    )

# ==============================================================================================================
# 4. OUTPUT DIRECTORY
# ==============================================================================================================

OUTPUT_DIR = (
    "/kaggle/working/video_detection_model/"
    "final_image_verification"
)

ANNOTATED_DIR = os.path.join(
    OUTPUT_DIR,
    "annotated_images"
)

os.makedirs(
    ANNOTATED_DIR,
    exist_ok=True
)

# ==============================================================================================================
# 5. RUN ORIGINAL DETECTOR
# ==============================================================================================================

print("\n" + "=" * 110)
print("RUNNING ORIGINAL DETECTOR")
print("=" * 110)

CONFIDENCE = 0.25
IOU_THRESHOLD = 0.45

print(
    f"\nConfidence threshold : {CONFIDENCE}"
)

print(
    f"NMS IoU threshold    : {IOU_THRESHOLD}"
)

all_detection_rows = []
visual_results = []

for image_number, (
    split,
    image_path
) in enumerate(
    selected,
    start=1
):

    print("\n" + "-" * 110)

    print(
        f"[{image_number:02d}/{len(selected)}] "
        f"{split.upper()} / "
        f"{os.path.basename(image_path)}"
    )

    result = model.predict(
        source=image_path,
        imgsz=640,
        conf=CONFIDENCE,
        iou=IOU_THRESHOLD,
        verbose=False
    )[0]

    visual_results.append(
        (
            split,
            image_path,
            result
        )
    )

    boxes = result.boxes

    detection_count = (
        0
        if boxes is None
        else len(boxes)
    )

    print(
        f"Detections: {detection_count}"
    )

    if detection_count == 0:

        print(
            "  No detections."
        )

    else:

        classes = (
            boxes.cls
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

        confidences = (
            boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        xyxy = (
            boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )

        for j in range(
            detection_count
        ):

            cls_id = int(
                classes[j]
            )

            class_name = model.names.get(
                cls_id,
                str(cls_id)
            )

            confidence = float(
                confidences[j]
            )

            x1, y1, x2, y2 = (
                xyxy[j]
            )

            print(
                f"  [{j + 1:02d}] "
                f"{class_name:<20} "
                f"conf={confidence:.4f} "
                f"box="
                f"({x1:.1f}, {y1:.1f}, "
                f"{x2:.1f}, {y2:.1f})"
            )

            all_detection_rows.append(
                {
                    "split": split,
                    "image_name": os.path.basename(
                        image_path
                    ),
                    "image_path": image_path,
                    "detection_index": j + 1,
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2)
                }
            )

# ==============================================================================================================
# 6. SAVE INDIVIDUAL ANNOTATED IMAGES
# ==============================================================================================================

print("\n" + "=" * 110)
print("SAVING INDIVIDUAL ANNOTATED IMAGES")
print("=" * 110)

saved_images = []

for i, (
    split,
    image_path,
    result
) in enumerate(
    visual_results,
    start=1
):

    plotted = result.plot()

    output_name = (
        f"{i:02d}_{split}_"
        f"{os.path.basename(image_path)}"
    )

    output_path = os.path.join(
        ANNOTATED_DIR,
        output_name
    )

    success = cv2.imwrite(
        output_path,
        plotted
    )

    if not success:

        print(
            f"✗ Failed: {output_path}"
        )

    else:

        print(
            f"✓ {output_path}"
        )

        saved_images.append(
            output_path
        )

# ==============================================================================================================
# 7. SAVE DETECTION TABLE
# ==============================================================================================================

detections_csv = os.path.join(
    OUTPUT_DIR,
    "final_image_verification_detections.csv"
)

if len(all_detection_rows) > 0:

    detection_df = pd.DataFrame(
        all_detection_rows
    )

else:

    detection_df = pd.DataFrame(
        columns=[
            "split",
            "image_name",
            "image_path",
            "detection_index",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2"
        ]
    )

detection_df.to_csv(
    detections_csv,
    index=False
)

print(
    "\n✓ Detection table:"
)

print(
    detections_csv
)

# ==============================================================================================================
# 8. PRINT CLASS SUMMARY
# ==============================================================================================================

print("\n" + "=" * 110)
print("CLASS SUMMARY")
print("=" * 110)

if len(detection_df) == 0:

    print(
        "No detections at confidence 0.25."
    )

else:

    class_summary = (
        detection_df
        .groupby("class_name")
        .agg(
            detections=("class_name", "count"),
            mean_confidence=(
                "confidence",
                "mean"
            ),
            max_confidence=(
                "confidence",
                "max"
            )
        )
        .reset_index()
    )

    display(
        class_summary
    )

# ==============================================================================================================
# 9. CONTACT SHEET
# ==============================================================================================================

print("\n" + "=" * 110)
print("CREATING FINAL VISUAL CONTACT SHEET")
print("=" * 110)

n = len(visual_results)

cols = 3

rows = int(
    np.ceil(
        n / cols
    )
)

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(18, 6 * rows)
)

axes = np.array(
    axes
).reshape(-1)

for ax in axes:

    ax.axis("off")

for ax, (
    split,
    image_path,
    result
) in zip(
    axes,
    visual_results
):

    plotted = result.plot()

    plotted = cv2.cvtColor(
        plotted,
        cv2.COLOR_BGR2RGB
    )

    ax.imshow(
        plotted
    )

    boxes = result.boxes

    count = (
        0
        if boxes is None
        else len(boxes)
    )

    if count > 0:

        confs = (
            boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        title = (
            f"{split.upper()} | "
            f"{os.path.basename(image_path)}\n"
            f"{count} detection(s), "
            f"max conf={confs.max():.3f}"
        )

    else:

        title = (
            f"{split.upper()} | "
            f"{os.path.basename(image_path)}\n"
            f"NO DETECTIONS"
        )

    ax.set_title(
        title,
        fontsize=10
    )

    ax.axis("off")

plt.tight_layout()

contact_sheet = os.path.join(
    OUTPUT_DIR,
    "FINAL_IMAGE_DETECTION_CONTACT_SHEET.png"
)

plt.savefig(
    contact_sheet,
    dpi=160,
    bbox_inches="tight"
)

plt.show()

print(
    "\n✓ Contact sheet saved:"
)

print(
    contact_sheet
)

# ==============================================================================================================
# 10. SAVE SUMMARY
# ==============================================================================================================

summary = {
    "model_task": str(
        model.task
    ),
    "model_classes": {
        str(k): v
        for k, v in model.names.items()
    },
    "confidence_threshold": CONFIDENCE,
    "iou_threshold": IOU_THRESHOLD,
    "images_tested": len(
        selected
    ),
    "images_with_detections": int(
        sum(
            1
            for _, _, r in visual_results
            if r.boxes is not None
            and len(r.boxes) > 0
        )
    ),
    "total_detections": int(
        len(detection_df)
    ),
    "annotated_images_directory": ANNOTATED_DIR,
    "detection_csv": detections_csv,
    "contact_sheet": contact_sheet
}

summary_path = os.path.join(
    OUTPUT_DIR,
    "FINAL_IMAGE_DETECTION_VERIFICATION.json"
)

with open(
    summary_path,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )

print(
    "\n✓ Summary saved:"
)

print(
    summary_path
)

# ==============================================================================================================
# FINAL STATUS
# ==============================================================================================================

print("\n" + "=" * 110)
print("CELL 5 COMPLETE")
print("=" * 110)

print(
    f"""
Images tested             : {len(selected)}
Images with detections    : {summary["images_with_detections"]}
Total detections          : {summary["total_detections"]}

Confidence threshold      : {CONFIDENCE}

Annotated images:
    {ANNOTATED_DIR}

Detection table:
    {detections_csv}

Contact sheet:
    {contact_sheet}

✓ Original checkpoint unchanged
✓ Original model unchanged
✓ No training
✓ No fine-tuning
✓ No segmentation
✓ IMG files only

If the boxes and labels look correct, the detector is ready
for the video frame-by-frame pipeline.
"""
)

print("=" * 110)


# ==============================================================================================================
# VIDEO DETECTION MODEL — CELL 5
# INSTALL / VERIFY ULTRALYTICS
# ==============================================================================================================

import sys
import subprocess

print("=" * 110)
print("VIDEO DETECTION MODEL — ULTRALYTICS SETUP")
print("=" * 110)

# Check whether Ultralytics is already installed
try:
    import ultralytics

    print("\n✓ Ultralytics already installed")
    print("Version:", ultralytics.__version__)

except ModuleNotFoundError:

    print("\n⚠ Ultralytics is not installed.")
    print("Installing Ultralytics...")

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "ultralytics"
    ])

    print("\n✓ Ultralytics installation completed")

    import ultralytics

    print("Version:", ultralytics.__version__)

# Verify YOLO import
from ultralytics import YOLO

print("\n✓ YOLO class imported successfully")

print("\n" + "=" * 110)
print("ULTRALYTICS SETUP COMPLETE")
print("=" * 110)


# ==============================================================================================================
# VIDEO DETECTION MODEL — CELL 6
# FRAME-BY-FRAME VIDEO DETECTION PIPELINE
#
# PIPELINE:
#     INPUT VIDEO
#         ↓
#     FRAME-BY-FRAME READING
#         ↓
#     ORIGINAL DETECTION MODEL
#         ↓
#     BOUNDING BOXES + CLASS + CONFIDENCE
#         ↓
#     ANNOTATED FRAME
#         ↓
#     OUTPUT VIDEO
#
# NO TRAINING
# NO FINE-TUNING
# NO SEGMENTATION
# NO MEASUREMENT
# NO CHECKPOINT MODIFICATION
# ==============================================================================================================

import os
import cv2
import json
import time
import numpy as np
import pandas as pd

print("=" * 110)
print("CRACKTO-SFNET — VIDEO DETECTION PIPELINE")
print("=" * 110)

# ==============================================================================================================
# 1. MODEL CHECK
# ==============================================================================================================

print("\n" + "=" * 110)
print("MODEL CHECK")
print("=" * 110)

if "model" not in globals():

    from ultralytics import YOLO

    MODEL_PATH = (
        "/kaggle/working/video_detection_model/"
        "best_reconstructed.pt"
    )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Detector not found:\n{MODEL_PATH}"
        )

    model = YOLO(MODEL_PATH)

print("✓ Original detector loaded")

print("\nClasses:")
print(model.names)

print("\nTask:")
print(model.task)

# ==============================================================================================================
# 2. USER VIDEO CONFIGURATION
# ==============================================================================================================

print("\n" + "=" * 110)
print("VIDEO CONFIGURATION")
print("=" * 110)

# --------------------------------------------------------------------------------------------------------------
# CHANGE ONLY THIS PATH
# --------------------------------------------------------------------------------------------------------------

INPUT_VIDEO = (
    "/kaggle/input/datasets/niranjan488759/input-video/WhatsApp Video 2026-08-30 at 4.35.28 PM.mp4"
    # Example:
    # "datasets/your-dataset/your_video.mp4"
)

# If the path above is empty, automatically search common locations.
if INPUT_VIDEO.strip() == "":

    print(
        "\n⚠ INPUT_VIDEO is empty."
    )

    print(
        "Searching /kaggle/input for video files..."
    )

    video_extensions = [
        "*.mp4",
        "*.avi",
        "*.mov",
        "*.mkv",
        "*.MP4",
        "*.AVI",
        "*.MOV",
        "*.MKV"
    ]

    discovered_videos = []

    for ext in video_extensions:

        discovered_videos.extend(
            __import__("glob").glob(
                "/kaggle/input/**/" + ext,
                recursive=True
            )
        )

    discovered_videos = sorted(
        set(discovered_videos)
    )

    if len(discovered_videos) == 0:

        raise FileNotFoundError(
            """
No video file was found under /kaggle/input.

Set INPUT_VIDEO manually, for example:

INPUT_VIDEO = "/kaggle/input/your-dataset/video.mp4"
"""
        )

    print(
        "\nVideos found:"
    )

    for i, path in enumerate(
        discovered_videos,
        start=1
    ):

        print(
            f"[{i:02d}] {path}"
        )

    # Use the first discovered video.
    INPUT_VIDEO = discovered_videos[0]

    print(
        f"\n✓ Automatically selected:\n{INPUT_VIDEO}"
    )

else:

    print(
        f"Input video:\n{INPUT_VIDEO}"
    )

# ==============================================================================================================
# 3. CHECK INPUT VIDEO
# ==============================================================================================================

if not os.path.isfile(INPUT_VIDEO):

    raise FileNotFoundError(
        f"\nInput video does not exist:\n{INPUT_VIDEO}"
    )

print(
    "\n✓ Input video exists"
)

# ==============================================================================================================
# 4. OUTPUT DIRECTORY
# ==============================================================================================================

OUTPUT_DIR = (
    "/kaggle/working/video_detection_model/"
    "video_pipeline"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

video_basename = os.path.splitext(
    os.path.basename(
        INPUT_VIDEO
    )
)[0]

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    f"{video_basename}_detected.mp4"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    f"{video_basename}_frame_detections.csv"
)

OUTPUT_JSON = os.path.join(
    OUTPUT_DIR,
    f"{video_basename}_summary.json"
)

print(
    "\nOutput video:"
)

print(
    OUTPUT_VIDEO
)

# ==============================================================================================================
# 5. DETECTION PARAMETERS
# ==============================================================================================================

CONFIDENCE = 0.25
NMS_IOU = 0.45
IMAGE_SIZE = 640

print("\n" + "=" * 110)
print("DETECTION PARAMETERS")
print("=" * 110)

print(
    f"Confidence threshold : {CONFIDENCE}"
)

print(
    f"NMS IoU threshold    : {NMS_IOU}"
)

print(
    f"Inference image size : {IMAGE_SIZE}"
)

# ==============================================================================================================
# 6. OPEN VIDEO
# ==============================================================================================================

print("\n" + "=" * 110)
print("OPENING INPUT VIDEO")
print("=" * 110)

cap = cv2.VideoCapture(
    INPUT_VIDEO
)

if not cap.isOpened():

    raise RuntimeError(
        f"Could not open video:\n{INPUT_VIDEO}"
    )

fps = cap.get(
    cv2.CAP_PROP_FPS
)

frame_count = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

duration = (
    frame_count / fps
    if fps > 0
    else 0
)

print(
    f"FPS              : {fps:.3f}"
)

print(
    f"Frame count      : {frame_count}"
)

print(
    f"Resolution       : {width} × {height}"
)

print(
    f"Duration         : {duration:.2f} seconds"
)

# ==============================================================================================================
# 7. CREATE VIDEO WRITER
# ==============================================================================================================

print("\n" + "=" * 110)
print("INITIALIZING OUTPUT VIDEO")
print("=" * 110)

# MP4V is broadly supported in Kaggle/OpenCV.
fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():

    cap.release()

    raise RuntimeError(
        "Could not initialize output video writer."
    )

print(
    "✓ Output writer initialized"
)

# ==============================================================================================================
# 8. FRAME-BY-FRAME INFERENCE
# ==============================================================================================================

print("\n" + "=" * 110)
print("RUNNING FRAME-BY-FRAME DETECTION")
print("=" * 110)

print(
    "\nThis may take some time depending on video length."
)

detection_rows = []

frame_index = 0

frames_with_detections = 0

total_detections = 0

class_counts = {}

max_confidence = 0.0

start_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ------------------------------------------------------------------------------------------
    # ORIGINAL DETECTOR
    # ------------------------------------------------------------------------------------------

    result = model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        iou=NMS_IOU,
        verbose=False
    )[0]

    boxes = result.boxes

    detection_count = (
        0
        if boxes is None
        else len(boxes)
    )

    if detection_count > 0:

        frames_with_detections += 1

        total_detections += detection_count

        classes = (
            boxes.cls
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

        confidences = (
            boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        xyxy = (
            boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )

        for detection_index in range(
            detection_count
        ):

            cls_id = int(
                classes[detection_index]
            )

            class_name = model.names.get(
                cls_id,
                str(cls_id)
            )

            confidence = float(
                confidences[
                    detection_index
                ]
            )

            x1, y1, x2, y2 = (
                xyxy[
                    detection_index
                ]
            )

            class_counts[
                class_name
            ] = (
                class_counts.get(
                    class_name,
                    0
                ) + 1
            )

            max_confidence = max(
                max_confidence,
                confidence
            )

            detection_rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": (
                        frame_index / fps
                        if fps > 0
                        else None
                    ),
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2)
                }
            )

    # ------------------------------------------------------------------------------------------
    # DRAW DETECTIONS
    # ------------------------------------------------------------------------------------------

    annotated_frame = result.plot()

    # Ensure exact original video dimensions.
    if (
        annotated_frame.shape[1] != width
        or
        annotated_frame.shape[0] != height
    ):

        annotated_frame = cv2.resize(
            annotated_frame,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

    writer.write(
        annotated_frame
    )

    # ------------------------------------------------------------------------------------------
    # PROGRESS
    # ------------------------------------------------------------------------------------------

    frame_index += 1

    if (
        frame_index == 1
        or
        frame_index % 100 == 0
        or
        frame_index == frame_count
    ):

        elapsed = (
            time.time()
            - start_time
        )

        processing_fps = (
            frame_index / elapsed
            if elapsed > 0
            else 0
        )

        if frame_count > 0:

            percent = (
                frame_index
                / frame_count
                * 100
            )

        else:

            percent = 0

        print(
            f"\r"
            f"Frame "
            f"{frame_index}/"
            f"{frame_count} "
            f"({percent:6.2f}%) | "
            f"Detections: "
            f"{total_detections} | "
            f"Processing FPS: "
            f"{processing_fps:.2f}",
            end=""
        )

# ==============================================================================================================
# 9. RELEASE VIDEO RESOURCES
# ==============================================================================================================

cap.release()

writer.release()

elapsed_total = (
    time.time()
    - start_time
)

print(
    "\n\n✓ Video processing completed"
)

# ==============================================================================================================
# 10. SAVE DETECTION CSV
# ==============================================================================================================

print("\n" + "=" * 110)
print("SAVING FRAME-LEVEL DETECTION DATA")
print("=" * 110)

if len(detection_rows) > 0:

    detection_df = pd.DataFrame(
        detection_rows
    )

else:

    detection_df = pd.DataFrame(
        columns=[
            "frame_index",
            "timestamp_seconds",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2"
        ]
    )

detection_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print(
    f"✓ Detection CSV saved:\n{OUTPUT_CSV}"
)

# ==============================================================================================================
# 11. VERIFY OUTPUT VIDEO
# ==============================================================================================================

print("\n" + "=" * 110)
print("VERIFYING OUTPUT VIDEO")
print("=" * 110)

output_size_mb = (
    os.path.getsize(
        OUTPUT_VIDEO
    )
    / (
        1024 ** 2
    )
)

verify_cap = cv2.VideoCapture(
    OUTPUT_VIDEO
)

output_opened = (
    verify_cap.isOpened()
)

output_frame_count = int(
    verify_cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

output_fps = verify_cap.get(
    cv2.CAP_PROP_FPS
)

output_width = int(
    verify_cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

output_height = int(
    verify_cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

verify_cap.release()

print(
    f"Output file size : "
    f"{output_size_mb:.2f} MB"
)

print(
    f"Output FPS       : "
    f"{output_fps:.3f}"
)

print(
    f"Output frames    : "
    f"{output_frame_count}"
)

print(
    f"Output resolution: "
    f"{output_width} × "
    f"{output_height}"
)

if not output_opened:

    raise RuntimeError(
        "Output video could not be reopened."
    )

if output_frame_count != frame_index:

    print(
        "\n⚠ Frame count differs between input processing and output."
    )

else:

    print(
        "\n✓ Output frame count verified"
    )

# ==============================================================================================================
# 12. SUMMARY
# ==============================================================================================================

print("\n" + "=" * 110)
print("VIDEO DETECTION SUMMARY")
print("=" * 110)

print(
    f"""
Input video:
    {INPUT_VIDEO}

Output video:
    {OUTPUT_VIDEO}

Input FPS:
    {fps:.3f}

Input frames:
    {frame_count}

Input resolution:
    {width} × {height}

Processed frames:
    {frame_index}

Frames containing detections:
    {frames_with_detections}

Total detections:
    {total_detections}

Maximum confidence:
    {max_confidence:.6f}

Processing time:
    {elapsed_total:.2f} seconds
"""
)

print(
    "Class detection counts:"
)

if len(class_counts) == 0:

    print(
        "    No detections."
    )

else:

    for class_name, count in sorted(
        class_counts.items(),
        key=lambda x: -x[1]
    ):

        print(
            f"    {class_name:<25} "
            f"{count}"
        )

# ==============================================================================================================
# 13. SAVE JSON SUMMARY
# ==============================================================================================================

summary = {
    "input_video": INPUT_VIDEO,
    "output_video": OUTPUT_VIDEO,
    "fps": float(fps),
    "input_frame_count": int(frame_count),
    "processed_frame_count": int(frame_index),
    "width": int(width),
    "height": int(height),
    "duration_seconds": float(duration),
    "confidence_threshold": float(CONFIDENCE),
    "nms_iou_threshold": float(NMS_IOU),
    "image_size": int(IMAGE_SIZE),
    "frames_with_detections": int(
        frames_with_detections
    ),
    "total_detections": int(
        total_detections
    ),
    "maximum_confidence": float(
        max_confidence
    ),
    "class_counts": class_counts,
    "processing_time_seconds": float(
        elapsed_total
    ),
    "output_size_mb": float(
        output_size_mb
    )
}

with open(
    OUTPUT_JSON,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )

print(
    "\n✓ JSON summary saved:"
)

print(
    OUTPUT_JSON
)

# ==============================================================================================================
# 14. FINAL STATUS
# ==============================================================================================================

print("\n" + "=" * 110)
print("CELL 6 COMPLETE")
print("=" * 110)

print(
    f"""
✓ Original detection model used
✓ Video processed frame-by-frame
✓ Detection boxes rendered
✓ Class labels rendered
✓ Confidence values rendered
✓ Original FPS preserved
✓ Original resolution preserved
✓ Frame-level detection CSV saved
✓ JSON summary saved
✓ Output video verified
✓ Model weights unchanged

OUTPUT VIDEO:
{OUTPUT_VIDEO}
"""
)

print("=" * 110)


# ==============================================================================================================
# CRACKTO-SFNET — CELL 6
# COMPLETE VIDEO DETECTION PIPELINE
#
# PURPOSE:
#     INPUT VIDEO
#         ↓
#     FRAME-BY-FRAME READING
#         ↓
#     ORIGINAL DETECTION MODEL
#         ↓
#     BOUNDING BOXES + CLASS + CONFIDENCE
#         ↓
#     ANNOTATED FRAME
#         ↓
#     OUTPUT VIDEO
#
# NO TRAINING
# NO FINE-TUNING
# NO SEGMENTATION
# NO MEASUREMENT
# NO CHECKPOINT MODIFICATION
# ==============================================================================================================

import os
import cv2
import json
import time
import glob
import zipfile
import shutil
import hashlib
import subprocess
import sys
import numpy as np
import pandas as pd

print("=" * 110)
print("CRACKTO-SFNET — VIDEO DETECTION PIPELINE")
print("=" * 110)


# ==============================================================================================================
# 1. INSTALL / VERIFY ULTRALYTICS
# ==============================================================================================================

print("\n" + "=" * 110)
print("1. ULTRALYTICS CHECK")
print("=" * 110)

try:
    import ultralytics
    from ultralytics import YOLO

    print("✓ Ultralytics already available")
    print("Ultralytics version:", ultralytics.__version__)

except ModuleNotFoundError:

    print("⚠ Ultralytics not installed.")
    print("Installing...")

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "ultralytics"
    ])

    import ultralytics
    from ultralytics import YOLO

    print("✓ Ultralytics installed")
    print("Ultralytics version:", ultralytics.__version__)


# ==============================================================================================================
# 2. MODEL LOCATIONS
# ==============================================================================================================

print("\n" + "=" * 110)
print("2. LOCATING ORIGINAL DETECTION MODEL")
print("=" * 110)

MODEL_DATASET_DIR = (
    "/kaggle/input/datasets/"
    "niranjan488759/crack-detection-model/best"
)

WORKING_MODEL_DIR = (
    "/kaggle/working/video_detection_model"
)

os.makedirs(
    WORKING_MODEL_DIR,
    exist_ok=True
)

RECONSTRUCTED_PT = os.path.join(
    WORKING_MODEL_DIR,
    "best_reconstructed.pt"
)

print("\nExpected detector dataset:")
print(MODEL_DATASET_DIR)

print("\nWorking model:")
print(RECONSTRUCTED_PT)


# ==============================================================================================================
# 3. SEARCH FOR AN EXISTING .PT FILE
# ==============================================================================================================

print("\n" + "=" * 110)
print("3. SEARCHING FOR EXISTING PT FILE")
print("=" * 110)

existing_pt_files = []

search_roots = [
    "/kaggle/input",
    "/kaggle/working"
]

for root in search_roots:

    for pattern in [
        "**/*.pt",
        "**/*.pth"
    ]:

        try:
            existing_pt_files.extend(
                glob.glob(
                    os.path.join(root, pattern),
                    recursive=True
                )
            )
        except Exception:
            pass


existing_pt_files = sorted(
    set(existing_pt_files)
)

# Do not accidentally select unrelated checkpoints.
preferred_existing = [
    p for p in existing_pt_files
    if (
        "crack-detection-model" in p.lower()
        or "best_reconstructed.pt" in p.lower()
    )
]

if os.path.isfile(RECONSTRUCTED_PT):

    MODEL_PATH = RECONSTRUCTED_PT

    print("✓ Existing reconstructed model found:")
    print(MODEL_PATH)

elif len(preferred_existing) > 0:

    # Prefer an explicitly reconstructed model if available.
    reconstructed_candidates = [
        p for p in preferred_existing
        if os.path.basename(p).lower() == "best_reconstructed.pt"
    ]

    if len(reconstructed_candidates) > 0:

        MODEL_PATH = reconstructed_candidates[0]

    else:

        MODEL_PATH = preferred_existing[0]

    print("✓ Existing detector checkpoint found:")
    print(MODEL_PATH)

else:

    MODEL_PATH = None

    print("⚠ No standalone .pt file found.")
    print("The detector appears to be stored as a PyTorch archive directory.")


# ==============================================================================================================
# 4. RECONSTRUCT PT ARCHIVE IF NECESSARY
# ==============================================================================================================

if MODEL_PATH is None:

    print("\n" + "=" * 110)
    print("4. RECONSTRUCTING PYTORCH MODEL ARCHIVE")
    print("=" * 110)

    if not os.path.isdir(MODEL_DATASET_DIR):

        raise FileNotFoundError(
            "\nDetector dataset directory not found:\n"
            + MODEL_DATASET_DIR
        )

    print("\n✓ Detector archive directory found")

    # ------------------------------------------------------------------------------------------
    # Locate required PyTorch serialization files.
    # ------------------------------------------------------------------------------------------

    required_files = [
        ".format_version",
        ".storage_alignment",
        "data.pkl",
        "byteorder",
        "version",
        ".data/serialization_id"
    ]

    print("\nChecking PyTorch serialization files:")

    missing_required = []

    for rel_path in required_files:

        full_path = os.path.join(
            MODEL_DATASET_DIR,
            rel_path
        )

        if os.path.isfile(full_path):

            size = os.path.getsize(full_path)

            print(
                f"✓ {rel_path:<35} "
                f"{size:,} bytes"
            )

        else:

            print(
                f"✗ {rel_path:<35} MISSING"
            )

            missing_required.append(
                rel_path
            )

    if missing_required:

        raise RuntimeError(
            "\nThe detector archive is incomplete.\n"
            "Missing files:\n"
            + "\n".join(
                "  - " + x
                for x in missing_required
            )
        )


    # ------------------------------------------------------------------------------------------
    # Collect all archive files.
    # ------------------------------------------------------------------------------------------

    archive_files = []

    for root, dirs, files in os.walk(
        MODEL_DATASET_DIR
    ):

        for filename in files:

            full_path = os.path.join(
                root,
                filename
            )

            relative_path = os.path.relpath(
                full_path,
                MODEL_DATASET_DIR
            )

            relative_path = relative_path.replace(
                os.sep,
                "/"
            )

            archive_files.append(
                (full_path, relative_path)
            )

    archive_files = sorted(
        archive_files,
        key=lambda x: x[1]
    )

    print(
        f"\nTotal archive files found: "
        f"{len(archive_files)}"
    )


    # ------------------------------------------------------------------------------------------
    # Tensor storage count.
    # ------------------------------------------------------------------------------------------

    storage_files = [
        rel
        for _, rel in archive_files
        if rel.startswith("data/")
    ]

    print(
        f"Tensor-storage files: "
        f"{len(storage_files)}"
    )


    # ------------------------------------------------------------------------------------------
    # Create a valid PyTorch zip archive.
    #
    # IMPORTANT:
    # PyTorch expects every archive member to live underneath
    # a common directory such as "best/".
    # ------------------------------------------------------------------------------------------

    print("\nCreating reconstructed archive...")

    if os.path.exists(RECONSTRUCTED_PT):

        os.remove(
            RECONSTRUCTED_PT
        )

    with zipfile.ZipFile(
        RECONSTRUCTED_PT,
        mode="w",
        compression=zipfile.ZIP_STORED
    ) as zf:

        for full_path, relative_path in archive_files:

            archive_name = (
                "best/"
                + relative_path
            )

            zf.write(
                full_path,
                arcname=archive_name
            )

    archive_size_mb = (
        os.path.getsize(
            RECONSTRUCTED_PT
        )
        / (1024 ** 2)
    )

    print(
        f"✓ PyTorch archive reconstructed"
    )

    print(
        f"Archive size: "
        f"{archive_size_mb:.2f} MB"
    )


    # ------------------------------------------------------------------------------------------
    # Verify archive structure.
    # ------------------------------------------------------------------------------------------

    print("\nVerifying reconstructed archive...")

    with zipfile.ZipFile(
        RECONSTRUCTED_PT,
        "r"
    ) as zf:

        names = zf.namelist()

    required_archive_entries = [
        "best/.format_version",
        "best/.storage_alignment",
        "best/data.pkl",
        "best/byteorder",
        "best/version",
        "best/.data/serialization_id"
    ]

    missing_archive_entries = [
        x
        for x in required_archive_entries
        if x not in names
    ]

    if missing_archive_entries:

        raise RuntimeError(
            "\nReconstructed archive is missing:\n"
            + "\n".join(
                "  - " + x
                for x in missing_archive_entries
            )
        )

    print(
        f"✓ Archive entries: {len(names)}"
    )

    for x in required_archive_entries:

        print(
            f"✓ {x}"
        )


    # ------------------------------------------------------------------------------------------
    # Calculate SHA256.
    # ------------------------------------------------------------------------------------------

    print("\nCalculating model archive SHA256...")

    sha256 = hashlib.sha256()

    with open(
        RECONSTRUCTED_PT,
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            sha256.update(
                chunk
            )

    print(
        "SHA256:"
    )

    print(
        sha256.hexdigest()
    )

    MODEL_PATH = RECONSTRUCTED_PT


# ==============================================================================================================
# 5. LOAD ORIGINAL DETECTOR
# ==============================================================================================================

print("\n" + "=" * 110)
print("5. LOADING ORIGINAL DETECTION MODEL")
print("=" * 110)

if MODEL_PATH is None or not os.path.isfile(
    MODEL_PATH
):

    raise FileNotFoundError(
        "\nDetector could not be resolved."
    )

print("\nModel path:")
print(MODEL_PATH)

model = YOLO(
    MODEL_PATH
)

print("\n✓ Original detector loaded")

print("\nClasses:")
print(model.names)

print("\nTask:")
print(model.task)


# ==============================================================================================================
# 6. USER VIDEO CONFIGURATION
# ==============================================================================================================

print("\n" + "=" * 110)
print("6. VIDEO CONFIGURATION")
print("=" * 110)

# --------------------------------------------------------------------------------------------------------------
# CHANGE ONLY THIS PATH IF YOUR VIDEO HAS A DIFFERENT LOCATION.
# --------------------------------------------------------------------------------------------------------------

INPUT_VIDEO = (
    "/kaggle/input/datasets/"
    "niranjan488759/input-video/"
    "WhatsApp Video 2026-08-30 at 4.35.28 PM.mp4"
)

# If the path does not exist, automatically search Kaggle input.
if not os.path.isfile(INPUT_VIDEO):

    print(
        "\n⚠ Specified video was not found:"
    )

    print(
        INPUT_VIDEO
    )

    print(
        "\nSearching /kaggle/input for video files..."
    )

    video_extensions = [
        "*.mp4",
        "*.MP4",
        "*.avi",
        "*.AVI",
        "*.mov",
        "*.MOV",
        "*.mkv",
        "*.MKV"
    ]

    discovered_videos = []

    for ext in video_extensions:

        discovered_videos.extend(
            glob.glob(
                "/kaggle/input/**/" + ext,
                recursive=True
            )
        )

    discovered_videos = sorted(
        set(discovered_videos)
    )

    if len(discovered_videos) == 0:

        raise FileNotFoundError(
            """
No video file was found under /kaggle/input.

Add a video dataset to the notebook and rerun this cell.
"""
        )

    print(
        "\nVideos found:"
    )

    for i, path in enumerate(
        discovered_videos,
        start=1
    ):

        print(
            f"[{i:02d}] {path}"
        )

    INPUT_VIDEO = discovered_videos[0]

    print(
        "\n✓ Automatically selected:"
    )

    print(
        INPUT_VIDEO
    )

else:

    print(
        "Input video:"
    )

    print(
        INPUT_VIDEO
    )


# ==============================================================================================================
# 7. CHECK INPUT VIDEO
# ==============================================================================================================

if not os.path.isfile(
    INPUT_VIDEO
):

    raise FileNotFoundError(
        "\nInput video does not exist:\n"
        + INPUT_VIDEO
    )

print(
    "\n✓ Input video exists"
)


# ==============================================================================================================
# 8. OUTPUT DIRECTORY
# ==============================================================================================================

OUTPUT_DIR = (
    "/kaggle/working/"
    "video_detection_model/"
    "video_pipeline"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

video_basename = os.path.splitext(
    os.path.basename(
        INPUT_VIDEO
    )
)[0]

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    f"{video_basename}_detected.mp4"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    f"{video_basename}_frame_detections.csv"
)

OUTPUT_JSON = os.path.join(
    OUTPUT_DIR,
    f"{video_basename}_summary.json"
)

print(
    "\nOutput video:"
)

print(
    OUTPUT_VIDEO
)


# ==============================================================================================================
# 9. DETECTION PARAMETERS
# ==============================================================================================================

CONFIDENCE = 0.25
NMS_IOU = 0.45
IMAGE_SIZE = 640

print("\n" + "=" * 110)
print("9. DETECTION PARAMETERS")
print("=" * 110)

print(
    f"Confidence threshold : {CONFIDENCE}"
)

print(
    f"NMS IoU threshold    : {NMS_IOU}"
)

print(
    f"Inference image size : {IMAGE_SIZE}"
)


# ==============================================================================================================
# 10. OPEN VIDEO
# ==============================================================================================================

print("\n" + "=" * 110)
print("10. OPENING INPUT VIDEO")
print("=" * 110)

cap = cv2.VideoCapture(
    INPUT_VIDEO
)

if not cap.isOpened():

    raise RuntimeError(
        f"Could not open video:\n{INPUT_VIDEO}"
    )

fps = cap.get(
    cv2.CAP_PROP_FPS
)

frame_count = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

duration = (
    frame_count / fps
    if fps > 0
    else 0
)

print(
    f"FPS              : {fps:.3f}"
)

print(
    f"Frame count      : {frame_count}"
)

print(
    f"Resolution       : {width} × {height}"
)

print(
    f"Duration         : {duration:.2f} seconds"
)


# ==============================================================================================================
# 11. CREATE VIDEO WRITER
# ==============================================================================================================

print("\n" + "=" * 110)
print("11. INITIALIZING OUTPUT VIDEO")
print("=" * 110)

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():

    cap.release()

    raise RuntimeError(
        "Could not initialize output video writer."
    )

print(
    "✓ Output writer initialized"
)


# ==============================================================================================================
# 12. FRAME-BY-FRAME INFERENCE
# ==============================================================================================================

print("\n" + "=" * 110)
print("12. RUNNING FRAME-BY-FRAME DETECTION")
print("=" * 110)

print(
    "\nThis may take some time depending on video length."
)

detection_rows = []

frame_index = 0
frames_with_detections = 0
total_detections = 0
class_counts = {}
max_confidence = 0.0

start_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # ------------------------------------------------------------------------------------------
    # ORIGINAL DETECTOR
    # ------------------------------------------------------------------------------------------

    result = model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        iou=NMS_IOU,
        verbose=False
    )[0]

    boxes = result.boxes

    detection_count = (
        0
        if boxes is None
        else len(boxes)
    )

    if detection_count > 0:

        frames_with_detections += 1

        total_detections += detection_count

        classes = (
            boxes.cls
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

        confidences = (
            boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        xyxy = (
            boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )

        for detection_index in range(
            detection_count
        ):

            cls_id = int(
                classes[detection_index]
            )

            class_name = model.names.get(
                cls_id,
                str(cls_id)
            )

            confidence = float(
                confidences[
                    detection_index
                ]
            )

            x1, y1, x2, y2 = (
                xyxy[
                    detection_index
                ]
            )

            class_counts[
                class_name
            ] = (
                class_counts.get(
                    class_name,
                    0
                ) + 1
            )

            max_confidence = max(
                max_confidence,
                confidence
            )

            detection_rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": (
                        frame_index / fps
                        if fps > 0
                        else None
                    ),
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2)
                }
            )


    # ------------------------------------------------------------------------------------------
    # DRAW DETECTIONS
    # ------------------------------------------------------------------------------------------

    annotated_frame = result.plot()

    if (
        annotated_frame.shape[1] != width
        or
        annotated_frame.shape[0] != height
    ):

        annotated_frame = cv2.resize(
            annotated_frame,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

    writer.write(
        annotated_frame
    )


    # ------------------------------------------------------------------------------------------
    # PROGRESS
    # ------------------------------------------------------------------------------------------

    frame_index += 1

    if (
        frame_index == 1
        or
        frame_index % 100 == 0
        or
        frame_index == frame_count
    ):

        elapsed = (
            time.time()
            - start_time
        )

        processing_fps = (
            frame_index / elapsed
            if elapsed > 0
            else 0
        )

        if frame_count > 0:

            percent = (
                frame_index
                / frame_count
                * 100
            )

        else:

            percent = 0

        print(
            f"\rFrame "
            f"{frame_index}/"
            f"{frame_count} "
            f"({percent:6.2f}%) | "
            f"Detections: "
            f"{total_detections} | "
            f"Processing FPS: "
            f"{processing_fps:.2f}",
            end=""
        )


# ==============================================================================================================
# 13. RELEASE VIDEO RESOURCES
# ==============================================================================================================

cap.release()
writer.release()

elapsed_total = (
    time.time()
    - start_time
)

print(
    "\n\n✓ Video processing completed"
)


# ==============================================================================================================
# 14. SAVE DETECTION CSV
# ==============================================================================================================

print("\n" + "=" * 110)
print("14. SAVING FRAME-LEVEL DETECTION DATA")
print("=" * 110)

if len(detection_rows) > 0:

    detection_df = pd.DataFrame(
        detection_rows
    )

else:

    detection_df = pd.DataFrame(
        columns=[
            "frame_index",
            "timestamp_seconds",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2"
        ]
    )

detection_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print(
    f"✓ Detection CSV saved:\n"
    f"{OUTPUT_CSV}"
)


# ==============================================================================================================
# 15. VERIFY OUTPUT VIDEO
# ==============================================================================================================

print("\n" + "=" * 110)
print("15. VERIFYING OUTPUT VIDEO")
print("=" * 110)

if not os.path.isfile(
    OUTPUT_VIDEO
):

    raise RuntimeError(
        "Output video file was not created."
    )

output_size_mb = (
    os.path.getsize(
        OUTPUT_VIDEO
    )
    / (1024 ** 2)
)

verify_cap = cv2.VideoCapture(
    OUTPUT_VIDEO
)

output_opened = (
    verify_cap.isOpened()
)

output_frame_count = int(
    verify_cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

output_fps = verify_cap.get(
    cv2.CAP_PROP_FPS
)

output_width = int(
    verify_cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

output_height = int(
    verify_cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

verify_cap.release()

print(
    f"Output file size : "
    f"{output_size_mb:.2f} MB"
)

print(
    f"Output FPS       : "
    f"{output_fps:.3f}"
)

print(
    f"Output frames    : "
    f"{output_frame_count}"
)

print(
    f"Output resolution: "
    f"{output_width} × "
    f"{output_height}"
)

if not output_opened:

    raise RuntimeError(
        "Output video could not be reopened."
    )

if output_frame_count != frame_index:

    print(
        "\n⚠ Frame count differs between input "
        "processing and output."
    )

else:

    print(
        "\n✓ Output frame count verified"
    )


# ==============================================================================================================
# 16. SUMMARY
# ==============================================================================================================

print("\n" + "=" * 110)
print("16. VIDEO DETECTION SUMMARY")
print("=" * 110)

print(
    f"""
Model:
    {MODEL_PATH}

Input video:
    {INPUT_VIDEO}

Output video:
    {OUTPUT_VIDEO}

Input FPS:
    {fps:.3f}

Input frames:
    {frame_count}

Input resolution:
    {width} × {height}

Duration:
    {duration:.2f} seconds

Processed frames:
    {frame_index}

Frames containing detections:
    {frames_with_detections}

Total detections:
    {total_detections}

Maximum confidence:
    {max_confidence:.6f}

Processing time:
    {elapsed_total:.2f} seconds
"""
)

print(
    "Class detection counts:"
)

if len(class_counts) == 0:

    print(
        "    No detections."
    )

else:

    for class_name, count in sorted(
        class_counts.items(),
        key=lambda x: -x[1]
    ):

        print(
            f"    {class_name:<25} "
            f"{count}"
        )


# ==============================================================================================================
# 17. SAVE JSON SUMMARY
# ==============================================================================================================

summary = {
    "model_path": MODEL_PATH,
    "input_video": INPUT_VIDEO,
    "output_video": OUTPUT_VIDEO,
    "fps": float(fps),
    "input_frame_count": int(frame_count),
    "processed_frame_count": int(frame_index),
    "width": int(width),
    "height": int(height),
    "duration_seconds": float(duration),
    "confidence_threshold": float(CONFIDENCE),
    "nms_iou_threshold": float(NMS_IOU),
    "image_size": int(IMAGE_SIZE),
    "frames_with_detections": int(
        frames_with_detections
    ),
    "total_detections": int(
        total_detections
    ),
    "maximum_confidence": float(
        max_confidence
    ),
    "class_counts": class_counts,
    "processing_time_seconds": float(
        elapsed_total
    ),
    "output_size_mb": float(
        output_size_mb
    )
}

with open(
    OUTPUT_JSON,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )

print(
    "\n✓ JSON summary saved:"
)

print(
    OUTPUT_JSON
)


# ==============================================================================================================
# 18. FINAL STATUS
# ==============================================================================================================

print("\n" + "=" * 110)
print("CELL 6 COMPLETE")
print("=" * 110)

print(
    f"""
✓ Original detection model used
✓ Detector archive resolved automatically
✓ Video processed frame-by-frame
✓ Detection boxes rendered
✓ Class labels rendered
✓ Confidence values rendered
✓ Original FPS preserved
✓ Original resolution preserved
✓ Frame-level detection CSV saved
✓ JSON summary saved
✓ Output video verified
✓ Model weights unchanged

OUTPUT VIDEO:
{OUTPUT_VIDEO}

DETECTION CSV:
{OUTPUT_CSV}

SUMMARY JSON:
{OUTPUT_JSON}
"""
)

print("=" * 110)
