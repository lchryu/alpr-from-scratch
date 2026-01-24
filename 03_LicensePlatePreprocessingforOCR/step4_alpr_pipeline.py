import os
from typing import Final
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

# ===================== CONFIG =====================
MODEL_PATH: Final[str] = "./model/best.pt"
IMAGE_PATH: Final[str] = "./img/multi_plates.png"

OUTPUT_DIR: Final[str] = "./outputs"
CROP_DIR: Final[str] = os.path.join(OUTPUT_DIR, "crops")
PRE_DIR: Final[str] = os.path.join(OUTPUT_DIR, "preprocess")
OCR_DIR: Final[str] = os.path.join(OUTPUT_DIR, "ocr")

STEPS: Final[list[str]] = [
    "01_gray",
    "02_resize",
    "03_blur",
    "04_thresh",
    "05_morph",
    "06_final"
]

# ===================== SETUP =====================
for d in [CROP_DIR, OCR_DIR]:
    os.makedirs(d, exist_ok=True)

for step in STEPS:
    os.makedirs(os.path.join(PRE_DIR, step), exist_ok=True)

# ===================== PREPROCESS =====================
def preprocess_plate(img: np.ndarray, scale: int = 2) -> dict[str, np.ndarray]:
    steps: dict[str, np.ndarray] = {}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps["01_gray"] = gray

    h, w = gray.shape
    resized = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    steps["02_resize"] = resized

    blur = cv2.GaussianBlur(resized, (5, 5), 0)
    steps["03_blur"] = blur

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    steps["04_thresh"] = thresh

    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    steps["05_morph"] = morph

    steps["06_final"] = morph
    return steps

# ===================== OCR =====================
def run_ocr(img: np.ndarray, reader: easyocr.Reader) -> tuple[str, float]:
    results = reader.readtext(img, detail=1)

    if len(results) == 0:
        return "", 0.0

    texts = []
    confs = []

    for (_, text, conf) in results:
        texts.append(text)
        confs.append(conf)

    final_text = "".join(texts)
    mean_conf = float(np.mean(confs))

    return final_text, mean_conf

# ===================== MAIN =====================
print("[INFO] Loading model...")
model = YOLO(MODEL_PATH)

print("[INFO] Loading image...")
img = cv2.imread(IMAGE_PATH)
assert img is not None, "Cannot read input image"

print("[INFO] Initializing OCR engine...")
reader = easyocr.Reader(['en'])

print("[INFO] Running detection...")
results = model(img)

plate_idx = 0

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        plate_img = img[y1:y2, x1:x2]
        if plate_img.size == 0:
            continue

        crop_path = os.path.join(CROP_DIR, f"plate_{plate_idx}.jpg")
        cv2.imwrite(crop_path, plate_img)

        steps = preprocess_plate(plate_img, scale=2)

        for step, step_img in steps.items():
            save_path = os.path.join(PRE_DIR, step, f"plate_{plate_idx}_{step}.jpg")
            cv2.imwrite(save_path, step_img)

        final_img = steps["06_final"]
        text, ocr_conf = run_ocr(final_img, reader)

        print("\n" + "=" * 60)
        print(f"[PLATE {plate_idx}]")
        print(f"Detection conf: {conf:.4f}")
        print(f"OCR text     : {text}")
        print(f"OCR conf     : {ocr_conf:.4f}")
        print("=" * 60)

        plate_idx += 1

print("\n====================== DONE ======================")
print("[INFO] Full ALPR pipeline completed.")
print("==================================================")
