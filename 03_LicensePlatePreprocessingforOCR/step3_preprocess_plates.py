import os

import cv2
import numpy as np

# ===================== CONFIG =====================
INPUT_DIR = "./outputs/crops"
OUTPUT_DIR = "./outputs/preprocess"

STEPS = [
    "01_gray",
    "02_resize",
    "03_blur",
    "04_thresh",
    "05_morph",
    "06_final"
]

# ===================== SETUP =====================
for step in STEPS:
    os.makedirs(os.path.join(OUTPUT_DIR, step), exist_ok=True)

# ===================== PREPROCESS FUNCTION =====================
def preprocess_plate(img, scale=2):
    steps = {}

    # 01 - Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps["01_gray"] = gray

    # 02 - Resize
    h, w = gray.shape
    resized = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    steps["02_resize"] = resized

    # 03 - Gaussian Blur
    blur = cv2.GaussianBlur(resized, (5, 5), 0)
    steps["03_blur"] = blur

    # 04 - Adaptive Threshold
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    steps["04_thresh"] = thresh

    # 05 - Morphology
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    steps["05_morph"] = morph

    # 06 - Final
    steps["06_final"] = morph

    return steps

# ===================== MAIN =====================
print("[INFO] Running preprocessing pipeline...")

if not os.path.exists(INPUT_DIR):
    print(f"[ERROR] Input directory '{INPUT_DIR}' does not exist!")
    print("[INFO] Run Step 2 (crop) first.")
    raise SystemExit(1)

for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(INPUT_DIR, fname)
    img = cv2.imread(img_path)

    if img is None:
        print(f"[WARN] Cannot read {fname}")
        continue

    print(f"[PROCESS] {fname}")

    results = preprocess_plate(img, scale=2)
    name = os.path.splitext(fname)[0]

    for step, result_img in results.items():
        save_path = os.path.join(OUTPUT_DIR, step, f"{name}_{step}.jpg")
        cv2.imwrite(save_path, result_img)

print("\n====================== DONE ======================")
print("[INFO] Preprocessing completed.")
print("[INFO] Check folder: outputs/preprocess")
print("==================================================")
